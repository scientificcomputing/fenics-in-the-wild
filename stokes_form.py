from mpi4py import MPI
import dolfinx
import scifem
import numpy as np
import ufl
import typing
import basix.ufl
import adios4dolfinx
import numpy.typing as npt
from pathlib import Path
from packaging.version import Version
import dolfinx.fem.petsc
from time import perf_counter

subdomains = typing.Literal["SAS", "LV", "V34"]
interfaces = typing.Literal["LV_PAR", "V34_PAR", "AM_U", "AM_L", "EXTERNAL"]


def define_subdomain_markers(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dolfinx.mesh.MeshTags,
    subdomain_map: dict[subdomains, tuple[int, ...]],
    interface_map: dict[interfaces, int],
    AM_U_function: typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.bool_]],
) -> dolfinx.mesh.MeshTags:
    """Tag facets for a brain mesh.

    Given a mesh where the lateral ventricles (LV), the subarachnoid space (SAS), the 3rd and 4th ventricles (V34)
    and the parenchyma (PAR) are tagged, define the interfaces between:
    1. LV and PAR
    2. V34 and PAR
    3. PAR and SAS
    4. All exterior facets that satisfies `AM_U_functions`
    5. All other external facets

    Args:
        mesh: The mesh to tag
        cell_tags: Cell markers for the different regions
        subdomain_map: Map from each subdomain to its tags.
        interface_map: Map from each tagged interface to an index used in the facet marker
        AM_U_function: Function describing the boundary to the upper part of the skull

    Returns:
        The corresponding facet_tag
    """
    # Sanity check, LV should not touch SAS
    assert (
        len(
            scifem.mesh.find_interface(
                cell_tags, subdomain_map["LV"], subdomain_map["SAS"]
            )
        )
        == 0
    )
    for m, val in interface_map.items():
        if val <= 0:
            raise ValueError(f"Interface marker {m} must be positive")

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facet_map = mesh.topology.index_map(mesh.topology.dim - 1)

    outer_parent_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    facet_vec = dolfinx.la.vector(facet_map)
    facet_marker = facet_vec.array
    parent_AM_U = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, AM_U_function
    )

    inflow_facets = scifem.mesh.find_interface(
        cell_tags, subdomain_map["PAR"], subdomain_map["LV"]
    )
    v34_walls = scifem.mesh.find_interface(
        cell_tags, subdomain_map["PAR"], subdomain_map["V34"]
    )

    internal_walls = scifem.mesh.find_interface(
        cell_tags, subdomain_map["PAR"], subdomain_map["SAS"]
    )

    facet_marker[outer_parent_facets] = interface_map["AM_L"]
    # Upper skull should always happen after lower skull
    facet_marker[parent_AM_U] = interface_map["AM_U"]

    facet_marker[internal_walls] = interface_map["PAR_SAS"]
    facet_marker[inflow_facets] = interface_map["LV_PAR"]
    facet_marker[v34_walls] = interface_map["V34_PAR"]
    facet_vec.scatter_forward()
    facet_marker = facet_marker.astype(np.int32)
    facet_pos = facet_marker > 0
    parent_ft = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        np.flatnonzero(facet_pos),
        facet_marker[facet_pos],
    )
    parent_ft.name = "interfaces_and_boundaries"
    return parent_ft


def tangent_projection(u, n):
    return u - ufl.dot(u, n) * n


def strong_bc_bdm_function(
    Q: dolfinx.fem.FunctionSpace,
    expr: ufl.core.expr.Expr,
    facets: npt.NDArray[np.int32],
) -> dolfinx.fem.Function:
    """
    n
    """
    domain = Q.mesh
    Q_el = Q.element
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    # Compute integration entities (cell, local_facet index) for all facets
    if Version(dolfinx.__version__) > Version("0.9.0"):
        boundary_entities = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.exterior_facet, domain.topology, facets
        )
    else:
        boundary_entities = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.exterior_facet,
            domain.topology,
            facets,
            domain.topology.dim - 1,
        )

    interpolation_points = Q_el.basix_element.x
    fdim = domain.topology.dim - 1

    c_el = domain.ufl_domain().ufl_coordinate_element()
    ref_top = c_el.reference_topology
    ref_geom = c_el.reference_geometry
    facet_types = set(
        basix.cell.subentity_types(domain.basix_cell())[domain.topology.dim - 1]
    )
    assert len(facet_types) == 1, "All facets must have the same topology"

    # Pull back interpolation points from reference coordinate element to facet reference element
    facet_cmap = basix.ufl.element(
        "Lagrange",
        facet_types.pop(),
        c_el.degree,
        shape=(domain.geometry.dim,),
        dtype=np.float64,
    )
    facet_cel = dolfinx.cpp.fem.CoordinateElement_float64(facet_cmap.basix_element._e)
    reference_facet_points = None
    for i, points in enumerate(interpolation_points[fdim]):
        geom = ref_geom[ref_top[fdim][i]]
        ref_points = facet_cel.pull_back(points, geom)
        # Assert that interpolation points are all equal on all facets
        if reference_facet_points is None:
            reference_facet_points = ref_points
        else:
            assert np.allclose(reference_facet_points, ref_points)
    # Create expression for BC
    normal_expr = dolfinx.fem.Expression(expr, reference_facet_points)

    points_per_entity = [sum(ip.shape[0] for ip in ips) for ips in interpolation_points]
    offsets = np.zeros(domain.topology.dim + 2, dtype=np.int32)
    offsets[1:] = np.cumsum(points_per_entity[: domain.topology.dim + 1])
    values_per_entity = np.zeros(
        (offsets[-1], domain.geometry.dim), dtype=dolfinx.default_scalar_type
    )
    entities = boundary_entities.reshape(-1, 2)
    values = np.zeros(entities.shape[0] * offsets[-1] * domain.geometry.dim)
    for i, entity in enumerate(entities):
        insert_pos = offsets[fdim] + reference_facet_points.shape[0] * entity[1]
        # Backwards compatibility
        try:
            normal_on_facet = normal_expr.eval(domain, entity.reshape(1, 2))
        except AttributeError:
            normal_on_facet = normal_expr.eval(domain, entity)

        # NOTE: evaluate within loop to avoid large memory requirements
        values_per_entity[insert_pos : insert_pos + reference_facet_points.shape[0]] = (
            normal_on_facet.reshape(-1, domain.geometry.dim)
        )
        values[
            i * offsets[-1] * domain.geometry.dim : (i + 1)
            * offsets[-1]
            * domain.geometry.dim
        ] = values_per_entity.reshape(-1)
    qh = dolfinx.fem.Function(Q)
    qh._cpp_object.interpolate(
        values.reshape(-1, domain.geometry.dim).T.copy(), boundary_entities[::2].copy()
    )
    qh.x.scatter_forward()

    return qh


def stokes_solver(
    mesh: dolfinx.mesh.Mesh,
    subdomains: dolfinx.mesh.MeshTags,
    facet_markers: dolfinx.mesh.MeshTags,
    domain_map: dict[subdomains, tuple[int, ...]],
    facet_map: dict[interfaces, int],
    mu: float = 7e-3,
    R0: float = 0,  # 1e4,
    sigma: float = 100.0,
    u_in: float = 4.63e-7,
    degree: int = 2,
) -> dolfinx.fem.Function:
    element_u = basix.ufl.element(
        basix.ElementFamily.BDM,
        mesh.basix_cell(),
        degree,
    )
    element_p = basix.ufl.element(
        basix.ElementFamily.P,
        mesh.basix_cell(),
        degree=degree - 1,
        discontinuous=True,
    )
    me = basix.ufl.mixed_element([element_u, element_p])
    W = dolfinx.fem.functionspace(mesh, me)

    if mesh.comm.rank == 0:
        cell_map = mesh.topology.index_map(mesh.topology.dim)
        vertex_map = mesh.topology.index_map(0)
        print(
            f"Num cells: {cell_map.size_global} Num vertices: {vertex_map.size_global}",
            f"\nNum dofs: {W.dofmap.index_map.size_global * W.dofmap.index_map_bs}",
        )

    # Create inflow boundary condition
    V, _ = W.sub(0).collapse()
    n = ufl.FacetNormal(mesh)
    ds = ufl.ds(domain=mesh, subdomain_data=facet_markers)

    ventricle_surface = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1)) * ds(
        facet_map["LV_PAR"]
    )
    A_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ventricle_surface))
    A = dolfinx.fem.Constant(mesh, mesh.comm.allreduce(A_local, op=MPI.SUM))
    U_in = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(u_in))
    inlet_expr = -U_in / A * n
    inflow_facets = facet_markers.indices[
        np.isin(facet_markers.values, facet_map["LV_PAR"])
    ]
    u_bc = strong_bc_bdm_function(
        V,
        inlet_expr,
        inflow_facets,
    )
    dofs = dolfinx.fem.locate_dofs_topological(
        (W.sub(0), V), mesh.topology.dim - 1, inflow_facets
    )
    bcs = [dolfinx.fem.dirichletbc(u_bc, dofs, W.sub(0))]

    walls = facet_markers.indices[
        np.isin(
            facet_markers.values,
            [facet_map["V34_PAR"], facet_map["PAR_SAS"], facet_map["AM_L"]],
        )
    ]
    u_wall = strong_bc_bdm_function(V, dolfinx.fem.Constant(mesh, 0.0) * n, walls)
    wall_dofs = dolfinx.fem.locate_dofs_topological(
        (W.sub(0), V), mesh.topology.dim - 1, walls
    )
    bcs.append(dolfinx.fem.dirichletbc(u_wall, wall_dofs, W.sub(0)))

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    dx = ufl.dx(domain=mesh, subdomain_data=subdomains)

    mu_c = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(mu))
    R = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(R0))
    a = mu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - ufl.div(v) * p * dx
    a += -ufl.div(u) * q * dx
    # Resistance outlet condition
    dAM_U = ds(facet_map["AM_U"])
    a += R * ufl.dot(u, n) * ufl.dot(v, n) * dAM_U

    # Wall condition (slip condition)
    sigma_c = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sigma))
    hF = ufl.FacetArea(mesh)
    tangential_nonslip_markers = (
        facet_map["V34_PAR"],
        facet_map["PAR_SAS"],
        facet_map["AM_L"],
        facet_map["LV_PAR"],
    )
    a += (
        -ufl.inner(ufl.dot(mu_c * ufl.grad(v), n), tangent_projection(u, n))
        * ds(tangential_nonslip_markers)
        - ufl.inner(ufl.dot(mu_c * ufl.grad(u), n), tangent_projection(v, n))
        * ds(tangential_nonslip_markers)
        + 2
        * mu_c
        * (sigma_c / hF)
        * ufl.inner(tangent_projection(u, n), tangent_projection(v, n))
        * ds(tangential_nonslip_markers)
    )
    # Weak enforcement of tangential continuiuty
    dS = ufl.dS(domain=mesh)
    a -= (
        -ufl.inner(
            ufl.dot(ufl.avg(mu * ufl.grad(u)), n("+")),
            ufl.jump(tangent_projection(v, n)),
        )
        * dS
    )
    a -= (
        -ufl.inner(
            ufl.dot(ufl.avg(mu * ufl.grad(v)), n("+")),
            ufl.jump(tangent_projection(u, n)),
        )
        * dS
    )
    hA = ufl.avg(2.0 * ufl.Circumradius(mesh))
    a += (
        2
        * mu_c
        * (sigma_c / hA)
        * ufl.inner(
            ufl.jump(tangent_projection(u, n)), ufl.jump(tangent_projection(v, n))
        )
        * dS
    )

    f = dolfinx.fem.Constant(
        mesh, dolfinx.default_scalar_type(np.zeros(mesh.geometry.dim))
    )
    L = ufl.inner(f, v) * dx

    # Create inflow bcs, enforced strongly
    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {
        "cffi_extra_compile_args": cffi_options,
        "cffi_libraries": ["m"],
    }
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": 100,
        "mat_mumps_icntl_24": 1,
        "mat_mumps_icntl_4": 2,
        # "mat_mumps_icntl_25": 1,
    }
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options=petsc_options, jit_options=jit_options
    )
    tic = perf_counter()
    wh = problem.solve()
    toc = perf_counter()
    if mesh.comm.rank == 0:
        print(f"Solving took: {toc - tic:.2e} seconds")
    return wh.sub(0).collapse(), wh.sub(1).collapse()


subdomain_map = {
    "PAR": (2,),
    "SAS": (1,),
    "LV": (3,),
    "V34": (4,),
}
interface_map = {
    "LV_PAR": 1,
    "V34_PAR": 2,
    "PAR_SAS": 5,
    "AM_U": 3,
    "AM_L": 4,
}


def compute_subdomain_exterior_cells(
    mesh: dolfinx.mesh.Mesh, ct: dolfinx.mesh.MeshTags, markers: tuple[int, ...]
):
    """Compute the exterior boundary of a set of subdomains.

    Args:
        mesh: Mesh to extract subdomains from
        ct: MeshTags object marking subdomains
        markers: The tags making up the "new" mesh
    Returns:
        Cells which has a facet on the exterior boundary of the subdomains.
    """
    # Find facets that are considered exterior
    subdomain_exterior_facets = scifem.mesh.compute_subdomain_exterior_facets(
        mesh, ct, markers
    )
    tdim = mesh.topology.dim
    assert ct.dim == tdim
    sub_cells = dolfinx.mesh.compute_incident_entities(
        mesh.topology,
        subdomain_exterior_facets,
        tdim - 1,
        tdim,
    )
    full_subdomain = ct.indices[
        np.isin(ct.values, np.asarray(markers, dtype=ct.values.dtype))
    ]
    cell_map = mesh.topology.index_map(tdim)
    return scifem.mesh.reverse_mark_entities(
        cell_map, np.intersect1d(full_subdomain, sub_cells)
    )


if __name__ == "__main__":
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "test_marius.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(ghost_mode=dolfinx.mesh.GhostMode.none)
        ct = xdmf.read_meshtags(mesh, name="mesh_tags")

    fluid_domains = subdomain_map["LV"] + subdomain_map["SAS"] + subdomain_map["V34"]

    # Refine parent mesh within ventricles
    num_refinements = 2
    if num_refinements == 0:
        refined_mesh = mesh
        refined_ct = ct
    for i in range(num_refinements):
        # Refine parent mesh within ventricles
        refine_cells = ct.indices[
            np.isin(
                ct.values,
                np.asarray(subdomain_map["V34"] + subdomain_map["LV"]),
            )
        ]

        # Find all cells associated with outer boundary (dura) and refine the cells they correspond to
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        fmap = mesh.topology.index_map(mesh.topology.dim - 1)
        exterior_facet_indices = scifem.mesh.reverse_mark_entities(
            fmap, dolfinx.mesh.exterior_facet_indices(mesh.topology)
        )
        boundary_cells = dolfinx.mesh.compute_incident_entities(
            mesh.topology,
            exterior_facet_indices,
            mesh.topology.dim - 1,
            mesh.topology.dim,
        )

        fluid_boundary_cells = compute_subdomain_exterior_cells(mesh, ct, fluid_domains)

        # For any further refinement, only refine the boundary of the fluid domains, not the interior
        if i < 1:
            cells_to_refine = np.unique(
                np.hstack([boundary_cells, fluid_boundary_cells, refine_cells])
            ).astype(np.int32)

        else:
            cells_to_refine = refine_cells

        edges_to_refine = dolfinx.mesh.compute_incident_entities(
            mesh.topology, cells_to_refine, mesh.topology.dim, 1
        )
        edge_map = mesh.topology.index_map(1)
        edges_to_refine = scifem.mesh.reverse_mark_entities(edge_map, edges_to_refine)
        refined_mesh, parent_cell, _ = dolfinx.mesh.refine(
            mesh,
            edges_to_refine,
            partitioner=None,
            option=dolfinx.mesh.RefinementOption.parent_cell,
        )
        refined_ct = dolfinx.mesh.transfer_meshtag(ct, refined_mesh, parent_cell)
        mesh = refined_mesh
        ct = refined_ct

    def upper_skull(x, upper_skull_z=0.027):
        return x[2] - 0.8 * x[1] > upper_skull_z

    parent_ft = define_subdomain_markers(
        refined_mesh, refined_ct, subdomain_map, interface_map, upper_skull
    )
    parent_ft.name = "interfaces_and_boundaries"
    comm = refined_mesh.comm
    with dolfinx.io.XDMFFile(comm, "refined.xdmf", "w") as xdmf:
        xdmf.write_mesh(refined_mesh)
        xdmf.write_meshtags(refined_ct, refined_mesh.geometry)
        xdmf.write_meshtags(parent_ft, refined_mesh.geometry)
    del refined_ct, parent_ft, refined_mesh
    comm.Barrier()

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "refined.xdmf", "r") as xdmf:
        refined_mesh = xdmf.read_mesh(ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
        refined_ct = xdmf.read_meshtags(refined_mesh, name="mesh_tags")
        refined_mesh.topology.create_connectivity(
            refined_mesh.topology.dim - 1, refined_mesh.topology.dim
        )
        refined_ft = xdmf.read_meshtags(refined_mesh, name="interfaces_and_boundaries")

    csf_mesh, cell_map, vertex_map, node_map, csf_markers = scifem.mesh.extract_submesh(
        refined_mesh,
        refined_ct,
        fluid_domains,
    )

    interface_marker, _ = scifem.transfer_meshtags_to_submesh(
        refined_ft, csf_mesh, vertex_map, cell_map
    )
    interface_marker.name = "interfaces"
    with dolfinx.io.XDMFFile(csf_mesh.comm, "csf.xdmf", "w") as xdmf:
        xdmf.write_mesh(csf_mesh)
        xdmf.write_meshtags(csf_markers, csf_mesh.geometry)
        csf_mesh.topology.create_connectivity(
            csf_mesh.topology.dim - 1, csf_mesh.topology.dim
        )
        xdmf.write_meshtags(interface_marker, csf_mesh.geometry)

    # --------------------- SIMPLE test setup ----------------------
    # N = 10
    # csf_mesh = dolfinx.mesh.create_unit_cube(
    #     MPI.COMM_WORLD,
    #     N,
    #     N,
    #     N,
    #     dolfinx.cpp.mesh.CellType.tetrahedron,
    #     ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    # )
    # cell_map = csf_mesh.topology.index_map(csf_mesh.topology.dim)
    # num_cells = cell_map.size_local + cell_map.num_ghosts
    # csf_markers = dolfinx.mesh.meshtags(
    #     csf_mesh,
    #     csf_mesh.topology.dim,
    #     np.arange(num_cells, dtype=np.int32),
    #     np.full(num_cells, subdomain_map["SAS"], dtype=np.int32),
    # )

    # csf_mesh.topology.create_connectivity(
    #     csf_mesh.topology.dim - 1, csf_mesh.topology.dim
    # )
    # facet_map = csf_mesh.topology.index_map(csf_mesh.topology.dim - 1)
    # num_facets = facet_map.size_local + facet_map.num_ghosts
    # values = np.full(num_facets, -1, dtype=np.int32)

    # tb = dolfinx.mesh.locate_entities_boundary(
    #     csf_mesh,
    #     csf_mesh.topology.dim - 1,
    #     lambda x: np.isclose(x[1], 0)
    #     | np.isclose(x[1], 1)
    #     | np.isclose(x[2], 1)
    #     | np.isclose(x[2], 0),
    # )
    # values[tb] = interface_map["PAR_SAS"]
    # lft = dolfinx.mesh.locate_entities_boundary(
    #     csf_mesh,
    #     csf_mesh.topology.dim - 1,
    #     lambda x: np.isclose(x[0], 1),
    # )
    # values[lft] = interface_map["AM_U"]
    # rft = dolfinx.mesh.locate_entities_boundary(
    #     csf_mesh,
    #     csf_mesh.topology.dim - 1,
    #     lambda x: np.isclose(x[0], 0),
    # )
    # values[rft] = interface_map["LV_PAR"]
    # indices = np.flatnonzero(values >= 0)
    # values = values[indices]
    # interface_marker = dolfinx.mesh.meshtags(
    #     csf_mesh, csf_mesh.topology.dim - 1, indices, values
    # )
    degree = 1
    uh, ph = stokes_solver(
        csf_mesh,
        csf_markers,
        interface_marker,
        subdomain_map,
        interface_map,
        mu=0.7e-3,
        u_in=4.63e-9,
        R0=1e4,
        degree=degree,
    )

    # For visualization of fluid flow within fluid cavities
    V_out = dolfinx.fem.functionspace(
        csf_mesh, ("DG", degree, (csf_mesh.geometry.dim,))
    )
    u_out = dolfinx.fem.Function(V_out, name="Velocity")
    u_out.interpolate(uh)
    with dolfinx.io.VTXWriter(csf_mesh.comm, "uh.bp", [u_out], engine="BP4") as bp:
        bp.write(0.0)

    # Map solution back onto the parent grid
    child_cells = np.arange(len(cell_map), dtype=np.int32)
    V_full = dolfinx.fem.functionspace(refined_mesh, uh.function_space.ufl_element())
    u_full = dolfinx.fem.Function(V_full, name="u")
    u_full.interpolate(uh, cells0=child_cells, cells1=cell_map)

    # Store solution and tags in checkpoint
    checkpoint_file = Path("checkpoint.bp")
    adios4dolfinx.write_mesh(checkpoint_file, refined_mesh)
    adios4dolfinx.write_meshtags(checkpoint_file, refined_mesh, refined_ct)
    adios4dolfinx.write_meshtags(checkpoint_file, refined_mesh, refined_ft)
    adios4dolfinx.write_function(checkpoint_file, u_full)
    adios4dolfinx.write_attributes(
        checkpoint_file, refined_mesh.comm, "cell_map", subdomain_map
    )
    adios4dolfinx.write_attributes(
        checkpoint_file, refined_mesh.comm, "facet_map", interface_map
    )
