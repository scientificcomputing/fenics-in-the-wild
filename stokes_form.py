from mpi4py import MPI
import dolfinx
import scifem
import numpy as np
import ufl
import typing
import basix.ufl
import numpy.typing as npt
from packaging.version import Version
import dolfinx.fem.petsc

subdomains = typing.Literal["SAS", "LV", "V34"]
interfaces = typing.Literal["LV_PAR", "V34_PAR", "AM_U", "AM_L", "EXTERNAL"]


def extract_submesh(mesh, entity_tag, tags: tuple[int, ...]):
    edim = entity_tag.dim
    mesh.topology.create_connectivity(edim, mesh.topology.dim)
    emap = mesh.topology.index_map(entity_tag.dim)
    marker = dolfinx.la.vector(emap)
    for tag in tags:
        marker.array[entity_tag.find(tag)] = 1
    marker.scatter_reverse(dolfinx.la.InsertMode.add)
    entities = np.flatnonzero(marker.array)

    # Extract submesh
    submesh, cell_map, vertex_map, node_map = dolfinx.mesh.create_submesh(
        mesh, edim, entities
    )

    # Transfer cell markers
    new_et, _ = scifem.transfer_meshtags_to_submesh(
        entity_tag, submesh, vertex_map, cell_map
    )
    new_et.name = entity_tag.name
    return submesh, cell_map, vertex_map, node_map, new_et


def interface(mesh, cell_tags, id_0, id_1):
    """Given to sets of cells, find the facets that are shared between them.

    Args:
        mesh: _description_
        cell_tags: _description_
        id_0: _description_
        id_1: _description_

    Returns:
        _description_
    """
    assert mesh.topology.dim == cell_tags.dim
    cell_map = mesh.topology.index_map(cell_tags.dim)

    # Find all cells on process that has cell with tag(s) id_0
    marker0 = dolfinx.la.vector(cell_map)
    marker0.array[:] = 0
    marker0.array[cell_tags.indices[np.isin(cell_tags.values, id_0)]] = 1
    marker0.scatter_reverse(dolfinx.la.InsertMode.add)
    marker0.scatter_forward()

    # Find all cells on process that has cell with tag(s) id_1
    marker1 = dolfinx.la.vector(cell_map)
    marker1.array[:] = 0
    marker1.array[cell_tags.indices[np.isin(cell_tags.values, id_1)]] = 1
    marker1.scatter_reverse(dolfinx.la.InsertMode.add)
    marker1.scatter_forward()

    # Find all facets connected to each domain
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    facets0 = dolfinx.mesh.compute_incident_entities(
        mesh.topology,
        np.flatnonzero(marker0.array),
        cell_tags.dim,
        cell_tags.dim - 1,
    )
    facets1 = dolfinx.mesh.compute_incident_entities(
        mesh.topology,
        np.flatnonzero(marker1.array),
        cell_tags.dim,
        cell_tags.dim - 1,
    )

    # Compute intersecting facets
    facet_map = mesh.topology.index_map(mesh.topology.dim - 1)
    facet_marker = dolfinx.la.vector(facet_map)
    facet_marker.array[:] = 0.0
    facet_marker.array[np.intersect1d(facets0, facets1)] = 1
    facet_marker.scatter_reverse(dolfinx.la.InsertMode.add)
    facet_marker.scatter_forward()
    return np.flatnonzero(facet_marker.array)


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
        len(interface(mesh, cell_tags, subdomain_map["LV"], subdomain_map["SAS"])) == 0
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

    inflow_facets = interface(
        mesh, cell_tags, subdomain_map["PAR"], subdomain_map["LV"]
    )
    v34_walls = interface(mesh, cell_tags, subdomain_map["PAR"], subdomain_map["V34"])

    internal_walls = interface(
        mesh, cell_tags, subdomain_map["PAR"], subdomain_map["SAS"]
    )

    facet_marker[outer_parent_facets] = interface_map["AM_L"]
    # Upper skull should always happen after lower skull
    facet_marker[parent_AM_U] = interface_map["AM_U"]

    facet_marker[internal_walls] = interface_map["PAR_SAS"]
    facet_marker[inflow_facets] = interface_map["LV_PAR"]
    facet_marker[v34_walls] = interface_map["V34_PAR"]
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


def create_inflow_function(
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
            mesh.topology.dim - 1,
        )

    interpolation_points = Q_el.basix_element.x
    fdim = domain.topology.dim - 1

    c_el = domain.ufl_domain().ufl_coordinate_element()
    ref_top = c_el.reference_topology
    ref_geom = c_el.reference_geometry
    facet_types = set(
        basix.cell.subentity_types(domain.basix_cell())[mesh.topology.dim - 1]
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
        normal_on_facet = normal_expr.eval(domain, entity)
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
    mu: float = 1.0,
    R0: float = 1e4,
    sigma: float = 1.0,
    u_in: float = 1e-4,
) -> dolfinx.fem.Function:
    element_u = basix.ufl.element(
        basix.ElementFamily.BDM,
        mesh.basix_cell(),
        2,
    )
    element_p = basix.ufl.element(
        basix.ElementFamily.P,
        mesh.basix_cell(),
        1,
        discontinuous=True,
    )
    me = basix.ufl.mixed_element([element_u, element_p])
    W = dolfinx.fem.functionspace(mesh, me)

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
    u_bc = create_inflow_function(
        V,
        inlet_expr,
        inflow_facets,
    )

    dofs = dolfinx.fem.locate_dofs_topological(
        (W.sub(0), V), mesh.topology.dim - 1, inflow_facets
    )
    bcs = [dolfinx.fem.dirichletbc(u_bc, dofs, W.sub(0))]

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
    for surface in ["V34_PAR", "PAR_SAS", "AM_L"]:
        marker = facet_map[surface]
        a += (
            -ufl.inner(ufl.dot(mu_c * ufl.grad(v), n), tangent_projection(u, n))
            * ds(marker)
            - ufl.inner(ufl.dot(mu_c * ufl.grad(u), n), tangent_projection(v, n))
            * ds(marker)
            + 2
            * mu_c
            * (sigma_c / hF)
            * ufl.inner(tangent_projection(u, n), tangent_projection(v, n))
            * ds(marker)
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
        * mu
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
    print(
        V.dofmap.index_map.size_global * V.dofmap.index_map_bs,
        mesh.topology.index_map(3).size_global,
        W.dofmap.index_map.size_global * W.dofmap.index_map_bs,
    )
    exit()
    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {
        "cffi_extra_compile_args": cffi_options,
        "cffi_libraries": ["m"],
    }
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options=petsc_options, jit_options=jit_options
    )
    wh = problem.solve()
    return wh.sub(0).collapse(), wh.sub(1).collapse()


if __name__ == "__main__":
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "test_marius.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
        ct = xdmf.read_meshtags(mesh, name="mesh_tags")

    def upper_skull(x, upper_skull_z=0.027):
        return x[2] - 0.8 * x[1] > upper_skull_z

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
    parent_ft = define_subdomain_markers(
        mesh, ct, subdomain_map, interface_map, upper_skull
    )
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "parent_facets.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(parent_ft, mesh.geometry)

    csf_mesh, cell_map, vertex_map, node_map, csf_markers = extract_submesh(
        mesh, ct, (1, 3, 4)
    )
    interface_marker, _ = scifem.transfer_meshtags_to_submesh(
        parent_ft, csf_mesh, vertex_map, cell_map
    )
    interface_marker.name = "interfaces"
    with dolfinx.io.XDMFFile(csf_mesh.comm, "csf.xdmf", "w") as xdmf:
        xdmf.write_mesh(csf_mesh)
        xdmf.write_meshtags(csf_markers, csf_mesh.geometry)
        csf_mesh.topology.create_connectivity(
            csf_mesh.topology.dim - 1, csf_mesh.topology.dim
        )
        xdmf.write_meshtags(interface_marker, csf_mesh.geometry)

    w_submesh = stokes_solver(
        csf_mesh, csf_markers, interface_marker, subdomain_map, interface_map
    )
