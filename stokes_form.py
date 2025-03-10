from mpi4py import MPI
import dolfinx
import scifem
import numpy as np
import ufl
import typing
import basix.ufl
import numpy.typing as npt

subdomains = typing.Literal["SAS", "LV", "V34"]
interfaces = typing.Literal["LV_PAR", "V34_PAR", "UPPER_SKULL", "EXTERNAL"]


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
    upper_skull_function: typing.Callable[
        [npt.NDArray[np.floating]], npt.NDArray[np.bool_]
    ],
) -> dolfinx.mesh.MeshTags:
    """Tag facets for a brain mesh.

    Given a mesh where the lateral ventricles (LV), the subarachnoid space (SAS), the 3rd and 4th ventricles (V34)
    and the parenchyma (PAR) are tagged, define the interfaces between:
    1. LV and PAR
    2. V34 and PAR
    3. PAR and SAS
    4. All exterior facets that satisfies `upper_skull_functions`
    5. All other external facets

    Args:
        mesh: The mesh to tag
        cell_tags: Cell markers for the different regions
        subdomain_map: Map from each subdomain to its tags.
        interface_map: Map from each tagged interface to an index used in the facet marker
        upper_skull_function: Function describing the boundary to the upper part of the skull

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
    parent_upper_skull = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, upper_skull_function
    )

    inflow_facets = interface(
        mesh, cell_tags, subdomain_map["PAR"], subdomain_map["LV"]
    )
    v34_walls = interface(mesh, cell_tags, subdomain_map["PAR"], subdomain_map["V34"])

    internal_walls = interface(
        mesh, cell_tags, subdomain_map["PAR"], subdomain_map["SAS"]
    )

    facet_marker[outer_parent_facets] = interface_map["LOWER_SKULL"]
    # Upper skull should always happen after lower skull
    facet_marker[parent_upper_skull] = interface_map["UPPER_SKULL"]

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


def stokes_solver(
    mesh: dolfinx.mesh.Mesh,
    subdomains: dolfinx.mesh.MeshTags,
    facet_markers: dolfinx.mesh.MeshTags,
    domain_map: dict[subdomains, tuple[int, ...]],
    facet_map: dict[interfaces, int],
    mu: float = 1.0,
    R0: float = 1e4,
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
    )
    me = basix.ufl.mixed_element([element_u, element_p])
    W = dolfinx.fem.functionspace(mesh, me)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    mu_c = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(mu))
    R = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(R0))
    a = mu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.div(u) * q * ufl.dx
    # a +=
    pass


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
        "UPPER_SKULL": 3,
        "LOWER_SKULL": 4,
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
