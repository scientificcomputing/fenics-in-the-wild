from mpi4py import MPI
import dolfinx
import scifem
import numpy as np


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


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "brain.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="mesh_tags")

csf_mesh, cell_map, vertex_map, node_map, new_et = extract_submesh(mesh, ct, (1, 5))
with dolfinx.io.XDMFFile(csf_mesh.comm, "csf.xdmf", "w") as xdmf:
    xdmf.write_mesh(csf_mesh)
    xdmf.write_meshtags(new_et, csf_mesh.geometry)
