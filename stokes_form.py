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
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "csf.xdmf", "r") as xdmf:
#     csf_mesh = xdmf.read_mesh()
# xdmf.write_meshtags(new_et, csf_mesh.geometry)


import basix.ufl
from scifem.xdmf import create_pointcloud

q_el = element = basix.ufl.quadrature_element(
    scheme="default", degree=2, cell=csf_mesh.basix_cell()
)
Q = dolfinx.fem.functionspace(csf_mesh, q_el)
tabs = Q.tabulate_dof_coordinates()
q = dolfinx.fem.Function(Q)


print(len(q.x.array))
import nibabel
import nibabel.affines as naff
from pathlib import Path
import time

image = nibabel.load("/root/data/mri2femii-chp2-dataset/Gonzo/mri/aseg.mgz")
nibabel.save(image, "aseg.nii")


# def read_mri_data(filename: Path, mesh: dolfinx.mesh.Mesh, tag: dolfinx.mesh.MeshTags, markers: tuple[int,...]):

data = image.get_fdata().astype(np.int32)
vox2ras = image.header.get_vox2ras_tkr()
ras2vox = np.linalg.inv(vox2ras)


# cells = new_et.indices[np.isin(new_et.values, (5,))]
# midpoints = dolfinx.mesh.compute_midpoints(csf_mesh, new_et.dim, cells)
midpoints = tabs

start = time.perf_counter()
ijk_vectorized = naff.apply_affine(ras2vox, midpoints)
# Round indices to nearest integer
ijk_rounded = np.rint(ijk_vectorized).astype("int")
cell_data = data[ijk_rounded[:, 0], ijk_rounded[:, 1], ijk_rounded[:, 2]]
end = time.perf_counter()
print("Elapsed time vectorized: ", end - start)
q.x.array[:] = cell_data
scifem.xdmf.create_pointcloud("create_pointcloud.xdmf", [q])

# cf = dolfinx.mesh.meshtags(csf_mesh, new_et.dim, cells, cell_data)
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "csf_voxel_data.xdmf", "w") as xdmf:
#     xdmf.write_mesh(csf_mesh)
#     xdmf.write_meshtags(cf, csf_mesh.geometry)
