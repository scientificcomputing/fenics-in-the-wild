import argparse
from mpi4py import MPI
import wildmeshing
import dolfinx
from pathlib import Path
import json
import numpy as np
import ufl
import basix.ufl
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", type=Path, default=os.environ["WILDFENICS_DATA_PATH"], required=False
)
parser.add_argument("--output_path", type=Path, default="brain.xdmf", required=False)
parser.add_argument("--quality", type=float, default=10, required=False)
parser.add_argument("--max_its", type=int, default=30, required=False)
parser.add_argument("--relative_edge_length", type=float, default=0.01, required=False)
parser.add_argument("--num_threads", type=int, default=3, required=False)
parser.add_argument("--epsilon", type=float, default=2.5e-4, required=False)
args = parser.parse_args()
folder = args.data_path
assert folder.exists(), "Could not find stl files"
tree = {
    "operation": "union",
    "left": str((folder / "final-dura.stl").absolute().as_posix()),
    "right": {
        "operation": "union",
        "left": {
            "operation": "union",
            "left": str((folder / "final-lhpial.stl").absolute().as_posix()),
            "right": str((folder / "final-rhpial.stl").absolute().as_posix()),
        },
        "right": {
            "operation": "union",
            "left": str((folder / "final-white.stl").absolute().as_posix()),
            "right": str((folder / "final-ventricles.stl").absolute().as_posix()),
        },
    },
}

tetra = wildmeshing.Tetrahedralizer(
    stop_quality=args.quality,
    max_its=args.max_its,
    edge_length_r=args.relative_edge_length,
    max_threads=args.num_threads,
    epsilon=args.epsilon,
)
tetra.load_csg_tree(json.dumps(tree))

# Create mesh
tetra.tetrahedralize()


point_array, cell_array, marker = tetra.get_tet_mesh()


mesh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD,
    cell_array.astype(np.int64),
    point_array,
    ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
)


local_entities, local_values = dolfinx.io.gmshio.distribute_entity_data(
    mesh,
    mesh.topology.dim,
    cell_array.astype(np.int64),
    marker.flatten().astype(np.int32),
)
adj = dolfinx.graph.adjacencylist(local_entities)
ct = dolfinx.mesh.meshtags_from_entities(
    mesh,
    mesh.topology.dim,
    adj,
    local_values.astype(np.int32, copy=False),
)

with dolfinx.io.XDMFFile(mesh.comm, args.output_path.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)
