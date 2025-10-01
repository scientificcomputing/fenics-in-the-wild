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
import typing


default_data_path = os.environ.get("WILDFENICS_DATA_PATH", "")

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=Path, default=default_data_path, required=False)
parser.add_argument("--output_path", type=Path, default="brain.xdmf", required=False)
parser.add_argument("--quality", type=float, default=10, required=False)
parser.add_argument("--max_its", type=int, default=30, required=False)
parser.add_argument("--relative_edge_length", type=float, default=0.03, required=False)
parser.add_argument("--num_threads", type=int, default=5, required=False)
parser.add_argument("--epsilon", type=float, default=0.00225, required=False)
args = parser.parse_args()
folder = args.data_path
assert folder.exists(), "Could not find stl files"
org_tree = {
    "operation": "union",
    "left": "skull.ply",
    "right": {
        "operation": "union",
        "left": "parenchyma_incl_ventr.ply",
        "right": {"operation": "union", "left": "LV.ply", "right": "V34.ply"},
    },
}


def add_folder_path(folder: Path, org_tree: dict[str, typing.Any]):
    """Modify a file tree structure to include the full path to the files

    Args:
        folder: Path object pointing to the folder containing the files
        org_tree: _description_

    Returns:
        _description_
    """
    tree = org_tree.copy()
    for key, value in org_tree.items():
        if isinstance(value, dict):
            tree[key] = add_folder_path(folder, value)
        else:
            if (folder / value).is_file():
                tree[key] = str((folder / value).absolute())
    return tree


tree = add_folder_path(folder, org_tree)

tetra = wildmeshing.Tetrahedralizer(
    stop_quality=args.quality,
    # Optimal energy is 4, not smaller than 8
    # stop energy "amips" value
    max_its=args.max_its,
    edge_length_r=args.relative_edge_length,  # Global edge length
    max_threads=args.num_threads,
    epsilon=args.epsilon,  # Envelope thickness (freedom to move from surface)
)

tetra.load_csg_tree(json.dumps(tree))

# Create mesh
tetra.tetrahedralize()


point_array, cell_array, marker = tetra.get_tet_mesh()


mesh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD,
    cells=cell_array.astype(np.int64),
    x=point_array,
    e=ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
)


local_entities, local_values = dolfinx.io.gmsh.distribute_entity_data(
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
