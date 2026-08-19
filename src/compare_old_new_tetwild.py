"""
Script to compare the performance of tetwild and wildmeshing on .ply files.
For information about input arguments, run:
    python src/compare_old_new_tetwild.py --help
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista
import simwild.simwild as wm
from pytetwild import tetrahedralize_pv

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Compare the performance of tetwild and wildmeshing on .ply files.",
)
parser.add_argument(
    "--max_threads", "-m", help="max number of threads", type=int, default=1
)
parser.add_argument("--eps", "-e", help="epsilon value", type=float, default=0.0005)
parser.add_argument(
    "--stop_quality", "-s", help="stop quality value", type=float, default=25.0
)
parser.add_argument(
    "--input",
    "-i",
    help="Path to a single .ply file OR a folder containing .ply files",
    type=Path,
    default=Path("results/test/surfaces/"),
)
parser.add_argument(
    "--result_dir",
    "-r",
    help="Path to the directory where results will be stored",
    type=Path,
    default=Path("results/comparison/"),
)


def run_legacy(
    ply_file: Path, max_threads: int, eps: float, stop_quality: float, result_dir: Path
):
    start_ftetwild = time.perf_counter()
    ref_in = pyvista.read(ply_file)
    mesh = tetrahedralize_pv(
        ref_in,
        epsilon=eps,
        edge_length_fac=eps * 50,
        coarsen=False,
        stop_energy=stop_quality,
        num_threads=max_threads,
        simplify=False,
        quiet=False,
    )
    end_ftetwild = time.perf_counter()
    if "marker" in mesh.cell_data:
        mesh.cell_data["gmsh:physical"] = np.asarray(
            mesh.cell_data["marker"], dtype=int
        )
    meshio_mesh = pyvista.to_meshio(mesh)
    om = create_mesh(meshio_mesh, "tetra")
    meshio.write(result_dir / "reference_meshio.msh", om, file_format="gmsh")
    return end_ftetwild - start_ftetwild


def plot_results(result_dir: Path, legacy_time: float, new_time: float):
    # Compute minimal dihedral angle for both meshes and compare them
    reference_quality = compute_minimal_dihedral_angle(
        result_dir / "reference_meshio.msh"
    )
    new_quality = compute_minimal_dihedral_angle(result_dir / "new_final.msh")
    # --- Modified Plotting Section ---
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Quality Histogram
    # Added alpha=0.7 to handle overlapping distributions better
    ax1.hist(
        reference_quality,
        bins=50,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        label="Reference",
    )
    ax1.hist(
        new_quality,
        bins=50,
        color="salmon",
        edgecolor="black",
        alpha=0.7,
        label="New",
    )
    ax1.legend()
    ax1.set_title("Minimum Dihedral Angle Distribution")
    ax1.set_xlabel("Minimum Dihedral Angle (Degrees)")
    ax1.set_ylabel("Number of Tetrahedra")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Subplot 2: Runtime Bar Chart
    bars = ax2.bar(
        ["Reference\n(ftetwild)", "New\n(wildmeshing)"],
        [legacy_time, new_time],
        color=["skyblue", "salmon"],
        edgecolor="black",
    )
    ax2.set_title("Runtime Comparison")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add text labels on top of the bars with the exact time
    ref_cells = len(reference_quality)
    new_cells = len(new_quality)
    cell_counts = [ref_cells, new_cells]
    for bar, count in zip(bars, cell_counts):
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.2f} s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval / 2,
            f"{count}\ncells",
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
        )
    plt.tight_layout()
    # Saved under the same file name, or you can update it to "comparison_results.png"
    plt.savefig(result_dir / "min_dihedral_angle_distribution.png")
    plt.close()


def compute_minimal_dihedral_angle(mesh_path: Path):
    mesh = pyvista.read(mesh_path)

    if 10 in mesh.celltypes:
        mesh = mesh.extract_cells(mesh.celltypes == 10)
    quality_mesh = mesh.cell_quality(quality_measure="min_angle")

    return quality_mesh.cell_data["min_angle"]


def create_mesh(mesh, cell_type):
    cells = mesh.get_cells_type(cell_type)
    cell_data = None
    if "gmsh:physical" in mesh.cell_data:
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points
    out_mesh = meshio.Mesh(
        points=points,
        cells={cell_type: cells},
        cell_data=None if cell_data is None else {"gmsh:physical": [cell_data]},
    )
    return out_mesh


if __name__ == "__main__":
    args = parser.parse_args()
    max_threads = args.max_threads
    eps = args.eps
    stop_quality = args.stop_quality
    input_path = args.input
    result_dir = args.result_dir

    if input_path.is_file():
        # Ensure it's actually a .ply file
        if input_path.suffix == ".ply":
            ply_files = [input_path]
        else:
            raise ValueError(f"Provided file '{input_path}' is not a .ply file.")
    elif input_path.is_dir():
        # Grab all .ply files in the directory
        ply_files = list(input_path.glob("*.ply"))
    else:
        raise FileNotFoundError(f"The path '{input_path}' does not exist.")

    results = {}
    for ply_file in ply_files:
        # Copy input .ply file to the result directory for reference
        in_name = Path(ply_file).absolute().stem
        result_dir = Path(f"comparison/{in_name}_{max_threads}_{eps}_{stop_quality}")
        result_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ply_file, result_dir / ply_file.name)

        legacy_time = run_legacy(ply_file, max_threads, eps, stop_quality, result_dir)

        # Run wildmeshing
        start_wm = time.perf_counter()
        j = {
            "application": "tetwild",
            "input": [str(ply_file)],
            "output": (result_dir / "new").absolute().as_posix(),
            "eps_rel": eps,
            "length_rel": eps * 50,
            "stop_energy": stop_quality,
            "coarsen_pass": False,
            "num_threads": max_threads,
            "write_vtu": False,
            "filter": "input",
            "skip_simplify": True,
        }
        wm.wildmeshing(j)
        end_wm = time.perf_counter()

        new_time = end_wm - start_wm

        print(f"Num threads: {max_threads}")
        print(f"tetwild time: {legacy_time:.2e} s")
        print(f"wildmeshing time: {new_time:.2e} s")
        results[f"{ply_file.stem}"] = {
            "max_threads": max_threads,
            "eps": eps,
            "stop_quality": stop_quality,
            "legacy_time": legacy_time,
            "new_time": new_time,
        }

        plot_results(result_dir, legacy_time, new_time)

    with open(result_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
