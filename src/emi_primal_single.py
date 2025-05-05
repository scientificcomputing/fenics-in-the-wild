from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from ufl import (
    inner,
    grad,
    TestFunctions,
    TrialFunctions,
    FacetNormal,
    MixedFunctionSpace,
    sin,
    pi,
    extract_blocks,
    Measure,
    SpatialCoordinate,
    cos,
    div,
)
import numpy as np
import numpy.typing as npt
import scifem
from packaging.version import Version


x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75


def interior_marker(x, tol=1e-12):
    lower_bound = lambda x, i, bound: x[i] >= bound - tol
    upper_bound = lambda x, i, bound: x[i] <= bound + tol
    return (
        lower_bound(x, 0, x_L)
        & lower_bound(x, 1, y_L)
        & upper_bound(x, 0, x_U)
        & upper_bound(x, 1, y_U)
    )


# Steps to set up submeshes and interface
M = 100
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, M, M, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)


interior_cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, interior_marker)

interior_marker = 2
exterior_marker = 3
cell_map = mesh.topology.index_map(mesh.topology.dim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
cell_marker = np.full(num_cells_local, exterior_marker, dtype=np.int32)
cell_marker[interior_cells] = interior_marker

ct = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), cell_marker
)

omega_i, interior_to_parent, _, _, _ = scifem.mesh.extract_submesh(
    mesh, ct, interior_marker
)
omega_e, exterior_to_parent, e_vertex_to_parent, _, _ = scifem.mesh.extract_submesh(
    mesh, ct, exterior_marker
)


gamma = scifem.mesh.find_interface(ct, interior_marker, exterior_marker)

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
facet_map = mesh.topology.index_map(mesh.topology.dim - 1)
num_facets_local = facet_map.size_local + facet_map.num_ghosts
facets = np.arange(num_facets_local, dtype=np.int32)
interface_marker = 4
boundary_marker = 5
marker = np.full_like(facets, -1, dtype=np.int32)
marker[gamma] = interface_marker
marker[exterior_facets] = boundary_marker
marker_filter = np.flatnonzero(marker != -1).astype(np.int32)
ft = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim - 1, marker_filter, marker[marker_filter]
)
ft.name = "interface_marker"

with dolfinx.io.XDMFFile(mesh.comm, "cell.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)

# Integration measures for volumes
dx = Measure("dx", domain=mesh, subdomain_data=ct)
dxI = dx(interior_marker)
dxE = dx(exterior_marker)


# Create integration measure for interface
# Interior marker is considered as ("+") restriction
def compute_interface_data(
    cell_tags: dolfinx.mesh.MeshTags, facet_indices: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """
    Compute interior facet integrals that are consistently ordered according to the `cell_tags`,
    such that the data `(cell0, facet_idx0, cell1, facet_idx1)` is ordered such that
    `cell_tags[cell0]`<`cell_tags[cell1]`, i.e the cell with the lowest cell marker is considered the
    "+" restriction".

    Args:
        cell_tags: MeshTags that must contain an integer marker for all cells adjacent to the `facet_indices`
        facet_indices: List of facets (local index) that are on the interface.
    Returns:
        The integration data.
    """
    # Future compatibilty check
    integration_args: tuple[int] | tuple
    if Version("0.10.0") <= Version(dolfinx.__version__):
        integration_args = ()
    else:
        fdim = cell_tags.dim - 1
        integration_args = (fdim,)
    idata = dolfinx.cpp.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.interior_facet,
        cell_tags.topology,
        facet_indices,
        *integration_args,
    )
    ordered_idata = idata.reshape(-1, 4).copy()
    switch = (
        cell_tags.values[ordered_idata[:, 0]] > cell_tags.values[ordered_idata[:, 2]]
    )
    if True in switch:
        ordered_idata[switch, :] = ordered_idata[switch][:, [2, 3, 0, 1]]
    return ordered_idata


ordered_integration_data = compute_interface_data(ct, ft.find(interface_marker))
parent_cells_plus = ordered_integration_data[:, 0]
parent_cells_minus = ordered_integration_data[:, 2]
mesh_to_interior = np.full(num_cells_local, -1, dtype=np.int32)
mesh_to_interior[interior_to_parent] = np.arange(
    len(interior_to_parent), dtype=np.int32
)
mesh_to_exterior = np.full(num_cells_local, -1, dtype=np.int32)
mesh_to_exterior[exterior_to_parent] = np.arange(
    len(exterior_to_parent), dtype=np.int32
)

entity_maps = {
    omega_i: mesh_to_interior,
    omega_e: mesh_to_exterior,
}
entity_maps[omega_i][parent_cells_minus] = entity_maps[omega_i][parent_cells_plus]
entity_maps[omega_e][parent_cells_plus] = entity_maps[omega_e][parent_cells_minus]

dGamma = Measure(
    "dS",
    domain=mesh,
    subdomain_data=[(2, ordered_integration_data.flatten())],
    subdomain_id=2,
)

element = ("Lagrange", 1)
Vi = dolfinx.fem.functionspace(omega_i, element)
Ve = dolfinx.fem.functionspace(omega_e, element)
W = MixedFunctionSpace(Vi, Ve)
vi, ve = TestFunctions(W)
ui, ue = TrialFunctions(W)

sigma_e = dolfinx.fem.Constant(omega_e, 2.0)
sigma_i = dolfinx.fem.Constant(omega_i, 1.0)
Cm = dolfinx.fem.Constant(mesh, 1.0)
dt = dolfinx.fem.Constant(mesh, 1e-8)

# Setup variational form
tr_ui = ui("+")
tr_ue = ue("-")
tr_vi = vi("+")
tr_ve = ve("-")

x, y = SpatialCoordinate(mesh)
ue_exact = sin(pi * (x + y))
ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
    pi * (y - y_L) * (y - y_U)
)

n = FacetNormal(mesh)
n_e = n("-")
Im = sigma_e * inner(grad(ue_exact), n_e)
T = Cm / dt
f = ui_exact - ue_exact - 1 / T * Im

a = sigma_e * inner(grad(ue), grad(ve)) * dxE
a += sigma_i * inner(grad(ui), grad(vi)) * dxI
a += T * (tr_ue - tr_ui) * tr_ve * dGamma
a += T * (tr_ui - tr_ue) * tr_vi * dGamma
L = T * inner(f, (tr_vi - tr_ve)) * dGamma
L -= div(sigma_e * grad(ue_exact)) * ve * dxE
L -= div(sigma_i * grad(ui_exact)) * vi * dxI
a_compiled = dolfinx.fem.form(extract_blocks(a), entity_maps=entity_maps)
L_compiled = dolfinx.fem.form(extract_blocks(L), entity_maps=entity_maps)

sub_tag, _ = scifem.mesh.transfer_meshtags_to_submesh(
    ft, omega_e, e_vertex_to_parent, exterior_to_parent
)
omega_e.topology.create_connectivity(omega_e.topology.dim - 1, omega_e.topology.dim)
bc_dofs = dolfinx.fem.locate_dofs_topological(
    Ve, omega_e.topology.dim - 1, sub_tag.find(boundary_marker)
)
u_bc = dolfinx.fem.Function(Ve)
u_bc.interpolate(lambda x: np.sin(np.pi * (x[0] + x[1])))

bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)


A = dolfinx.fem.petsc.assemble_matrix(a_compiled, kind="mpi", bcs=[bc])
A.assemble()
b = dolfinx.fem.petsc.assemble_vector(L_compiled, kind="mpi")
bcs1 = dolfinx.fem.bcs_by_block(
    dolfinx.fem.extract_function_spaces(a_compiled, 1), [bc]
)
dolfinx.fem.petsc.apply_lifting(b, a_compiled, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(L_compiled), [bc])
dolfinx.fem.petsc.set_bc(b, bcs0)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.setErrorIfNotConverged(True)

ui = dolfinx.fem.Function(Vi)
ue = dolfinx.fem.Function(Ve)
x = b.duplicate()
ksp.solve(b, x)
x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dolfinx.fem.petsc.assign(x, [ui, ue])


with dolfinx.io.VTXWriter(omega_i.comm, "uh_i.bp", [ui], engine="BP5") as bp:
    bp.write(0.0)
with dolfinx.io.VTXWriter(omega_i.comm, "uh_e.bp", [ue], engine="BP5") as bp:
    bp.write(0.0)


error_ui = dolfinx.fem.form(
    inner(ui - ui_exact, ui - ui_exact) * dxI, entity_maps=entity_maps
)
error_ue = dolfinx.fem.form(
    inner(ue - ue_exact, ue - ue_exact) * dxE, entity_maps=entity_maps
)
local_ui = dolfinx.fem.assemble_scalar(error_ui)
local_ue = dolfinx.fem.assemble_scalar(error_ue)
global_ui = np.sqrt(mesh.comm.allreduce(local_ui, op=MPI.SUM))
global_ue = np.sqrt(mesh.comm.allreduce(local_ue, op=MPI.SUM))
print(f"L2(ui): {global_ui:.2e}\n L2(ue): {global_ue:.2e}")
