# # Primal single-domain formulation

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
import scifem


x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75


def lower_bound(x, i, bound, tol=1e-12):
    return x[i] >= bound - tol


def upper_bound(x, i, bound, tol=1e-12):
    return x[i] <= bound + tol


def interior_marker(x, tol=1e-12):
    return (
        lower_bound(x, 0, x_L, tol=tol)
        & lower_bound(x, 1, y_L, tol=tol)
        & upper_bound(x, 0, x_U, tol=tol)
        & upper_bound(x, 1, y_U, tol=tol)
    )


# Steps to set up submeshes and interface
M = 400
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

# Integration measures for volumes
dx = Measure("dx", domain=mesh, subdomain_data=ct)
dxI = dx(interior_marker)
dxE = dx(exterior_marker)

ordered_integration_data = scifem.mesh.compute_interface_data(
    ct, ft.find(interface_marker)
)
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
dt = dolfinx.fem.Constant(mesh, 1.0e-2)

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


P = sigma_e * inner(grad(ue), grad(ve)) * dxE
P += sigma_i * inner(grad(ui), grad(vi)) * dxI
P += inner(ui, vi) * dxI

bc_P = dolfinx.fem.dirichletbc(0.0, bc_dofs, Ve)

ui = dolfinx.fem.Function(Vi, name="ui")
ue = dolfinx.fem.Function(Ve, name="ue")

entity_maps = [interior_to_parent, exterior_to_parent]
petsc_options = {
    "ksp_type": "cg",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_rtol": 1e-12,
    "ksp_atol": 1e-12,
    "ksp_monitor": None,
    "ksp_norm_type": "preconditioned",
    "ksp_error_if_not_converged": True,
}
problem = dolfinx.fem.petsc.LinearProblem(
    extract_blocks(a),
    extract_blocks(L),
    P=extract_blocks(P),
    u=[ui, ue],
    bcs=[bc],
    petsc_options=petsc_options,
    petsc_options_prefix="primal_single_",
    entity_maps=entity_maps,
)
problem.solve()
num_iterations = problem.solver.getIterationNumber()
converged_reason = problem.solver.getConvergedReason()
PETSc.Sys.Print(f"Solver converged in: {num_iterations} with reason {converged_reason}")

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
L2_ui = np.sqrt(scifem.assemble_scalar(error_ui, entity_maps=entity_maps))
L2_ue = np.sqrt(scifem.assemble_scalar(error_ue, entity_maps=entity_maps))
PETSc.Sys.Print(f"L2(ui): {L2_ui:.2e}\n L2(ue): {L2_ue:.2e}")
