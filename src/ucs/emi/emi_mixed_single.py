# # Mixed single-domain formulation

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
    inv,
    dot,
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

omega_i, interior_to_parent, _, _, _ = scifem.extract_submesh(mesh, ct, interior_marker)
omega_e, exterior_to_parent, e_vertex_to_parent, _, _ = scifem.extract_submesh(
    mesh, ct, exterior_marker
)


gamma = scifem.find_interface(ct, interior_marker, exterior_marker)

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

ordered_integration_data = scifem.compute_interface_data(ct, ft.find(interface_marker))

# Integration measures for volumes
dx = Measure("dx", domain=mesh, subdomain_data=ct)
dGamma = Measure(
    "dS",
    domain=mesh,
    subdomain_data=[(2, ordered_integration_data.flatten())],
    subdomain_id=2,
)
ds = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=boundary_marker)

V = dolfinx.fem.functionspace(mesh, ("DG", 0))
S = dolfinx.fem.functionspace(mesh, ("RT", 1))
W = MixedFunctionSpace(V, S)
sigma = dolfinx.fem.Function(V)

sigma_e = 2.0
sigma_i = 1.0
sigma.interpolate(
    lambda x: np.full(x.shape[1], sigma_e), cells0=ct.find(exterior_marker)
)
sigma.interpolate(
    lambda x: np.full(x.shape[1], sigma_i), cells0=ct.find(interior_marker)
)
sigma.x.scatter_forward()

u, J = TrialFunctions(W)
q, tau = TestFunctions(W)
Cm = dolfinx.fem.Constant(mesh, 1.0)
dt = dolfinx.fem.Constant(mesh, 1e-4)
T = Cm / dt

n = FacetNormal(mesh)
n_i = n("+")
n_e = n("-")
a = inner(inv(sigma) * J, tau) * dx
a += inv(T) * dot(J("+"), n_i) * dot(tau("+"), n_i) * dGamma
a -= u * div(tau) * dx
a -= q * div(J) * dx

x, y = SpatialCoordinate(mesh)
ue_exact = sin(pi * (x + y))

ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
    pi * (y - y_L) * (y - y_U)
)


Im_exact = sigma_e * inner(grad(ue_exact), n_e)
f = ui_exact - ue_exact - 1 / T * Im_exact
dxE = dx(exterior_marker)
dxI = dx(interior_marker)
L = -f * inner(tau("+"), n_i) * dGamma
L -= -div(sigma_e * grad(ue_exact)) * q * dxE
L -= -div(sigma_i * grad(ui_exact)) * q * dxI
L -= ue_exact * dot(tau, n) * ds

P = inv(sigma) * inner(J, tau) * dx
P += div(J) * div(tau) * dx
P += inv(T) * dot(J("+"), n_i) * dot(tau("+"), n_i) * dGamma
P += inner(u, q) * dx

petsc_options = {
    "ksp_type": "minres",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
    "ksp_monitor": None,
    "ksp_rtol": 1e-12,
    "ksp_atol": 1e-12,
    "ksp_norm_type": "preconditioned",
}
u = dolfinx.fem.Function(V)
tau = dolfinx.fem.Function(S)
problem = dolfinx.fem.petsc.LinearProblem(
    extract_blocks(a),
    extract_blocks(L),
    u=[u, tau],
    bcs=[],
    P=extract_blocks(P),
    petsc_options=petsc_options,
    petsc_options_prefix="mixed_single_",
)
problem.solve()


with dolfinx.io.VTXWriter(omega_i.comm, "u.bp", [u], engine="BP5") as bp:
    bp.write(0.0)


error_ui = inner(u - ui_exact, u - ui_exact) * dxI
error_ue = inner(u - ue_exact, u - ue_exact) * dxE
L2_ui = np.sqrt(scifem.assemble_scalar(error_ui))
L2_ue = np.sqrt(scifem.assemble_scalar(error_ue))
PETSc.Sys.Print(f"L2(ui): {L2_ui:.2e}\n L2(ue): {L2_ue:.2e}")
