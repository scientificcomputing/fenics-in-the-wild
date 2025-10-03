# # Mixed single-domain formulation
# ```{note}
# This examples uses many of the same concepts as [the primal-single example](./emi_primal_single)
# and [the primal-mixed example](./emi_primal_mixed).
# It is recommended to read through those examples first, as this example will only focus on new concepts
# required for the mixed dimensional formulation.
# ```
#
# In this example, we consider the formulation from Chapter 5.2.2 of {cite}`emi-Kuchta2021emi`.
#
# Find $\mathbf{J} \in S$, $u\in V$ such that
#
# $$
# \int_\Omega \sigma^{-1} \mathbf{J} \cdot \boldsymbol{\tau}~\mathrm{d}x +
# \int_\Gamma \frac{\Delta t}{C_m} (\mathbf{J}\cdot \mathbf{n}_i)(\boldsymbol{\tau}\cdot \mathbf{n}_i)~\mathrm{d}s -
# \int_\Omega u \nabla \cdot \boldsymbol{\tau}~\mathrm{d}x &= - \int_\Gamma f \boldsymbol{\tau}\cdot \mathbf{n}_i~\mathrm{d}s\\
# - \int_\Omega (\nabla \cdot \mathbf{J}) q~\mathrm{d}x &= \int_{\Omega_e} f_e q~\mathrm{d}x + \int_{\Omega_i} f_i q~\mathrm{d}x
# $$
# for all $\boldsymbol{\tau}\in S$ and $q \in V$.
#
# As in the previous EMI-examples, we start by importing the necessary modules and define the domain and interface markers.

# +
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

M = 132
x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75
sigma_e = 2.0
sigma_i = 1.0
interior_marker = 2
exterior_marker = 3
interface_marker = 4
boundary_marker = 5


def lower_bound(x, i, bound, tol=1e-12):
    return x[i] >= bound - tol


def upper_bound(x, i, bound, tol=1e-12):
    return x[i] <= bound + tol


def omega_interior_marker(x, tol=1e-12):
    return (
        lower_bound(x, 0, x_L, tol=tol)
        & lower_bound(x, 1, y_L, tol=tol)
        & upper_bound(x, 0, x_U, tol=tol)
        & upper_bound(x, 1, y_U, tol=tol)
    )


omega = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD,
    M,
    M,
    ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    diagonal=dolfinx.mesh.DiagonalType.right,
)
tdim = omega.topology.dim
interior_cells = dolfinx.mesh.locate_entities(omega, tdim, omega_interior_marker)
cell_map = omega.topology.index_map(tdim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
cell_marker = np.full(num_cells_local, exterior_marker, dtype=np.int32)
cell_marker[interior_cells] = interior_marker
ct = dolfinx.mesh.meshtags(
    omega, tdim, np.arange(num_cells_local, dtype=np.int32), cell_marker
)
gamma_facets = scifem.find_interface(ct, interior_marker, exterior_marker)

omega.topology.create_connectivity(tdim - 1, tdim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(omega.topology)
facet_map = omega.topology.index_map(tdim - 1)
num_facets_local = facet_map.size_local + facet_map.num_ghosts
facets = np.arange(num_facets_local, dtype=np.int32)
marker = np.full_like(facets, -1, dtype=np.int32)
marker[gamma_facets] = interface_marker
marker[exterior_facets] = boundary_marker
marker_filter = np.flatnonzero(marker != -1).astype(np.int32)
ft = dolfinx.mesh.meshtags(omega, tdim - 1, marker_filter, marker[marker_filter])
ft.name = "interface_marker"
# -

# ```{note} Integration measures
# Note that since we are working on a single mesh, we can use the standard {py:class}`Measure<ufl.Measure>`
# with appropriate `subdomain_data` and `subdomain_id` to restrict the integration domains.
# ```

dx = Measure("dx", domain=omega, subdomain_data=ct)
dGamma = Measure("dS", domain=omega, subdomain_data=ft, subdomain_id=interface_marker)
ds = Measure("ds", domain=omega, subdomain_data=ft, subdomain_id=boundary_marker)

# We define the function-spaces as before

V = dolfinx.fem.functionspace(omega, ("DG", 0))
S = dolfinx.fem.functionspace(omega, ("RT", 1))
W = MixedFunctionSpace(V, S)

# Next, as we would like to unify the treatment of the spatially varying `sigma`, we
# use a discontinuous function space of piecewise constants to represent the conductivity.
# We use {py:meth}`Function.interpolate<dolfinx.fem.Function.interpolate>` with the
# input argument `cells0` to only perform interpolation on those cells marked with
# `interior_marker` and `exterior_marker`, separately.
# After interpolation, we call {py:meth}`scatter_forward<dolfinx.la.Vector.scatter_forward>`
# to ensure that ghost values are updated.

sigma = dolfinx.fem.Function(V)
sigma.interpolate(
    lambda x: np.full(x.shape[1], sigma_e), cells0=ct.find(exterior_marker)
)
sigma.interpolate(
    lambda x: np.full(x.shape[1], sigma_i), cells0=ct.find(interior_marker)
)
sigma.x.scatter_forward()


Cm = dolfinx.fem.Constant(omega, 1.0)
dt = dolfinx.fem.Constant(omega, 1e-4)
T = Cm / dt

u, J = TrialFunctions(W)
q, tau = TestFunctions(W)

n = FacetNormal(omega)
n_i = n("+")
n_e = n("-")
a = inner(inv(sigma) * J, tau) * dx
a += inv(T) * dot(J("+"), n_i) * dot(tau("+"), n_i) * dGamma
a -= u * div(tau) * dx
a -= q * div(J) * dx

x, y = SpatialCoordinate(omega)
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


if dolfinx.has_adios2:
    with dolfinx.io.VTXWriter(omega.comm, "u.bp", [u], engine="BP5") as bp:
        bp.write(0.0)


error_ui = inner(u - ui_exact, u - ui_exact) * dxI
error_ue = inner(u - ue_exact, u - ue_exact) * dxE
L2_ui = np.sqrt(scifem.assemble_scalar(error_ui))
L2_ue = np.sqrt(scifem.assemble_scalar(error_ue))
PETSc.Sys.Print(f"L2(ui): {L2_ui:.2e}\n L2(ue): {L2_ue:.2e}")
