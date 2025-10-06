# # Mixed mixed-domain formulation
# ```{note}
# This examples uses many of the same concepts as [the primal-single example](./emi_primal_single),
# [the primal-mixed example](./emi_primal_mixed) and [the mixed-single example](./emi_mixed_single).
# It is recommended to read through those examples first, as this example will only focus on new concepts
# required for the mixed dimensional formulation.
# ```
#
# As in the previous EMI-examples, we consider the formulation from Chapter 5.3.2 of {cite}`emimm-Kuchta2021emi`.
#
# Find $\mathbf{J} \in S(\Omega)$, $u\in V(\Omega)$ and $v \in W(\Gamma)$ such that
#
# $$
# \int_\Omega \sigma^{-1} \mathbf{J} \cdot \boldsymbol{\tau}~\mathrm{d}x
# - \int_\Omega u \nabla \cdot \boldsymbol{\tau}~\mathrm{d}x
# + \int_\Gamma v (\boldsymbol{\tau}\cdot \mathbf{n}_i)~\mathrm{d}s
# &= 0 \\
# - \int_\Omega (\nabla \cdot \mathbf{J}) q~\mathrm{d}x &=
# \int_{\Omega_e} f_e q~\mathrm{d}x + \int_{\Omega_i} f_i q~\mathrm{d}x \\
# \int_\Gamma (\mathbf{J}\cdot \mathbf{n}_i) w~\mathrm{d}s
# - \int_\Gamma \frac{C_m}{\Delta t} v w ~\mathrm{d}s
# &= -\frac{C_m}{\Delta t} \int_\Gamma f w ~\mathrm{d}s
# $$
#
# for all $\boldsymbol{\tau}\in S(\Omega)$, $q \in V(\Omega)$ and $w \in W(\Gamma)$.

# We first import the necessary modules and define the domain and interface markers

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


x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75
interior_marker = 2
exterior_marker = 3


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


# Steps to set up submeshes and interface
M = 400
omega = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, M, M, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
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


gamma = scifem.find_interface(ct, interior_marker, exterior_marker)

omega.topology.create_connectivity(tdim - 1, tdim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(omega.topology)
facet_map = omega.topology.index_map(tdim - 1)
num_facets_local = facet_map.size_local + facet_map.num_ghosts
facets = np.arange(num_facets_local, dtype=np.int32)
interface_marker = 4
boundary_marker = 5
marker = np.full_like(facets, -1, dtype=np.int32)
marker[gamma] = interface_marker
marker[exterior_facets] = boundary_marker
marker_filter = np.flatnonzero(marker != -1).astype(np.int32)
ft = dolfinx.mesh.meshtags(omega, tdim - 1, marker_filter, marker[marker_filter])
ft.name = "interface_marker"

Gamma, interface_to_parent, _, _, _ = scifem.extract_submesh(
    omega, ft, interface_marker
)
# -

# Next, we define integration measures in the same way as in {ref}`one-sided-integrals`

# +
ordered_integration_data = scifem.compute_interface_data(ct, ft.find(interface_marker))
i_indices = np.array([0, 1]) if interior_marker < exterior_marker else np.array([2, 3])
onesided_integration_data = (
    interface_marker,
    ordered_integration_data[:, i_indices].flatten(),
)
exterior_indices = dolfinx.fem.compute_integration_domains(
    dolfinx.fem.IntegralType.exterior_facet, omega.topology, ft.find(boundary_marker)
)
exterior_integration_data = (boundary_marker, exterior_indices)

ds = Measure(
    "ds",
    domain=omega,
    subdomain_data=[exterior_integration_data, onesided_integration_data],
)
dGamma_i = ds(interface_marker)
ds_ext = ds(boundary_marker)
dx = Measure("dx", domain=omega, subdomain_data=ct)
# -

# We define the function spaces and $\sigma$ as in [the previous examples](./emi_mixed_single)

# +
S = dolfinx.fem.functionspace(omega, ("RT", 2))
V = dolfinx.fem.functionspace(omega, ("DG", 1))
W = dolfinx.fem.functionspace(Gamma, ("DG", 1))
Z = MixedFunctionSpace(S, V, W)

T = dolfinx.fem.functionspace(omega, ("DG", 0))
sigma = dolfinx.fem.Function(T)
sigma_e = 2.0
sigma_i = 1.0
sigma.interpolate(
    lambda x: np.full(x.shape[1], sigma_e), cells0=ct.find(exterior_marker)
)
sigma.interpolate(
    lambda x: np.full(x.shape[1], sigma_i), cells0=ct.find(interior_marker)
)
sigma.x.scatter_forward()

# -

# The variational forms are defined as in the previous examples

# +
J, u, v = TrialFunctions(Z)
tau, q, w = TestFunctions(Z)
Cm = dolfinx.fem.Constant(omega, 1.0)
dt = dolfinx.fem.Constant(omega, 1e-4)
T = Cm / dt

n = FacetNormal(omega)
a = inner(inv(sigma) * J, tau) * dx
a += v * dot(tau, n) * dGamma_i
a -= u * div(tau) * dx
a += w * dot(J, n) * dGamma_i
a -= q * div(J) * dx
a -= T * v * w * dGamma_i
x, y = SpatialCoordinate(omega)
ue_exact = sin(pi * (x + y))

ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
    pi * (y - y_L) * (y - y_U)
)


Im_exact = sigma_e * inner(grad(ue_exact), -n)
f = ui_exact - ue_exact - 1 / T * Im_exact
dxE = dx(exterior_marker)
dxI = dx(interior_marker)
L = -T * f * w * dGamma_i
L -= -div(sigma_e * grad(ue_exact)) * q * dxE
L -= -div(sigma_i * grad(ui_exact)) * q * dxI
L -= ue_exact * dot(tau, n) * ds
# -

# And the problem is solved with a direct method

# +
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
    "ksp_monitor": None,
}

u = dolfinx.fem.Function(V)
tau = dolfinx.fem.Function(S)
v = dolfinx.fem.Function(W)
entity_maps = {interface_to_parent}
problem = dolfinx.fem.petsc.LinearProblem(
    extract_blocks(a),
    extract_blocks(L),
    u=[tau, u, v],
    bcs=[],
    petsc_options=petsc_options,
    petsc_options_prefix="mixed_mixed_",
    entity_maps=entity_maps,
)
problem.solve()
# -

# We output the solution to file, if ADIOS2 is available.

if dolfinx.has_adios2:
    with dolfinx.io.VTXWriter(omega.comm, "u_mixed_mixed.bp", [u], engine="BP5") as bp:
        bp.write(0.0)

# We compute the $L^2$-error of the solution in the interior and exterior domains.

error_ui = inner(u - ui_exact, u - ui_exact) * dxI
error_ue = inner(u - ue_exact, u - ue_exact) * dxE
L2_ui = np.sqrt(scifem.assemble_scalar(error_ui))
L2_ue = np.sqrt(scifem.assemble_scalar(error_ue))
PETSc.Sys.Print(f"L2(ui): {L2_ui:.5e}\nL2(ue): {L2_ue:.5e}")

# ```{bibliography}
# :filter: cited
# :labelprefix:
# :keyprefix: emimm-
# ```
