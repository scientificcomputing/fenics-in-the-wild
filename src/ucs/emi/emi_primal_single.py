# # Primal single-domain formulation

# In this example, we consider the formulation from Chapter 5.2.1 of {cite}`emi-Kuchta2021emi`.
#
# Find $u_i\in V_i=V(\Omega_i)$ and $u_e\in V_e=V(\Omega_e)$ such that
#
# $$
# \int_{\Omega_e} \sigma_e \nabla u_e \cdot \nabla v_e~\mathrm{d}x +
# \int_\Gamma \frac{C_m}{\Delta t} (u_e - u_i) v_e ~\mathrm{d}s &=
# - \frac{C_m}{\Delta t} \int_\Gamma f v_e ~\mathrm{d}s \\
# \int_{\Omega_i} \sigma_i \nabla u_i \cdot \nabla v_i~\mathrm{d}x
# + \int_\Gamma \frac{C_m}{\Delta t} (u_i - u_e) v_i ~\mathrm{d}s &=
# \frac{C_m}{\Delta t} \int_\Gamma f v_i ~\mathrm{d}s
# $$
#
# for all $v_e\in V_e$ and $v_i\in V_i$.

# We start by importing the necessary libraries

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

# Next, we define the intracellular and extracellular domain.
# In the following examples, we will use $\Omega_e = [x_L, x_U]\times[y_L, y_U]$
# and $\Omega_i = [0,1]^2\setminus\Omega_e$.
# We define the bounds and make a set of convenience functions to marker the domains.

# +
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


# -

# With the convenience functions we can set up $\Omega$

M = 400
omega = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, M, M, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)

# We create a {py:class}`MeshTags<dolfinx.mesh.MeshTags>` objects that marks the
# interior cells with `interior_marker`, and the remaining (exterior) cells with `exterior_marker`.

interior_cells = dolfinx.mesh.locate_entities(
    omega, omega.topology.dim, interior_marker
)
interior_marker = 2
exterior_marker = 3
cell_map = omega.topology.index_map(omega.topology.dim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
cell_marker = np.full(num_cells_local, exterior_marker, dtype=np.int32)
cell_marker[interior_cells] = interior_marker
ct = dolfinx.mesh.meshtags(
    omega, omega.topology.dim, np.arange(num_cells_local, dtype=np.int32), cell_marker
)

# Next, we create a {py:class}`Mesh<dolfinx.mesh.Mesh>` for $\Omega_i$ and $\Omega_e$
# using {py:func}`scifem.extract_submesh`:

omega_i, interior_to_parent, _, _, _ = scifem.extract_submesh(
    omega, ct, interior_marker
)
omega_e, exterior_to_parent, e_vertex_to_parent, _, _ = scifem.extract_submesh(
    omega, ct, exterior_marker
)

# We identity the facets on the interface $\Gamma$ using {py:func}`scifem.find_interface`.

gamma = scifem.find_interface(ct, interior_marker, exterior_marker)

# We create a {py:class}`MeshTags<dolfinx.mesh.MeshTags>` object that marks the
# facets on $\Gamma$ with `interface_marker` and additionally  marks all exterior facets with `boundary_marker`.
# Any facet not on $\Gamma$ or $\partial\Omega$ is excluded from the
# {py:class}`MeshTags<dolfinx.mesh.MeshTags>` object.

omega.topology.create_connectivity(omega.topology.dim - 1, omega.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(omega.topology)
facet_map = omega.topology.index_map(omega.topology.dim - 1)
num_facets_local = facet_map.size_local + facet_map.num_ghosts
facets = np.arange(num_facets_local, dtype=np.int32)
interface_marker = 4
boundary_marker = 5
marker = np.full_like(facets, -1, dtype=np.int32)
marker[gamma] = interface_marker
marker[exterior_facets] = boundary_marker
marker_filter = np.flatnonzero(marker != -1).astype(np.int32)
ft = dolfinx.mesh.meshtags(
    omega, omega.topology.dim - 1, marker_filter, marker[marker_filter]
)
ft.name = "interface_marker"

# For the volume integrals, we create an integration {py:class}`measure<ufl.Measure>` that integrates
# over $\Omega$. However, we create a restriction of the integration measure to $\Omega_i$ and $\Omega_e$ using the
# py calling the measure with the appropriate marker. **Note** that this would result in a zero integration measure
# if we had not passed `ct` into the initializer of `dx`.

dx = Measure("dx", domain=omega, subdomain_data=ct)
dxI = dx(interior_marker)
dxE = dx(exterior_marker)

# ## Setting up the mixed function space and variational form
# We use {py:class}`MixedFunctionSpace<ufl.MixedFunctionSpace>` to create the mixed function space

element = ("Lagrange", 1)
Vi = dolfinx.fem.functionspace(omega_i, element)
Ve = dolfinx.fem.functionspace(omega_e, element)
W = MixedFunctionSpace(Vi, Ve)
vi, ve = TestFunctions(W)
ui, ue = TrialFunctions(W)


# Next, for the interface integrals, we want to create a {py:class}`measure<ufl.Measure>` that integrates over $\Gamma$.
# However, as $\Gamma$ connects to two cells, one from $\Omega_i$ and one from $\Omega_e$, we need to define
# the appropriate restrictions to compute the jump of $u_e$ and $u_i$.


ordered_integration_data = scifem.compute_interface_data(ct, ft.find(interface_marker))
dGamma = Measure(
    "dS",
    domain=omega,
    subdomain_data=[(2, ordered_integration_data.flatten())],
    subdomain_id=2,
)

sigma_e = dolfinx.fem.Constant(omega_e, 2.0)
sigma_i = dolfinx.fem.Constant(omega_i, 1.0)
Cm = dolfinx.fem.Constant(omega, 1.0)
dt = dolfinx.fem.Constant(omega, 1.0e-2)

# Setup variational form
tr_ui = ui("+")
tr_ue = ue("-")
tr_vi = vi("+")
tr_ve = ve("-")

x, y = SpatialCoordinate(omega)
ue_exact = sin(pi * (x + y))
ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
    pi * (y - y_L) * (y - y_U)
)

n = FacetNormal(omega)
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

sub_tag, _ = scifem.transfer_meshtags_to_submesh(
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


# ```{bibliography}
# :filter: cited
# :labelprefix:
# :keyprefix: emi-
# ```
