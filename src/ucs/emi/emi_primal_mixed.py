# # Primal mixed-domain formulation
# ```{note}
# This examples uses many of the same concepts as [the primal-single example](./emi_primal_single).
# It is recommended to read through that example first, as this example will only focus on new concepts
# required for the mixed dimensional formulation.
# ```
#
# In this example, we consider the formulation from Chapter 5.2.1 of {cite}`emi-Kuchta2021emi`.
#
# Find $u_i\in V_i=V(\Omega_i)$ and $u_e\in V_e=V(\Omega_e)$, $I_m \in Q(\Gamma)$ such that
#
# $$
# \int_{\Omega_e} \sigma_e \nabla u_e \cdot \nabla v_e~\mathrm{d}x -
# \int_\Gamma I_m v_e ~\mathrm{d}s &=
# \int_{\Omega_e} f_e v_e ~\mathrm{d}x\\
# \int_{\Omega_i} \sigma_i \nabla u_i \cdot \nabla v_i~\mathrm{d}x +
# \int_\Gamma I_m v_i ~\mathrm{d}s &=
# \int_{\Omega_i} f_i v_i ~\mathrm{d}x\\
# \int_{\Gamma} (u_i - u_e) j_m ~\mathrm{d}s - \frac{\Delta t}{C_m}I_m j_m~\mathrm{d}s &=
# \int_{\Gamma} f j_m ~\mathrm{d}s
# $$
#
# for all $v_e\in V_e$, $v_i\in V_i$ and $j_m \in Q(\Gamma)$.

# As in [the primal-single example](./emi_primal_single), we start by importing the necessary modules and define
# the intracellular and extracellular domains and create appropriate surface markers.

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
)
import numpy as np
import scifem

M = 132
x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75
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
omega_i, interior_to_parent, _, _, _ = scifem.extract_submesh(
    omega, ct, interior_marker
)
omega_e, exterior_to_parent, e_vertex_to_parent, _, _ = scifem.extract_submesh(
    omega, ct, exterior_marker
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

# As we in this example require a function space on $\Gamma$, we create a
# {py:class}`Mesh<dolfinx.mesh.Mesh>` object for the interface $\Gamma$ using
# {py:func}`scifem.extract_submesh`.

Gamma, interface_to_parent, _, _, _ = scifem.extract_submesh(
    omega, ft, interface_marker
)

# We define the mixed function space and appropriate test and trial functions.

element = ("Lagrange", 2)
Q = dolfinx.fem.functionspace(Gamma, element)
Vi = dolfinx.fem.functionspace(omega_i, element)
Ve = dolfinx.fem.functionspace(omega_e, element)
W = MixedFunctionSpace(Vi, Ve, Q)
vi, ve, jm = TestFunctions(W)
ui, ue, Im = TrialFunctions(W)


# We create the consistently oriented integration data for the integrals, as in {ref}`consistent_restrictions`.

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"
dx = Measure("dx", domain=omega, subdomain_data=ct)
dxI = dx(interior_marker)
dxE = dx(exterior_marker)
ordered_integration_data = scifem.compute_interface_data(ct, ft.find(interface_marker))
dGamma = Measure(
    "dS",
    domain=omega,
    subdomain_data=[(2, ordered_integration_data.flatten())],
    subdomain_id=2,
)
tr_ui = ui(i_res)
tr_ue = ue(e_res)
tr_vi = vi(i_res)
tr_ve = ve(e_res)

# (mixed-assembly)=
# ### Assembly of mixed dimensional forms
# As we in the variational forms have terms that couple functions defined on the pairs $\Omega_i$ and $\Gamma$,
# $\Omega_e$ and $\Gamma$, we need to ensure that the assembly is done correctly.
# We will perform the integration over $\Gamma$ as an interior facet integral on the full mesh $\Omega$,
# similarly to what we did in [the primal-single example](./emi_primal_single).
# However, as $I_m$ and $j_m$ are only defined on $\Gamma$, and doesn't have a natural extension to the interior
# of either $\Omega_i$ or $\Omega_e$, we have to use the `"+"` restriction to ensure that
# {py:mod}`ufl` interprets the functions correctly.

Im = Im("+")
jm = jm("+")

# ## Symbolic variational formulation in {py:mod}`ufl`
# As in {ref}`manufactured_solutions`, we define manufactured solutions $u_i$, $u_e$ and corresponding
# right hand side source terms $f_i$, $f_e$ and $f$

# +
sigma_e = dolfinx.fem.Constant(omega_e, 2.0)
sigma_i = dolfinx.fem.Constant(omega_i, 1.0)
Cm = dolfinx.fem.Constant(omega, 1.0)
dt = dolfinx.fem.Constant(omega, 1e-2)

x, y = SpatialCoordinate(omega)
ue_exact = sin(pi * (x + y))
ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
    pi * (y - y_L) * (y - y_U)
)

n = FacetNormal(omega)
n_e = n("-")
Im_exact = sigma_e * inner(grad(ue_exact), n_e)
T = Cm / dt
f = ui_exact - ue_exact - 1 / T * Im_exact
f_e = -div(sigma_e * grad(ue_exact))
f_i = -div(sigma_i * grad(ui_exact))

a = sigma_e * inner(grad(ue), grad(ve)) * dxE
a += sigma_i * inner(grad(ui), grad(vi)) * dxI
a += Im * (tr_vi - tr_ve) * dGamma
a += (tr_ui - tr_ue) * jm * dGamma + -1 / T * Im * jm * dGamma
L = inner(f, jm) * dGamma
L += f_e * ve * dxE
L += f_i * vi * dxI
# -

# We also transfer the mesh tags to the exterior mesh to apply the Dirichlet boundary conditions

# +
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
# -

# We can now solve the problem using {py:class}`LinearProblem<dolfinx.fem.petsc.LinearProblem>`.

ui = dolfinx.fem.Function(Vi, name="ui")
ue = dolfinx.fem.Function(Ve, name="ue")
Imh = dolfinx.fem.Function(Q, name="Im")
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_monitor": None,
    "ksp_error_if_not_converged": True,
}
entity_maps = [interface_to_parent, exterior_to_parent, interior_to_parent]
problem = dolfinx.fem.petsc.LinearProblem(
    extract_blocks(a),
    extract_blocks(L),
    u=[ui, ue, Imh],
    bcs=[bc],
    petsc_options=petsc_options,
    petsc_options_prefix="primal_mixed_",
    entity_maps=entity_maps,
)
problem.solve()

# Optional visualization with {py:class}`adios2.Adios`

if dolfinx.has_adios2:
    with dolfinx.io.VTXWriter(omega_i.comm, "uh_i.bp", [ui], engine="BP5") as bp:
        bp.write(0.0)
    with dolfinx.io.VTXWriter(omega_i.comm, "uh_e.bp", [ue], engine="BP5") as bp:
        bp.write(0.0)
    with dolfinx.io.VTXWriter(Gamma.comm, "Imh.bp", [Imh], engine="BP5") as bp:
        bp.write(0.0)

# Compute $L^2$ errors for each potential

error_ui = inner(ui - ui_exact, ui - ui_exact) * dxI
error_ue = inner(ue - ue_exact, ue - ue_exact) * dxE
L2_ui = np.sqrt(scifem.assemble_scalar(error_ui, entity_maps=entity_maps))
L2_ue = np.sqrt(scifem.assemble_scalar(error_ue, entity_maps=entity_maps))
PETSc.Sys.Print(f"L2(ui): {L2_ui:.2e}\nL2(ue): {L2_ue:.2e}")

# ```{note}
# To do a proper error analysis for $I_m$, we would need the broken norm $H^{-1/2}(\Gamma)$.
# This is not implemented yet.
# ```
