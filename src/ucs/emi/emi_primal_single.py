# # Primal single-domain formulation

# In this example, we consider the formulation from Chapter 5.2.1 of {cite}`emi-Kuchta2021emi`.
#
# Find $u_i\in V_i=V(\Omega_i)$ and $u_e\in V_e=V(\Omega_e)$ such that
#
# $$
# \int_{\Omega_e} \sigma_e \nabla u_e \cdot \nabla v_e~\mathrm{d}x +
# \int_\Gamma \frac{C_m}{\Delta t} (u_e - u_i) v_e ~\mathrm{d}s &=
# \int_{\Omega_e} f_e v_e ~\mathrm{d}x
# - \frac{C_m}{\Delta t} \int_\Gamma f v_e ~\mathrm{d}s \\
# \int_{\Omega_i} \sigma_i \nabla u_i \cdot \nabla v_i~\mathrm{d}x
# + \int_\Gamma \frac{C_m}{\Delta t} (u_i - u_e) v_i ~\mathrm{d}s &=
# \int_{\Omega_i} f_i v_i ~\mathrm{d}x
# + \frac{C_m}{\Delta t} \int_\Gamma f v_i ~\mathrm{d}s
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
# We create each of the spaces, and make them a {py:class}`mixed function space <ufl.MixedFunctionSpace>`, which
# we can extract {py:func}`test<ufl.TestFunctions>` and {py:func}`trial<ufl.TrialFunctions>` functions from.

element = ("Lagrange", 1)
Vi = dolfinx.fem.functionspace(omega_i, element)
Ve = dolfinx.fem.functionspace(omega_e, element)
W = MixedFunctionSpace(Vi, Ve)
vi, ve = TestFunctions(W)
ui, ue = TrialFunctions(W)

# Next, for the interface integrals, we want to create a {py:class}`measure<ufl.Measure>` that integrates over $\Gamma$.
# However, as $\Gamma$ connects to two cells, one from $\Omega_i$ and one from $\Omega_e$, we need to define
# the appropriate restrictions of $u_e$, $u_i$, $v_e$, and $v_i$ to the interface.
# This is done by calling  {py:func}`scifem.compute_interface_data`, which returns the integration data order such that
# the "+" side of the [restriction](https://docs.fenicsproject.org/ufl/main/manual/form_language.html#restriction-v-and-v)
# corresponds to the smallest cell tag of `interior_marker` and `exterior_marker`.
# We pass in the ordered integration data to the initializer of a {py:class}`measure<ufl.Measure>`.

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"
ordered_integration_data = scifem.compute_interface_data(ct, ft.find(interface_marker))
interface_tag = 2
dGamma = Measure(
    "dS",
    domain=omega,
    subdomain_data=[(interface_tag, ordered_integration_data.flatten())],
    subdomain_id=interface_tag,
)

# Next, we define the trace operators on the interface for each of the functions.

tr_ui = ui(i_res)
tr_ue = ue(e_res)
tr_vi = vi(i_res)
tr_ve = ve(e_res)

# ## Variational formulation

# We define the problem parameters as {py:class}`Constant<dolfinx.fem.Constant>` objects.
# This is to make the code efficient, enven if we change the parameter values later.

sigma_e = dolfinx.fem.Constant(omega_e, 2.0)
sigma_i = dolfinx.fem.Constant(omega_i, 1.0)
Cm = dolfinx.fem.Constant(omega, 1.0)
dt = dolfinx.fem.Constant(omega, 1.0e-2)

# Next, we define the exact solution of each of the potentials
x, y = SpatialCoordinate(omega)
ue_exact = sin(pi * (x + y))
ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
    pi * (y - y_L) * (y - y_U)
)

# We also define corresponding right hand side source terms $f$, $f_i$, $f_e$ that fulfills the
# strong form of the equations, given the exact solutions above. This is called the
# *method of manufactured solutions*.

n = FacetNormal(omega)
n_e = n(e_res)
Im = sigma_e * inner(grad(ue_exact), n_e)
T = Cm / dt
f = ui_exact - ue_exact - 1 / T * Im
f_e = -div(sigma_e * grad(ue_exact))
f_i = -div(sigma_i * grad(ui_exact))


# We can then define the variational formulation with the bilinear form `a` and linear form `L`

a = sigma_e * inner(grad(ue), grad(ve)) * dxE
a += sigma_i * inner(grad(ui), grad(vi)) * dxI
a += T * (tr_ue - tr_ui) * tr_ve * dGamma
a += T * (tr_ui - tr_ue) * tr_vi * dGamma
L = T * inner(f, (tr_vi - tr_ve)) * dGamma
L += f_e * ve * dxE
L += f_i * vi * dxI

# We impose a Dirichlet boundary condition on the outer boundary of $\Omega_e$.
# To do this, we transfer the facet tags (`ft`) from the parent mesh to the submesh (`omega_e`)
# by calling {py:func}`scifem.transfer_meshtags_to_submesh`.

sub_tag, _ = scifem.transfer_meshtags_to_submesh(
    ft, omega_e, e_vertex_to_parent, exterior_to_parent
)

# With the tags transferred onto the submesh, we can locate the degree of freedom (dofs)
# that we want to constrain with {py:func}`dolfinx.fem.locate_dofs_topological`

omega_e.topology.create_connectivity(omega_e.topology.dim - 1, omega_e.topology.dim)
bc_dofs = dolfinx.fem.locate_dofs_topological(
    Ve, omega_e.topology.dim - 1, sub_tag.find(boundary_marker)
)

# We interpolate the exact solution into the appropriate function space
# and create a {py:class}`DirichletBC<dolfinx.fem.DirichletBC>` object using
# the {py:func}`dolfinx.fem.dirichletbc` constructor.

u_bc = dolfinx.fem.Function(Ve)
u_bc.interpolate(lambda x: np.sin(np.pi * (x[0] + x[1])))
bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)

# ## Preconditioning of the system
# {cite}`emi-Kuchta2021prec` suggests using the following preconditioner for the system:
#
# $$
# P &=
# \begin{pmatrix}
#    \int_{\Omega_i} \sigma_i \nabla u_i \cdot \nabla v_i + u_i v_i ~\mathrm{d}x & 0 \\
#    0 & \int_{\Omega_e} \sigma_e \nabla u_e \cdot \nabla v_e ~\mathrm{d}x
# \end{pmatrix}
# $$

P = sigma_e * inner(grad(ue), grad(ve)) * dxE
P += sigma_i * inner(grad(ui), grad(vi)) * dxI
P += inner(ui, vi) * dxI

# ## Solving the system
# We use the {py:class}`LinearProblem<dolfinx.fem.petsc.LinearProblem>` class to set up
# a solver for the linear system. We use {py:func}`extract_blocks<ufl.extract_blocks>`
# to extract the block structure of the bilinear form, linear form, and preconditioner.

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

# We check the numbers of iterations and the {py:class}`reason<petsc4py.PETSc.KSP.ConvergedReason>` for convergence

num_iterations = problem.solver.getIterationNumber()
converged_reason = problem.solver.getConvergedReason()
PETSc.Sys.Print(f"Solver converged in: {num_iterations} with reason {converged_reason}")

# If we have {py:class}`adios2.Adios` installed, we can save the solution to a file that can be visualized
# in Paraview

if dolfinx.has_adios2:
    with dolfinx.io.VTXWriter(omega_i.comm, "uh_i.bp", [ui], engine="BP5") as bp:
        bp.write(0.0)
    with dolfinx.io.VTXWriter(omega_i.comm, "uh_e.bp", [ue], engine="BP5") as bp:
        bp.write(0.0)


# Finally, we compute the $L^2$ error of each of the potentials

error_ui = dolfinx.fem.form(
    inner(ui - ui_exact, ui - ui_exact) * dxI, entity_maps=entity_maps
)
error_ue = dolfinx.fem.form(
    inner(ue - ue_exact, ue - ue_exact) * dxE, entity_maps=entity_maps
)
L2_ui = np.sqrt(scifem.assemble_scalar(error_ui, entity_maps=entity_maps))
L2_ue = np.sqrt(scifem.assemble_scalar(error_ue, entity_maps=entity_maps))

PETSc.Sys.Print(f"L2(ui): {L2_ui:.2e}\nL2(ue): {L2_ue:.2e}")


# ```{bibliography}
# :filter: cited
# :labelprefix:
# :keyprefix: emi-
# ```
