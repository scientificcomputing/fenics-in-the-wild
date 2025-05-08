from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from sys import argv
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
    ln
)
import numpy as np
import numpy.typing as npt
import scifem
from packaging.version import Version

x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75


def interior_marker_function(x, tol=1e-12):
    lower_bound = lambda x, i, bound: x[i] >= bound - tol
    upper_bound = lambda x, i, bound: x[i] <= bound + tol
    return (
        lower_bound(x, 0, x_L)
        & lower_bound(x, 1, y_L)
        & upper_bound(x, 0, x_U)
        & upper_bound(x, 1, y_U)
    )


# Steps to set up submeshes and interface
M = int(argv[1])
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, M, M, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)


interior_cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, interior_marker_function)

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

membrane_integration_tag = 2
dGamma = Measure(
    "dS",
    domain=mesh,
    subdomain_data=[(membrane_integration_tag, ordered_integration_data.flatten())],
    subdomain_id=membrane_integration_tag,
)

interface_markers = (membrane_integration_tag,)

# Ionic models
class Passive(object):
    tags = interface_markers
    def __init__(self) -> None:
        pass

    def _eval(self, ion_number: int):
        return phi_m_
passive_model = Passive()
ionic_models = [passive_model]

# Species information
chloride = {"Di" : 2.03,#e-9, # Intracellular diffusion coefficient
            "De" : 2.03,#e-9, # Intracellular diffusion coefficient
            "z"  : -1,  # Valence
            "ki_init" : 137, # Initial intracellular concentration
            "ke_init" : 104, # Initial extracellular concentration
            "f_i" : 0.0, # Intracellular source term
            "f_e" : 0.0, # Extracellular source term
            "name" : "Cl" 
}
potassium = {"Di" : 1.96,#e-9, # Intracellular diffusion coefficient
             "De" : 1.96,#e-9, # Intracellular diffusion coefficient
             "z"  : 1,  # Valence
             "ki_init" : 125, # Initial intracellular concentration
             "ke_init" : 4, # Initial extracellular concentration
             "f_i" : 0.0, # Intracellular source term
             "f_e" : 0.0, # Extracellular source term
             "name" : "K" 
}
sodium = {"Di" : 1.33,#e-9, # Intracellular diffusion coefficient
          "De" : 1.33,#e-9, # Intracellular diffusion coefficient
          "z"  : 1,  # Valence
          "ki_init" : 12, # Initial intracellular concentration
          "ke_init" : 100, # Initial extracellular concentration
          "f_i" : 0.0, # Intracellular source term
          "f_e" : 0.0, # Extracellular source term
          "name" : "Na" 
}
ion_list = [chloride, potassium, sodium]
num_ions = len(ion_list)

# Functions spaces
element = ("Lagrange", 1)
V  = dolfinx.fem.functionspace(mesh, element)
Vi = dolfinx.fem.functionspace(omega_i, element)
Ve = dolfinx.fem.functionspace(omega_e, element)
interior_spaces = [Vi for _ in range(num_ions+1)]
exterior_spaces = [Ve for _ in range(num_ions+1)]
spaces = (interior_spaces+exterior_spaces)
W = MixedFunctionSpace(*spaces)
u = TrialFunctions(W)
v = TestFunctions(W)

uh_ = [dolfinx.fem.Function(space) for space in spaces] # Previous timestep functions
phi_m_ = dolfinx.fem.Function(V) # Previous timestep membrane potential function
phi_i_init = -0.06774 # Initial intracellular potential (V)
phi_e_init = 0.0 # Initial extracellular potential (V)
phi_m_.x.array[:] = phi_i_init - phi_e_init

# Extract potential functions
phi_i  = u[num_ions] # Intracellular trial function
vphi_i = v[num_ions] # Intracellular test  function
phi_e  = u[2*num_ions+1] # Extracellular trial function
vphi_e = v[2*num_ions+1] # Extracellular test  function

# Define traces
i_res = "+"
e_res = "-"
tr_phi_i  = phi_i(i_res)
tr_phi_e  = phi_e(e_res)
tr_vphi_i = vphi_i(i_res)
tr_vphi_e = vphi_e(e_res)

# Constants
sigma_e = dolfinx.fem.Constant(omega_e, 2.0)
sigma_i = dolfinx.fem.Constant(omega_i, 1.0)
Cm = dolfinx.fem.Constant(mesh, 1.0)
dt = dolfinx.fem.Constant(mesh, 1.e-3)
t = 0.0
temp      = 300   # temperature (K)
faraday   = 96485 # Faraday's constant (C/mol)
gas_const = 8.314 # Gas constant (J/(K*mol))
psi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(gas_const*temp/faraday))

x, y = SpatialCoordinate(mesh)
ue_exact = sin(pi * (x + y))
ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
    pi * (y - y_L) * (y - y_U)
)

n = FacetNormal(mesh)
n_e = n(e_res)
Im = sigma_e * inner(grad(ue_exact), n_e)
T = Cm / (faraday*dt)
f = ui_exact - ue_exact - 1 / T * Im

# Initialize some variables
alpha_i_sum = 0; alpha_e_sum = 0 # Alpha fractions
J_phi_i     = 0; J_phi_e     = 0 # Total intra-/extracellular fluxes
I_ch = dict.fromkeys(interface_markers, 0)
# Calculate ion channel currents
for idx, ion in enumerate(ion_list):
    # Get attributes and potential functions
    z  = ion['z']
    Di = ion['Di']
    De = ion['De']
    uh_i = uh_[idx]
    uh_e = uh_[num_ions+1+idx]

    if np.isclose(t, 0.0):
        uh_i.x.array[:] = ion['ki_init']
        uh_e.x.array[:] = ion['ke_init']

    # Add contribution to alpha fraction
    alpha_i_sum += Di * z**2 * uh_i     # Intracellular contribution
    alpha_e_sum += De * z**2 * uh_e # Extracellular contribution

    # Update ion channel currents
    ion['I_ch'] = dict.fromkeys(interface_markers)

    for model in ionic_models:
        for interface_tag in model.tags:
            ion['I_ch'][interface_tag] = model._eval(idx)     # Calculate channel current
            I_ch[interface_tag] += ion['I_ch'][interface_tag] # Add contribution to total channel current

# Setup variational form
# The membrane trace contributions
a  = T * (tr_phi_e - tr_phi_i) * tr_vphi_e * dGamma
a += T * (tr_phi_i - tr_phi_e) * tr_vphi_i * dGamma
L  = div(sigma_e * grad(ue_exact)) * vphi_e * dxE
L -= div(sigma_i * grad(ui_exact)) * vphi_i * dxI

# Ion specific parts
for idx, ion in enumerate(ion_list):

    # Get ion attributes
    z  = ion['z']
    Di = ion['Di']
    De = ion['De']
    I_ch_k = ion['I_ch']

    # Set intracellular ion functions
    ki  = u[idx]       # Trial function
    vki = v[idx]       # Test function
    ki_ = uh_[idx] # Previous solution

    # Set extracellular ion functions
    ke  = u[num_ions+1+idx]       # Trial function
    vke = v[num_ions+1+idx]       # Test function
    ke_ = uh_[num_ions+1+idx] # Previous solution

    # Set alpha fraction - the fraction of ion-specific capacitor currents
    alpha_i = Di * z**2 * ki_ / alpha_i_sum
    alpha_e = De * z**2 * ke_ / alpha_e_sum

    # Ion fluxes
    Ji = -Di * grad(ki) - (Di*z/psi) * ki_ * grad(phi_i)
    Je = -De * grad(ke) - (De*z/psi) * ke_ * grad(phi_e)

    # Add contributions to total fluxes
    J_phi_i += z*Ji
    J_phi_e += z*Je

    # A couple of useful constants
    C_i = Cm*alpha_i(i_res) / (faraday*z)
    C_e = Cm*alpha_e(e_res) / (faraday*z)

    # Weak form - equation for ki
    a += ki * vki * dxI - dt * inner(Ji, grad(vki)) * dxI
    a += C_i * inner(tr_phi_i, vki(i_res)) * dGamma
    a -= C_i * inner(tr_phi_e, vki(i_res)) * dGamma
    L += ki_ * vki * dxI

    # Weak form - equation for ke
    a += ke * vke * dxE - dt * inner(Je, grad(vke)) * dxE
    a += C_e * inner(tr_phi_e, vke(e_res)) * dGamma
    a -= C_e * inner(tr_phi_i, vke(e_res)) * dGamma
    L += ke_ * vke * dxE

    # Ion channel currents
    for interface_tag in interface_markers:
        L -= (dt * I_ch_k[interface_tag] - alpha_i(i_res) * Cm * phi_m_) / (faraday*z) * vki(i_res) * dGamma(interface_tag)
        L += (dt * I_ch_k[interface_tag] - alpha_e(e_res) * Cm * phi_m_) / (faraday*z) * vke(e_res) * dGamma(interface_tag)

    # Source terms
    L += inner(ion['f_i'], vki) * dxI
    L += inner(ion['f_e'], vke) * dxE

# Weak form - potential equations
a -= inner(J_phi_i, grad(vphi_i)) * dxI
a -= inner(J_phi_e, grad(vphi_e)) * dxE

# Membrane currents
for interface_tag in interface_markers:
    L += (I_ch[interface_tag]/faraday - T * phi_m_) * (tr_vphi_e - tr_vphi_i) * dGamma(interface_tag)
    
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

# P  = sigma_e * inner(grad(phi_e), grad(vphi_e)) * dxE
# P += sigma_i * inner(grad(phi_i), grad(vphi_i)) * dxI
# P += inner(phi_i, vphi_i) * dxI
# P_compiled = dolfinx.fem.form(extract_blocks(P), entity_maps=entity_maps)
# bc_P = dolfinx.fem.dirichletbc(0.0, bc_dofs, Ve)
# B = dolfinx.fem.petsc.assemble_matrix(P_compiled, kind="mpi", bcs=[bc_P])
# B.assemble()

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)#, B)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
ksp.setFromOptions()
# ksp.setTolerances(1e-12, 1e-12)
# ksp.setMonitor(lambda ksp, its, rnorm: print(f"Iteration: {its}, residual: {rnorm}"))
# ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
# ksp.setErrorIfNotConverged(True)

x = b.duplicate()
ksp.solve(b, x)
x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dolfinx.fem.petsc.assign(x, uh_)
ui = uh_[num_ions]
ue = uh_[2*num_ions+1]

num_iterations = ksp.getIterationNumber()
converged_reason = ksp.getConvergedReason()
print(f"Solver converged in: {num_iterations} with reason {converged_reason}")

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
print(f"L2(ui): {global_ui:.2e}\nL2(ue): {global_ue:.2e}")