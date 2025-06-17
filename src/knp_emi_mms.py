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
    extract_blocks,
    Measure,
    dot,
    Coefficient
)
import numpy as np
import numpy.typing as npt
import scifem
from packaging.version import Version
from knp_emi_exact_solutions import ExactSolutionsKNPEMI

x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75
z_L = 0.25
z_U = 0.75

def interior_marker_function(x, tol=1e-12):
    lower_bound = lambda x, i, bound: x[i] >= bound - tol
    upper_bound = lambda x, i, bound: x[i] <= bound + tol
    return (
        lower_bound(x, 0, x_L)
        & lower_bound(x, 1, y_L)
        & lower_bound(x, 2, z_L)
        & upper_bound(x, 0, x_U)
        & upper_bound(x, 1, y_U)
        & upper_bound(x, 2, z_U)
    )

# Steps to set up submeshes and interface
M = int(argv[1])
mesh = dolfinx.mesh.create_unit_cube(
    MPI.COMM_WORLD, M, M, M, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
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

omega_i, interior_to_parent, i_vertex_to_parent, _, _ = scifem.mesh.extract_submesh(
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

sub_tag_e, _ = scifem.mesh.transfer_meshtags_to_submesh(
    ft, omega_e, e_vertex_to_parent, exterior_to_parent
)
omega_e.topology.create_connectivity(omega_e.topology.dim - 1, omega_e.topology.dim)

with dolfinx.io.XDMFFile(mesh.comm, "cell.xdmf", "w") as xdmf:
    xdmf.write_mesh(omega_e)
    xdmf.write_meshtags(sub_tag_e, omega_e.geometry)

# Integration measures for volumes
dx = Measure("dx", domain=mesh, subdomain_data=ct)
dxI = dx(interior_marker)
dxE = dx(exterior_marker)

# Exterior facet integral measure
ds = Measure("ds", domain=mesh, subdomain_data=ft)

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
    subdomain_data=[(interface_marker, ordered_integration_data.flatten())],
    subdomain_id=interface_marker,
)

interface_markers = (interface_marker,)

# Functions spaces
element = ("Lagrange", 1)
V  = dolfinx.fem.functionspace(mesh, element)
Vi = dolfinx.fem.functionspace(omega_i, element)
Ve = dolfinx.fem.functionspace(omega_e, element)

# Species information
sodium = {"Di" : dolfinx.fem.Constant(mesh, 1.0), # Intracellular diffusion coefficient
          "De" : dolfinx.fem.Constant(mesh, 1.0), # Intracellular diffusion coefficient
          "z"  : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1)),  # Valence
          "f_i" : dolfinx.fem.Function(Vi), # Intracellular source term
          "f_e" : dolfinx.fem.Function(Ve), # Extracellular source term
          "name" : "Na" 
}
potassium = {"Di" : dolfinx.fem.Constant(mesh, 1.0), # Intracellular diffusion coefficient
             "De" : dolfinx.fem.Constant(mesh, 1.0), # Intracellular diffusion coefficient
             "z"  : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1)),  # Valence
             "f_i" : dolfinx.fem.Function(Vi), # Intracellular source term
             "f_e" : dolfinx.fem.Function(Ve), # Extracellular source term
             "name" : "K" 
}
chloride = {"Di" : dolfinx.fem.Constant(mesh, 1.0), # Intracellular diffusion coefficient
            "De" : dolfinx.fem.Constant(mesh, 1.0), # Intracellular diffusion coefficient
            "z"  : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(-1)),  # Valence
            "f_i" : dolfinx.fem.Function(Vi), # Intracellular source term
            "f_e" : dolfinx.fem.Function(Ve), # Extracellular source term
            "name" : "Cl" 
}
ion_list = [sodium, potassium, chloride]
num_ions = len(ion_list)
names = ["Na_i", "K_i", "Cl_i", "phi_i", "Na_e", "K_e", "Cl_e", "phi_e"]
interior_spaces = [Vi.clone() for _ in range(num_ions+1)]
exterior_spaces = [Ve.clone() for _ in range(num_ions+1)]
spaces = (interior_spaces+exterior_spaces)
W = MixedFunctionSpace(*spaces)
u = TrialFunctions(W)
v = TestFunctions(W)
uh_ = [dolfinx.fem.Function(space) for space in spaces] # Previous timestep functions

# Extract potential functions
phi_i  = u[num_ions] # Intracellular trial function
vphi_i = v[num_ions] # Intracellular test  function
phi_e  = u[-1] # Extracellular trial function
vphi_e = v[-1] # Extracellular test  function

# Define traces
i_res = "+"
e_res = "-"
tr_phi_i  = phi_i(i_res)
tr_phi_e  = phi_e(e_res)
tr_vphi_i = vphi_i(i_res)
tr_vphi_e = vphi_e(e_res)

# Constants
t = dolfinx.fem.Constant(mesh, 0.0) # Time
timestep = 1.0e-5
t_end = timestep 
dt = dolfinx.fem.Constant(mesh, timestep)

# Get exact solutions
exact_solutions = ExactSolutionsKNPEMI(mesh, t)
exact_solutions_funcs, exact_source_terms = exact_solutions.get_mms_terms()
phi_i_exact = exact_solutions_funcs["phi_i"]
phi_e_exact = exact_solutions_funcs["phi_e"]
phi_m_ = exact_solutions_funcs["phi_i_init"] - exact_solutions_funcs["phi_e_init"]

# Set constants
gas_const = temp = 1.0
faraday = dolfinx.fem.Constant(mesh, 1.0)
sigma_e = dolfinx.fem.Constant(omega_e, dolfinx.default_scalar_type(1.0))
sigma_i = dolfinx.fem.Constant(omega_i, dolfinx.default_scalar_type(1.0))
Cm = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
psi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(gas_const*temp/faraday.value))
n = FacetNormal(mesh)

# Passive ionic model
class Passive(object):
    tags = interface_markers
    def __init__(self) -> None:
        pass

    def _eval(self):
        return phi_m_
passive_model = Passive()
ionic_models = [passive_model]

def setup_variational_form():

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
    
        # Set initial conditions
        uh_i.interpolate(dolfinx.fem.Expression(
                                exact_solutions_funcs[ion["name"]+"_i"],
                                uh_i.function_space.element.interpolation_points
                                )
                            )
        uh_e.interpolate(dolfinx.fem.Expression(
                                    exact_solutions_funcs[ion["name"]+"_e"],
                                    uh_e.function_space.element.interpolation_points
                                )
                            )

        # Add contribution to alpha fraction
        alpha_i_sum += Di * z**2 * uh_i     # Intracellular contribution
        alpha_e_sum += De * z**2 * uh_e # Extracellular contribution

        # Set ion channel currents
        ion['I_ch'] = dict.fromkeys(interface_markers)

        for model in ionic_models:
            for interface_tag in model.tags:
                ion['I_ch'][interface_tag] = model._eval()     # Calculate channel current
                I_ch[interface_tag] += ion['I_ch'][interface_tag] # Add contribution to total channel current

    # Set initial conditions for the potentials    
    uh_[num_ions].interpolate(dolfinx.fem.Expression(
                                phi_i_exact,
                                uh_[num_ions].function_space.element.interpolation_points
                                )
                            )
    uh_[2*num_ions+1].interpolate(dolfinx.fem.Expression(
                                    phi_e_exact,
                                    uh_[2*num_ions+1].function_space.element.interpolation_points
                                )
                            )

    # Setup variational form   
    # Initialize 
    a = 0; L = 0

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
            L -= 1/(faraday*z) * (dt*I_ch_k[interface_tag] - alpha_i(i_res)*Cm*phi_m_) * vki(i_res) * dGamma(interface_tag)
            L += 1/(faraday*z) * (dt*I_ch_k[interface_tag] - alpha_e(e_res)*Cm*phi_m_) * vke(e_res) * dGamma(interface_tag)

        # Source terms
        L += dt * inner(ion["f_i"], vki) * dxI
        L += dt * inner(ion["f_e"], vke) * dxE

        ##------ MMS TERMS START ------##
        # Concentration source terms
        L += dt * inner(exact_source_terms[f"f_{ion['name']}_i"], vki) * dxI # Intracellular
        L += dt * inner(exact_source_terms[f"f_{ion['name']}_e"], vke) * dxE # Extracellular

        # Membrane current correction
        for interface_tag in interface_markers:
            L += dt/(faraday*z) * alpha_i(i_res) * inner(exact_source_terms[f"f_phi_{ion['name']}"], vki(i_res)) * dGamma(interface_tag)
            L -= dt/(faraday*z) * alpha_e(e_res) * inner(exact_source_terms[f"f_phi_{ion['name']}"], vke(e_res)) * dGamma(interface_tag)
            L -= dt/(faraday*z) * alpha_e(e_res) * inner(exact_source_terms["f_gamma"], vke(e_res)) * dGamma(interface_tag)

        # Exterior boundary fluxes
        L -= dt * inner(dot(exact_source_terms[f"J_{ion['name']}_e"], n), vke) * ds
        L += faraday*z * inner(dot(exact_source_terms[f"J_{ion['name']}_e"], n), vphi_e) * ds
        ##------ MMS TERMS END ------##
            
    # Weak form - potential equations
    a -= dt * inner(J_phi_i, grad(vphi_i)) * dxI
    a -= dt * inner(J_phi_e, grad(vphi_e)) * dxE

    # Trace terms
    a += Cm/faraday * (tr_phi_i - tr_phi_e) * tr_vphi_i * dGamma
    a += Cm/faraday * (tr_phi_e - tr_phi_i) * tr_vphi_e * dGamma

    # Membrane currents
    for interface_tag in interface_markers:
        L += (1/faraday) * (dt*I_ch[interface_tag] - Cm*phi_m_) * (tr_vphi_e - tr_vphi_i) * dGamma(interface_tag)

    ##------ MMS TERMS START ------##
    # Potential source terms
    L -= dt * inner(exact_source_terms["f_phi_i"], vphi_i) * dxI # Intracellular
    L -= dt * inner(exact_source_terms["f_phi_e"], vphi_e) * dxE # Extracellular

    # Membrane current correction
    for interface_tag in interface_markers:
        L += dt * inner(exact_source_terms["f_phi_m"], tr_vphi_i - tr_vphi_e) * dGamma(interface_tag)
        L -= dt * inner(exact_source_terms["f_gamma"], tr_vphi_e) * dGamma(interface_tag)
    ##------ MMS TERMS END ------##
        
    a_compiled = dolfinx.fem.form(extract_blocks(a), entity_maps=entity_maps)
    L_compiled = dolfinx.fem.form(extract_blocks(L), entity_maps=entity_maps)

    return a_compiled, L_compiled

def setup_preconditioner_form():

    # Initialize
    J_phi_i = 0; J_phi_e = 0
    P = 0

    # Ion specific parts
    for idx, ion in enumerate(ion_list):

        # Get ion attributes
        z  = ion['z']
        Di = ion['Di']
        De = ion['De']

        # Set intracellular ion functions
        ki  = u[idx]       # Trial function
        vki = v[idx]       # Test function
        ki_ = uh_[idx] # Previous solution

        # Set extracellular ion functions
        ke  = u[num_ions+1+idx]       # Trial function
        vke = v[num_ions+1+idx]       # Test function
        ke_ = uh_[num_ions+1+idx] # Previous solution

        P += dt * inner(Di*grad(ki), grad(vki)) * dxI + ki*vki * dxI
        P += dt * inner(De*grad(ke), grad(vke)) * dxE + ke*vke * dxE

        # Ion fluxes
        Ji =  - (Di*z/psi) * ki_ * grad(phi_i)
        Je =  - (De*z/psi) * ke_ * grad(phi_e)

        # Add contributions to total fluxes
        J_phi_i += z*Ji
        J_phi_e += z*Je

    # Potential terms
    P -= dt * inner(J_phi_i, grad(vphi_i)) * dxI + (Cm/faraday) * inner(tr_phi_i, tr_vphi_i) * dGamma
    P -= dt * inner(J_phi_e, grad(vphi_e)) * dxE + (Cm/faraday) * inner(tr_phi_e, tr_vphi_e) * dGamma

    P_compiled = dolfinx.fem.form(extract_blocks(P), entity_maps=entity_maps)

    return P_compiled

def get_error_forms(uh: list[dolfinx.fem.Function], uh_exact: list[Coefficient]) \
    -> list[dolfinx.fem.Form]:
    forms = []
    for idx, sol in zip(range(len(uh)), uh):
        exact_sol = uh_exact[names[idx]]
        dx_error = dxI if idx <= 3 else dxE
        error_L2 = inner(sol - exact_sol, sol - exact_sol) * dx_error
        forms.append(dolfinx.fem.form(error_L2, entity_maps=entity_maps))
    
    return forms

def calculate_errors(uh: list[dolfinx.fem.Function], uh_exact: list[Coefficient]):
    error_forms = get_error_forms(uh, uh_exact)
    for error_form, name in zip(error_forms, names):
        error = np.sqrt(
                    mesh.comm.allreduce(
                        dolfinx.fem.assemble_scalar(error_form),
                        op=MPI.SUM
                    )
                )
        print(f"L2 error {name} = {error:.2e}")

def assemble_system():
    """ Assemble linear system matrix, right-hand side vector, and
        apply boundary conditions.
    """
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a_compiled, bcs=bcs)
    A.assemble()
    with b.localForm() as loc: loc.set(0.0)
    dolfinx.fem.petsc.assemble_vector(b, L_compiled)
    bcs1 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(a_compiled, 1), bcs
    )
    dolfinx.fem.petsc.apply_lifting(b, a_compiled, bcs=bcs1)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(L_compiled), bcs)
    dolfinx.fem.petsc.set_bc(b, bcs0)

#----- Variational form and BCs -----#
a_compiled, L_compiled = setup_variational_form()

bc_e_dofs = dolfinx.fem.locate_dofs_topological(
    Ve, omega_e.topology.dim-1, sub_tag_e.find(boundary_marker)
)
bcs = []

# BC for potential
u_bc_e = dolfinx.fem.Function(exterior_spaces[-1])
u_bc_e.interpolate(dolfinx.fem.Expression(
                    phi_e_exact,
                    Ve.element.interpolation_points
                )
            )
bcs.append(dolfinx.fem.dirichletbc(u_bc_e, bc_e_dofs))

# BCs for concentrations
for idx, ion in enumerate(ion_list):
    exact_func = exact_solutions_funcs[ion["name"]+"_e"]
    func = dolfinx.fem.Function(exterior_spaces[idx])
    func.interpolate(dolfinx.fem.Expression(
                    exact_func,
                    Ve.element.interpolation_points
                )
            )
    bc   = dolfinx.fem.dirichletbc(func, bc_e_dofs)
    bcs.append(bc)

#----- Solver setup -----#
A = dolfinx.fem.petsc.create_matrix(a_compiled, kind="mpi") # System matrix
b = dolfinx.fem.petsc.create_vector(L_compiled, kind="mpi") # RHS vector
x = b.duplicate() # Solution vector

ksp = PETSc.KSP().create(mesh.comm)

use_direct_solver = False
if use_direct_solver:
    # Configure PETSc solver: direct solver using MUMPS
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
else:
    # Assemble preconditioner matrix
    P_compiled = setup_preconditioner_form()
    B = dolfinx.fem.petsc.assemble_matrix(P_compiled, kind="mpi", bcs=bcs)
    B.assemble()

    # Configure PETSc solver: iterative solver using GMRES with Hypre
    ksp.setOperators(A, B)
    ksp.setType("gmres")
    ksp.getPC().setType("hypre")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setTolerances(1.0e-7)
    ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)

    opts = PETSc.Options()
    opts.setValue("ksp_max_it", 1000)
    opts.setValue("ksp_initial_guess_nonzero", True)
    opts.setValue("pc_hypre_boomeramg_max_iter", 1)
    if mesh.geometry.dim==3: opts.setValue("pc_hypre_boomeramg_strong_threshold", 0.5)

ksp.setMonitor(lambda ksp, its, rnorm: print(f"Iteration: {its}, residual: {rnorm}"))
ksp.setErrorIfNotConverged(True)
ksp.setFromOptions()

# Prepare output
comm = MPI.COMM_WORLD
bp_names = ["Na_i.bp", "K_i.bp", "Cl_i.bp", "phi_i.bp",
            "Na_e.bp", "K_e.bp", "Cl_e.bp", "phi_e.bp"
        ]
bps = [dolfinx.io.VTXWriter(comm, bp_names[i], [uh_[i]], engine="BP5") for i in range(len(uh_))]
[bp.write(t) for bp in bps]

# Increment time and assemble system
t.value += dt.value 
print(f"Time = {t.value}")
assemble_system()

# Solve and update
ksp.solve(b, x)
print(f"Solver converged in: {ksp.getIterationNumber()} with reason {ksp.getConvergedReason()}")
x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dolfinx.fem.petsc.assign(x, uh_)

# Calculate errors
calculate_errors(uh_, exact_solutions_funcs)

# Write output and close files
[bp.write(t) for bp in bps]
[bp.close() for bp in bps]