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
    div,
    dot
)
import numpy as np
import numpy.typing as npt
import scifem
from packaging.version import Version
from knp_emi_mms import ExactSolutionsKNPEMI

x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75
MMS_VERIFICATION = int(argv[2])

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

sub_tag_i, _ = scifem.mesh.transfer_meshtags_to_submesh(
    ft, omega_i, i_vertex_to_parent, interior_to_parent
)
omega_i.topology.create_connectivity(omega_i.topology.dim - 1, omega_i.topology.dim)

with dolfinx.io.XDMFFile(mesh.comm, "cell.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)

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

# Species information
sodium = {"Di" : 1.33,#e-9, # Intracellular diffusion coefficient
          "De" : 1.33,#e-9, # Intracellular diffusion coefficient
          "z"  : 1,  # Valence
          "ki_init" : 12, # Initial intracellular concentration
          "ke_init" : 100, # Initial extracellular concentration
          "f_i" : 0.0, # Intracellular source term
          "f_e" : 0.0, # Extracellular source term
          "name" : "Na" 
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
chloride = {"Di" : 2.03,#e-9, # Intracellular diffusion coefficient
            "De" : 2.03,#e-9, # Intracellular diffusion coefficient
            "z"  : -1,  # Valence
            "ki_init" : 137, # Initial intracellular concentration
            "ke_init" : 104, # Initial extracellular concentration
            "f_i" : 0.0, # Intracellular source term
            "f_e" : 0.0, # Extracellular source term
            "name" : "Cl" 
}
ion_list = [sodium, potassium, chloride]
ion_list = [sodium, chloride, potassium]
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
phi_i_ = dolfinx.fem.Function(Vi)
phi_e_ = dolfinx.fem.Function(Ve)
phi_i_init = -0.06774 # Initial intracellular potential (V)
phi_e_init = 0.0 # Initial extracellular potential (V)

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
t = 0.0
t_end = 1.0
if MMS_VERIFICATION:
    temp = faraday = gas_const = sigma_e_val = sigma_i_val = Cm_val = 1
    for ion in ion_list:
        ion["Di"] = 1.0
        ion["De"] = 1.0
    timestep = t_end - t

else:
    temp      = 300   # temperature (K)
    faraday   = 96485 # Faraday's constant (C/mol)
    gas_const = 8.314 # Gas constant (J/(K*mol))
    Cm_val = 1.0
    sigma_e_val = 2.0
    sigma_i_val = 1.0
    timestep  = 1e-2

sigma_e = dolfinx.fem.Constant(omega_e, dolfinx.default_scalar_type(sigma_e_val))
sigma_i = dolfinx.fem.Constant(omega_i, dolfinx.default_scalar_type(sigma_i_val))
Cm = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(Cm_val))
psi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(gas_const*temp/faraday))
n = FacetNormal(mesh)
n_e = n(e_res)
dt = dolfinx.fem.Constant(mesh, timestep)
num_timesteps = int(t_end / dt.value)
T = Cm / (faraday*dt)

# Get exact solutions
exact_solutions = ExactSolutionsKNPEMI(mesh)

def get_exact_solutions(time: float):
    exact_solutions.t = time
    exact_solutions_expr = exact_solutions.get_exact_solutions()
    exact_solutions_funcs = dict.fromkeys(exact_solutions_expr)
    exact_source_terms = exact_solutions.get_source_terms()
    for function, key in zip(uh_, exact_solutions_funcs.keys()):
        func = function.copy()
        space = func.function_space
        expr = dolfinx.fem.Expression(exact_solutions_expr[key], space.element.interpolation_points)
        func.interpolate(expr)
        exact_solutions_funcs[key] = func
    
    return exact_solutions_expr, exact_source_terms

def project_onto_membrane(u: dolfinx.fem.Function):
    return None

def setup_variational_form(time: float):

    if MMS_VERIFICATION:
        exact_solutions_funcs, exact_source_terms = get_exact_solutions(time)
        phi_m_ = exact_solutions_funcs["phi_i"] - exact_solutions_funcs["phi_e"]
    else:
        phi_m_ = project_onto_membrane(phi_i_ - phi_e_)

    # Initialize some variables
    alpha_i_sum = 0; alpha_e_sum = 0 # Alpha fractions
    J_phi_i     = 0; J_phi_e     = 0 # Total intra-/extracellular fluxes
    I_ch = dict.fromkeys(interface_markers, 0)

    # Ionic models
    class Passive(object):
        tags = interface_markers
        def __init__(self) -> None:
            pass

        def _eval(self, ion_number: int):
            return phi_m_
    passive_model = Passive()
    ionic_models = [passive_model]

    # Calculate ion channel currents
    for idx, ion in enumerate(ion_list):
        # Get attributes and potential functions
        z  = ion['z']
        Di = ion['Di']
        De = ion['De']
        uh_i = uh_[idx]
        uh_e = uh_[num_ions+1+idx]

        if np.isclose(time, 0.0):
            if MMS_VERIFICATION:
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
            else:
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

    if np.isclose(time, 0.0):
        if MMS_VERIFICATION:
            phi_i_ = exact_solutions_funcs["phi_i"]
            uh_[num_ions].interpolate(dolfinx.fem.Expression(
                                        exact_solutions_funcs["phi_i"],
                                        uh_[num_ions].function_space.element.interpolation_points
                                        )
                                    )
            phi_e_ = exact_solutions_funcs["phi_e"]
            uh_[2*num_ions+1].interpolate(dolfinx.fem.Expression(
                                            exact_solutions_funcs["phi_e"],
                                            uh_[2*num_ions+1].function_space.element.interpolation_points
                                        )
                                    )
        else:
            phi_i_.x.array[:] = phi_i_init
            phi_e_.x.array[:] = phi_e_init

    # Setup variational form
    # The membrane trace contributions
    a  = T * (tr_phi_e - tr_phi_i) * tr_vphi_e * dGamma
    a += T * (tr_phi_i - tr_phi_e) * tr_vphi_i * dGamma
    L = 0
    if MMS_VERIFICATION:
        L += div(sigma_e * grad(exact_solutions_funcs["phi_e"])) * vphi_e * dxE
        L -= div(sigma_i * grad(exact_solutions_funcs["phi_i"])) * vphi_i * dxI

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
        L += inner(ion["f_i"], vki) * dxI
        L += inner(ion["f_e"], vke) * dxE

        if MMS_VERIFICATION:
            # Concentration source terms
            L += dt * inner(exact_source_terms[f"f_{ion['name']}_i"], vki) * dxI # Intracellular
            L += dt * inner(exact_source_terms[f"f_{ion['name']}_e"], vke) * dxE # Extracellular

            # Membrane current correction
            L += dt/(faraday*z) * alpha_i(i_res) * inner(exact_source_terms["f_phi_m"](i_res), vki(i_res)) * dGamma
            L -= dt/(faraday*z) * alpha_e(e_res) * inner(exact_source_terms["f_phi_m"](e_res), vke(e_res)) * dGamma
            L -= dt/(faraday*z) * alpha_e(e_res) * inner(exact_source_terms["f_gamma"](e_res), vke(e_res)) * dGamma

            # Exterior boundary fluxes
            L -= dt * inner(dot(exact_source_terms[f"J_{ion['name']}_e"], n), vke) * ds
            L += faraday*z * inner(dot(exact_source_terms[f"J_{ion['name']}_e"], n), vphi_e) * ds
            
    # Weak form - potential equations
    a -= inner(J_phi_i, grad(vphi_i)) * dxI
    a -= inner(J_phi_e, grad(vphi_e)) * dxE

    # Membrane currents
    for interface_tag in interface_markers:
        L += (I_ch[interface_tag]/faraday - T * phi_m_) * (tr_vphi_e - tr_vphi_i) * dGamma(interface_tag)

    if MMS_VERIFICATION:
        # Potential source terms
        L -= inner(exact_source_terms["f_phi_i"], vphi_i) * dxI # Intracellular
        L -= inner(exact_source_terms["f_phi_e"], vphi_e) * dxE # Extracellular

        # Membrane current correction
        L += inner(exact_source_terms["f_phi_m"](i_res), vphi_i(i_res)) * dGamma
        L -= inner(exact_source_terms["f_phi_m"](e_res), vphi_e(e_res)) * dGamma
        L -= inner(exact_source_terms["f_gamma"](e_res), vphi_e(e_res)) * dGamma
        
    a_compiled = dolfinx.fem.form(extract_blocks(a), entity_maps=entity_maps)
    L_compiled = dolfinx.fem.form(extract_blocks(L), entity_maps=entity_maps)

    return a_compiled, L_compiled

bc_e_dofs = dolfinx.fem.locate_dofs_topological(
    Ve, omega_e.topology.dim-1, sub_tag_e.find(boundary_marker)
)
bcs = []
u_bc_e = dolfinx.fem.Function(Ve)
if MMS_VERIFICATION:
    exact_solutions_funcs, _ = get_exact_solutions(t)
    bc_e = exact_solutions_funcs["phi_e"]
    u_bc_e.interpolate(dolfinx.fem.Expression(
                        bc_e,
                        u_bc_e.function_space.element.interpolation_points
                    )
                )
    # BCs for concentrations
    bc_funcs_ions = []
    for idx, ion in enumerate(ion_list):
        exact_func = exact_solutions_funcs[ion["name"]+"_e"]
        func = dolfinx.fem.Function(Ve)
        func.interpolate(dolfinx.fem.Expression(
                        exact_func,
                        Ve.element.interpolation_points
                    )
                )
        bc_funcs_ions.append(func)
        dofs = dolfinx.fem.locate_dofs_topological(Ve, ft.dim, sub_tag_e.find(boundary_marker))
        bc   = dolfinx.fem.dirichletbc(func, dofs)
        bcs.append(bc)
else:
    bc_e = lambda x: np.sin(np.pi * (x[0] + x[1]))
    u_bc_e.interpolate(bc_e)
bcs.append(dolfinx.fem.dirichletbc(u_bc_e, bc_e_dofs))

a_compiled, L_compiled = setup_variational_form(time=t)

A = dolfinx.fem.petsc.create_matrix(a_compiled, kind="mpi")
b = dolfinx.fem.petsc.create_vector(L_compiled, kind="mpi")

def assemble_system():
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix_mat(A, a_compiled, bcs=bcs)
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

# Prepare output
comm = MPI.COMM_WORLD
bp_names = ["Na_i.bp", "K_i.bp", "Cl_i.bp", "phi_i.bp",
            "Na_e.bp", "K_e.bp", "Cl_e.bp", "phi_e.bp"
        ]
bps = [dolfinx.io.VTXWriter(comm, bp_names[i], [uh_[i]], engine="BP5") for i in range(len(uh_))]
[bp.write(t) for bp in bps]

for _ in range(num_timesteps):
    t += dt.value # Increment time
    print(f"Time = {t}")

    if MMS_VERIFICATION:
        exact_solutions_funcs, _ = get_exact_solutions(t)
        u_bc_e.interpolate(dolfinx.fem.Expression(
                        exact_solutions_funcs["phi_e"],
                        u_bc_e.function_space.element.interpolation_points
                    )
                )
        
    # Setup variational form and assemble system
    a_compiled, L_compiled = setup_variational_form(time=t)
    assemble_system()

    # Solve and update
    ksp.solve(b, x)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dolfinx.fem.petsc.assign(x, uh_)
    
    phi_i_.x.array[:] = uh_[num_ions].x.array.copy()
    phi_e_.x.array[:] = uh_[2*num_ions+1].x.array.copy()

    num_iterations = ksp.getIterationNumber()
    converged_reason = ksp.getConvergedReason()
    print(f"Solver converged in: {num_iterations} with reason {converged_reason}")

    [bp.write(t) for bp in bps]

[bp.close() for bp in bps]

if MMS_VERIFICATION:
    phi_i_exact = exact_solutions_funcs["phi_i"]
    phi_e_exact = exact_solutions_funcs["phi_e"]
    error_ui = dolfinx.fem.form(
        inner(phi_i_ - phi_i_exact, phi_i_ - phi_i_exact) * dxI, entity_maps=entity_maps
    )
    error_ue = dolfinx.fem.form(
        inner(phi_e_ - phi_e_exact, phi_e_ - phi_e_exact) * dxE, entity_maps=entity_maps
    )
    local_ui = dolfinx.fem.assemble_scalar(error_ui)
    local_ue = dolfinx.fem.assemble_scalar(error_ue)
    global_ui = np.sqrt(mesh.comm.allreduce(local_ui, op=MPI.SUM))
    global_ue = np.sqrt(mesh.comm.allreduce(local_ue, op=MPI.SUM))
    print(f"L2(ui): {global_ui:.2e}\nL2(ue): {global_ue:.2e}")