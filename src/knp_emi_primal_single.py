from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from sys import argv
from ufl import (
    inner,
    grad,
    TestFunctions,
    TrialFunctions,
    TestFunction,
    TrialFunction,
    FacetNormal,
    MixedFunctionSpace,
    extract_blocks,
    Measure,
    Coefficient,
    ln,
    exp
)
import numpy as np
import numpy.typing as npt
import scifem
import matplotlib.pyplot as plt
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

# Functions spaces
element = ("Lagrange", 1)
V  = dolfinx.fem.functionspace(mesh, element)
Vi = dolfinx.fem.functionspace(omega_i, element)
Ve = dolfinx.fem.functionspace(omega_e, element)

# Species information
sodium = {"Di" : dolfinx.fem.Constant(mesh, 1.33e-9), # Intracellular diffusion coefficient
          "De" : dolfinx.fem.Constant(mesh, 1.33e-9), # Intracellular diffusion coefficient
          "z"  : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1)),  # Valence
          "init_i" : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(12)), # Initial intracellular concentration
          "init_e" : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(100)), # Initial extracellular concentration
          "name" : "Na" 
}
potassium = {"Di" : dolfinx.fem.Constant(mesh, 1.96e-9), # Intracellular diffusion coefficient
             "De" : dolfinx.fem.Constant(mesh, 1.96e-9), # Intracellular diffusion coefficient
             "z"  : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1)),  # Valence
             "init_i" : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(125)), # Initial intracellular concentration
             "init_e" : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(4)), # Initial extracellular concentration
             "name" : "K" 
}
chloride = {"Di" : dolfinx.fem.Constant(mesh, 2.03e-9), # Intracellular diffusion coefficient
            "De" : dolfinx.fem.Constant(mesh, 2.03e-9), # Intracellular diffusion coefficient
            "z"  : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(-1)),  # Valence
            "init_i" : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(137)), # Initial intracellular concentration
            "init_e" : dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(104)), # Initial extracellular concentration
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

# Get previous timestep potential functions and
# set initial conditions for the potentials
phi_i_ = uh_[num_ions]
phi_e_ = uh_[2*num_ions+1]
phi_i_init = dolfinx.fem.Constant(mesh, -0.06774) # Initial intracellular potential (V)
phi_e_init = dolfinx.fem.Constant(mesh, 0.0)      # Initial extracellular potential (V)
phi_rest     = dolfinx.fem.Constant(mesh, -0.065) # Resting membrane potential (V)

# Add source terms to ions
for idx, ion in enumerate(ion_list):
    ion["f_i"] = dolfinx.fem.Function(uh_[idx].function_space) # Intracellular source term
    ion["f_e"] = dolfinx.fem.Function(uh_[num_ions+1+idx].function_space) # Extracellular source term

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
temp      = 300   # temperature (K)
faraday   = 96485 # Faraday's constant (C/mol)
gas_const = 8.314 # Gas constant (J/(K*mol))
Cm_val = 1.0
sigma_e_val = 2.0
sigma_i_val = 1.0
timestep  = 1e-1
t_end = 1.0
phi_m_ = dolfinx.fem.Function(V)

sigma_e = dolfinx.fem.Constant(omega_e, dolfinx.default_scalar_type(sigma_e_val))
sigma_i = dolfinx.fem.Constant(omega_i, dolfinx.default_scalar_type(sigma_i_val))
Cm = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(Cm_val))
psi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(gas_const*temp/faraday))
n = FacetNormal(mesh)
dt = dolfinx.fem.Constant(mesh, timestep)
num_timesteps = int(t_end / dt.value)
t = dolfinx.fem.Constant(mesh, 0.0) # Time

# Ionic models
class Passive(object):
    tags = interface_markers
    stimulus_tags = None

    def __init__(self) -> None:
        pass

    def _eval(self, ion_idx: int):
        return phi_m_

class HodginHuxley(object):
    tags = interface_markers # Interface tags
    stimulus_tags = interface_markers
    time_steps_ODE = 25

    # Initial gating variable values [dimensionless]
    n_init = 0.27622914792
    m_init = 0.03791834627
    h_init = 0.68848921811

    # Stimulus current paramters
    g_Na_bar  = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1200))  # Na max conductivity [S/m**2]
    g_K_bar   = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(360))   # K max conductivity [S/m**2]
    a_syn     = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.002)) # Synaptic time constant [s]
    g_syn_bar = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(40))    # Synaptic conductivity [S/m**2]

    # Leak current conductivities [S/m**2]
    g_leak = {"Na" : dolfinx.fem.Constant(mesh, 2.0*0.5),
              "K"  : dolfinx.fem.Constant(mesh, 8.0*0.5),
              "Cl" : dolfinx.fem.Constant(mesh, 0.0)
            }

    def __init__(self) -> None:
        pass

    def __str__(self):
        return f"Hodgkin-Huxley"

    def _init(self):
        
        # Initialize gating variables in a continuous 
        # first-order Lagrange finite element space
        G = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        self.n = dolfinx.fem.Function(G)
        self.m = dolfinx.fem.Function(G)
        self.h = dolfinx.fem.Function(G)

        self.n.x.array[:] = self.n_init
        self.m.x.array[:] = self.m_init
        self.h.x.array[:] = self.h_init

    def _eval(self, ion_idx: int):
        """ Evaluate and return the channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        I_ch : float
            The value of the passive channel current.
        """
        ion = ion_list[ion_idx]
        ion["g_k"] = self.g_leak[ion["name"]] # Get the leak current for the current ion

        # Add stimulus and gating variable terms
        if ion["name"]=="Na":
            ion["g_k"] += self.g_Na_bar * self.m**3 * self.h
        elif ion["name"]=="K":
            ion["g_k"] += self.g_K_bar * self.n**4
        
        # Calculate channel current
        I_ch = ion["g_k"] * (phi_m_ - ion["E"])
        
        return I_ch
    
    def g_syn(self):
        condition = lambda t: dolfinx.default_scalar_type(1 if float(t) % 0.2 else 0)
        return self.g_syn_bar * exp(-condition(t) / self.a_syn)

    def _add_stimulus(self):
        """ Evaluate and return the stimulus part of the channel current for the Sodium ion. """
        ion = ion_list[0]
        assert ion["name"]=="Na"

        return self.g_syn() * (phi_m_ - ion["E"])

    def update_gating_variables(self):

        dt_ode = dt.value / self.time_steps_ODE # ODE timestep size

        V_M = 1000 * (phi_m_.x.array - phi_rest.value) # Convert membrane potential to mV
        print("V_M = ", V_M)

        alpha_n = 0.01e3 * (10.0 - V_M) / (np.exp((10.0 - V_M) / 10.0) - 1.0)
        beta_n = 0.125e3 * np.exp(-V_M / 80.0)
        alpha_m = 0.1e3 * (25.0 - V_M) / (np.exp((25.0 - V_M) / 10.0) - 1)
        beta_m = 4.0e3 * np.exp(-V_M / 18.0)
        alpha_h = 0.07e3 * np.exp(-V_M / 20.0)
        beta_h = 1.0e3 / (np.exp((30.0 - V_M) / 10.0) + 1)

        tau_y_n = 1.0 / (alpha_n + beta_n)
        tau_y_m = 1.0 / (alpha_m + beta_m)
        tau_y_h = 1.0 / (alpha_h + beta_h)

        y_inf_n = alpha_n * tau_y_n
        y_inf_m = alpha_m * tau_y_m
        y_inf_h = alpha_h * tau_y_h

        y_exp_n = np.exp(-dt_ode / tau_y_n)
        y_exp_m = np.exp(-dt_ode / tau_y_m)
        y_exp_h = np.exp(-dt_ode / tau_y_h)

        for _ in range(self.time_steps_ODE):
            self.n.x.array[:] = y_inf_n + (self.n.x.array.copy() - y_inf_n) * y_exp_n
            self.m.x.array[:] = y_inf_m + (self.m.x.array.copy() - y_inf_m) * y_exp_m
            self.h.x.array[:] = y_inf_h + (self.h.x.array.copy() - y_inf_h) * y_exp_h

passive_model = Passive()
hodgin_huxley_model = HodginHuxley()
hodgin_huxley_model._init()
ionic_models = [hodgin_huxley_model]

# Setup functions for the variational form
# and the preconditioner
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
        if isinstance(ion['init_i'], dolfinx.fem.Constant):
            uh_i.x.array[:] = ion['init_i'].value
        else:
            uh_i.interpolate(dolfinx.fem.Expression(
                                    ion['init_i'],
                                    uh_i.function_space.element.interpolation_points
                                    )
                                )
        if isinstance(ion['init_e'], dolfinx.fem.Constant):
            uh_e.x.array[:] = ion['init_e'].value
        else:
            uh_e.interpolate(dolfinx.fem.Expression(
                                    ion['init_e'],
                                    uh_e.function_space.element.interpolation_points
                                    )
                                )

        # Add contribution to alpha fraction
        alpha_i_sum += Di * z**2 * uh_i     # Intracellular contribution
        alpha_e_sum += De * z**2 * uh_e # Extracellular contribution

        # Calculate and update Nernst potential for current ion
        ion['E'] = (psi/z) * ln(uh_i / uh_e)

        # Set ion channel currents
        ion['I_ch'] = dict.fromkeys(interface_markers)

        for model in ionic_models:
            for interface_tag in model.tags:
                ion['I_ch'][interface_tag] = model._eval(idx)     # Calculate channel current

                if model.stimulus_tags is not None and interface_tag in model.stimulus_tags:
                        if ion['name']=='Na' and model.__str__()=='Hodgkin-Huxley':
                            ion['I_ch'][interface_tag] += model._add_stimulus()
                
                I_ch[interface_tag] += ion['I_ch'][interface_tag] # Add contribution to total channel current

    # Set initial conditions for the potentials
    if isinstance(phi_i_init, dolfinx.fem.Constant):
        uh_[num_ions].x.array[:] = phi_i_init.value
    else:    
        uh_[num_ions].interpolate(dolfinx.fem.Expression(
                                    phi_i_init,
                                    uh_[num_ions].function_space.element.interpolation_points
                                    )
                                )
    if isinstance(phi_e_init, dolfinx.fem.Constant):
        uh_[2*num_ions+1].x.array[:] = phi_e_init.value
    else:  
        uh_[2*num_ions+1].interpolate(dolfinx.fem.Expression(
                                        phi_e_init,
                                        uh_[2*num_ions+1].function_space.element.interpolation_points
                                    )
                                )

    # Setup bilinear and linear form   
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
            
    # Weak form - potential equations
    a -= dt * inner(J_phi_i, grad(vphi_i)) * dxI
    a -= dt * inner(J_phi_e, grad(vphi_e)) * dxE

    # Trace terms
    a += Cm/faraday * (tr_phi_i - tr_phi_e) * tr_vphi_i * dGamma
    a += Cm/faraday * (tr_phi_e - tr_phi_i) * tr_vphi_e * dGamma

    # Membrane currents
    for interface_tag in interface_markers:
        L += (1/faraday) * (dt*I_ch[interface_tag] - Cm*phi_m_) * (tr_vphi_e - tr_vphi_i) * dGamma(interface_tag)
        
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

def get_point_and_cell_on_interface():
    x = np.array([0.25, 0.5, 0.0])
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x)
    cc = colliding_cells.links(0)[0]
    
    return x, cc

def setup_projection_problem(phi_m: dolfinx.fem.Function, phi_diff: dolfinx.fem.Function):
    V = phi_m.function_space
    dx = Measure('dx', domain=V.mesh)
    eta, zeta = TrialFunction(V), TestFunction(V)
    a = inner(eta, zeta) * dx
    L = inner(phi_diff, zeta) * dx
    a_compiled = dolfinx.fem.form(a, entity_maps=entity_maps)
    L_compiled = dolfinx.fem.form(L, entity_maps=entity_maps)

    return dolfinx.fem.petsc.LinearProblem(a_compiled, L_compiled, bcs=[], u=phi_m)

projection_problem = setup_projection_problem(phi_m_, phi_i_(i_res)-phi_e_(i_res))

#----- Variational form and BCs -----#
a_compiled, L_compiled = setup_variational_form()

bc_e_dofs = dolfinx.fem.locate_dofs_topological(
    Ve, omega_e.topology.dim-1, sub_tag_e.find(boundary_marker)
)
bcs = []

# BC for potential
u_bc_e = dolfinx.fem.Function(exterior_spaces[-1])
# u_bc_e.interpolate(dolfinx.fem.Expression(
#                     phi_e_exact,
#                     Ve.element.interpolation_points
#                 )
#             )
bcs.append(dolfinx.fem.dirichletbc(u_bc_e, bc_e_dofs))

# BCs for concentrations
for idx, ion in enumerate(ion_list):
    # exact_func = exact_solutions_funcs[ion["name"]+"_e"]
    func = dolfinx.fem.Function(exterior_spaces[idx])
    # func.interpolate(dolfinx.fem.Expression(
    #                 exact_func,
    #                 Ve.element.interpolation_points
    #             )
    #         )
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
times = [0.0]

point, cell = get_point_and_cell_on_interface()
projection_problem.solve()
Vm = [phi_m_.eval(point, cell)]
print(Vm)

for _ in range(num_timesteps):
    # Increment time
    t.value += dt.value 
    times.append(t.value)
    print(f"Time = {t.value}")

    # Update gating variables
    hodgin_huxley_model.update_gating_variables()

    # Assemble system
    assemble_system()

    # Solve and update
    ksp.solve(b, x)
    print(f"Solver converged in: {ksp.getIterationNumber()} with reason {ksp.getConvergedReason()}")
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dolfinx.fem.petsc.assign(x, uh_)

    # Update membrane potential
    projection_problem.solve()
    Vm.append(phi_m_.eval(point, cell))

    # Write output
    [bp.write(t) for bp in bps]

# Close output files
[bp.close() for bp in bps]

# Plot membrane potential
Vm = np.array(Vm)
times = np.array(times)

plt.plot(times, Vm)
plt.show()