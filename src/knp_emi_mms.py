import dolfinx
from ufl import (sin,
                 cos,
                 exp,
                 pi,
                 SpatialCoordinate,
                 FacetNormal,
                 div,
                 grad,
                 dot,
                 diff,
                 variable
                )

class ExactSolutionsKNPEMI:
    def __init__(self, mesh: dolfinx.mesh.Mesh):
        self.mesh = mesh
        self.x, self.y = SpatialCoordinate(mesh)
        self.t = 0 # Time
        self.n = FacetNormal(mesh)


    def get_exact_solutions(self):
        x, y, t = self.x, self.y, self.t # Ease notation

        # sodium (Na) concentration
        Na_i_exact = 0.7 + 0.3 * sin(2*pi * x) * sin(2*pi * y) * exp(
            -t
        )
        Na_e_exact = 1.0 + 0.6 * sin(2*pi * x) * sin(2*pi * y) * exp(
            -t
        )
        # potassium (K) concentration
        K_i_exact = 0.3 + 0.3 * sin(2*pi * x) * sin(2*pi * y) * exp(
            -t
        )
        K_e_exact = 1.0 + 0.2 * sin(2*pi * x) * sin(2*pi * y) * exp(
            -t
        )
        # chloride (Cl) concentration
        Cl_i_exact = 1.0 + 0.6 * sin(2*pi * x) * sin(2*pi * y) * exp(
            -t
        )
        Cl_e_exact = 2.0 + 0.8 * sin(2*pi * x) * sin(2*pi * y) * exp(
            -t
        )
        # potentials
        phi_i_exact = cos(2*pi * x) * cos(2*pi * y) * (1 + exp(-t))
        phi_e_exact = cos(2*pi * x) * cos(2*pi * y)
    
        exact_solutions = {"Na_i"  : Na_i_exact,
                           "Na_e"  : Na_e_exact,
                           "K_i"   : K_i_exact,
                           "K_e"   : K_e_exact,
                           "Cl_i"  : Cl_i_exact,
                           "Cl_e"  : Cl_e_exact,
                           "phi_i" : phi_i_exact,
                           "phi_e" : phi_e_exact
        }

        return exact_solutions

    def get_source_terms(self):
        # Valences
        z_Na = 1
        z_K  = 1
        z_Cl = -1
        n = self.n # Facet normal vector
        t = variable(self.t) # Time
        exact_solutions = self.get_exact_solutions()
        exact_gradients = dict.fromkeys(exact_solutions)
        exact_fluxes = dict.fromkeys(exact_solutions)
        for key, function in zip(exact_solutions.keys(), exact_solutions.values()):
            exact_gradients[key] = grad(function)

        # Membrane potential
        phi_m_exact = exact_solutions["phi_i"] - exact_solutions["phi_e"]

        # Compartmental fluxes = -grad(k_r) + z_k * k_r * grad(phi_r)
        J_Na_i = -exact_gradients["Na_i"] - z_Na*exact_solutions["Na_i"] * exact_gradients["phi_i"]
        J_Na_e = -exact_gradients["Na_e"] - z_Na*exact_solutions["Na_e"] * exact_gradients["phi_e"]
        J_K_i  = -exact_gradients["K_i"]  - z_K *exact_solutions["K_i"]  * exact_gradients["phi_i"]
        J_K_e  = -exact_gradients["K_e"]  - z_K *exact_solutions["K_e"]  * exact_gradients["phi_e"]
        J_Cl_i = -exact_gradients["Cl_i"] - z_Cl*exact_solutions["Cl_i"] * exact_gradients["phi_i"]
        J_Cl_e = -exact_gradients["Cl_e"] - z_Cl*exact_solutions["Cl_e"] * exact_gradients["phi_e"]

        # Source terms per species = dk_r/dt + div(J_k_r)
        f_Na_i = diff(exact_solutions["Na_i"], t) + div(J_Na_i)
        f_Na_e = diff(exact_solutions["Na_e"], t) + div(J_Na_e)
        f_K_i  = diff(exact_solutions["K_i"],  t) + div(J_K_i )
        f_K_e  = diff(exact_solutions["K_e"],  t) + div(J_K_e )
        f_Cl_i = diff(exact_solutions["Cl_i"], t) + div(J_Cl_i)
        f_Cl_e = diff(exact_solutions["Cl_e"], t) + div(J_Cl_e)

        # Potential source terms = -F * sum_k(z_k * div(J_k_r))
        f_phi_i = -(z_Na*div(J_Na_i) + z_K*div(J_K_i) + z_Cl*div(J_Cl_i))
        f_phi_e = -(z_Na*div(J_Na_e) + z_K*div(J_K_e) + z_Cl*div(J_Cl_e))

        # Total intracellular membrane flux = F * sum_k(z^k * J_k_i)
        # and intracellular membrane currents = dot(total_flux_intra, n_i)
        total_flux_intra = z_Na*J_Na_i + z_K*J_K_i + z_Cl*J_Cl_i
        Im_intra = dot(total_flux_intra, n)

        # Total extracellular membrane flux = -F * sum_k(z^k * J_k_i)
        # and extracellular membrane currents = dot(total_flux_extra, n_e) = -dot(total_flux_intra, n_i) = -Im_intra
        total_flux_extra = -(z_Na*J_Na_e + z_K*J_K_e + z_Cl*J_Cl_e)
        Im_extra = -Im_intra

        # Ion channel currents
        Ich_Na = phi_m_exact
        Ich_K  = phi_m_exact
        Ich_Cl = phi_m_exact
        Ich = Ich_Na + Ich_K + Ich_Cl

        # Equation for the membrane potential source term: f = Cm*d(phi_m)/dt - (Im - Ich) 
        # where we choose Im = F * sum_k(z^k * dot(J_i_k, n_i)) = total_flux_intra
        f_phi_m = diff(phi_m_exact, t) + Ich - Im_intra

        # Coupling condition for Im: Im_intra = Im_extra + f
        # which yields f = Im_intra - Im_extra
        f_gamma = Im_intra - Im_extra

        source_terms = {"f_Na_i"  : f_Na_i,
                        "f_K_i"   : f_K_i,
                        "f_Cl_i"  : f_Cl_i,
                        "f_phi_i" : f_phi_i,
                        "f_Na_e"  : f_Na_e,
                        "f_K_e"   : f_K_e,
                        "f_Cl_e"  : f_Cl_e,
                        "f_phi_e" : f_phi_e,
                        "f_phi_m" : f_phi_m,
                        "f_gamma" : f_gamma,
                        "J_Na_e"  : J_Na_e,
                        "J_K_e"   : J_K_e,
                        "J_Cl_e"  : J_Cl_e
        }
        
        return source_terms