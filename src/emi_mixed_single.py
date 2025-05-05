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
    dot
)
import numpy as np
import numpy.typing as npt
import scifem
from packaging.version import Version


x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75


def interior_marker(x, tol=1e-12):
    lower_bound = lambda x, i, bound: x[i] >= bound - tol
    upper_bound = lambda x, i, bound: x[i] <= bound + tol
    return (
        lower_bound(x, 0, x_L)
        & lower_bound(x, 1, y_L)
        & upper_bound(x, 0, x_U)
        & upper_bound(x, 1, y_U)
    )


# Steps to set up submeshes and interface
M = 100
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, M, M, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)


interior_cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, interior_marker)

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

# Integration measures for volumes
dx = Measure("dx", domain=mesh, subdomain_data=ct)
dGamma = Measure(
    "dS",
    domain=mesh,
    subdomain_data=[(2, ordered_integration_data.flatten())],
    subdomain_id=2,
)
ds = Measure("ds", domain=mesh, subdomain_data=ft,
            subdomain_id=boundary_marker)

V = dolfinx.fem.functionspace(mesh, ("DG", 0))
S = dolfinx.fem.functionspace(mesh, ("RT", 1))
W = MixedFunctionSpace(V, S)
sigma = dolfinx.fem.Function(V)

sigma_e = 2.0
sigma_i = 1.0
sigma.interpolate(lambda x: np.full(x.shape[1],sigma_e), cells0=ct.find(exterior_marker))
sigma.interpolate(lambda x: np.full(x.shape[1],sigma_i), cells0=ct.find(interior_marker))
sigma.x.scatter_forward()

u, J = TrialFunctions(W)
q, tau = TestFunctions(W)
Cm = dolfinx.fem.Constant(mesh, 1.0)
dt = dolfinx.fem.Constant(mesh, 1e-4)
T = Cm / dt

n = FacetNormal(mesh)
n_i = n("+")
n_e = n("-")
a = inner(inv(sigma)*J, tau) * dx
a += inv(T)*dt * dot(J("+"), n_i) * dot(tau("+"), n_i) * dGamma
a -= u * div(tau) * dx
a -= q * div(J) * dx

x, y = SpatialCoordinate(mesh)
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
a_compiled = dolfinx.fem.form(extract_blocks(a))
L_compiled = dolfinx.fem.form(extract_blocks(L))


A = dolfinx.fem.petsc.assemble_matrix(a_compiled, kind="mpi", bcs=[])
A.assemble()
b = dolfinx.fem.petsc.assemble_vector(L_compiled, kind="mpi")
# bcs1 = dolfinx.fem.bcs_by_block(
#     dolfinx.fem.extract_function_spaces(a_compiled, 1), [bc]
# )
# dolfinx.fem.petsc.apply_lifting(b, a_compiled, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# bcs0 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(L_compiled), [bc])
# dolfinx.fem.petsc.set_bc(b, bcs0)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.setErrorIfNotConverged(True)

u = dolfinx.fem.Function(V)
tau = dolfinx.fem.Function(S)
x = b.duplicate()
ksp.solve(b, x)
x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dolfinx.fem.petsc.assign(x, [u, tau])


with dolfinx.io.VTXWriter(omega_i.comm, "u.bp", [u], engine="BP5") as bp:
    bp.write(0.0)


error_ui = dolfinx.fem.form(
    inner(u - ui_exact, u - ui_exact) * dxI
)
error_ue = dolfinx.fem.form(
    inner(u - ue_exact, u - ue_exact) * dxE
)
local_ui = dolfinx.fem.assemble_scalar(error_ui)
local_ue = dolfinx.fem.assemble_scalar(error_ue)
global_ui = np.sqrt(mesh.comm.allreduce(local_ui, op=MPI.SUM))
global_ue = np.sqrt(mesh.comm.allreduce(local_ue, op=MPI.SUM))
print(f"L2(ui): {global_ui:.2e}\n L2(ue): {global_ue:.2e}")
