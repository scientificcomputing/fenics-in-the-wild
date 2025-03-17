# Installation instructions

This webpage uses several Python packages:

- [FEniCS/DOLFINx](https://github.com/FEniCS/dolfinx/) - A software for modelling partial differential equations ({term}`PDE`) with the finite element method ({term}`FEM`).
- [Scifem](https://github.com/scientificcomputing/scifem) - Convenience wrappers for on top of DOLFINx for scientific computing.
- [Wildmeshing](https://github.com/wildmeshing/wildmeshing-python) - A mesh generation software.

We recommend using either conda or docker for the installation

## Docker

Use the official DOLFINx [docker images](https://github.com/FEniCS/dolfinx/pkgs/container/dolfinx%2Flab/287932396?tag=stable)
Then use PIP to install scifem and wildmeshing with

```bash
python3 -m pip install scifem wildmeshing
```

## Conda

Create a conda enviroment based on the following `environment.yml` file:

```yaml
name: fenicsinthewild
channels:
  - conda-forge
dependencies:
  - fenics-dolfinx
  - pyvista
  - pip
  - git
  - scipy
  - trame-client
  - trame-vtk
  - trame-server
  - trame-vuetify
  - trame
  - ipywidgets
  - sphinx>=6.0.0
  - scifem
  - pip:
      - wildmeshing
variables:
  PYVISTA_TRAME_SERVER_PROXY_PREFIX: "/proxy/"
  PYVISTA_TRAME_SERVER_PROXY_ENABLED: "True"
  PYVISTA_OFF_SCREEN: false
  PYVISTA_JUPYTER_BACKEND: "html"
  LIBGL_ALWAYS_SOFTWARE: 1
  OMP_NUM_THREADS: 1
```
