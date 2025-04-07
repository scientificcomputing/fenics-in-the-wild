# Building the webpage

To build the webpage, install [Docker](https://www.docker.com/)
and build the docker image with the following instructions

```bash
docker build -f ./docker/Dockerfile.webpage -t wildfenicswebpagebuilder .
```

and then build the webpage with:

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --shm-size=512m wildfenicswebpagebuilder
```

docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --shm-size=512m wildfenicswebpagebuilder

The webpage is then build in [\_build/html/index.html](_build/html/index.html).

## Local development environment

Add the entrypoint `/bin/bash` to the above command, i.e.

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --shm-size=512m --entrypoint=/bin/bash wildfenicswebpagebuilder
```

### Jupyter lab development

To use jupyter lab with container call

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --entrypoint=/bin/bash -p 8888:8888 wildfenicswebpagebuilder
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

# Installation with spack

Clone spack from its git repository (https://github.com/spack/spack)

```bash
. ./spack/share/spack/setup-env.sh
spack env create fenicsx-stable
spack env activate fenicsx-stable
spack add py-pip py-h5py py-scipy py-pyvista
spack add py-fenics-dolfinx fenics-dolfinx+adios2 ^adios2+python ^petsc+mumps+int64 cflags="-O3" fflags="-O3"
spack add py-scikit-build-core py-nanobind
spack install
python3 -m pip install adios4dolfinx wildmeshing git+https://github.com/scientificcomputing/scifem.git
```

Currently one has to work around:
https://github.com/ornladios/ADIOS2/issues/4485
by locating where ADIOS2 puts its Python package and add it to the Python path.

```

```
