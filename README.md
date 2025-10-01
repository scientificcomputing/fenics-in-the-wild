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
spack env create fenicsx-main
spack env activate fenicsx-main
spack add py-pip py-h5py py-scipy py-pyvista
spack add py-fenics-dolfinx@main ^fenics-dolfinx+adios2 ^adios2+python ^petsc+mumps+int64 cflags="-O3" fflags="-O3"
spack add py-nanobind py-adios4dolfinx@main py-scifem@main+adios2+biomed
spack install
python3 -m pip install wildmeshing
```
