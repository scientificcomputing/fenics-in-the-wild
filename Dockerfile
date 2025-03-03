FROM ghcr.io/fenics/dolfinx/lab:stable


# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
# 24.04 adds `ubuntu` as uid 1000;
# remove it if it already exists before creating our user
RUN id -nu ${NB_UID} && userdel --force $(id -nu ${NB_UID}) || true; \
    useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME=/home/${NB_USER}

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

WORKDIR ${HOME}/data
# Download stl files
RUN apt-get update && apt-get install -y wget unzip
RUN wget -nc https://zenodo.org/records/14536218/files/mri2femii-chp2-dataset.tar.gz
RUN tar xvzf mri2femii-chp2-dataset.tar.gz -C ${WILDFENICS_EXTRACT_PATH}
ENV WILDFENICS_EXTRACT_PATH="/root/data/"
ENV WILDFENICS_DATA_PATH=${WILDFENICS_EXTRACT_PATH}/mri2femii-chp2-dataset/Gonzo/output

WORKDIR ${HOME}
ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="/proxy/"
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"
ENV PYVISTA_JUPYTER_BACKEND="html"
ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1

RUN python3 -m pip install .[dev]

USER ${NB_USER}
ENTRYPOINT []