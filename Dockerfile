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
ENV WILDFENICS_EXTRACT_PATH="${HOME}/src/stl_files"
ENV WILDFENICS_DATA_PATH=${WILDFENICS_EXTRACT_PATH}/mhornkjol-mri2fem-ii-chapter-3-code-ff74dab/stl_files
# Download stl files
RUN apt-get update && apt-get install -y wget unzip
RUN wget -nc https://zenodo.org/records/10808334/files/mhornkjol/mri2fem-ii-chapter-3-code-v1.0.0.zip && \
    unzip mri2fem-ii-chapter-3-code-v1.0.0.zip -d ${WILDFENICS_EXTRACT_PATH}


ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="/proxy/"
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"
ENV PYVISTA_JUPYTER_BACKEND="html"
ENV DISPLAY=:99
RUN python3 -m pip install .[dev]

USER ${NB_USER}
ENTRYPOINT []