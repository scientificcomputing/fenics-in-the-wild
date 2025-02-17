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

RUN python3 -m pip install .

# Download stl files
RUN apt-get update && apt-get install -y wget unzip
RUN wget -nc https://zenodo.org/records/10808334/files/mhornkjol/mri2fem-ii-chapter-3-code-v1.0.0.zip && \
    unzip mri2fem-ii-chapter-3-code-v1.0.0.zip -d stl_files

ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="/proxy/"
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"

USER ${NB_USER}
CMD ["python3", "-m", "jupyter", "book", "build" "."]
ENTRYPOINT []