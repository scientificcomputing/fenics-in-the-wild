FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=wildfenics
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install apt dependencies
USER $USERNAME
RUN sudo apt-get update && \
    sudo apt-get upgrade -y && \
    sudo apt-get install -y python3-dev \
    python3-venv \
    python3-pip \
    git

# Create virtual environment
ENV VIRTUAL_ENV=fenics-in-the-wild
ENV PATH=/home/${USERNAME}/${VIRTUAL_ENV}/bin:$PATH
RUN python3 -m venv /home/${USERNAME}/${VIRTUAL_ENV}
RUN python3 -m pip install setuptools[pyproject.toml]

# Install doc build dependencies
WORKDIR /home/${USERNAME}

COPY pyproject.toml .
RUN python3 -m pip install . -v


CMD ["jupyter", "book", "build", "."]