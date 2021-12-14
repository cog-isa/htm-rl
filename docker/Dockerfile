# =========== 1. HTM-CORE-FORK ================

# Based on the original Dockerfile from the htm.core repo

FROM amd64/python:3.9-slim as htm-core-fork

# ==> 1.2 Install htm.core installation prereqs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        make \
        cmake \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

# ==> 1.3 Init python virtual env
RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

# ==> 1.4
WORKDIR /app/deps
RUN git clone --depth 1 -b master https://github.com/ZhekaHauska/htm.core.git

# ==> 1.6 pip install reqs and htm.core
WORKDIR /app/deps/htm.core
RUN pip install --no-cache-dir --upgrade \
        setuptools \
        pip \
        wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

#! TODO merge with prev after testing
RUN rm -rf * \
    && rm -rf .*


# ============= 2. HIMA ==================
FROM amd64/python:3.9-slim as hima

# ==> 2.1.2 Install common prereqs
RUN DEBIAN_FRONTEND=noninteractive \
        apt-get update \
    && apt-get install -y \
        ca-certificates \
        wget \
        tar \
        xz-utils \
        git \
        zsh \
    && rm -rf /var/lib/apt/lists/*

# ==> 2.1.3 Install zsh with oh-my-zsh and set it as default shell
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
SHELL ["/bin/zsh", "-c"]

# ==> 2.2 get CoppeliaSim to `/app/deps/CoppeliaSimXXX`
WORKDIR /app/deps/

# ==> 2.2.1 Set env vars
ARG CoppeliaSimFilename=CoppeliaSim_Edu_V4_2_0_Ubuntu20_04

ENV COPPELIASIM_ROOT /app/deps/$CoppeliaSimFilename
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/platforms
ENV QT_QPA_PLATFORM_PLUGIN_PATH $COPPELIASIM_ROOT

# ==> 2.2.2 Get CoppeliaSim archive, extract, remove
RUN wget https://www.coppeliarobotics.com/files/$CoppeliaSimFilename.tar.xz

#! TODO merge with prev after testing
RUN tar -xf $CoppeliaSimFilename.tar.xz \
    && rm $CoppeliaSimFilename.tar.xz

# ==> 2.3.1 Install prereqs for PyRep
RUN DEBIAN_FRONTEND=noninteractive \
        apt-get update \
    && apt-get install -y \
        libglib2.0-0  \
        libgl1-mesa-glx \
        xcb \
        "^libxcb.*" \
        libx11-xcb-dev \
        libglu1-mesa-dev \
        libxrender-dev \
        libxi6 \
        libdbus-1-3 \
        libfontconfig1 \
        qtbase5-dev qtdeclarative5-dev libqt5webkit5-dev libsqlite3-dev qttools5-dev-tools \
        # qtchooser qt5-qmake qtbase5-dev-tools \
        # qt5-default \
        libffi-dev \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

# ==> 2.3.2 Clone PyRep
RUN git clone --depth 1 --branch master https://github.com/stepjam/PyRep.git

# ==========> 2.4 Start building Python Virtual Env

# ==> 2.5 Copy venv with `htm.core` from `htm-core-fork` image
COPY --from=htm-core-fork /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip and other setup tools are copied updated from htm-core-fork

# ==> 2.6 Install PyRep

WORKDIR /app/deps/PyRep
RUN pip install --no-cache-dir --upgrade \
        cython \
    && pip install --no-cache-dir --upgrade \
        -r requirements.txt \
    && pip install --no-cache-dir --upgrade \
        .

#! TODO merge with prev
WORKDIR /app/deps
RUN rm -r PyRep

# ==> 2.7 Install HIMA
WORKDIR /app

