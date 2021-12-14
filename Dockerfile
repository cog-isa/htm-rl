# Stage 1: build of htm.core on the target platform
FROM amd64/python:3.9-slim as build

RUN apt-get update && apt-get install -y \
    #? pip installed later
    cmake \
    make \
    g++ \
    git \
    python3-dev \
    python3-pip \
    # py3-numpy \
    zsh \
  && rm -rf /var/lib/apt/lists/*

RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
SHELL ["/bin/zsh", "-c"]

# RUN ln -s /usr/bin/python3 /usr/local/bin/python && python --version
  # && ln -s /usr/bin/pip3 /usr/local/bin/pip
RUN pip install --no-cache-dir --upgrade setuptools pip wheel

WORKDIR /app

# Stage 1.1: clone and cd to htm.core
RUN git clone --depth 1 -b master https://github.com/ZhekaHauska/htm.core.git

WORKDIR /app/htm.core

# Stage 1.2: pip install htm.core
# RUN python -m pip uninstall -y htm.core
RUN pip install --no-cache-dir \
  # Explicitly specify --cache-dir, --build, and --no-clean so that build
  # artifacts may be extracted from the container later.  Final built python
  # packages can be found in /usr/local/src/htm.core/bindings/py/dist
  # --cache-dir /usr/local/src/htm.core/pip-cache \
  # --build /usr/local/src/htm.core/pip-build \
  # --no-clean \
  -r requirements.txt

# RUN mkdir -p build/scripts && \
#     cd build/scripts && \
#     cmake ../.. -DCMAKE_BUILD_TYPE=Release -DBINDING_BUILD=Python3 && \
#     make -j4 && make install

RUN pip install .
# RUN python setup.py install

# State 2: Setup CoppeliaSim + PyRep

# RUN apt-get update && apt-get install -y \
RUN DEBIAN_FRONTEND=noninteractive \
    apt update \
  && apt install -y \
    ca-certificates

RUN DEBIAN_FRONTEND=noninteractive \
    apt update \
  && apt install -y \
    zsh \
    wget \
    libglib2.0-0  \
    libgl1-mesa-glx \
    xcb \
    "^libxcb.*" \
    libx11-xcb-dev

RUN DEBIAN_FRONTEND=noninteractive \
    apt update \
  && apt install -y \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi6 \
    libdbus-1-3 \
    libfontconfig1

RUN DEBIAN_FRONTEND=noninteractive \
    apt update \
  && apt install -y \
    xvfb \
    tar \
    git \
    python3-pip

RUN DEBIAN_FRONTEND=noninteractive \
    apt update \
  && apt install -y \
    qtbase5-dev qtdeclarative5-dev libqt5webkit5-dev libsqlite3-dev qttools5-dev-tools \
    # qtchooser qt5-qmake qtbase5-dev-tools \
    # qt5-default \
    libffi-dev

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app/deps/
ARG CoppeliaSimFilename=CoppeliaSim_Edu_V4_2_0_Ubuntu20_04

ENV COPPELIASIM_ROOT /app/deps/$CoppeliaSimFilename
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/platforms
ENV QT_QPA_PLATFORM_PLUGIN_PATH $COPPELIASIM_ROOT

RUN wget https://www.coppeliarobotics.com/files/$CoppeliaSimFilename.tar.xz && \
  tar -xf $CoppeliaSimFilename.tar.xz

RUN git clone --depth 1 --branch master https://github.com/stepjam/PyRep.git

WORKDIR /app/deps/PyRep
RUN pip install cython
RUN pip install -r requirements.txt
RUN pip install . 


# Stage 3: Setup HIMA

WORKDIR /app
RUN pip install --no-cache-dir \
  numpy \
  matplotlib \
  jupyterlab \
  ruamel.yaml \
  tqdm \
  wandb \
  mock \
  imageio \
  seaborn

RUN pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN git clone --depth 1 --branch master https://github.com/cog-isa/htm-rl.git

WORKDIR /app/htm-rl/htm_rl
RUN pip install -e .

WORKDIR /app/htm-rl/htm_rl/htm_rl
