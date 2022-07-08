# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.234.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.10-bullseye"
ARG CUDA_SUPPORT="nocuda"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT} as base-amd64-nocuda

# Install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 --disable-pip-version-check --no-cache-dir install \
    opencv-python-headless==4.5.5.62 \
    Shapely==1.8.1.post1 \
    onnxruntime==1.10.0 \
    Pillow==8.4.0

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY iot-edge-solution/modules/samplemodule/requirements.txt /tmp/pip-tmp/requirements.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt
RUN rm -rf /tmp/pip-tmp


FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT} as base-arm64-nocuda

# Install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    ffmpeg \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 --disable-pip-version-check --no-cache-dir install \
    opencv-python-headless==4.5.5.62 \
    Shapely==1.8.1.post1 \
    onnxruntime==1.10.0 \
    Pillow==8.4.0

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY iot-edge-solution/modules/samplemodule/requirements.txt /tmp/pip-tmp/requirements.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt
RUN rm -rf /tmp/pip-tmp


FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 as base-amd64-cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CPATH="${CPATH}:/usr/local/cuda-11.0/targets/aarch64-linux/include"
ENV LIBRARY_PATH="${LIBRARY_PATH}:/usr/local/cuda-11.0/targets/aarch64-linux/lib"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    python3-pip \
    build-essential \
    python3-dev \
    nano \
    curl \
    wget \
    python3-opencv \
    python3-libnvinfer \
    python3-shapely \
    python3-venv \
    netbase \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir setuptools Cython wheel
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir Pillow==8.4.0
RUN pip3 install --no-cache-dir opencv-python
RUN pip3 install --no-cache-dir pycuda

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/local/bin/python

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

COPY .devcontainer/scripts/*.sh .devcontainer/scripts/*.env /tmp/library-scripts/


# Setup default python tools in a venv via pipx to avoid conflicts
RUN pip uninstall black -y
ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=/usr/local/py-utils/bin
ENV PATH=${PATH}:${PIPX_BIN_DIR}
RUN bash /tmp/library-scripts/python-tools.sh "${PIPX_HOME}" "${USERNAME}" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*
RUN ln -s $PIPX_HOME/bin/black /usr/local/bin/black

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY iot-edge-solution/modules/samplemodule/requirements.txt /tmp/pip-tmp/requirements.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt
RUN rm -rf /tmp/pip-tmp

FROM nvcr.io/nvidia/l4t-base:r32.4.4 as base-arm64-cuda

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

# Install prerequisites (many of these are for numpy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    libopenmpi2 \
    openmpi-bin \
    openmpi-common \
    gfortran \
    cmake \
    nano \
    curl \
    gnupg2 \
    dirmngr \
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install python packages
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir setuptools Cython wheel
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir pycuda six
RUN pip3 install --no-cache-dir boto3 pandas
RUN pip3 install --no-cache-dir Pillow==8.4.0
# alias python3 -> python
RUN rm /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    ln -s /usr/bin/python3 /usr/local/bin/python

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-scipy \
    python3-sklearn \
    python3-sklearn-lib \
    python3-shapely \
    python3-venv \
    netbase \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install OpenCV (with CUDA)#
ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

RUN apt-get purge -y '*opencv*' || echo "previous OpenCV installation not found" && \
    mkdir opencv && \
    cd opencv && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${OPENCV_URL} -O ${OPENCV_DEB} && \
    tar -xzvf ${OPENCV_DEB} && \
    dpkg -i --force-depends *.deb && \
    apt-get update && \
    apt-get install -y -f --no-install-recommends && \
    dpkg -i *.deb && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    cd ../ && \
    rm -rf opencv && \
    cp -r /usr/include/opencv4 /usr/local/include/opencv4 && \
    cp -r /usr/lib/python3.6/dist-packages/cv2 /usr/local/lib/python3.6/dist-packages/cv2

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

COPY .devcontainer/scripts/*.sh .devcontainer/scripts/*.env /tmp/library-scripts/

# Setup default python tools in a venv via pipx to avoid conflicts
RUN pip uninstall black -y
ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=/usr/local/py-utils/bin
ENV PATH=${PATH}:${PIPX_BIN_DIR}
RUN bash /tmp/library-scripts/python-tools.sh "${PIPX_HOME}" "${USERNAME}" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*
RUN ln -s $PIPX_HOME/bin/black /usr/local/bin/black

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY iot-edge-solution/modules/samplemodule/requirements.txt /tmp/pip-tmp/requirements.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt
RUN rm -rf /tmp/pip-tmp

FROM base-${TARGETARCH}-${CUDA_SUPPORT} AS base

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# need to add user to group video to access GPU
RUN usermod -a -G video $USERNAME
RUN echo "Final - ARCH is equal to ${TARGETARCH}"