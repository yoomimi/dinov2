FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1

# -----------------------
# System packages
# -----------------------
RUN apt-get update && apt-get install -y \
    git wget curl ca-certificates \
    software-properties-common \
    ffmpeg \
    tmux \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Python 3.10 (correct way on focal)
# -----------------------
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# pip for python3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3.10

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --upgrade pip


# -----------------------
# PyTorch (CUDA 11.3)
# -----------------------
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu113

# -----------------------
# Python deps (DINOv2 feature extraction)
# -----------------------
RUN pip install \
    timm \
    einops \
    opencv-python \
    tqdm \
    scipy \
    scikit-learn \
    pillow \
    h5py \
    psutil

# -----------------------
# Workspace
# -----------------------
WORKDIR /workspace
