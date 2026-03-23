FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

LABEL org.opencontainers.image.title="nlolib-cuda-multigpu-dev"
LABEL org.opencontainers.image.description="CUDA multi-GPU validation image for nlolib"
LABEL org.opencontainers.image.vendor="nlolib"

RUN apt-get update && apt-get install -y --no-install-recommends \
  cmake \
  ninja-build \
  build-essential \
  pkg-config \
  python3 \
  git \
  libfftw3-dev \
  libopenblas-dev \
  libvulkan-dev \
  glslang-tools \
  spirv-tools \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/nlolib
