FROM ubuntu:24.04

LABEL org.opencontainers.image.title="nlolib-cpu-dev"
LABEL org.opencontainers.image.description="CPU-only validation image for nlolib"
LABEL org.opencontainers.image.vendor="nlolib"

RUN apt-get update && apt-get install -y --no-install-recommends \
  cmake \
  ninja-build \
  build-essential \
  pkg-config \
  python3 \
  libfftw3-dev \
  libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/nlolib
