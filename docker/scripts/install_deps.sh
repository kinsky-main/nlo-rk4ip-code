#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends \
  cmake \
  ninja-build \
  build-essential \
  pkg-config \
  libfftw3-dev \
  libvulkan-dev \
  glslang-tools \
  vulkan-tools \
  spirv-tools
rm -rf /var/lib/apt/lists/*
