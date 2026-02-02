#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends \
  cmake \
  ninja-build \
  build-essential \
  pkg-config \
  libfftw3-dev \
  python3-cffi
rm -rf /var/lib/apt/lists/*
