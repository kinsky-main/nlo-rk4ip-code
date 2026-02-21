#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-wheel-linux}"
CONFIG="${2:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

cmake -S . -B "$BUILD_DIR" \
  -DNLO_INSTALL_GIT_HOOKS=OFF \
  -DNLO_BUMP_PATCH_ON_BUILD=OFF \
  -DBUILD_TESTING=OFF \
  -DNLOLIB_BUILD_DOCS=OFF \
  -DNLOLIB_BUILD_BENCHMARKS=OFF \
  -DNLOLIB_BUILD_EXAMPLES=OFF \
  -DNLO_SQLITE_USE_FETCHCONTENT=ON \
  -DNLO_ENABLE_VULKAN_BACKEND=ON \
  -DNLO_ENABLE_VKFFT=ON

matlab -batch "addpath('matlab'); package_mltbx('${BUILD_DIR}', '${CONFIG}')"
