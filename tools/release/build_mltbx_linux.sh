#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-wheel-linux}"
CONFIG="${2:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$(dirname "${BASH_SOURCE[0]}")/cmake_options.sh"

nlo_configure_verified "$BUILD_DIR" \
  INSTALL_GIT_HOOKS=OFF \
  BUMP_PATCH_ON_BUILD=OFF \
  BUILD_TESTING=OFF \
  NLOLIB_BUILD_DOCS=OFF \
  NLOLIB_BUILD_BENCHMARKS=OFF \
  NLOLIB_BUILD_EXAMPLES=OFF \
  SQLITE_USE_FETCHCONTENT=ON \
  ENABLE_VULKAN_BACKEND=ON \
  ENABLE_VKFFT=ON

cmake --build "$BUILD_DIR" --config "$CONFIG"

# package_mltbx reads the version back out of CMakeLists.txt to name the bundle
# and stamp its metadata, so the tree must already carry the version being
# released. Fail here rather than ship a bundle named after something else.
python3 tools/release/check_versions.py
matlab -batch "addpath('matlab'); package_mltbx('${BUILD_DIR}', '${CONFIG}')"
