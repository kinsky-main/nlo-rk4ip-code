#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-wheel-linux}"
CONFIG="${2:-Release}"

# The toolbox is built from this repository plus the core nlolib sources,
# which core_dir.sh locates: ./core (tools/fetch_core.sh), or the parent
# directory when this repository is the matlab/ submodule of the core
# repository.
source "$(dirname "${BASH_SOURCE[0]}")/core_dir.sh"
source "$(dirname "${BASH_SOURCE[0]}")/cmake_options.sh"
cd "$BINDING_ROOT"

NLO_SOURCE_DIR="$CORE_DIR" nlo_configure_verified "$BUILD_DIR" \
  NLOLIB_MATLAB_DIR="$BINDING_ROOT" \
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

# package_mltbx names the bundle and stamps its metadata from the VERSION file.
matlab -batch "addpath('${BINDING_ROOT}'); package_mltbx('${BUILD_DIR}', '${CONFIG}')"
