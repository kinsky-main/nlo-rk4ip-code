#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-wheel-macos}"
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
  ENABLE_VULKAN_BACKEND=OFF \
  ENABLE_VKFFT=OFF

cmake --build "$BUILD_DIR" --config "$CONFIG"

PKG_DIR="$REPO_ROOT/python/nlolib"
mkdir -p "$PKG_DIR"

for candidate in \
  "$REPO_ROOT/python/libnlolib.dylib" \
  "$BUILD_DIR/src/libnlolib.dylib" \
  "$BUILD_DIR/src/$CONFIG/libnlolib.dylib"; do
  if [[ -f "$candidate" ]]; then
    cp -f "$candidate" "$PKG_DIR/libnlolib.dylib"
    break
  fi
done

if [[ ! -f "$PKG_DIR/libnlolib.dylib" ]]; then
  echo "Failed to locate libnlolib.dylib after build." >&2
  exit 1
fi

python3 -m pip install --upgrade build
python3 -m build

if command -v delocate-wheel >/dev/null 2>&1; then
  for wheel in dist/*.whl; do
    delocate-wheel -w dist/ "$wheel"
  done
fi

echo "Built wheels in dist/"
