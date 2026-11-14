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

PKG_DIR="$REPO_ROOT/python/nlolib"
mkdir -p "$PKG_DIR"

for candidate in \
  "$REPO_ROOT/python/libnlolib.so" \
  "$BUILD_DIR/src/libnlolib.so" \
  "$BUILD_DIR/src/$CONFIG/libnlolib.so"; do
  if [[ -f "$candidate" ]]; then
    cp -f "$candidate" "$PKG_DIR/libnlolib.so"
    break
  fi
done

if [[ ! -f "$PKG_DIR/libnlolib.so" ]]; then
  echo "Failed to locate libnlolib.so after build." >&2
  exit 1
fi

python3 -m pip install --upgrade build
python3 -m build

# repair rewrites the tag from the build machine's bare linux_x86_64 to the
# manylinux tag the binary has actually earned, under a new filename. The
# unrepaired wheel is removed so dist/ holds only publishable wheels -- PyPI
# rejects a linux_x86_64 tag outright.
if command -v auditwheel >/dev/null 2>&1; then
  for wheel in dist/*-linux_*.whl; do
    [[ -e "$wheel" ]] || continue
    auditwheel repair "$wheel" -w dist/
    rm -f "$wheel"
  done
else
  echo "auditwheel not found: dist/ holds unrepaired wheels that PyPI will reject." >&2
fi

echo "Built wheels in dist/"
