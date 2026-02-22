#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-wheel-linux}"
CONFIG="${2:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

cmake -S . -B "$BUILD_DIR" \
  -DNLO_INSTALL_GIT_HOOKS=OFF \
  -DNLO_BUMP_PATCH_ON_BUILD=ON \
  -DBUILD_TESTING=OFF \
  -DNLOLIB_BUILD_DOCS=OFF \
  -DNLOLIB_BUILD_BENCHMARKS=OFF \
  -DNLOLIB_BUILD_EXAMPLES=OFF \
  -DNLO_SQLITE_USE_FETCHCONTENT=ON \
  -DNLO_ENABLE_VULKAN_BACKEND=ON \
  -DNLO_ENABLE_VKFFT=ON

cmake --build "$BUILD_DIR" --config "$CONFIG"
python3 - <<'PY'
import pathlib
import re

root = pathlib.Path(".")
cmake_text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
match = re.search(r"project\([^)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", cmake_text)
if not match:
    raise RuntimeError("Could not parse CMake project version")
version = match.group(1)

pyproject = root / "pyproject.toml"
text = pyproject.read_text(encoding="utf-8")
updated, count = re.subn(
    r'(?m)^version\s*=\s*"[0-9]+\.[0-9]+\.[0-9]+"$',
    f'version = "{version}"',
    text,
    count=1,
)
if count != 1:
    raise RuntimeError("Could not update pyproject version")
pyproject.write_text(updated, encoding="utf-8")
print(version)
PY

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

if command -v auditwheel >/dev/null 2>&1; then
  for wheel in dist/*.whl; do
    auditwheel repair "$wheel" -w dist/
  done
fi

echo "Built wheels in dist/"
