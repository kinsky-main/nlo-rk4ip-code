#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-wheel-macos}"
CONFIG="${2:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$(dirname "${BASH_SOURCE[0]}")/cmake_options.sh"

nlo_configure_verified "$BUILD_DIR" \
  INSTALL_GIT_HOOKS=OFF \
  BUMP_PATCH_ON_BUILD=ON \
  BUILD_TESTING=OFF \
  NLOLIB_BUILD_DOCS=OFF \
  NLOLIB_BUILD_BENCHMARKS=OFF \
  NLOLIB_BUILD_EXAMPLES=OFF \
  SQLITE_USE_FETCHCONTENT=ON \
  ENABLE_VULKAN_BACKEND=OFF \
  ENABLE_VKFFT=OFF

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
matlab -batch "addpath('matlab'); package_mltbx('${BUILD_DIR}', '${CONFIG}')"
