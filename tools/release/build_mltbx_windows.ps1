param(
    [string]$BuildDir = "build-wheel-win",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

cmake -S . -B $BuildDir `
  -DNLO_INSTALL_GIT_HOOKS=OFF `
  -DNLO_BUMP_PATCH_ON_BUILD=ON `
  -DBUILD_TESTING=OFF `
  -DNLOLIB_BUILD_DOCS=OFF `
  -DNLOLIB_BUILD_BENCHMARKS=OFF `
  -DNLOLIB_BUILD_EXAMPLES=OFF `
  -DNLO_SQLITE_USE_FETCHCONTENT=ON `
  -DNLO_ENABLE_VULKAN_BACKEND=ON `
  -DNLO_ENABLE_VKFFT=ON

cmake --build $BuildDir --config $Config
@'
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
'@ | python -
matlab -batch "addpath('matlab'); package_mltbx('$BuildDir', '$Config')"
