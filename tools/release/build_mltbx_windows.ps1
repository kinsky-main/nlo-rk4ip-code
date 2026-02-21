param(
    [string]$BuildDir = "build-wheel-win",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

cmake -S . -B $BuildDir `
  -DNLO_INSTALL_GIT_HOOKS=OFF `
  -DNLO_BUMP_PATCH_ON_BUILD=OFF `
  -DBUILD_TESTING=OFF `
  -DNLOLIB_BUILD_DOCS=OFF `
  -DNLOLIB_BUILD_BENCHMARKS=OFF `
  -DNLOLIB_BUILD_EXAMPLES=OFF `
  -DNLO_SQLITE_USE_FETCHCONTENT=ON `
  -DNLO_ENABLE_VULKAN_BACKEND=ON `
  -DNLO_ENABLE_VKFFT=ON

matlab -batch "addpath('matlab'); package_mltbx('$BuildDir', '$Config')"
