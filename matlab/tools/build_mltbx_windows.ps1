param(
    [string]$BuildDir = "build-mltbx-win",
    [string]$Config = "Release",
    # Core nlolib checkout to build from. Defaults to .\core (the checkout the
    # workflows and tools/fetch_core.sh create at the revision CORE_REF pins),
    # then to the parent directory, for when this repository is the matlab\
    # submodule of the core repository.
    [string]$CoreDir = $env:NLOLIB_CORE_DIR
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not $CoreDir) {
    foreach ($candidate in @((Join-Path $repoRoot "core"), (Join-Path $repoRoot ".."))) {
        if (Test-Path (Join-Path $candidate "src\nlolib.h")) { $CoreDir = $candidate; break }
    }
}
if (-not $CoreDir -or -not (Test-Path (Join-Path $CoreDir "src\nlolib.h"))) {
    throw "No core nlolib checkout found. Run tools/fetch_core.sh, or pass -CoreDir."
}
$CoreDir = (Resolve-Path $CoreDir).Path

# The toolbox version is this repository's own; package_mltbx reads the same file.
$version = (Get-Content -Raw -Encoding utf8 (Join-Path $repoRoot "VERSION")).Trim()
if ($version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') {
    throw "Expected MAJOR.MINOR.PATCH in VERSION, found '$version'"
}

# NOTE: these option names are NOT prefixed. They were renamed from NLO_* in
# commit 2669906; a -D for a name the project never reads is silently ignored
# (it lands in the cache as UNINITIALIZED), so a stale prefix here disables the
# setting instead of failing. The verification below enforces that each option
# actually reached the project with the value we asked for.
$cmakeOptions = [ordered]@{
    INSTALL_GIT_HOOKS       = "OFF"
    BUMP_PATCH_ON_BUILD     = "OFF"
    BUILD_TESTING           = "OFF"
    NLOLIB_BUILD_DOCS       = "OFF"
    NLOLIB_BUILD_BENCHMARKS = "OFF"
    NLOLIB_BUILD_EXAMPLES   = "OFF"
    SQLITE_USE_FETCHCONTENT = "ON"
    ENABLE_VULKAN_BACKEND   = "ON"
    ENABLE_VKFFT            = "ON"
}

$cmakeArgs = @("-S", $CoreDir, "-B", $BuildDir, "-DNLOLIB_MATLAB_DIR=$repoRoot")
foreach ($name in $cmakeOptions.Keys) {
    $cmakeArgs += "-D$name=$($cmakeOptions[$name])"
}
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

# Verify only the options this script passes. A blanket scan for UNINITIALIZED
# entries would also flag stale ones left in an existing cache by earlier runs,
# which CMake never removes.
$cacheLines = Get-Content (Join-Path $repoRoot "$BuildDir/CMakeCache.txt")
$optionProblems = @()
foreach ($name in $cmakeOptions.Keys) {
    $pattern = "^" + [regex]::Escape($name) + ":"
    $entry = $cacheLines | Where-Object { $_ -match $pattern } | Select-Object -First 1
    if (-not $entry) {
        $optionProblems += "$name : not present in CMakeCache.txt"
    }
    elseif ($entry -match ":UNINITIALIZED=") {
        $optionProblems += "$name : ignored by the project (renamed or misspelled) -- $entry"
    }
    else {
        $actual = ($entry -replace '^[^=]*=', '')
        if ($actual -ne $cmakeOptions[$name]) {
            $optionProblems += "$name : expected $($cmakeOptions[$name]), cache has $actual"
        }
    }
}
if ($optionProblems.Count -gt 0) {
    throw ("CMake options did not take effect:`n  " + ($optionProblems -join "`n  "))
}

cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }

matlab -batch "addpath('$repoRoot'); package_mltbx('$BuildDir', '$Config')"
if ($LASTEXITCODE -ne 0) { throw "MATLAB toolbox packaging failed" }

$artifact = Join-Path $repoRoot "dist/nlolib-$version-win64.mltbx"
if (-not (Test-Path $artifact)) {
    throw "Expected toolbox artifact was not produced: $artifact"
}
Write-Host "Toolbox artifact: $artifact"
