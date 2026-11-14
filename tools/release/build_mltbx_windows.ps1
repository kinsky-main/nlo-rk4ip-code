param(
    [string]$BuildDir = "build-mltbx-win",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

function Get-ProjectVersion {
    $cmakeText = Get-Content -Raw -Encoding utf8 (Join-Path $repoRoot "CMakeLists.txt")
    $match = [regex]::Match($cmakeText, 'project\([^)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)')
    if (-not $match.Success) {
        throw "Could not parse CMake project version"
    }
    return $match.Groups[1].Value
}

$version = Get-ProjectVersion

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

$cmakeArgs = @("-S", ".", "-B", $BuildDir)
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

# package_mltbx reads the version back out of CMakeLists.txt to name the bundle
# and stamp its metadata, so the tree must already carry the version being
# released. Fail here rather than ship a bundle named after something else.
python tools/release/check_versions.py
if ($LASTEXITCODE -ne 0) { throw "Project version and its packaged copies disagree" }

matlab -batch "addpath('matlab'); package_mltbx('$BuildDir', '$Config')"
if ($LASTEXITCODE -ne 0) { throw "MATLAB toolbox packaging failed" }

$artifact = Join-Path $repoRoot "dist/nlolib-$version-win64.mltbx"
if (-not (Test-Path $artifact)) {
    throw "Expected toolbox artifact was not produced: $artifact"
}
Write-Host "Toolbox artifact: $artifact"
