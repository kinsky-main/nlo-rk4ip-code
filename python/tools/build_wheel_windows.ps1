param(
    [string]$BuildDir = "build-wheel-win",
    [string]$Config = "Release",
    # Core nlolib checkout to build from. Defaults to .\core (the checkout the
    # workflows and tools/fetch_core.sh create at the revision CORE_REF pins),
    # then to the parent directory, for when this repository is the python\
    # submodule of the core repository.
    [string]$CoreDir = $env:NLOLIB_CORE_DIR
)

$ErrorActionPreference = "Stop"
# Native commands (cmake, python, delvewheel) do not trip ErrorActionPreference,
# so without this a failed build falls straight through to the staging steps and
# the script reports success. PowerShell 7+; harmless on 5.1, where the explicit
# exit-code check below still applies.
$PSNativeCommandUseErrorActionPreference = $true

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

cmake -S $CoreDir -B $BuildDir `
  "-DNLOLIB_PYTHON_DIR=$repoRoot" `
  -DINSTALL_GIT_HOOKS=OFF `
  -DBUMP_PATCH_ON_BUILD=OFF `
  -DBUILD_TESTING=OFF `
  -DNLOLIB_BUILD_DOCS=OFF `
  -DNLOLIB_BUILD_BENCHMARKS=OFF `
  -DNLOLIB_BUILD_EXAMPLES=OFF `
  -DSQLITE_USE_FETCHCONTENT=ON `
  -DENABLE_VULKAN_BACKEND=ON `
  -DENABLE_VKFFT=ON

cmake --build $BuildDir --config $Config

$pkgDir = Join-Path $repoRoot "nlolib"
New-Item -ItemType Directory -Force -Path $pkgDir | Out-Null

$dllCandidates = @(
    (Join-Path $repoRoot "nlolib.dll"),
    (Join-Path $repoRoot ("{0}\nlolib.dll" -f $Config)),
    (Join-Path $repoRoot ("$BuildDir\src\$Config\nlolib.dll"))
)

$dllPath = $null
foreach ($candidate in $dllCandidates) {
    if (Test-Path $candidate) {
        $dllPath = $candidate
        break
    }
}
if (-not $dllPath) {
    throw "Could not locate nlolib.dll after build."
}

Copy-Item -Force $dllPath (Join-Path $pkgDir "nlolib.dll")

$sqliteCandidates = @(
    (Join-Path (Split-Path $dllPath -Parent) "sqlite3.dll"),
    (Join-Path $repoRoot "sqlite3.dll"),
    (Join-Path $repoRoot "Release\sqlite3.dll"),
    (Join-Path $repoRoot "Debug\sqlite3.dll")
)
foreach ($sqlitePath in $sqliteCandidates) {
    if (Test-Path $sqlitePath) {
        Copy-Item -Force $sqlitePath (Join-Path $pkgDir "sqlite3.dll")
        break
    }
}

if (-not (Test-Path (Join-Path $pkgDir "sqlite3.dll"))) {
    $foundSqlite = Get-ChildItem -Recurse -File -Filter sqlite3.dll $BuildDir, $repoRoot -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($foundSqlite) {
        Copy-Item -Force $foundSqlite.FullName (Join-Path $pkgDir "sqlite3.dll")
    }
}

python -m pip install --upgrade build
python -m build

# vulkan-1.dll is delay-loaded and belongs to the user's graphics driver, so it
# is excluded rather than vendored; nlolib probes for the loader before touching
# any entry point, so its absence stays benign.
#
# Repaired into a staging directory and moved back, not written straight to
# dist\: delvewheel keeps the win_amd64 filename, so repairing in place makes it
# copy the wheel onto itself and fail with WinError 32.
if (Get-Command delvewheel -ErrorAction SilentlyContinue) {
    $repaired = Join-Path $repoRoot "dist\repaired"
    New-Item -ItemType Directory -Force -Path $repaired | Out-Null
    Get-ChildItem dist\*.whl | ForEach-Object {
        delvewheel repair --exclude vulkan-1.dll --add-path $pkgDir --wheel-dir $repaired $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "delvewheel repair failed for $($_.Name) with exit code $LASTEXITCODE" }
        Move-Item -Force (Join-Path $repaired $_.Name) $_.FullName
    }
    Remove-Item -Recurse -Force $repaired
} else {
    Write-Warning "delvewheel not found: dist\ holds unrepaired wheels."
}

Write-Host "Built wheels in dist/"
