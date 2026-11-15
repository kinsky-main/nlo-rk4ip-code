param(
    [string]$BuildDir = "build-wheel-win",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
# Native commands (cmake, python, delvewheel) do not trip ErrorActionPreference,
# so without this a failed build falls straight through to the staging steps and
# the script reports success. PowerShell 7+; harmless on 5.1, where the explicit
# exit-code check below still applies.
$PSNativeCommandUseErrorActionPreference = $true

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

cmake -S . -B $BuildDir `
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

$pkgDir = Join-Path $repoRoot "python\nlolib"
New-Item -ItemType Directory -Force -Path $pkgDir | Out-Null

$dllCandidates = @(
    (Join-Path $repoRoot "python\nlolib.dll"),
    (Join-Path $repoRoot ("python\{0}\nlolib.dll" -f $Config)),
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
    (Join-Path $repoRoot "python\sqlite3.dll"),
    (Join-Path $repoRoot "python\Release\sqlite3.dll"),
    (Join-Path $repoRoot "python\Debug\sqlite3.dll")
)
foreach ($sqlitePath in $sqliteCandidates) {
    if (Test-Path $sqlitePath) {
        Copy-Item -Force $sqlitePath (Join-Path $pkgDir "sqlite3.dll")
        break
    }
}

if (-not (Test-Path (Join-Path $pkgDir "sqlite3.dll"))) {
    $foundSqlite = Get-ChildItem -Recurse -File -Filter sqlite3.dll $BuildDir, "$repoRoot\python" -ErrorAction SilentlyContinue | Select-Object -First 1
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
