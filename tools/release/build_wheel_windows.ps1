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

cmake --build $BuildDir --config $Config --target nlolib

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

if (Get-Command delvewheel -ErrorAction SilentlyContinue) {
    Get-ChildItem dist\*.whl | ForEach-Object {
        delvewheel repair --add-path $pkgDir --wheel-dir dist $_.FullName
    }
}

Write-Host "Built wheels in dist/"
