# nlolib

`nlolib` is a C99 nonlinear optics library with CPU CBLAS and Vulkan compute backends, plus Python, MATLAB, and Julia bindings.

Full API documentation is available at https://kinsky-main.github.io/nlo-rk4ip-code/.

## Prerequisites

### Required for core build

- CMake `3.28.3+`
- C99 compiler toolchain
- FFTW build prerequisites (handled by CMake fetch/in-tree build)
- OpenBLAS/CBLAS dependency (auto-resolved by CMake from system install, fetched source, or fetched Windows binary)
- Vulkan toolchain only when `NLO_ENABLE_VULKAN_BACKEND=ON`:
  - Vulkan loader library (runtime + link-time)
  - Vulkan headers (auto-discovered or fetched by CMake)
  - Vulkan SDK/glslang development components that provide CMake target `Vulkan::glslang`
  - `glslangValidator` on `PATH` (or available via `VULKAN_SDK`)

### Optional but commonly needed

- Python 3 (required if `BUILD_TESTING=ON` because Python tests are added)
- Doxygen (for docs target)
- Graphviz (recommended for Doxygen call/directory graphs)
- Network access for CMake to fetch the SQLite amalgamation archive and, when needed, OpenBLAS sources/binaries

## Quick Start (Windows)

```powershell
cmake -S . -B build `
  -DNLO_INSTALL_GIT_HOOKS=OFF `
  -DNLO_BUMP_PATCH_ON_BUILD=OFF

cmake --build build --config Debug
ctest --test-dir build --build-config Debug --output-on-failure
```

Run the C demo:

```powershell
cmake --build build --config Release --target runtime_temporal_demo_c
.\build\examples\Release\runtime_temporal_demo_c.exe
```

## Quick Start (Linux)

Install toolchain and runtime prerequisites (Ubuntu/Debian example):

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake ninja-build build-essential pkg-config \
  libopenblas-dev \
  libvulkan-dev glslang-tools vulkan-tools spirv-tools \
  python3
```

Configure, build, and test:

```bash
cmake -S . -B build \
  -DNLO_INSTALL_GIT_HOOKS=OFF \
  -DNLO_BUMP_PATCH_ON_BUILD=OFF

cmake --build build
ctest --test-dir build --output-on-failure
```

Repo helper script (Linux/macOS shell environments):

```bash
./setup.sh --build --cmake-arg -DNLO_INSTALL_GIT_HOOKS=OFF --cmake-arg -DNLO_BUMP_PATCH_ON_BUILD=OFF
```

Script flags:

- `--build-dir <dir>` sets configure/build directory
- `--build` runs build after configure
- `--no-install` skips apt dependency install step
- `--cmake-arg <arg>` passes additional CMake cache/options

## Build Configuration Options

Current top-level CMake options and cache variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NLO_ENABLE_RK4_DEBUG_DIAGNOSTICS` | `ON` | Enables RK4/dispersion debug diagnostics in `Debug` builds. |
| `NLO_INSTALL_GIT_HOOKS` | `ON` | Installs project pre-commit hook for automatic version bump behavior. |
| `NLOLIB_BUILD_DOCS` | `ON` | Enables `docs` target when Doxygen is found. |
| `NLOLIB_BUILD_BENCHMARKS` | `ON` | Builds benchmark targets in `benchmarks/`. |
| `NLOLIB_BUILD_EXAMPLES` | `ON` | Builds example targets in `examples/`. |
| `NLOLIB_BUILD_MATLAB_TESTS` | `OFF` | Enables optional MATLAB parser/runtime tests under `tests/matlab` when MATLAB is available. |
| `NLO_ENABLE_VULKAN_BACKEND` | `ON` | Enables Vulkan backend and shader compilation path. |
| `NLO_ENABLE_VKFFT` | `ON` | Enables VkFFT FFT path (auto-forced `OFF` when Vulkan backend is disabled). |
| `NLO_BUMP_PATCH_ON_BUILD` | `ON` | Adds `nlo_patch_bump_on_build` target to patch-bump version on successful build. |
| `NLO_SQLITE_USE_FETCHCONTENT` | `ON` | Deprecated compatibility toggle; SQLite amalgamation is always used. |
| `NLO_CBLAS_PREFER_SYSTEM` | `ON` on Linux/macOS, `OFF` on Windows | Prefer a system OpenBLAS install before fetching OpenBLAS. |
| `NLO_OPENBLAS_URL` | OpenBLAS `v0.3.30` tarball | OpenBLAS source archive used for in-tree static CBLAS builds on non-Windows platforms. |
| `NLO_OPENBLAS_WINDOWS_URL` | OpenBLAS `v0.3.30` x64 zip | Prebuilt OpenBLAS package fetched for Windows builds. |
| `NLO_VULKAN_HEADERS_URL` | Khronos main zip | Vulkan-Headers fetch URL fallback when headers are not local. |
| `NLO_SQLITE_AMALGAMATION_URL` | sqlite.org zip | SQLite amalgamation fetch URL fallback. |
| `NLO_FFTW_GIT_TAG` | `fftw-3.3.10` | FFTW tarball tag used in CMake fetch/build. |
| `BUILD_TESTING` | `ON` (via CTest default) | Enables test targets under `tests/`. |

Build configuration types:

- `Debug`
- `Release`
- `RelWithDebInfo`
- `MinSizeRel`

Single-config generators use `CMAKE_BUILD_TYPE` (defaults to `Debug` if not set).  
Multi-config generators use `--config <type>` during build/test.

## Dependency Resolution Behavior

- FFTW is resolved through CMake FetchContent and linked as static FFTW3 target.
- CPU CBLAS is resolved through OpenBLAS:
  - Linux/macOS prefer a system OpenBLAS install when `NLO_CBLAS_PREFER_SYSTEM=ON`, otherwise CMake fetches and builds a static OpenBLAS.
  - Windows fetches a pinned prebuilt OpenBLAS package, links the provided import library, and copies the matching DLL beside built binaries.
- When `NLO_ENABLE_VKFFT=ON`, VkFFT is fetched via CMake FetchContent for Vulkan FFT integration.
- When `NLO_ENABLE_VULKAN_BACKEND=ON`, Vulkan headers are found via `find_package(Vulkan)` or fetched from `NLO_VULKAN_HEADERS_URL`.
- When `NLO_ENABLE_VULKAN_BACKEND=ON`, Vulkan loader must be present locally (`vulkan`/`vulkan-1` library).
- When `NLO_ENABLE_VULKAN_BACKEND=ON`, `glslangValidator` is required to compile compute shaders to SPIR-V during build.
- When `NLO_ENABLE_VKFFT=ON`, CMake target `Vulkan::glslang` must be resolvable for FFT backend linking.
- SQLite is linked from the fetched amalgamation as a static library (no external `sqlite3.dll` runtime dependency).

## Runtime Operator Semantics

- Runtime operators use symbols: `A` (field), `w` (frequency/spatial-frequency), `I` (`|A|^2`), `D` (dispersion factor), `V` (potential), `h` (half-step exponent).
- `nonlinear_expr` is interpreted as the full nonlinear RHS `N(A)` and is written directly by the solver.
- Legacy multiplier-form nonlinear expressions must be migrated to include `A`.
- `runtime.nonlinear_model` selects nonlinear execution mode:
  - `0` (`NLO_NONLINEAR_MODEL_EXPR`): evaluate `nonlinear_expr` (default).
  - `1` (`NLO_NONLINEAR_MODEL_KERR_RAMAN`): built-in Kerr + delayed Raman model with optional self-steepening.
- `NLO_NONLINEAR_MODEL_KERR_RAMAN` parameters (Python `RuntimeOperators`, MATLAB `cfg.runtime`):
  - `nonlinear_gamma` Kerr coefficient `gamma`.
  - `raman_fraction` delayed Raman fraction `f_R` in `[0, 1]`.
  - `raman_tau1`, `raman_tau2` default Raman kernel shape constants (`0.0122`, `0.0320`).
  - `raman_response_time` optional custom time-domain Raman response (`num_time_samples` complex values); when omitted, the normalized default response is generated from `tau1/tau2`.
  - `shock_omega0` enables self-steepening when `> 0` (set `0` to disable).
- Current limitation: `NLO_NONLINEAR_MODEL_KERR_RAMAN` is supported for temporal-only runs (not tensor-coupled `tensor_nt*tensor_nx*tensor_ny` grids).

Common migration examples:

- `i*gamma*I` -> `i*gamma*A*I`
- `i*gamma*I + i*V` -> `i*A*(gamma*I + V)`

## Adaptive Error Control Semantics

- Reference formulation: Balac-Mahé embedded ERK4(3)-IP local-defect control.
- Solver keeps the embedded pair construction in RK4IP form (`A^[4]`, `A^[3]`) and controls steps with:
  - `delta_rel = sqrt(sum(|A^[4]-A^[3]|^2) / sum((a_floor + |A^[4]|)^2))`
  - accepted when `delta_rel <= error_tolerance`
  - adaptive update uses `h_next = clamp(h * (error_tolerance / delta_rel)^(1/4), h_min, h_max)`
- Default defect floor is `a_floor = 1e-14`.
- Python `rtol` is an alias for `error_tolerance`; both now map to this relative defect threshold.
- `step_history.error` stores the same relative defect (`delta_rel`) for each accepted adaptive step.
- Previous behavior (removed): absolute L2-like defect scaling `sqrt(dt * sum |A^[4]-A^[3]|^2)` compared directly to `error_tolerance`, which made tolerance behavior depend strongly on signal scale and grid size.
- Min-step safeguard: if `h_min` is reached while `delta_rel > error_tolerance`, the run continues with constrained steps and emits a warning.

## Build Targets and Usage

Build the shared library:

```bash
cmake --build build --target nlolib
```

Build examples:

```bash
cmake -S . -B build -DNLOLIB_BUILD_EXAMPLES=ON
cmake --build build --target runtime_temporal_demo_c --config Release
```

Build benchmarks:

```bash
cmake -S . -B build -DNLOLIB_BUILD_BENCHMARKS=ON
cmake --build build --target bench_solver_backend --config Release
```

Run temporal benchmark example:

```powershell
.\build\benchmarks\Release\bench_solver_backend.exe --backend=both --sizes=1024,4096 --warmup=2 --runs=8 --csv=benchmarks/results/solver_backend.csv
```

Run tensor scaling benchmark planning pass:

```powershell
.\build\benchmarks\Release\bench_solver_backend.exe --scenario=tensor3d_scaling --dry-run --tensor-scales=8,16,32,64 --planner-host-bytes=40000000 --planner-gpu-bytes=5000000
```

Run tensor scaling benchmark execution:

```powershell
.\build\benchmarks\Release\bench_solver_backend.exe --scenario=tensor3d_scaling --backend=both --tensor-scales=8,16,32,64 --warmup=1 --runs=3 --csv=benchmarks/results/tensor_backend_scaling.csv --storage-dir=benchmarks/results/tensor_storage
```

Tensor scaling reports three regions:

- `gpu_fit`: active tensor working set fits both GPU and system memory, so CPU and GPU both run normally.
- `host_fit_only`: active tensor working set fits system memory but exceeds the effective GPU budget, so CPU runs and GPU is reported as an expected limit.
- `output_spill`: active tensor working set still fits execution, but the captured output volume is forced above the effective host-memory budget and spilled through SQLite-backed snapshot storage.

## Running Tests

Configure with tests enabled:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Debug
```

List tests discovered by CTest:

```bash
ctest --test-dir build -N
```

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

Run all tests for a specific config (multi-config generators):

```powershell
ctest --test-dir build --build-config Debug --output-on-failure
```

Run subsets with regex:

```bash
ctest --test-dir build -R "^test_fft$" --output-on-failure
ctest --test-dir build -R "^test_python_.*" --output-on-failure
ctest --test-dir build -R "^test_nlo_.*" --output-on-failure
```

Run the new Python callable-translation test directly:

```bash
ctest --test-dir build -R "^test_python_runtime_expr_translation$" --output-on-failure
ctest --test-dir build -R "^test_julia_api_smoke$" --output-on-failure
```

Enable and run optional MATLAB parser/runtime tests:

```powershell
cmake -S . -B build `
  -DBUILD_TESTING=ON `
  -DNLOLIB_BUILD_MATLAB_TESTS=ON
ctest --test-dir build --build-config Debug -R "^test_matlab_runtime_handle_parser$" --output-on-failure
```

```bash
cmake -S . -B build \
  -DBUILD_TESTING=ON \
  -DNLOLIB_BUILD_MATLAB_TESTS=ON
ctest --test-dir build -R "^test_matlab_runtime_handle_parser$" --output-on-failure
```

Current test groups:

- FFT tests: `test_nlo_complex`, `test_fft`
- Core tests: `test_core_state_alloc`
- Numerics/backend tests: `test_nlo_numerics`, `test_nlo_vector_backend`, `test_nlo_vector_backend_vulkan`
- Python tests: `test_python_api_smoke`, `test_python_operator_regression`, `test_python_storage_chunking`
- Python translation test: `test_python_runtime_expr_translation`
- Julia tests: `test_julia_api_smoke`
- MATLAB tests (optional): `test_matlab_runtime_handle_parser` when `NLOLIB_BUILD_MATLAB_TESTS=ON` and MATLAB is found

Note: Python tests require a discoverable Python interpreter at CMake configure time.

## Generating Doxygen Docs

Enable docs and configure:

```bash
cmake -S . -B build -DNLOLIB_BUILD_DOCS=ON
```

Build docs target:

```bash
cmake --build build --target docs --config Release
```

Output is generated in:

- `build/docs/html/index.html`

Doc generation details:

- If Doxygen is not found, the `docs` target is not added.
- `doxygen-awesome-css` is fetched automatically during configure when docs are enabled.
- Doxygen input is `src/` (`*.h`, `*.c`), with call and directory graphs enabled.

### Publish Docs to GitHub Pages (Manual Trigger)

The repository includes a manual GitHub Actions workflow at
`.github/workflows/docs-pages.yml` that builds docs and deploys
`build-docs/docs/html` to GitHub Pages.

One-time GitHub Pages setup can be done with the existing secret token:

```powershell
$token = Get-Secret -Name GithubAdminToken -AsPlainText
$origin = git remote get-url origin
$null = $origin -match 'github\.com[:/](?<owner>[^/]+)/(?<repo>[^/.]+)(\.git)?$'
$owner = $Matches.owner
$repo = $Matches.repo
$headers = @{
  Authorization = "Bearer $token"
  Accept = "application/vnd.github+json"
  "User-Agent" = "nlolib-pages-setup"
  "X-GitHub-Api-Version" = "2022-11-28"
}
$body = @{ build_type = "workflow" } | ConvertTo-Json
try {
  Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$owner/$repo/pages" -Headers $headers -Body $body -ContentType "application/json"
} catch {
  Invoke-RestMethod -Method Put -Uri "https://api.github.com/repos/$owner/$repo/pages" -Headers $headers -Body $body -ContentType "application/json"
}
```

```bash
token="$GITHUB_TOKEN"
origin="$(git remote get-url origin)"
owner="$(echo "$origin" | sed -E 's#.*github.com[:/]([^/]+)/([^/.]+)(\\.git)?#\\1#')"
repo="$(echo "$origin" | sed -E 's#.*github.com[:/]([^/]+)/([^/.]+)(\\.git)?#\\2#')"
payload='{"build_type":"workflow"}'
curl -L -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${token}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${owner}/${repo}/pages" \
  -d "${payload}" || \
curl -L -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${token}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${owner}/${repo}/pages" \
  -d "${payload}"
```

For Linux/macOS, set `GITHUB_TOKEN` to a token with Pages admin permission
before running the command.

Then run the workflow from GitHub Actions UI:

- Workflow: `docs-pages`
- Trigger: `Run workflow`
- Branch: `main`

After a successful run, the site is published at:

- `https://<owner>.github.io/<repo>/`

Troubleshooting:

- If configure says Doxygen is missing, install Doxygen and rerun.
- First Pages deployment may require a short propagation delay before URL is live.

## Python Usage

`python/CMakeLists.txt` places the built shared library next to the Python package for direct `ctypes` loading.

Example environment setup:

```powershell
$env:PYTHONPATH="$PWD/python"
$env:NLOLIB_LIBRARY="$PWD/python/nlolib.dll"
python examples/python/runtime_temporal_demo.py
```

On Linux:

```bash
export PYTHONPATH="$PWD/python"
export NLOLIB_LIBRARY="$PWD/python/libnlolib.so"
python3 examples/python/runtime_temporal_demo.py
```

Tensor-grid backend timing example:

```powershell
python examples/python/tensor_backend_scaling_rk4ip.py --gpu-fit-scales 8,16,32 --host-fit-scales 48,64 --spill-scale 32 --spill-records 128,256,512
```

Build wheel artifacts manually:

```powershell
.\tools\release\build_wheel_windows.ps1 -BuildDir build-wheel-win -Config Release
```

```bash
./tools/release/build_wheel_linux.sh build-wheel-linux Release
./tools/release/build_wheel_macos.sh build-wheel-macos Release
```

These release scripts use CMake-managed patch bumps (`x.y.Z`) and then sync
`pyproject.toml` from `CMakeLists.txt` so Python and MATLAB distribution
versions stay aligned.

## MATLAB Usage

Stage MATLAB package artifacts:

```powershell
cmake --build build --config Release --target matlab_stage
```

Then in MATLAB:

```matlab
addpath('build/matlab_toolbox');
nlolib_setup();
api = nlolib.NLolib();
```

From source tree without staging:

```matlab
addpath('matlab');
nlolib_setup();
```

`matlab_stage` copies:

- `matlab/+nlolib`
- `matlab/nlolib_setup.m`
- `examples/matlab`
- shared library into `build/matlab_toolbox/lib`
- `src/nlolib_matlab.h` into `build/matlab_toolbox/lib`

You can override runtime library lookup with `NLOLIB_LIBRARY`.

Create a toolbox bundle manually:

```powershell
matlab -batch "addpath('matlab'); package_mltbx('build', 'Release')"
```

```bash
matlab -batch "addpath('matlab'); package_mltbx('build', 'Release')"
```

Additional manual publication steps are documented in `docs/release_manual.md`.

## Julia Usage

Stage the Julia package artifacts:

```powershell
cmake --build build --config Release --target julia_stage
```

```bash
cmake --build build --config Release --target julia_stage
```

Then in Julia:

```julia
using Pkg
Pkg.activate("build/julia_package")
using NLOLib

NLOLib.load()
```

From the source tree without staging, point Julia at a built shared library:

```powershell
$env:NLOLIB_LIBRARY="$PWD\\python\\Release\\nlolib.dll"
julia --project=julia
```

```bash
export NLOLIB_LIBRARY="$PWD/python/libnlolib.so"
julia --project=julia
```

The Julia wrapper is intentionally low-level and performance-first:

- results are returned as typed dense arrays instead of nested language objects
- `propagate` returns records as a column-major matrix shaped `(num_time_samples, num_records)`
- `propagate!` writes directly into caller-owned output buffers
- pointer-backed helper views use Julia FFI primitives (`ccall`, `GC.@preserve`, `unsafe_wrap`, `reinterpret`)

`julia_stage` copies:

- `julia/src`
- `julia/test`
- `julia/Project.toml`
- shared library into `build/julia_package/lib`
- `src/nlolib_matlab.h` into `build/julia_package/lib`

## Preset Workflow (Optional)

This repo includes `CMakeUserPresets.json` with a local Windows preset example:

```powershell
cmake --preset local-custom-path-debug
cmake --build --preset local-custom-path-debug-build
ctest --preset local-custom-path-debug-test
```
