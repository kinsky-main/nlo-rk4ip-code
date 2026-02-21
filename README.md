# nlolib

`nlolib` is a C99 nonlinear optics library with CPU SIMD and Vulkan compute backends, plus Python and MATLAB bindings.

## Prerequisites

### Required for core build

- CMake `3.28.3+`
- C99 compiler toolchain
- Vulkan loader library (runtime + link-time)
- Vulkan headers (auto-discovered or fetched by CMake)
- Vulkan SDK/glslang development components that provide CMake target `Vulkan::glslang`
- `glslangValidator` on `PATH` (or available via `VULKAN_SDK`)

### Optional but commonly needed

- Python 3 (required if `BUILD_TESTING=ON` because Python tests are added)
- Doxygen (for docs target)
- Graphviz (recommended for Doxygen call/directory graphs)
- SQLite3 dev package (or let CMake fetch SQLite amalgamation)

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
| `NLO_BUMP_PATCH_ON_BUILD` | `ON` | Adds `nlo_patch_bump_on_build` target to patch-bump version on successful build. |
| `NLO_SQLITE_USE_FETCHCONTENT` | `OFF` | If `ON`, always fetch SQLite amalgamation; otherwise try system SQLite first. |
| `NLO_CPU_SIMD_LEVEL` | `AUTO` | CPU SIMD mode: `AUTO`, `AVX2`, `AVX`, `SCALAR`. |
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
- VkFFT is fetched via CMake FetchContent for FFT backend integration.
- Vulkan headers are found via `find_package(Vulkan)` or fetched from `NLO_VULKAN_HEADERS_URL`.
- Vulkan loader must be present locally (`vulkan`/`vulkan-1` library).
- `glslangValidator` is required to compile compute shaders to SPIR-V during build.
- CMake target `Vulkan::glslang` must be resolvable for FFT backend linking.
- SQLite is discovered from system/Conda hints unless `NLO_SQLITE_USE_FETCHCONTENT=ON`; fallback fetch is supported.

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

Run benchmark example:

```powershell
.\build\benchmarks\Release\bench_solver_backend.exe --backend=both --sizes=1024,4096 --warmup=2 --runs=8 --csv=benchmarks/results/solver_backend.csv
```

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

Current test groups:

- FFT tests: `test_nlo_complex`, `test_fft`
- Core tests: `test_core_state_alloc`
- Numerics/backend tests: `test_nlo_numerics`, `test_nlo_vector_backend`, `test_nlo_simd_wrapper`, `test_nlo_vector_backend_vulkan`
- Python tests: `test_python_api_smoke`, `test_python_operator_regression`, `test_python_storage_chunking`

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

## Preset Workflow (Optional)

This repo includes `CMakeUserPresets.json` with a local Windows preset example:

```powershell
cmake --preset local-custom-path-debug
cmake --build --preset local-custom-path-debug-build
ctest --preset local-custom-path-debug-test
```
