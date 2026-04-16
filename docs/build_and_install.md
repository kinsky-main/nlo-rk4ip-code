# Build and Install

## Prerequisites

### Required for core build

- CMake `3.28.3+`
- C99 compiler toolchain
- FFTW build prerequisites handled by CMake
- OpenBLAS/CBLAS dependency resolved by CMake from system install, fetched
  source, or fetched Windows binary
- Vulkan toolchain only when `ENABLE_VULKAN_BACKEND=ON`

### Optional but commonly needed

- Python 3 when `BUILD_TESTING=ON`
- Doxygen for the `docs` target
- Graphviz for call/include graphs

## Quick Start

### Windows

```powershell
cmake -S . -B build `
  -DINSTALL_GIT_HOOKS=OFF `
  -DBUMP_PATCH_ON_BUILD=OFF

cmake --build build --config Debug
ctest --test-dir build --build-config Debug --output-on-failure
```

### Linux

```bash
cmake -S . -B build \
  -DINSTALL_GIT_HOOKS=OFF \
  -DBUMP_PATCH_ON_BUILD=OFF

cmake --build build
ctest --test-dir build --output-on-failure
```

## Build Configuration

Important top-level options:

- `NLOLIB_BUILD_DOCS`
- `NLOLIB_BUILD_BENCHMARKS`
- `NLOLIB_BUILD_EXAMPLES`
- `NLOLIB_BUILD_MATLAB_TESTS`
- `ENABLE_VULKAN_BACKEND`
- `ENABLE_VKFFT`
- `BUILD_TESTING`

## Doxygen Docs

Enable docs and configure:

```bash
cmake -S . -B build -DNLOLIB_BUILD_DOCS=ON
```

Build docs:

```bash
cmake --build build --target docs --config Release
```

Generated output:

- `build/docs/html/index.html`

Docs notes:

- If Doxygen is not found, the `docs` target is not added.
- HTML formulas render through MathJax; local LaTeX is not required.
- The rendered landing page comes from `docs/index.md`.

## Related Guides

- [Python Binding Guide](../python/README.md)
- [MATLAB Binding Guide](../matlab/README.md)
- [Julia Binding Guide](../julia/README.md)
