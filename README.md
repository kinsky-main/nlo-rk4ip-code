# nlolib

`nlolib` is a C99 nonlinear optics library with CPU CBLAS and Vulkan compute
backends, plus Python, MATLAB, and Julia bindings.

Full rendered documentation is available at
[kinsky-main.github.io/nlo-rk4ip-code](https://kinsky-main.github.io/nlo-rk4ip-code/).

## What to Read

- Rendered docs home: `docs/index.md`
- Runtime operator mathematics: `docs/runtime_operators.md`
- Build and install guide: `docs/build_and_install.md`
- Python binding guide: `python/README.md`
- MATLAB binding guide: `matlab/README.md`
- Julia binding guide: `julia/README.md`

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

## Default Operator Forms

The public runtime-operator configuration uses the reference forms:

- `D(w) = i c0 w^2 - c1`
- `N(A) = i A (c2 |A|^2 + V)`

For the common quadratic GLSE form:

- `D(w) = i (beta2 / 2) w^2 - alpha_amp`
- `c0 = beta2 / 2`
- `c1 = alpha_amp`
- `c2 = gamma`

If your loss coefficient is written as the common power-loss term
`-alpha_pow A / 2`, pass `c1 = alpha_pow / 2`.

Higher-order dispersion uses the scalar constant table `constants[]` through
successive symbols `c0`, `c1`, `c2`, ... rather than an array-valued `c0`.

## Docs Build

```bash
cmake -S . -B build -DNLOLIB_BUILD_DOCS=ON
cmake --build build --target docs --config Release
```

Generated output:

- `build/docs/html/index.html`

## Bindings

- Python package and docs-facing guide: `python/README.md`
- MATLAB package guide: `matlab/README.md`
- Julia package guide: `julia/README.md`

The PyPI/package readme remains `python/README_PYPI.md`.

## Python Benchmarks

Run the nlolib-only tensor CPU vs GPU benchmark with:

```powershell
python examples/python/tensor_backend_scaling_nlolib_rk4ip.py
```

The mixed nlolib/MMTools runtime comparison remains:

```powershell
python examples/python/tensor_backend_scaling_rk4ip.py
```
