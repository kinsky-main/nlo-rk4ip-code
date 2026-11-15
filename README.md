# nlolib

`nlolib` is a C99 nonlinear optics library with CPU and Vulkan compute
backends, plus Python, MATLAB, and Julia bindings.

Full rendered documentation is available at
[kinsky-main.github.io/nlo-rk4ip-code](https://kinsky-main.github.io/nlo-rk4ip-code/).

## What to Read

- Rendered docs home: `docs/index.md`
- Runtime operator mathematics: `docs/runtime_operators.md`
- Stored runs and the run database layout: `docs/nlolib_run_database.md`
- Build and install guide: `docs/build_and_install.md`
- Python binding guide: `python/README.md`
- MATLAB binding guide: `matlab/README.md`
- Julia binding guide: `julia/README.md`
- Julia release and registration guide: `docs/release_julia.md`

## Quick Start

### Windows

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build --build-config Debug --output-on-failure
```

### Linux

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### What CI Runs

`.github/workflows/ci.yml` runs on every pull request against `main` and on
every push to `main`:

- a source-only job that needs nothing but a Python interpreter, so it answers
  in under a minute:

  ```bash
  python tools/release/check_versions.py
  python tools/write_sbom.py --check
  python tools/check_api_parity.py
  ```

- a Linux job that configures and builds with the Vulkan backend on, then runs
  the native test suite:

  ```bash
  ctest --test-dir build --output-on-failure -E '^test_(python|julia)_'
  ```

The runners have no GPU, so the Vulkan cases skip their device work and only the
CPU paths are exercised. The `test_python_*` and `test_julia_*` cases are
excluded there and are not covered by CI: run the full `ctest` above locally
before opening a pull request that touches the Python or Julia bindings.

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

## Stored Runs

Passing `sqlite_path` (C `storage_options`, or the `sqlite_path` keyword of the
bindings) stores the run in a shared run database, normally
`<store>/nlolib/nlolib.sqlite`. Each run streams into
`<store>/nlolib/staging/` and is merged into the shared file under a lock file
when it completes, so many machines can write to one database on a network
share. Every stored run gets a row in the table `runs`, following the
ModelStore run index contract (kind `nlolib`), and by default keeps per-record
power, spectrum and fluence maps plus the final field; raw records are opt-in
with `record_mode` (`reduced`, `raw`, `reduced_and_raw`) and `raw_stride_t/x/y`.
In-memory results of `propagate` are unchanged.

`storage_run_key` returns the key a run will be stored under, so a script can
skip runs with `storage_has_run`; `storage_recover_staging` merges or discards
staging files left by interrupted runs. Tables, blob encoding, axes, units and
the lock protocol are specified in `docs/nlolib_run_database.md`.

## Bindings

- Python package and docs-facing guide: `python/README.md`
- MATLAB package guide: `matlab/README.md`
- Julia package guide: `julia/README.md`

The PyPI/package readme remains `python/README_PYPI.md`.

The Julia package installs from the General registry with `] add NLOLib`, which
downloads a prebuilt artifact instead of compiling anything; a built checkout
also works through the `julia_stage` target. Both paths, the release flow and
registration are in `docs/release_julia.md`.

## Python Releases

Wheels are published to PyPI by `.github/workflows/release-pypi.yml` when a
bare version tag (no `v` prefix) is pushed. One wheel per platform, Linux and
Windows x86-64, each smoke-tested on a clean runner before anything is
uploaded. Run the workflow manually for a dry run that builds and checks the
wheels but publishes nothing.

To build a wheel locally:

```bash
./tools/release/build_wheel_linux.sh build-wheel-linux Release
```

```powershell
.\tools\release\build_wheel_windows.ps1 -BuildDir build-wheel-win -Config Release
```

Both need the Vulkan loader and `glslangValidator`. See
[docs/release_python.md](docs/release_python.md) for the full story: the wheel
tagging, the platform matrix and its glibc floor, Trusted Publishing setup,
and why macOS is not published.

## Python Benchmarks

Run the nlolib-only tensor CPU vs GPU benchmark with:

```powershell
python examples/python/tensor_backend_scaling_nlolib_rk4ip.py
```

The nlolib-only runtime plot fits the GPU curve in two regions: a smaller
ringbuffer-friendly region and a larger transfer-limited region.

The mixed nlolib/MMTools runtime comparison remains:

```powershell
python examples/python/tensor_backend_scaling_rk4ip.py
```

It translates MMTools scalar mode counts into GRIN-effective nlolib grid
sizes, then writes both state-vector-size and scalar-mode-count runtime
plots for the comparison.
