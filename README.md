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
- Vulkan backend behaviour, JIT and runtime switches: `docs/gpu_backend.md`
- Bindings and how they are released: [Bindings](#bindings) below

## Getting the Source

The Python, MATLAB and Julia bindings are separate repositories, mounted here
as the `python/`, `matlab/` and `julia/` git submodules:

```bash
git clone --recurse-submodules https://github.com/kinsky-main/nlo-rk4ip-code.git
```

In an existing clone, `git submodule update --init` fetches them. The core
library builds without them; a binding that is not checked out is skipped, along
with its targets and tests.

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

  `check_api_parity.py` compares the three bindings with the shared public
  surface in `tests/api_surface.json` (see `docs/binding_api_vocabulary.md`)
  and fails on any difference.

- a Linux job that configures and builds with the Vulkan backend on, then runs
  the native test suite:

  ```bash
  ctest --test-dir build --output-on-failure -E '^test_(python|julia)_'
  ```

The runners have no GPU, so the Vulkan cases skip their device work and only the
CPU paths are exercised. The `test_python_*` and `test_julia_*` cases are
excluded here; each binding repository runs its own suite in its own CI, against
the core revision it pins. Run the full `ctest` above locally before opening a
pull request that changes behaviour the bindings rely on.

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
staging files left by interrupted runs. With `defer_merge` (or
`NLOLIB_STORE_DEFER_MERGE=1`) a run leaves its finished staging file for one
designated host to merge with the `nlolib_store_merge` tool built next to the
library:

```bash
nlolib_store_merge --watch 60 /path/to/store
```

On Linux and macOS, run databases are written with SQLite's `unix-none` VFS
under Nlolib's own writer lock, because SQLite's Unix locking cannot commit on
SMB mounts (`NLOLIB_STORE_VFS` overrides). Tables, blob encoding, axes, units
and the lock protocol are specified in `docs/nlolib_run_database.md`.

## Bindings

Each binding is its own repository with its own version, tests, examples and
release pipeline, and is mounted here as a submodule:

| Binding | Submodule | Repository | Published to |
| --- | --- | --- | --- |
| Python | `python/` | [nlolib-python](https://github.com/kinsky-main/nlolib-python) | PyPI (`pip install nlolib`) |
| MATLAB | `matlab/` | [nlolib-matlab](https://github.com/kinsky-main/nlolib-matlab) | File Exchange / GitHub releases (`nlolib.mltbx`) |
| Julia | `julia/` | [NLOLib.jl](https://github.com/kinsky-main/NLOLib.jl) | General registry (`] add NLOLib`) |

The three bindings share one public API, each in its own language's idiom:
one `propagate` entry point, operators written as plain functions, the same
option and field names, and lowercase strings for backend and other enum
values. [docs/binding_api_vocabulary.md](docs/binding_api_vocabulary.md) is the
reference. In Python:

```python
import nlolib

result = nlolib.propagate(
    nlolib.PulseSpec(samples=field0, delta_time=dt),
    lambda A, w: 1j * (0.5 * beta2) * w * w,   # linear operator
    lambda A, I: 1j * gamma * A * I,           # nonlinear operator
    propagation_distance=1.0,
    num_records=2,
    exec_options=nlolib.ExecutionOptions(backend_type="auto"),
)
```

Each binding ships the same twelve examples under its `examples/` directory,
starting with `quickstart`.

Inside this repository a binding builds against the tree it is mounted in:
`cmake --build build` puts the shared library beside the Python package, and
the `matlab_stage` and `julia_stage` targets stage the other two. On its own, a
binding repository builds against the core commit named in its `CORE_REF` file,
by configuring this repository with `-DNLOLIB_PYTHON_DIR`, `-DNLOLIB_MATLAB_DIR`
or `-DNLOLIB_JULIA_DIR` pointing at the binding checkout.

A change that touches both sides lands in three steps: merge the core change
here, point the binding's `CORE_REF` at it (with the binding change) and merge
that, then update the submodule pointer here. Releases are cut from the binding
repositories; see [docs/release_manual.md](docs/release_manual.md).

## Python Benchmarks

Run the nlolib-only tensor CPU vs GPU benchmark with:

```powershell
python python/examples/tensor_backend_scaling_nlolib.py
```

The nlolib-only runtime plot fits the GPU curve in two regions: a smaller
ringbuffer-friendly region and a larger transfer-limited region.

The mixed nlolib/MMTools runtime comparison remains:

```powershell
python python/examples/tensor_backend_scaling.py
```

It translates MMTools scalar mode counts into GRIN-effective nlolib grid
sizes, then writes both state-vector-size and scalar-mode-count runtime
plots for the comparison.
