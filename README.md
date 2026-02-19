# TODO List
- [x] Fix double declarations and definitions of vector operations in backend and numerics
- [x] Assess whether SPIR-V is required for GPU backend or if Vulkan compute shaders are sufficient. Ideally remove as it is a runtime dependency.
- [x] Investigate GPU benchmark failure.
- [x] Investigate why RK4 solver is exploding to NaNs generally.
- [ ] Write db io for larger datasets and implement checkpointing in solver when problem size exceeds system memory limits.
- [ ] Add more benchmarks and diagnostics, e.g. per-kernel timings, memory usage, RK4 intermediate state dumps, etc.
- [ ] Expand solver backend to support more unified kernel operations (Complex Potentials) and more flexible data layouts
- [x] Implement MATLAB interface for solver with string API for defining input operators.
- [ ] Extension: Add MPA solver for coupled mode problems.
- [ ] Extension: OpenMP backend for multi-core CPU parallelism.
- [x] Implement dispersion operator custom functions, currently staging function is not used.
- [x] Combine grin vector operator into dispersion and nonlinear operator expressions.
- [x] Remove old API and just have custom functions for dispersion and nonlinearity, this removes the complications of additional vector operators.
- [ ] Implement new GRIN operations using preferred kernels for (2+1)D operations.


## FFTW

FFTW is a required dependency and is always built from source and linked statically.
Vulkan (including loader, headers, and `glslangValidator`) is also required.

```powershell
cmake -S . -B build -DNLO_INSTALL_GIT_HOOKS=OFF
cmake --build build --config Debug
```

## Benchmarks

Build benchmark targets by enabling `NLOLIB_BUILD_BENCHMARKS`:

```powershell
cmake -S . -B build -DNLOLIB_BUILD_BENCHMARKS=ON
cmake --build build --target bench_solver_backend --config Release
```

Run CPU vs GPU end-to-end solver benchmark:

```powershell
.\build\benchmarks\Release\bench_solver_backend.exe --backend=both --sizes=1024,4096 --warmup=2 --runs=8 --csv=benchmarks/results/solver_backend.csv
```

Output:
- Console summary statistics per backend and size
- CSV rows in `benchmarks/results/solver_backend.csv`

Callable-vs-string runtime expression overhead benchmark (CPU):

```powershell
$env:PYTHONPATH="$PWD/python"
$env:NLOLIB_LIBRARY="$PWD/python/Release/nlolib.dll"
python benchmarks/python/bench_runtime_callable.py --n 4096 --runs 5
```

## MATLAB API

MATLAB wrappers are provided in the `matlab/+nlolib` namespace package. They call the nlolib C shared library directly via `loadlibrary`/`calllib` — no Python or MEX dependency.

### Install from .mltbx (recommended)

Download the latest `nlolib.mltbx` from the [GitHub Releases](../../releases) page and double-click it in MATLAB, or run:

```matlab
matlab.addons.install('nlolib.mltbx');
```

### MATLAB Development workflow

#### Build from source

```powershell
cmake -S . -B build
cmake --build build --config Release --target matlab_stage
```

Then add the staged output to the MATLAB path and run setup:

```matlab
addpath('build/matlab_toolbox');
nlolib_setup();
```

From a source checkout (without staging), use:

```matlab
addpath('matlab');
nlolib_setup();
```

`nlolib_setup()` adds the MATLAB package plus `examples/matlab` to the path.  
The `matlab_stage` target copies `+nlolib/*.m`, `nlolib_setup.m`, `examples/matlab`, the shared library, and `nlolib_matlab.h` into `build/matlab_toolbox/`. You can also set the `NLOLIB_LIBRARY` environment variable to the full path of `nlolib.dll` / `libnlolib.so` to override automatic library discovery.

#### Create MATLAB package from source

- Home -> Add-Ons -> Package Toolbox
- Toolbox folder: C:\Users\Wenzel\Final Year Project\nlolib\build\matlab_toolbox
- Include these:
    - +nlolib/
    - nlolib_setup.m
    - examples/matlab/
    - lib/nlolib.dll
    - lib/nlolib_matlab.h
- Save project as: C:\Users\Wenzel\Final Year Project\nlolib\matlab\nlolib.prj
- Click Package -> output nlolib.mltbx

#### Prerequisites

- MATLAB R2019b or later (loadlibrary with C99 header support).
- A GPU driver that ships the Vulkan loader (standard on NVIDIA, AMD, and Intel desktop drivers). No Vulkan SDK is needed at runtime.

### Quick start

```matlab
api = nlolib.NLolib();
cfg = struct(...); % see examples/matlab/runtime_temporal_demo.m
records = api.propagate(cfg, field0, 2);
```

Runtime operators accept either explicit strings (`dispersion_expr`, `nonlinear_expr`) or MATLAB function handles (`dispersion_fn`, `nonlinear_fn`) that are translated to the internal expression grammar at runtime.
