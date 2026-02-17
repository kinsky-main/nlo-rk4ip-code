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

MATLAB wrappers are provided in `matlab/+nlolib` and call into the Python ctypes bindings (`python/nlolib_ctypes.py`) with no MEX dependency.

```matlab
addpath("matlab")
api = nlolib.NLolib();
cfg = struct(...); % see examples/matlab/runtime_temporal_demo.m
records = api.propagate(cfg, field0, 2);
```

Runtime operators accept either explicit strings (`dispersion_expr`, `nonlinear_expr`) or MATLAB function handles (`dispersion_fn`, `nonlinear_fn`) that are translated to the internal expression grammar at runtime.
