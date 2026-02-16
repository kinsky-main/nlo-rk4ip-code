# TODO List
- [ ] Fix double declarations and definitions of vector operations in backend and numerics
- [ ] Assess whether SPIR-V is required for GPU backend or if Vulkan compute shaders are sufficient. Ideally remove as it is a runtime dependency.
- [ ] Investigate GPU benchmark failure.
- [ ] Investigate why RK4 solver is exploding to NaNs generally.
- [ ] Write db io for larger datasets and implement checkpointing in solver when problem size exceeds system memory limits.
- [ ] Add more benchmarks and diagnostics, e.g. per-kernel timings, memory usage, RK4 intermediate state dumps, etc.

## FFTW

When `NLO_ENABLE_FFTW=ON`, FFTW is built from source and linked statically by default.

```powershell
cmake -S . -B build -DNLO_INSTALL_GIT_HOOKS=OFF
cmake --build build --config Debug
```

To use a preinstalled FFTW instead, disable the in-tree static build:

```powershell
cmake -S . -B build -DNLO_BUILD_FFTW_FROM_SOURCE=OFF -DFFTW3_ROOT=C:/libs/fftw
```

## Benchmarks

Build benchmark targets by enabling `NLOLIB_BUILD_BENCHMARKS`:

```powershell
cmake -S . -B build-bench -DNLOLIB_BUILD_BENCHMARKS=ON
cmake --build build-bench --target bench_solver_backend --config Release
```

Run CPU vs GPU end-to-end solver benchmark:

```powershell
.\build-bench\benchmarks\Release\bench_solver_backend.exe --backend=both --sizes=1024,4096 --warmup=2 --runs=8 --csv=benchmarks/results/solver_backend.csv
```

Output:
- Console summary statistics per backend and size
- CSV rows in `benchmarks/results/solver_backend.csv`
