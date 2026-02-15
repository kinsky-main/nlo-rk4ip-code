# TODO List
- [ ] Fix double declarations and definitions of vector operations in backend and numerics
- [ ] Assess whether SPIR-V is required for GPU backend or if Vulkan compute shaders are sufficient. Ideally remove as it is a runtime dependency.
- [ ] Investigate GPU benchmark failure.
- [ ] Investigate why RK4 solver is exploding to NaNs generally.

## Benchmarks

Build benchmark targets by enabling `NLOLIB_BUILD_BENCHMARKS`:

```powershell
cmake -S . -B build-bench -DNLOLIB_BUILD_BENCHMARKS=ON
cmake --build build-bench --target bench_solver_backend --config Release
```

Run CPU vs GPU end-to-end solver benchmark:

```powershell
$env:PATH = "$env:PATH;C:/libs/fftw"
.\build-bench\benchmarks\Release\bench_solver_backend.exe --backend=both --sizes=1024,4096 --warmup=2 --runs=8 --csv=benchmarks/results/solver_backend.csv
```

Output:
- Console summary statistics per backend and size
- CSV rows in `benchmarks/results/solver_backend.csv`
