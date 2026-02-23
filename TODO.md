# Final Year Project TODO List for NLOLib

## Main Tasks
- [x] Fix double declarations and definitions of vector operations in backend and numerics
- [x] Assess whether SPIR-V is required for GPU backend or if Vulkan compute shaders are sufficient. Ideally remove as it is a runtime dependency.
- [x] Investigate GPU benchmark failure.
- [x] Investigate why RK4 solver is exploding to NaNs generally.
- [x] Write db io for larger datasets and implement checkpointing in solver when problem size exceeds system memory limits.
- [x] Expand solver backend to support more unified kernel operations (Complex Potentials) and more flexible data layouts
- [x] Implement MATLAB interface for solver with string API for defining input operators.
- [x] Implement dispersion operator custom functions, currently staging function is not used.
- [x] Combine grin vector operator into dispersion and nonlinear operator expressions.
- [x] Remove old API and just have custom functions for dispersion and nonlinearity, this removes the complications of additional vector operators.
- [x] Implement new GRIN operations using preferred kernels for (2+1)D operations.
- [ ] Add more benchmarks and diagnostics, e.g. per-kernel timings, memory usage, RK4 intermediate state dumps, etc.
- [x] Package and distribute library for Windows and Linux.
- [ ] Go over soliton analytical solution and ensure it is correct as there is still an oscillatory L2 error in the solver compared to analytical solution.
- [ ] Fix MATLAB output for installed packages where progress is not printed.
- [ ] Improve printout for progress of solver, showing estimated time of completion.
- [ ] Remove redundant wrappers and functions from MATLAB and Python bindings, duplications of factor expressions and dispersion expressions.
- [ ] Remove bloat in nlolib.c which results in 3 pointless function wrappers for the same operations.
- [ ] Remove bloat in wrappers which mirrors public API functions in nlolib.c bloat.
- [ ] Fix soliton example, actually find original reference for analytical solution and ensure it is correct.
- [ ] Check linear drift solution to see why error is bouncing around 0.6 z units, there is some characteristic in the error which should not be there.

## Potentially Required Tasks

- [ ] Implement more efficient GPU memory management and data transfer strategies, e.g. pinned memory, async transfers, etc.
- [ ] Implement arbitrary kernel combinations to chain together operations into single GPU kernels for better performance.
- [ ] Move transverse operator into main nonlinear and dispersion operators, may be required for certain potentials.
- [ ] `query_runtime_limits` should return accurate grid size limits for GPU and CPU backend.
- [ ] Loading bar with estimated time remaining for long-running simulations.
- [ ] Refactor state.c into respective init files.
- [ ] Remap (3+1)D problems to use full size tensors rather than flattening to 1D arrays, this will also require implementation of dedicated GPU kernels for (3+1)D problems.
- [ ] Find a better way to implement the parser for operator expressions, current letter style parsing is not very robust and also relies heavily on the wrapper interpreting function handless correctly, ideally would have a more general parser which can handle arbitrary number of variables and coefficients.

## Extensions

- [ ] Add Massively Parallel Algorithm solver mode for coupled mode problems.
- [ ] OpenMP backend for multi-core CPU parallelism.
- [ ] Add example problem documentation on the physics of the problems (Do this in the report first ;)).
