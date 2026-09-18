# MATLAB Binding Guide

The MATLAB package exposes the native wrapper through the class
`nlolib.NLolib`, using `loadlibrary`/`calllib` with no Python or MEX dependency.

For the full walkthrough — pulse and operator specifications, options,
telemetry, coupled 2D/3D runs, logging, storage, and troubleshooting — see the
[MATLAB User Guide](../docs/matlab_user_guide.md). This page covers the binding
itself: install, build-tree usage, tests, and design notes.

## Install

Option A — toolbox (recommended). Download `nlolib.mltbx` from the release page
and double-click it, or:

```matlab
matlab.addons.install('nlolib.mltbx');
```

Option B — build tree:

```powershell
cmake --build build --config Release --target matlab_stage
```

```matlab
addpath('build/matlab_toolbox');
nlolib_setup();
api = nlolib.NLolib();
```

Requires MATLAB R2019b or later and nothing else — no C compiler, no Visual C++
redistributable, no Vulkan runtime. The packaged toolbox ships a prebuilt
`loadlibrary` prototype, so the header is never parsed on the client.

The shared library (`nlolib.dll` / `libnlolib.so` / `libnlolib.dylib`) must be
reachable at runtime; the wrapper searches common staged and build-tree
locations, and `NLOLIB_LIBRARY` overrides the search. In a build tree with no
staged prototype the wrapper falls back to parsing `nlolib_matlab.h`, which
does need a compiler — run `matlab/generate_library_prototype.m` to avoid it.

## Minimal Math-to-API Example

This example uses the quadratic GLSE mapping

\f[
D(\omega)=i\left(\frac{\beta_2}{2}\right)\omega^2-\frac{\alpha}{2},
\qquad
N(A)=i\gamma A|A|^2
\f]

so the runtime constants are

- `c0 = beta2 / 2`
- `c1 = alpha / 2`
- `c2 = gamma`

```matlab
beta2 = -0.02;
alpha = 0.10;
gamma = 1.20;

cfg.runtime.dispersion_factor_expr = "i*c0*w*w-c1";
cfg.runtime.nonlinear_expr = "i*c2*A*I";
% num_constants is derived from numel(cfg.runtime.constants); do not set it.
cfg.runtime.constants = [0.5 * beta2, 0.5 * alpha, gamma];

api = nlolib.NLolib();
```

## Quick Start

The high-level entry point derives the solver schedule from a preset, so a run
needs only a pulse, two operators, and a propagation distance:

```matlab
n  = 512;
dt = 0.02;
t  = ((0:(n - 1)) - 0.5 * (n - 1)) * dt;

pulse = struct( ...
    'samples',    exp(-(t / 0.25).^2) .* exp(-1i * 8.0 * t), ...
    'delta_time', dt);

linearOperator = struct('expr',   "i*beta2*w*w - loss", ...
                        'params', struct('beta2', -0.005, 'loss', 0.0));
nonlinearOperator = struct('expr',   "i*gamma*A*I", ...
                           'params', struct('gamma', 0.01));

options = struct('propagation_distance', 1.0, ...
                 'records',              128, ...
                 'preset',               "balanced");

api    = nlolib.NLolib();
result = api.propagate(pulse, linearOperator, nonlinearOperator, options);

imagesc(t, result.z_axis, abs(result.records).^2);
```

The equivalent low-level call passes a flat config struct straight through:

```matlab
result = api.propagate(cfg, field0, 128, struct('backend_type', 2));
```

Both forms are documented in
[The Two API Levels](../docs/matlab_user_guide.md#levels).

## Examples

Self-contained scripts that run against an installed toolbox:

- [`examples/matlab/standalone_transverse_grin_beam.m`](../examples/matlab/standalone_transverse_grin_beam.m)
  — transverse x-y propagation with a GRIN potential grid.
- [`examples/matlab/standalone_spatiotemporal_kerr_bullet.m`](../examples/matlab/standalone_spatiotemporal_kerr_bullet.m)
  — coupled 3+1D Kerr propagation validated against Gaussian beam/pulse laws.

Repo-helper examples (run `addpath('examples/matlab')` first) cover callable
runtime operators, soliton and linear-drift validation, GRIN fibre transverse
phase, and coupled dispersion/nonlinearity/diffraction. The full index is in the
[user guide](../docs/matlab_user_guide.md#examples).

## Public API Entry Points

| Symbol | Purpose |
|---|---|
| `nlolib.NLolib` | wrapper class: `propagate`, `query_runtime_limits`, logging, progress, perf counters, storage availability |
| `nlolib.translate_runtime_handle` | function handle → runtime expression plus captured constants |
| `nlolib.prepare_sim_config` | MATLAB config struct → `simulation_config` / `runtime_operator_params` libstructs |
| `nlolib.pack_complex_array` | complex vector → `nlo_complex*` buffer |
| `nlolib.unpack_records` | `nlo_complex*` records → complex matrix |
| `nlolib_setup` | add package, examples, and library directories to the MATLAB path |

## Tests

| CTest name | Needs MATLAB? | Covers |
|---|---|---|
| `test_matlab_binding_types` | no | Every `libstruct('T')`/`libpointer('TPtr')` name resolves to a typedef in `src/nlolib_matlab.h` |
| `test_matlab_header_abi` | no | Compile-time `sizeof`/`offsetof` parity between `nlolib_matlab.h` and the canonical headers |
| `test_matlab_bindings` | yes | Runtime-handle translation plus an end-to-end FFI smoke suite (propagate, step history, runtime limits, log buffer, perf counters) |

Run the MATLAB-free checks alone with:

```bash
ctest --test-dir build -R "test_matlab_binding_types|test_matlab_header_abi"
```

## Design Notes

`libstruct(TYPE)` with no initialiser creates an *empty* object that `calllib`
marshals as `NULL`; output-parameter calls must use
`NLolib.zeroed_libstruct()`. Reading `.Value` on a struct pointer returns only
the first element, and `setdatatype()` rejects struct pointers outright, so
array read-back uses pointer arithmetic (`ptr = ptr + 1`).

`prepare_sim_config` returns a `keepalive` cell array holding the `libpointer`
objects that back the pointer fields. It must stay in scope for as long as the
config is used, otherwise MATLAB frees the buffers underneath the library.

Why the binding is not generated with `clibgen`:
[docs/matlab_binding_clibgen_evaluation.md](../docs/matlab_binding_clibgen_evaluation.md).

## Related Reference

- [MATLAB User Guide](../docs/matlab_user_guide.md)
- [Runtime Operator Mathematics](../docs/runtime_operators.md)
- @ref matlab_binding
