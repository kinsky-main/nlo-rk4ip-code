# MATLAB Binding Guide

The MATLAB package exposes the native wrapper through the class
`nlolib.NLolib`.

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

## Build-Tree Usage

```powershell
cmake --build build --config Release --target matlab_stage
```

Then in MATLAB:

```matlab
addpath('build/matlab_toolbox');
nlolib_setup();
api = nlolib.NLolib();
```

## Public API Entry Points

- `nlolib.NLolib`
- `nlolib.translate_runtime_handle`
- `nlolib.prepare_sim_config`
- `nlolib.pack_complex_array`
- `nlolib.unpack_records`

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

Why the binding is not generated with `clibgen`:
[docs/matlab_binding_clibgen_evaluation.md](../docs/matlab_binding_clibgen_evaluation.md).

## Related Reference

- @ref matlab_binding
