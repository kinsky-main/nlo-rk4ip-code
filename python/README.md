# Python Binding Guide

The Python package exposes the user-facing `nlolib` wrapper around the native
shared library.

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

```python
import nlolib

beta2 = -0.02
alpha = 0.10
gamma = 1.20

runtime = nlolib.RuntimeOperators(
    dispersion_factor_expr="i*c0*w*w-c1",
    nonlinear_expr="i*c2*A*I",
    constants=[0.5 * beta2, 0.5 * alpha, gamma],
)
```

## Public API Entry Points

- `nlolib.NLolib`
- `nlolib.RuntimeOperators`
- `nlolib.PulseSpec`
- `nlolib.OperatorSpec`
- `nlolib.propagate`
- `nlolib.query_runtime_limits`
- `nlolib.translate_callable`
- `nlolib.perf_profile_read`, `nlolib.perf_profile_reset`,
  `nlolib.perf_profile_set_enabled`, `nlolib.perf_profile_is_enabled`
- `nlolib.__version__` -- the project version, from the installed distribution
  metadata or, in a source checkout, from `pyproject.toml`

## Checking Whether the GPU JIT Ran

On the Vulkan backend each operator program is compiled to a single GPU kernel
at runtime, and falls back to the interpreter when it cannot be. The fallback is
silent: results stay correct, only slower. Two counters say which happened.

```python
api = nlolib.NLolib()
api.perf_profile_reset()
api.propagate(cfg, field0, 1,
              nlolib.default_execution_options(nlolib.NLO_VECTOR_BACKEND_AUTO))

snapshot = api.perf_profile_read()
print(snapshot.operator_jit_programs)   # 3: one kernel per active operator program
print(snapshot.operator_jit_fallbacks)  # 0: nothing was left on the interpreter
```

Unlike the timing counters in the same snapshot, these two need no
`perf_profile_set_enabled(True)`, so a release build can be checked as it
stands. Both stay zero wherever the JIT is never reached -- the CPU backend, or
a build configured `ENABLE_VULKAN_BACKEND=OFF` -- which is not the same as a
program falling back. See [GPU backend behaviour](../docs/gpu_backend.md).

## Build-Tree Usage

```powershell
python -m pip install -r examples\python\requirements.txt
$env:PYTHONPATH="$PWD/python"
$env:NLOLIB_LIBRARY="$PWD/python/nlolib.dll"
python examples/python/runtime_temporal_demo.py
```

```bash
python3 -m pip install -r examples/python/requirements.txt
export PYTHONPATH="$PWD/python"
export NLOLIB_LIBRARY="$PWD/python/libnlolib.so"
python3 examples/python/runtime_temporal_demo.py
```

## Related Reference

- @ref python_binding
