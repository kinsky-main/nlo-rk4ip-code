# nlolib

Python package for the nlolib nonlinear-optics runtime: a split-step (RK4IP)
propagator for pulses and beams in nonlinear media, with a CPU backend and an
optional Vulkan GPU backend.

```bash
pip install nlolib
```

Wheels are published for Linux x86-64 and Windows x86-64 and work on any
CPython 3.9 or later. The native library ships inside the wheel, so nothing is
compiled on install.

Full documentation, including the Python guide and API reference:
https://kinsky-main.github.io/nlo-rk4ip-code/

## Quick start

```python
import nlolib

limits = nlolib.query_runtime_limits()
print(limits.max_num_time_samples_runtime)

def linear(A, w):
    return 0.0

def nonlinear(A, I):
    return 0.0

pulse = nlolib.PulseSpec(samples=[0j] * 64, delta_time=0.01, pulse_period=0.64)
result = nlolib.propagate(
    pulse,
    nlolib.OperatorSpec(fn=linear),
    nlolib.OperatorSpec(fn=nonlinear),
    propagation_distance=0.1,
    t_eval=[0.0, 0.05, 0.1],
)
print(result.status, result.message, len(result.records))
```

Operators can also be given as expression strings with named constants. For
the GLSE with dispersion `beta2`, loss `alpha` and Kerr coefficient `gamma`:

```python
runtime = nlolib.RuntimeOperators(
    dispersion_factor_expr="i*c0*w*w-c1",
    nonlinear_expr="i*c2*A*I",
    constants=[0.5 * beta2, 0.5 * alpha, gamma],
)
```

## CPU and GPU

- The CPU backend works out of the box.
- The Vulkan backend is used when a Vulkan driver is present; without one the
  CPU backend runs unchanged.
- On Vulkan, operator programs are compiled to single GPU kernels at runtime
  and fall back to an interpreter silently when they cannot be. After a run,
  `nlolib.perf_profile_read()` reports which happened:

  ```python
  snapshot = nlolib.perf_profile_read()
  snapshot.operator_jit_programs   # kernels compiled; 0 on CPU
  snapshot.operator_jit_fallbacks  # programs left on the interpreter
  ```

## Examples

The [`examples/`](https://github.com/kinsky-main/nlolib-python/tree/main/examples)
directory of the source repository holds runnable scripts: soliton and
self-phase-modulation validation, Raman scattering, GRIN-fibre propagation,
runtime-callable operators, and SQLite snapshot storage. They need the extra
packages in `examples/requirements.txt`.

## Links

- Source and issues: https://github.com/kinsky-main/nlolib-python
- Core C library: https://github.com/kinsky-main/nlo-rk4ip-code
- Working on the package itself: [docs/development.md](https://github.com/kinsky-main/nlolib-python/blob/main/docs/development.md)

## License

GPL-2.0-or-later.
