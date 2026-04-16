# nlolib (Python)

Python package for the nlolib nonlinear-optics runtime.

Full rendered documentation, including the Python guide and API reference, is
available at https://kinsky-main.github.io/nlo-rk4ip-code/.

For the docs-facing Python usage guide in this repository, see `python/README.md`.

## Runtime behavior

- Wheels are intended to bundle `nlolib` and non-system dependencies required by the wrapper.
- CPU execution path is expected to work out-of-the-box.
- GPU/Vulkan execution is optional and depends on platform driver/runtime availability.

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
