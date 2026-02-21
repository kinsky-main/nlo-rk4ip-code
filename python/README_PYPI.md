# nlolib (Python)

Python package for the nlolib nonlinear-optics runtime.

## Runtime behavior

- Wheels are intended to bundle `nlolib` and non-system dependencies required by the wrapper.
- CPU execution path is expected to work out-of-the-box.
- GPU/Vulkan execution is optional and depends on platform driver/runtime availability.

## Quick start

```python
import nlolib

api = nlolib.NLolib()
limits = api.query_runtime_limits()
print(limits.max_num_time_samples_runtime)
```
