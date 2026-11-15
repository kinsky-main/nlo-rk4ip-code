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
- `nlolib.storage_run_key`, `nlolib.storage_has_run`, `nlolib.storage_recover_staging` --
  run keys and the shared run database (`sqlite_path=...`, `record_mode=...`);
  layout in `docs/nlolib_run_database.md`
- `nlolib.__version__` -- the project version, from the installed distribution
  metadata or, in a source checkout, from `pyproject.toml`

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
