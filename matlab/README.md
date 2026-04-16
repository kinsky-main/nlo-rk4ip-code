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
cfg.runtime.num_constants = uint64(3);
cfg.runtime.constants(1:3) = [0.5 * beta2, 0.5 * alpha, gamma];

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

## Related Reference

- @ref matlab_binding
