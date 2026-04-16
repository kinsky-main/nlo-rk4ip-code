# Julia Binding Guide

The Julia package exposes the native wrapper through the exported `NLOLib`
module.

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

```julia
using NLOLib

beta2 = -0.02
alpha = 0.10
gamma = 1.20

phys = NLOLib.physics_config(
    dispersion_factor_expr = "i*c0*w*w-c1",
    nonlinear_expr = "i*c2*A*I",
    constants = [0.5 * beta2, 0.5 * alpha, gamma],
)
```

## Build-Tree Usage

```bash
cmake --build build --config Release --target julia_stage
```

Then in Julia:

```julia
using Pkg
Pkg.activate("build/julia_package")
using NLOLib
NLOLib.load()
```

## Public API Entry Points

- `NLOLib.NLolib`
- `NLOLib.RuntimeOperators`
- `NLOLib.PulseSpec`
- `NLOLib.physics_config`
- `NLOLib.query_runtime_limits`
- `NLOLib.propagate`
- `NLOLib.propagate!`

## Related Reference

- @ref julia_binding
