# NLOLib.jl

Julia package for the nlolib nonlinear-optics runtime: a split-step (RK4IP)
propagator for pulses and beams in nonlinear media, with a CPU backend and an
optional Vulkan GPU backend.

## Install

```julia
using Pkg
Pkg.add("NLOLib")
```

The native library and its runtime dependencies arrive as a Julia artifact, so
nothing is compiled on install and no Vulkan SDK is needed. Linux x86-64 and
Windows x86-64 are supported, on Julia 1.10 or later. On Linux the Vulkan
loader (`libvulkan.so.1`) must be present because it is a plain shared-library
dependency; on Windows it is optional. Without a Vulkan driver the CPU backend
runs unchanged.

## Quick start

Operators are expression strings with named constants. For the GLSE with
dispersion `beta2`, loss `alpha` and Kerr coefficient `gamma`:

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

The main entry points are `NLOLib.NLolib`, `NLOLib.RuntimeOperators`,
`NLOLib.PulseSpec`, `NLOLib.physics_config`, `NLOLib.query_runtime_limits`,
`NLOLib.propagate` and `NLOLib.propagate!`.

`NLOLib.load()` resolves the library from an explicit path, then
`ENV["NLOLIB_LIBRARY"]`, then the installed artifact.

## Examples

The [`examples/`](examples) directory is a Julia project of its own with
runnable scripts: second-order and high-order GRIN solitons, Raman scattering,
and 3D tensor dispersion.

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/second_order_soliton_rk4ip.jl
```

## Documentation

- [Full documentation and API reference](https://kinsky-main.github.io/nlo-rk4ip-code/)
- [Binding guide](docs/guide.md)
- [Working on the package itself](docs/development.md) and
  [releasing](docs/release.md)

## Links

- Source and issues: https://github.com/kinsky-main/NLOLib.jl
- Core C library: https://github.com/kinsky-main/nlo-rk4ip-code

## License

GPL-2.0-or-later.
