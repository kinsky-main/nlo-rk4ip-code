# nlolib for MATLAB

MATLAB toolbox for the nlolib nonlinear-optics runtime: a split-step (RK4IP)
propagator for pulses and beams in nonlinear media, with a CPU backend and an
optional Vulkan GPU backend. It supports temporal, transverse (x-y) and coupled
3+1D propagation, with operators given as expression strings or MATLAB function
handles.

## Install

Install **nlolib** from the Add-On Explorer, or download `nlolib.mltbx` from
the [latest release](https://github.com/kinsky-main/nlolib-matlab/releases/latest)
and double-click it, or:

```matlab
matlab.addons.install('nlolib.mltbx');
```

Requires MATLAB R2023b or later on Windows or Linux (x86-64), and nothing else:
no C compiler, no Visual C++ redistributable, no Vulkan runtime. The toolbox
ships the native library and a prebuilt `loadlibrary` prototype for both
platforms.

## Quick start

A run needs a pulse, two operators and a propagation distance; the solver
schedule comes from a preset.

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

## Examples

The toolbox installs its examples; `nlolib_setup()` adds them to the path.
These three are self-contained and run as they stand:

- `standalone_transverse_grin_beam` — transverse x-y propagation with a GRIN
  potential grid.
- `standalone_spatiotemporal_kerr_bullet` — coupled 3+1D Kerr propagation
  validated against Gaussian beam and pulse laws.
- `grin_fiber_soliton_fission` — higher-order soliton fission in the
  fundamental mode of a GRIN fibre.

Further examples cover function-handle operators, soliton and linear-drift
validation, and coupled dispersion, nonlinearity and diffraction.

## CPU and GPU

The CPU backend always works. The Vulkan backend is used when a Vulkan driver
is present. On Vulkan each operator program is compiled to a single GPU kernel
at runtime and falls back to an interpreter silently when it cannot be; two
counters say which happened:

```matlab
api.perf_profile_reset();
result = api.propagate(pulse, linearOperator, nonlinearOperator, options);

snapshot = api.perf_profile_read();
fprintf('%d JIT programs, %d interpreter fallbacks\n', ...
        snapshot.operator_jit_programs, snapshot.operator_jit_fallbacks);
```

## What is in the toolbox

| Symbol | Purpose |
|---|---|
| `nlolib.NLolib` | wrapper class: `propagate`, `query_runtime_limits`, logging, progress, performance counters, storage |
| `nlolib.constants` | C API enum values and limits by name |
| `nlolib.tensor_record_view` | coupled record row → `(nt, ny, nx)` volume |
| `nlolib.translate_runtime_handle` | function handle → runtime expression plus captured constants |
| `nlolib_setup` | add the package, examples and library folders to the MATLAB path |

## Documentation

- [User guide](docs/user_guide.md) — pulse and operator specifications,
  options, telemetry, coupled 2D/3D runs, logging, storage, troubleshooting.
  Also installed with the toolbox, under `docs/`.
- [Full documentation and API reference](https://kinsky-main.github.io/nlo-rk4ip-code/)
- [Binding internals, tests and packaging](docs/binding_guide.md) and
  [working on the toolbox itself](docs/development.md)

## Links

- Source and issues: https://github.com/kinsky-main/nlolib-matlab
- Core C library: https://github.com/kinsky-main/nlo-rk4ip-code

## License

GPL-2.0-or-later.
