# MATLAB User Guide

This guide is a task-oriented walkthrough of the MATLAB binding: how to install
it, how to describe a problem, how to run it, and how to read the result. The
symbol-level reference lives in @ref matlab_binding; the mathematics of the
runtime operators is in [Runtime Operator Mathematics](runtime_operators.md).

Everything below runs against `nlolib.NLolib`, a thin `loadlibrary`/`calllib`
wrapper over the native shared library. There is no Python or MEX dependency.

- [Install and Verify](#install)
- [The Two API Levels](#levels)
- [Quick Start](#quickstart)
- [Describing the Pulse](#pulse)
- [Describing the Operators](#operators)
- [Propagation Options](#options)
- [Reading the Result](#result)
- [Coupled 2D and 3D Runs](#tensor)
- [Low-Level Configuration API](#lowlevel)
- [Solver Telemetry and Step History](#telemetry)
- [Logging, Progress, and Perf Counters](#logging)
- [Sizing a Run with query_runtime_limits](#limits)
- [SQLite Storage](#storage)
- [Troubleshooting](#troubleshooting)
- [Example Scripts](#examples)

<a name="install"></a>

## Install and Verify

### Prerequisites

- MATLAB R2019b or later.
- A C compiler visible to MATLAB. `loadlibrary` parses `nlolib_matlab.h` at
  load time and needs one on Windows; check with `mex -setup C` if the load
  fails while complaining about the header.
- For the Vulkan backend, a GPU driver shipping the Vulkan loader (standard
  NVIDIA / AMD / Intel desktop drivers include it). The CPU backend needs
  nothing extra.

### Option A — toolbox install (recommended)

Download `nlolib.mltbx` from the GitHub release page and double-click it, or:

```matlab
matlab.addons.install('nlolib.mltbx');
```

The toolbox bundles the package, the shared library, `nlolib_matlab.h`, and the
example scripts.

### Option B — build tree

```powershell
cmake -S . -B build
cmake --build build --config Release --target matlab_stage
```

```matlab
addpath('build/matlab_toolbox');
nlolib_setup();
```

`nlolib_setup` adds the package and example folders to the MATLAB path and, on
Windows, prepends the staged library directories to `PATH` so the loader can
resolve `nlolib.dll`. Pass `nlolib_setup(true)` to persist the path with
`savepath`.

### Library discovery

The constructor searches, in order:

1. an explicit path — `nlolib.NLolib('C:\path\to\nlolib.dll')`;
2. the `NLOLIB_LIBRARY` environment variable;
3. staged and build-tree locations relative to the package
   (`lib/`, `build/matlab_toolbox/lib/`, `build/src/<config>/`, `python/`, ...),
   newest file first.

Set the override from MATLAB with `setenv('NLOLIB_LIBRARY', ...)` before
constructing the object.

### Verify the install

```matlab
api = nlolib.NLolib();          % throws if the library cannot be loaded
disp(api.query_runtime_limits); % prints device budget and record limits
disp(api.storage_is_available); % true when SQLite storage is compiled in
```

Constructing `nlolib.NLolib` twice reuses the already-loaded library. Release it
explicitly with the static `nlolib.NLolib.unload()`; the destructor deliberately
does not, since several wrapper objects may share one loaded library.

<a name="levels"></a>

## The Two API Levels

`propagate` accepts two calling conventions and picks between them by inspecting
the first argument.

| Level | Call | Chosen when |
|---|---|---|
| High-level | `api.propagate(pulse, linearOp, nonlinearOp, options)` | first argument has `samples` and `delta_time` and no `num_time_samples` |
| Low-level | `api.propagate(cfg, field, numRecords, ...)` | anything else |

The high-level form derives step sizes, error tolerance, record count, and the
frequency grid from a preset and the pulse; the low-level form passes a flat
config struct through to the C API essentially untouched. Start high-level;
drop to the low-level form when you need to pin solver parameters or use
runtime features the facade does not expose (Raman, explicit `wt`/`kx`/`ky`
axes, split `linear_factor_expr`/`linear_expr` operators).

<a name="quickstart"></a>

## Quick Start

A chirped Gaussian in a purely dispersive medium, using the built-in operator
presets:

```matlab
api = nlolib.NLolib();

n  = 512;
dt = 0.02;
t  = ((0:(n - 1)) - 0.5 * (n - 1)) * dt;

pulse = struct();
pulse.samples    = exp(-(t / 0.25).^2) .* exp(-1i * 8.0 * t);
pulse.delta_time = dt;

options = struct();
options.propagation_distance = 1.0;

% "gvd" -> i*beta2*w^2 - loss, "kerr" -> i*gamma*A*|A|^2, with preset constants
result = api.propagate(pulse, "gvd", "kerr", options);

fprintf('%d records of %d samples\n', size(result.records));
fprintf('power in=%.6e out=%.6e\n', ...
        sum(abs(pulse.samples).^2), sum(abs(result.final).^2));
```

The same run with explicit physical coefficients — the usual starting point for
real work:

```matlab
beta2 = -0.01;    % GVD  [ps^2/m]
gamma = 0.01;     % Kerr [1/W/m]
alpha = 0.0;      % amplitude loss [1/m]

linearOperator = struct( ...
    'expr',   "i*beta2*w*w - loss", ...
    'params', struct('beta2', 0.5 * beta2, 'loss', 0.5 * alpha));

nonlinearOperator = struct( ...
    'expr',   "i*A*(gamma*I + V)", ...
    'params', struct('gamma', gamma));

options = struct();
options.propagation_distance = 0.506;
options.records              = 160;
options.preset               = "balanced";

result = api.propagate(pulse, linearOperator, nonlinearOperator, options);
```

Note the factors of one half: the solver applies \f$D(\omega)\f$ as written, so
the GLSE term \f$i(\beta_2/2)\partial_t^2\f$ maps to `beta2 -> beta2/2`, and a
power-loss coefficient \f$\alpha_{\mathrm{pow}}\f$ maps to
`loss -> alpha_pow/2`. See [Runtime Operator
Mathematics](runtime_operators.md).

<a name="pulse"></a>

## Describing the Pulse

The pulse struct describes the launch field and the grid it lives on.

| Field | Required | Meaning |
|---|---|---|
| `samples` | yes | complex launch field, flattened to a row vector |
| `delta_time` | yes | temporal sample spacing, > 0 |
| `pulse_period` | no | temporal window; default `nt * delta_time` |
| `frequency_grid` | no | complex angular-frequency axis, one entry per temporal sample; default is the FFT-order grid for `nt` and `delta_time` |
| `tensor_nt`, `tensor_nx`, `tensor_ny` | no | coupled-run shape; all three together |
| `tensor_layout` | no | `0` = `TENSOR_LAYOUT_XYT_T_FAST` (t fastest, then y, then x) |
| `delta_x`, `delta_y` | no | transverse sample spacing; default `1.0` |
| `spatial_frequency_grid` | no | explicit transverse frequency grid, length `nx*ny` or the full volume |
| `potential_grid` | no | complex potential \f$V\f$, length `nx*ny` or the full volume |

`samples` is reshaped to a row vector internally, so any orientation works. The
default frequency grid is FFT-ordered (non-negative frequencies first, then the
negative half), matching what the library expects:

```matlab
idx   = 0:(n - 1);
half  = floor((n - 1) / 2);
omega = (2 * pi / (n * dt)) * (idx - n * (idx > half));
pulse.frequency_grid = complex(omega, zeros(1, n));
```

Supplying it explicitly is worthwhile whenever you also want the same axis for
plotting or spectral post-processing.

<a name="operators"></a>

## Describing the Operators

Both operator arguments accept a preset name, an expression struct, or a
function handle struct.

### Presets

| Argument | Preset | Expression | Default params |
|---|---|---|---|
| linear | `"gvd"`, `"default"` | `i*beta2*w*w-loss` | `beta2 = -0.5`, `loss = 0` |
| linear | `"none"` | `0` | — |
| nonlinear | `"kerr"`, `"default"` | `i*gamma*A*I` | `gamma = 1.0` |
| nonlinear | `"none"` | `0` | — |

A preset name may be overridden by supplying `params` alongside it, and any
string that is not a known preset is taken as a literal expression:

```matlab
linearOperator = struct('expr', "gvd", 'params', struct('beta2', -0.005, 'loss', 0.0));
```

### Expression operators

`expr` is compiled by the runtime expression compiler. Available symbols:

| Symbol | Meaning | Valid in |
|---|---|---|
| `A` | current field value | linear factor, nonlinear |
| `I` | \f$\lvert A\rvert^2\f$ | nonlinear |
| `V` | potential / auxiliary term | nonlinear |
| `w` | temporal angular frequency | linear factor |
| `wt`, `kx`, `ky` | tensor frequency axes | linear factor (tensor runs) |
| `t`, `x`, `y` | tensor coordinate axes | linear factor (tensor runs) |
| `i` | imaginary unit | everywhere |
| `c0`, `c1`, ... | runtime constants | everywhere |
| `exp`, `log`, `sqrt`, `sin`, `cos` | intrinsics | everywhere |

`params` binds names to constants. Each named parameter is substituted for a
`cN` token and appended to the runtime constant table; linear constants are
allocated first, then nonlinear ones, so you never index `cN` by hand:

```matlab
% "i*beta2*w*w-loss" with params beta2, loss  ->  "i*c0*w*w-c1", constants [c0 c1]
% "i*gamma*A*I"      with params gamma        ->  "i*c2*A*I",    constants [... c2]
```

`params` may also be a plain numeric vector, in which case the expression must
already reference `c0`, `c1`, ... directly. The table holds at most 16
constants (`RUNTIME_OPERATOR_CONSTANTS_MAX`); exceeding that raises
`nlolib:tooManyRuntimeConstants`.

### Function-handle operators

Supply `fn` instead of `expr` and the handle is translated to an expression by
`nlolib.translate_runtime_handle`. Argument positions determine the symbol
binding:

| Context | Handle signature |
|---|---|
| linear (dispersion factor) | `@(A, w)` — one or two arguments |
| nonlinear | `@(A, I, V)` — one to three arguments |
| dispersion (low-level `dispersion_fn`) | `@(A, D, h, w)` — one to four arguments |

```matlab
beta2 = 0.05;
linearOperator = struct('fn', @(A, w) 1i * (beta2 / 2.0) * (w .* w));
```

Rules that matter in practice:

- Element-wise operators are normalised (`.*` → `*`, `.^` → `^`, `./` → `/`),
  and `1i`/`1j` literals become `i`. The body must still be a single
  expression — no statements, no branching, no calls beyond the intrinsics
  above.
- Any identifier that is not a reserved symbol is captured from the handle's
  closure as a real scalar constant (here, `beta2`). A captured value that is
  non-scalar, complex, or non-finite is an error.
- Pin values explicitly with `runtime.constant_bindings` (a struct or
  `containers.Map`), or disable capture with
  `runtime.auto_capture_constants = false`, in the low-level config.
- `expr` and `fn` are mutually exclusive on the same operator, and `params` is
  not supported alongside `fn`.

You can inspect the translation directly, which is the fastest way to debug a
handle:

```matlab
[expr, constants] = nlolib.translate_runtime_handle( ...
    @(A, w) 1i * (beta2 / 2.0) * (w .* w), "dispersion_factor");
% expr      = 'i*(c0/2.0)*(w*w)'
% constants = [0.05]
```

<a name="options"></a>

## Propagation Options

| Field | Default | Meaning |
|---|---|---|
| `propagation_distance` | *required*, > 0 | total \f$z\f$ to propagate |
| `preset` | `"balanced"` | solver defaults, see below |
| `records` | from preset | number of \f$z\f$ samples to record |
| `output` | `"dense"` | `"final"` forces `records = 1` |
| `exec_options` | `struct()` | backend and telemetry selection |
| `storage` | none | SQLite storage options |

### Presets

Step sizes scale with the propagation distance \f$L\f$:

| Preset | starting step | max step | min step | tolerance | records |
|---|---|---|---|---|---|
| `"fast"` | \f$L/120\f$ | \f$L/12\f$ | \f$L/4000\f$ | `5e-6` | 64 |
| `"balanced"` | \f$L/200\f$ | \f$L/25\f$ | \f$L/20000\f$ | `1e-6` | 128 |
| `"accuracy"` | \f$L/400\f$ | \f$L/50\f$ | \f$L/80000\f$ | `1e-7` | 192 |

The solver is adaptive, so these are bounds and an initial guess, not a fixed
schedule. Use the low-level API when you need exact control.

### Execution options

`options.exec_options` is passed through to `execution_options`, plus a few
MATLAB-only keys handled by the wrapper.

| Field | Default | Meaning |
|---|---|---|
| `backend_type` | `2` | `0` CPU, `1` Vulkan, `2` auto |
| `fft_backend` | `0` | `0` auto, `1` FFTW, `2` VkFFT |
| `device_heap_fraction` | `0.70` | fraction of the device budget the solver may use |
| `record_ring_target` | `0` | record ring size hint; `0` lets the library choose |
| `forced_device_budget_bytes` | `0` | override the detected device budget |
| `capture_step_history` | `false` | fill `result.step_history` |
| `step_history_capacity` | `200000` | step events to preallocate |
| `matlab_stream_logs` | `false` | drain the runtime log buffer after the call |
| `matlab_log_buffer_bytes` | `262144` | buffer size used when streaming logs |
| `matlab_progress_stream` | auto | `0` stderr, `1` stdout, `2` both; defaults to *both* when MATLAB has no desktop |
| `matlab_debug` | `false` | attach output-buffer probe details to failures |

```matlab
execOptions = struct();
execOptions.backend_type          = 1;         % Vulkan
execOptions.fft_backend           = 2;         % VkFFT
execOptions.capture_step_history  = true;
execOptions.step_history_capacity = uint64(200000);

options.exec_options = execOptions;
```

<a name="result"></a>

## Reading the Result

`propagate` returns a struct:

| Field | Type | Meaning |
|---|---|---|
| `records` | `numRecords x numSamples` complex | recorded fields, one row per \f$z\f$ sample |
| `final` | `1 x numSamples` complex | last row of `records`, `[]` when nothing was returned |
| `z_axis` | numeric | \f$z\f$ positions, `linspace(0, L, numRecords)` |
| `step_history` | struct | solver telemetry, see below |
| `meta` | struct | run metadata |

`meta` carries `output`, `records`, `records_requested`, `records_written`,
`records_returned`, `storage_enabled`, `coupled`, and `step_history_dropped`;
the high-level path adds `preset`, the low-level path adds `backend_requested`,
and either adds `storage_result` when storage is enabled.

`records_written` can be smaller than `records_requested` — the library reduces
the count for fixed-step runs, explicit-\f$z\f$ schedules, and callback-aborted
runs — so size plots off `size(result.records, 1)` rather than the request:

```matlab
intensity = abs(result.records).^2;
imagesc(t, result.z_axis, intensity);
set(gca, 'YDir', 'normal');
xlabel('t'); ylabel('z'); colorbar;

power = sum(intensity, 2);
fprintf('power drift = %.3e\n', abs(power(end) - power(1)) / power(1));
```

<a name="tensor"></a>

## Coupled 2D and 3D Runs

Setting `tensor_nt`, `tensor_nx`, and `tensor_ny` switches the run to a coupled
field of `tensor_nt * tensor_nx * tensor_ny` points. `samples` must be
flattened in `TENSOR_LAYOUT_XYT_T_FAST` order — t fastest, then y, then x —
which is exactly MATLAB's own ordering for an `(nt, ny, nx)` array, so a plain
`reshape(field0, 1, [])` is correct and `reshape(row, [ny, nx])` inverts it for
a transverse sheet.

Diffraction is encoded in the linear operator through `kx`/`ky`; the transverse
axes are generated by the library from `delta_x` and `delta_y` unless you supply
`spatial_frequency_grid` (or `kx_axis`/`ky_axis` in the low-level config).

### Transverse-only (x-y sheet)

Set `tensor_nt = 1`. The temporal fields are then unused but still required:

```matlab
[xx, yy] = meshgrid(x, y);                       % both ny-by-nx
field0    = complex(exp(-((xx - x0).^2 + yy.^2) / w^2), 0);
potential = grinDepth * (xx.^2 + yy.^2);         % V(x,y)

pulse = struct();
pulse.samples        = reshape(field0, 1, []);
pulse.tensor_nt      = 1;
pulse.tensor_nx      = nx;
pulse.tensor_ny      = ny;
pulse.tensor_layout  = 0;
pulse.delta_x        = dx;
pulse.delta_y        = dy;
pulse.delta_time     = 1.0;                      % unused when tensor_nt == 1
pulse.pulse_period   = 1.0;                      % unused when tensor_nt == 1
pulse.frequency_grid = complex(0.0, 0.0);        % one entry per tensor_nt
pulse.potential_grid = reshape(complex(potential, 0), 1, []);

linearOperator    = struct('expr',   "i*beta_t*(kx*kx + ky*ky)", ...
                           'params', struct('beta_t', -1.0 / (2.0 * k0)));
nonlinearOperator = struct('expr', "i*A*V");     % consume the potential only

result = api.propagate(pulse, linearOperator, nonlinearOperator, options);

sheets = zeros(size(result.records, 1), ny, nx);
for idx = 1:size(result.records, 1)
    sheets(idx, :, :) = reshape(result.records(idx, :), [ny, nx]);
end
```

See [`examples/matlab/standalone_transverse_grin_beam.m`](../examples/matlab/standalone_transverse_grin_beam.m)
for the complete script, which checks the beam centroid against
\f$x_0\cos(gz)\f$.

### Full 3+1D (t, x, y)

```matlab
field0 = zeros(nt, ny, nx);
for idx = 1:nt
    field0(idx, :, :) = temporal(idx) * spatial;   % spatial is ny-by-nx
end

pulse = struct();
pulse.samples        = reshape(complex(field0, 0), 1, []);
pulse.tensor_nt      = nt;
pulse.tensor_nx      = nx;
pulse.tensor_ny      = ny;
pulse.tensor_layout  = 0;
pulse.delta_time     = dt;
pulse.pulse_period   = nt * dt;
pulse.delta_x        = dx;
pulse.delta_y        = dy;
pulse.frequency_grid = complex(omega, zeros(1, nt));   % one entry per tensor_nt

linearOperator = struct( ...
    'expr',   "i*(b2*wt*wt + bt*(kx*kx + ky*ky))", ...
    'params', struct('b2', 0.5 * beta2, 'bt', -1.0 / (2.0 * k0)));
```

Note that the temporal symbol is `wt`, not `w`, once the run is a tensor run.
Full script: [`examples/matlab/standalone_spatiotemporal_kerr_bullet.m`](../examples/matlab/standalone_spatiotemporal_kerr_bullet.m).

Memory grows as the product of the three dimensions times the record count —
check the budget with [`query_runtime_limits`](#limits) before launching a large
grid.

<a name="lowlevel"></a>

## Low-Level Configuration API

```matlab
result = api.propagate(cfg, inputField, numRecordedSamples);
result = api.propagate(cfg, inputField, numRecordedSamples, execOptions);
result = api.propagate(cfg, inputField, numRecordedSamples, execOptions, storageOptions);
```

The fourth argument is treated as storage options when it has an `sqlite_path`
field, otherwise as execution options.

### Required config fields

`num_time_samples`, `propagation_distance`, `starting_step_size`,
`max_step_size`, `min_step_size`, `error_tolerance`, `pulse_period`,
`delta_time`, `frequency_grid`.

`num_time_samples` must equal `numel(inputField)` — for a tensor run that is the
whole volume, `tensor_nt * tensor_nx * tensor_ny`, not the temporal length. A
mismatch raises `nlolib:inputFieldLengthMismatch` rather than a bare
`INVALID_ARGUMENT` from the C dimension resolver.

### Optional config fields

`tensor_nt`, `tensor_nx`, `tensor_ny`, `tensor_layout`, `delta_x`, `delta_y`,
`spatial_frequency_grid`, `kx_axis`, `ky_axis`, `potential_grid`, `wt_axis`,
and `runtime`.

`wt_axis`, `kx_axis`, and `ky_axis` are left unset unless you supply them, which
is what tells the library to generate the axes itself — do not pass `[]` to
"clear" them.

### The runtime sub-struct

| Field | Meaning |
|---|---|
| `dispersion_factor_expr` / `dispersion_factor_fn` | \f$D\f$, applied as \f$\exp(hD)\f$ |
| `dispersion_expr` / `dispersion_fn` | full dispersion operator |
| `linear_factor_expr`, `linear_expr` | split linear operator form |
| `potential_expr` | potential term |
| `nonlinear_expr` / `nonlinear_fn` | \f$N(A)\f$ |
| `constants` | scalar constant table, at most 16 entries |
| `constant_bindings` | explicit name → value map for handle translation |
| `auto_capture_constants` | set `false` to reject closure capture |
| `nonlinear_model` | `0` expression model, Kerr+Raman selector otherwise |
| `nonlinear_gamma`, `raman_fraction`, `raman_tau1`, `raman_tau2`, `shock_omega0` | built-in Kerr+Raman parameters |
| `raman_response_time` | explicit complex Raman response samples |

`num_constants` is derived from `numel(runtime.constants)`; do not set it.

```matlab
beta2 = -0.02;  alpha = 0.10;  gamma = 1.20;

cfg = struct();
cfg.num_time_samples     = n;
cfg.propagation_distance = 1.0;
cfg.starting_step_size   = 5e-3;
cfg.max_step_size        = 4e-2;
cfg.min_step_size        = 5e-5;
cfg.error_tolerance      = 1e-6;
cfg.delta_time           = dt;
cfg.pulse_period         = n * dt;
cfg.frequency_grid       = complex(omega, zeros(1, n));
cfg.runtime = struct( ...
    'dispersion_factor_expr', "i*c0*w*w-c1", ...
    'nonlinear_expr',         "i*c2*A*I", ...
    'constants',              [0.5 * beta2, 0.5 * alpha, gamma]);

result = api.propagate(cfg, field0, 128, struct('backend_type', 2));
```

`nlolib.prepare_sim_config(cfg)` performs the config → `libstruct` translation
on its own if you want to inspect or reuse it:

```matlab
[simCfgPtr, physicsCfgPtr, keepalive] = nlolib.prepare_sim_config(cfg);
```

`keepalive` holds the `libpointer` objects backing the pointer fields — it must
stay in scope for as long as the pointers are used, or MATLAB will free the
buffers underneath the library.

<a name="telemetry"></a>

## Solver Telemetry and Step History

Enable capture through the execution options and read `result.step_history`:

```matlab
execOptions = struct('capture_step_history', true, ...
                     'step_history_capacity', uint64(200000));
options.exec_options = execOptions;

result  = api.propagate(pulse, linearOperator, nonlinearOperator, options);
history = result.step_history;

fprintf('%d steps, %d dropped, mean step %.3e\n', ...
        numel(history.step_index), history.dropped, mean(history.step_size));
semilogy(history.z, history.error);
xlabel('z'); ylabel('local error estimate');
```

| Field | Meaning |
|---|---|
| `step_index` | solver step counter |
| `z` | position at the start of the step |
| `step_size` | step actually taken |
| `next_step_size` | step proposed for the following iteration |
| `error` | local error estimate |
| `dropped` | events the library could not record — raise `step_history_capacity` |
| `capacity` | capacity the buffer was allocated with |

A non-zero `dropped` means the history is truncated, not that the run failed.

<a name="logging"></a>

## Logging, Progress, and Perf Counters

### Logs

```matlab
api.set_log_level(3);                  % 0 ERROR, 1 WARN, 2 INFO, 3 DEBUG
api.set_log_buffer(uint64(256 * 1024));
api.clear_log_buffer();

result = api.propagate(pulse, linearOperator, nonlinearOperator, options);

api.tail_logs();                       % print and consume
text = api.read_log_buffer(false);     % read without consuming
api.set_log_file('run.log', true);     % also mirror to a file, append
```

Setting `exec_options.matlab_stream_logs = true` does the buffer setup and the
drain for you, and attaches the captured log text to the error message when a
propagation fails — the fastest way to diagnose a failing run.

### Progress

```matlab
api.set_progress_options(true, 10, false);   % enabled, every 10%, not on step adjust
api.set_progress_stream(2);                  % 0 stderr, 1 stdout, 2 both
```

When MATLAB runs without a desktop (`-batch`, `-nodesktop`), `propagate`
selects stream mode *both* automatically so progress survives redirection; set
`exec_options.matlab_progress_stream` to override.

### Perf counters

```matlab
api.perf_profile_set_enabled(true);
api.perf_profile_reset();

result   = api.propagate(pulse, linearOperator, nonlinearOperator, options);
snapshot = api.perf_profile_read();

fprintf('dispersion %.1f ms over %d calls, nonlinear %.1f ms over %d calls\n', ...
        snapshot.dispersion_ms, snapshot.dispersion_calls, ...
        snapshot.nonlinear_ms,  snapshot.nonlinear_calls);
```

The snapshot also reports GPU dispatch counts, host/device copy counts and
bytes, and upload/download totals — useful when comparing FFT backends or
tuning `device_heap_fraction`.

Every logging and perf method raises `nlolib:logUnavailable` or
`nlolib:perfProfileUnavailable` when the loaded library was built without the
corresponding entry point, so guard them if you ship against multiple builds.

<a name="limits"></a>

## Sizing a Run with query_runtime_limits

```matlab
limits = api.query_runtime_limits(cfg, execOptions);

fprintf('max time samples          : %d\n', limits.max_num_time_samples_runtime);
fprintf('max in-memory records     : %d\n', limits.max_num_recorded_samples_in_memory);
fprintf('max records with storage  : %d\n', limits.max_num_recorded_samples_with_storage);
fprintf('working set / budget      : %.2f / %.2f GiB\n', ...
        limits.estimated_required_working_set_bytes / 2^30, ...
        limits.estimated_device_budget_bytes / 2^30);
```

Both arguments are optional; called bare it reports the device budget for a
trivial config. `storage_available` mirrors `api.storage_is_available()`.

Query before a large tensor run and clamp `records` to
`max_num_recorded_samples_in_memory`, or enable storage.

<a name="storage"></a>

## SQLite Storage

Storage streams records to a database instead of holding them all in memory. It
is only present when the library was built with SQLite support — check
`api.storage_is_available()` first, or you get `nlolib:storageUnavailable`.

```matlab
storage = struct();
storage.sqlite_path = 'run.sqlite';    % required
storage.run_id      = 'sweep-01';
storage.chunk_records = uint64(64);
storage.return_records = false;        % stream only, keep memory flat

options.storage = storage;             % high-level
result = api.propagate(pulse, linearOperator, nonlinearOperator, options);

% low-level: pass as the fifth argument
result = api.propagate(cfg, field0, 4096, execOptions, storage);

disp(result.meta.storage_result);
```

| Field | Meaning |
|---|---|
| `sqlite_path` | database path, required and non-empty |
| `run_id` | run label; generated when omitted |
| `sqlite_max_bytes` | database size cap, `0` for unlimited |
| `chunk_records` | records per write batch |
| `cap_policy` | `0` stop writes at the cap, `1` fail the run |
| `log_final_output_field_to_db` | also persist the final field |
| `return_records` | `false` skips the in-memory record buffer entirely |

`result.meta.storage_result` reports `run_id`, `records_captured`,
`records_spilled`, `chunks_written`, `db_size_bytes`, and `truncated`. With
`return_records = false`, `result.records` is empty by design — read the data
back from the database.

<a name="troubleshooting"></a>

## Troubleshooting

| Identifier | Cause | Fix |
|---|---|---|
| `nlolib:libraryNotFound` | no `nlolib.dll` / `libnlolib.so` on any search path | set `NLOLIB_LIBRARY`, or pass the path to the constructor |
| `nlolib:headerNotFound` | `nlolib_matlab.h` missing next to the library | rebuild the `matlab_stage` target, or reinstall the toolbox |
| `nlolib:libraryLoadFailed` | every candidate failed to load | check the per-candidate reasons in the message; usually a header-parse failure (check `mex -setup C`) or an architecture mismatch |
| `nlolib:loadlibraryWarnings` | header parsed with warnings | non-fatal on its own, but unresolved types fail later — treat it as a real signal |
| `nlolib:simConfigTypeUnavailable` | stale or duplicate parsed type table | `nlolib.NLolib.unload(); clear classes;` then confirm only one `nlolib` package is on the path |
| `nlolib:inputFieldLengthMismatch` | `cfg.num_time_samples ~= numel(inputField)` | for tensor runs set it to `nt*nx*ny` |
| `nlolib:invalidPulseSpec` | bad shape or missing `samples`/`delta_time` | `numel(samples)` must equal `tensor_nt*tensor_nx*tensor_ny` |
| `nlolib:invalidOperatorSpec` | both `expr` and `fn`, `params` with `fn`, or an unknown preset | pick one form; a non-preset string is used verbatim as an expression |
| `nlolib:tooManyRuntimeConstants` | more than 16 constants | fold constants together in the expression |
| `nlolib:propagateFailed` | the C call returned non-OK | rerun with `exec_options.matlab_stream_logs = true` and `set_log_level(3)` |
| `nlolib:legacyApiRemoved` | `time_nt`/`spatial_nx`/`spatial_ny`/`transverse_*` | migrate to `tensor_nt`/`tensor_nx`/`tensor_ny` and encode diffraction in the linear operator |
| `nlolib:storageUnavailable` | library built without SQLite | check `api.storage_is_available()` |
| `nlolib:logUnavailable`, `nlolib:perfProfileUnavailable` | entry point absent in this build | guard optional calls |

`unknown identifier '<name>' in runtime handle` means a handle referenced a
variable that is neither a reserved symbol nor a real scalar in its closure —
bind it through `runtime.constant_bindings`, or switch to `expr` with `params`.

When a run misbehaves numerically rather than failing, the usual sequence is:
tighten `preset` to `"accuracy"`, enable `capture_step_history` and look for the
step size pinned at `min_step_size`, then check power/energy drift across
`result.records`.

<a name="examples"></a>

## Example Scripts

The self-contained scripts run against an installed toolbox with no repo
helpers; the rest use `setup_matlab_example_environment` and the `+backend`
plotting helpers.

| Script | Shows |
|---|---|
| [`standalone_transverse_grin_beam.m`](../examples/matlab/standalone_transverse_grin_beam.m) | transverse x-y run, potential grid, analytical centroid check |
| [`standalone_spatiotemporal_kerr_bullet.m`](../examples/matlab/standalone_spatiotemporal_kerr_bullet.m) | full 3+1D Kerr propagation against Gaussian beam/pulse laws |
| [`runtime_temporal_demo.m`](../examples/matlab/runtime_temporal_demo.m) | function-handle linear operator, log buffer, progress options |
| [`second_order_soliton_rk4ip.m`](../examples/matlab/second_order_soliton_rk4ip.m) | analytical soliton validation with step-history telemetry |
| [`linear_drift_rk4ip.m`](../examples/matlab/linear_drift_rk4ip.m) | purely dispersive drift with an analytical reference |
| [`runtime_callable_operator_rk4ip.m`](../examples/matlab/runtime_callable_operator_rk4ip.m) | callable runtime operators end to end |
| [`coupled_dispersion_nonlinearity_diffraction_rk4ip.m`](../examples/matlab/coupled_dispersion_nonlinearity_diffraction_rk4ip.m) | coupled 3+1D dispersion, nonlinearity, and diffraction |
| [`grin_fiber_xy_rk4ip.m`](../examples/matlab/grin_fiber_xy_rk4ip.m) | GRIN transverse phase validation |

Run one from the repository with:

```matlab
addpath('examples/matlab');
runtime_temporal_demo
```

Plots are written under `examples/matlab/output/<example-name>/`.

## Related Reference

- @ref matlab_binding — generated symbol reference
- [MATLAB Binding Guide](../matlab/README.md) — build-tree usage, tests, design notes
- [Runtime Operator Mathematics](runtime_operators.md)
- [Build and Install](build_and_install.md)
- @ref c_api — the underlying C entry points
