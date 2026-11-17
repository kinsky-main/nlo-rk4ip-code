# Binding API vocabulary

The Python, MATLAB and Julia bindings share one public vocabulary and one
structure, each written in its own language's idiom. This page is the
reference for that surface. The same content as data is
[`tests/api_surface.json`](../tests/api_surface.json), which
`tools/check_api_parity.py` compares with the three binding sources.

Status: this is the approved target. The bindings are being moved to it; until
they are, the checker reports the differences without failing
(`--surface report`, the default). `--surface enforce` makes them fail.

```bash
python tools/check_api_parity.py                     # constants enforced, surface reported
python tools/check_api_parity.py --surface-summary   # difference counts only
python tools/check_api_parity.py --surface enforce   # fail on any surface difference
```

## 1. Shape

- One entry point: `propagate(pulse, linear_operator, nonlinear_operator, <options>)`.
- No client class. Everything is a module or package function; `load(path)`
  selects a specific shared library.
- One result type, `PropagationResult`.
- Operators are plain functions. Free names in them are captured from the
  enclosing scope; `constant_bindings` names values explicitly.
- Option names are the C API's names.
- Every enum-style value is a lowercase string. There are no public integer
  constants.
- The raw C structs, the config builders and the pack/unpack helpers are
  private.

The public surface is 28 names, the same in each binding: 17 functions, 9 data
types and 2 error types. MATLAB has the 17 functions plus `unload`, uses plain
structs for the data types and error identifiers for the errors.

## 2. Functions

| Function | Parameters (positional first, then options by name) | Returns |
|---|---|---|
| `load` | `path` (optional) | |
| `loaded_library_path` | | path |
| `propagate` | `pulse`, `linear_operator`, `nonlinear_operator`; options below | `PropagationResult` |
| `query_runtime_limits` | options `pulse`, `exec_options` | `RuntimeLimits` |
| `storage_is_available` | | boolean |
| `tensor_record_view` | `record`, `nt`, `nx`, `ny` | one record reshaped to (t, y, x) |
| `set_log_file` | `path`; option `append=false` | |
| `set_log_buffer` | `capacity_bytes=262144` | |
| `clear_log_buffer` | | |
| `read_log_buffer` | options `consume=true`, `max_bytes=262144` | string |
| `set_log_level` | `level="info"` | |
| `set_progress_options` | options `enabled=true`, `milestone_percent=5`, `emit_on_step_adjust=false` | |
| `set_progress_stream` | `stream_mode="stderr"` | |
| `perf_profile_set_enabled` | `enabled=true` | |
| `perf_profile_is_enabled` | | boolean |
| `perf_profile_reset` | | |
| `perf_profile_read` | | `PerfProfileSnapshot` |

An option is a Python keyword argument, a Julia keyword argument (after `;`)
and a MATLAB name=value argument.

### `propagate` options, in order

| Option | Default | Note |
|---|---|---|
| `propagation_distance` | required | |
| `preset` | `"balanced"` | `"fast"`, `"balanced"`, `"accuracy"` |
| `starting_step_size`, `max_step_size`, `min_step_size`, `error_tolerance` | from the preset | |
| `num_records` | from the preset | |
| `record_z` | unset | explicit record positions |
| `return_records` | `true` | |
| `potential_operator` | unset | tensor runs only |
| `linear_step_operator` | unset | the per-step linear operator, C `linear_expr` / `dispersion_expr` |
| `constant_bindings` | unset | name to value |
| `nonlinear_model` | `"expr"` | `"expr"`, `"kerr_raman"` |
| `nonlinear_gamma`, `raman_fraction`, `shock_omega0` | `0.0` | |
| `raman_tau1`, `raman_tau2` | `0.0122`, `0.0320` | |
| `raman_response_time` | unset | |
| `exec_options` | unset | `ExecutionOptions` |
| `storage` | unset | `StorageOptions` |
| `capture_step_history` | `false` | |
| `step_history_capacity` | `200000` | |
| `progress_callback` | unset | not in MATLAB yet (known gap) |

MATLAB also has `live_progress`, `progress_style` and `stream_logs`, which
drive its Command Window output.

Presets (step sizes are `propagation_distance` divided by the number shown):

| Preset | starting | max | min | `error_tolerance` | `num_records` |
|---|---|---|---|---|---|
| `fast` | 120 | 12 | 4000 | 5e-6 | 64 |
| `balanced` | 200 | 25 | 20000 | 1e-6 | 128 |
| `accuracy` | 400 | 50 | 80000 | 1e-7 | 192 |

### Which records come back

One rule, in every binding; there is no `output` option.

1. `record_z` given: dense output at exactly those positions, for any length
   including one. `num_records` must be unset or equal to its length.
2. `record_z` unset and `num_records == 1`: the final field only.
3. Otherwise: `num_records` uniformly spaced positions from 0 to
   `propagation_distance`.

Known core discrepancy, not resolved: for a `record_z` with exactly one
position, rule 1 says the field at that position comes back. Core `9417e92`
does not do that. `src/nlolib.c` (the `num_recorded_samples == 1u` branches)
downloads the current field, which is the final field at
`propagation_distance`, into the only record whenever exactly one record is
requested, in either output mode. So today a single position `p` returns the
field at `propagation_distance` unless `p` equals it. Either the rule or the
core has to change; until then a binding must report in `z_axis` what was
actually returned.

## 3. Types

| Type | Fields, in order |
|---|---|
| `PulseSpec` | `samples`, `delta_time`, `pulse_period`, `frequency_grid`, `tensor_nt`, `tensor_nx`, `tensor_ny`, `delta_x`, `delta_y`, `spatial_frequency_grid`, `potential_grid`, `wt_axis`, `kx_axis`, `ky_axis` |
| `ExecutionOptions` | `backend_type="auto"`, `fft_backend="auto"`, `device_heap_fraction=0.70`, `record_ring_target=0`, `forced_device_budget_bytes=0` |
| `StorageOptions` | `sqlite_path`, `run_id`, `sqlite_max_bytes=0`, `chunk_records=0`, `cap_policy="stop_writes"`, `log_final_output_field_to_db=false` |
| `PropagationResult` | `records`, `z_axis`, `final`, `step_history`, `storage_result`, `meta` |
| `meta` | `preset`, `records_requested`, `records_written`, `backend_requested`, `coupled`, `status` |
| `StepHistory` | `step_index`, `z`, `step_size`, `next_step_size`, `error`, `dropped`, `capacity` |
| `StorageResult` | `run_id`, `records_captured`, `records_spilled`, `chunks_written`, `db_size_bytes`, `truncated` |
| `ProgressInfo` | `event_type`, `step_index`, `reject_attempt`, `z`, `z_end`, `percent`, `step_size`, `next_step_size`, `error`, `elapsed_seconds`, `eta_seconds` |
| `RuntimeLimits` | the six fields of the C `runtime_limits` |
| `PerfProfileSnapshot` | the eighteen fields of the C `nlo_perf_profile_snapshot` |

`records` keeps each language's layout: a list of rows in Python, records x
samples in MATLAB, samples x records in Julia. `step_history` is unset unless
`capture_step_history`; `storage_result` is unset unless `storage` was given.

The checker compares field names, not value types, so these are stated here:

- Flags are booleans in every binding, not the C `int`:
  `RuntimeLimits.storage_available`, `StorageResult.truncated`, `meta.coupled`.
  `PerfProfileSnapshot` has no flags; its fields are milliseconds and counts.
- `pulse` must be the `PulseSpec` type, which in MATLAB is a struct with the
  `PulseSpec` field names. A dict, mapping or NamedTuple is the "bad pulse"
  error. The same holds for `exec_options` and `storage`.

## 4. Enum values are strings

The public value is the lowercase string. The binding maps it to the C integer
privately, and an unknown string is the "bad option value" error.

| Where | Strings (C integer) |
|---|---|
| `ExecutionOptions.backend_type`, `meta.backend_requested` | `"cpu"` (0), `"vulkan"` (1), `"auto"` (2) |
| `ExecutionOptions.fft_backend` | `"auto"` (0), `"fftw"` (1), `"vkfft"` (2) |
| `propagate` `nonlinear_model` | `"expr"` (0), `"kerr_raman"` (1) |
| `StorageOptions.cap_policy` | `"stop_writes"` (0), `"fail"` (1) |
| `set_log_level` `level` | `"error"` (0), `"warn"` (1), `"info"` (2), `"debug"` (3) |
| `set_progress_stream` `stream_mode` | `"stderr"` (0), `"stdout"` (1), `"both"` (2) |
| `ProgressInfo.event_type` (returned) | `"accepted"` (0), `"rejected"` (1), `"finish"` (2) |
| `meta.status`, error `status` (returned) | `"ok"` (0), `"invalid_argument"` (1), `"allocation_failed"` (2), `"not_implemented"` (3), `"aborted"` (4) |

The integer constants (`VECTOR_BACKEND_*`, `FFT_BACKEND_*`,
`NONLINEAR_MODEL_*`, `STORAGE_DB_CAP_POLICY_*`, `NLOLIB_LOG_LEVEL_*`,
`NLOLIB_PROGRESS_STREAM_*`, `PROGRESS_EVENT_*`, `NLOLIB_STATUS_*`,
`PROPAGATE_OUTPUT_*`, `TENSOR_LAYOUT_XYT_T_FAST`) and the two limits
(`NT_MAX`, `STORAGE_RUN_ID_MAX`) are no longer public. `query_runtime_limits`
replaces `NT_MAX`; a `run_id` longer than 63 characters is a bad option value.

## 5. Errors

| Condition | Python | Julia | MATLAB identifier |
|---|---|---|---|
| Operator is not a function (strings included) | `TypeError` | `ArgumentError` | `nlolib:invalidOperator` |
| Operator cannot be translated, or uses a symbol its context does not allow | `ValueError` | `ArgumentError` | `nlolib:invalidOperator` |
| Bad pulse | `ValueError` | `ArgumentError` | `nlolib:invalidPulse` |
| Bad option value, including an unknown enum string | `ValueError` | `ArgumentError` | `nlolib:invalidOption` |
| Unknown option name | `TypeError` (native) | `MethodError` (native) | MATLAB's own name=value error |
| Library not found or not loadable | `OSError` | `NLOLibError` | `nlolib:libraryLoadFailed` |
| Feature not in this build | `NlolibError` | `NLOLibError` | `nlolib:unavailable` |
| C call returned a non-OK status | `NlolibError` (`.status`) | `NLOLibError` (`.status`) | `nlolib:statusFailed` |
| Aborted by the progress callback | `PropagationAbortedError` (`.result`) | `PropagationAbortedError` (`.result`) | `nlolib:propagationAborted` |

`NlolibError` derives from `RuntimeError`. MATLAB raises these seven
identifiers and no other `nlolib:` identifier.

What `.status` holds:

| Case | `.status` |
|---|---|
| A C call returned a non-OK status | that status string, for example `"invalid_argument"` |
| Feature not in this build (no C call was made) | `"not_implemented"` |
| Library not found or not loadable (no C call was made) | unset: `nothing` in Julia; Python raises `OSError`, which has no `status` |

MATLAB errors carry no status field. The identifier says which case it is, and
`nlolib:statusFailed` names the status string in its message.

## 6. How operators are lowered

Each operator function is translated once, at setup, into the expression text
the C core compiles. The core compiles a different set of expressions for the
two run types (`src/core/init_state_builder.c`), and its parser allows a
different set of symbols in each context
(`src/physics/operator_program_compile.c`, `context_allows_symbol_*`):

- temporal run (`pulse.tensor_nt` unset): `dispersion_factor_expr`,
  `dispersion_expr`, `nonlinear_expr`. The linear and potential expressions are
  ignored.
- tensor run (`tensor_nt`, `tensor_nx`, `tensor_ny` set): `linear_factor_expr`,
  `linear_expr`, `potential_expr`, `nonlinear_expr`. The dispersion expressions
  are used only as a fallback when the linear ones are unset.

The rule, the same in all three bindings:

1. The binding picks the context by run type and writes the text to the one C
   field the core compiles for that run type. It leaves the other field unset.
2. `potential_operator` on a temporal run, or together with
   `pulse.potential_grid`, is a bad option value: the core would ignore it.
3. Parameters bind to runtime symbols as described under "How parameters
   bind" below.

| Operator | Temporal run | Tensor run |
|---|---|---|
| `linear_operator` | `dispersion_factor_expr`, context `dispersion_factor` | `linear_factor_expr`, context `linear_factor` |
| `linear_step_operator` | `dispersion_expr`, context `dispersion` | `linear_expr`, context `linear` |
| `potential_operator` | not allowed | `potential_expr`, context `potential` |
| `nonlinear_operator` | `nonlinear_expr`, context `nonlinear` | the same |

| Context | Parameters, in binding order | Alias | What the C parser allows |
|---|---|---|---|
| `dispersion_factor` | `A`, `w` | `wt` = `w` | `A`, `w`, `wt` |
| `dispersion` | `A`, `D`, `h`, `w`, `V` | `wt` = `w` | `A`, `D`, `h`, `V`, `w`, `wt` |
| `linear_factor` | `A`, `wt`, `kx`, `ky`, `x`, `y`, `t` | `w` = `wt` | the same, and `w` (see below) |
| `linear` | `A`, `D`, `h`, `wt`, `kx`, `ky`, `x`, `y`, `t`, `V` | `w` = `wt` | the same, and `w` (see below) |
| `potential` | `x`, `y`, `t` | | `A`, `x`, `y`, `t` |
| `nonlinear` | `A`, `I`, `V` | | `A`, `I`, `V`, `x`, `y`, `t` |

Functions: `exp`, `log`, `sqrt`, `sin`, `cos`. Imaginary unit: `i`. Named
constants: `c0` to `c15`.

### How parameters bind

The runtime symbols are `A`, `I`, `D`, `V`, `h`, `w`, `wt`, `kx`, `ky`, `x`,
`y`, `t`. A context's symbols are its parameters in the table above plus its
alias. "The error" below is "operator cannot be translated", raised by the
binding before the C call.

- (a) First, every parameter whose name is one of the context's symbols binds
  to that symbol, wherever it stands in the parameter list.
- (b) Then the remaining parameters bind, left to right, to the next symbol in
  the context's order that is not bound yet. More parameters than symbols is
  the error.
- (c) A parameter named after a runtime symbol the context does not provide is
  the error. It is never rebound positionally.
- (d) Two parameters that resolve to the same symbol are the error.
- (e) `w` and `wt` are one symbol in the four linear contexts. The binding
  emits the context's own spelling: `w` for a temporal run, `wt` for a tensor
  run.
- (f) Runtime symbols enter only through parameters. A free name in the body
  is a captured constant, whatever it is called.

| Operator | Context | Result |
|---|---|---|
| `@(B, A)` | `dispersion_factor` | `A` binds to `A` by name (a); `B` takes the next unused symbol, `w` (b) |
| `@(A, omega)` | `dispersion_factor` or `linear_factor` | `omega` is not a runtime symbol, so it takes the next unused symbol: `w` in a temporal run, `wt` in a tensor run (b) |
| `@(A, w, wt, kx, ky)` | `linear_factor` | error (d): `w` and `wt` are the same symbol |
| `@(A, w, kx, ky)` | `linear_factor` | `A`, `wt`, `kx`, `ky`; `w` is rewritten to `wt` (e) |
| `@(A, kx)` | `dispersion_factor` | error (c): a temporal run does not provide `kx` |

On `w` in the tensor contexts: `context_allows_symbol_w` is true only for the
two dispersion contexts. The parser's branch for the identifier `w` then tests
`context_allows_symbol_wt` and emits the `wt` opcode, so the C core does
compile `w` as `wt` in `linear_factor` and `linear`. The bindings rewrite `w`
to `wt` themselves and do not depend on that.

What changes from the bindings as they stood: Python lowered the linear
operator in the `linear_factor` context for every run and wrote the text to
both C fields; MATLAB lowered it in `dispersion_factor` for every run; Julia
already chose by run type. All three listed `kx`, `ky`, `x`, `y`, `t` for the
dispersion contexts and `wt`, `kx`, `ky` for `potential`, which the C parser
rejects, so those mistakes surfaced as a generic failure from the core. The C
parser also accepts `x`, `y`, `t` in `nonlinear` (they have grids only in a
tensor run); the bindings do not expose that.

## 7. The same call in each language

```python
result = nlolib.propagate(
    nlolib.PulseSpec(samples=field0, delta_time=dt),
    lambda A, w: 1j * 0.5 * beta2 * w * w,
    lambda A, I: 1j * gamma * A * I,
    propagation_distance=1.0, num_records=128,
    exec_options=nlolib.ExecutionOptions(backend_type="auto"),
    storage=nlolib.StorageOptions(sqlite_path="run.sqlite"),
)
```

```julia
result = propagate(
    PulseSpec(samples = field0, delta_time = dt),
    (A, w) -> im * 0.5 * beta2 * w^2,
    (A, I) -> im * gamma * A * I;
    propagation_distance = 1.0, num_records = 128,
    exec_options = ExecutionOptions(backend_type = "auto"),
    storage = StorageOptions(sqlite_path = "run.sqlite"),
)
```

```matlab
result = nlolib.propagate( ...
    struct('samples', field0, 'delta_time', dt), ...
    @(A, w) 1i * 0.5 * beta2 * w.^2, ...
    @(A, I) 1i * gamma * A .* I, ...
    propagation_distance=1.0, num_records=128, ...
    exec_options=struct('backend_type', "auto"), ...
    storage=struct('sqlite_path', "run.sqlite"));
```

## 8. Old to new

### All three bindings

| Old | New |
|---|---|
| client class `NLolib` and its methods | module / package functions; `load(path)` |
| `records=`, `num_recorded_samples=` | `num_records` |
| `t_eval`, `explicit_record_z` | `record_z` |
| `output`, `output_mode` | removed; see "Which records come back" |
| integer constants | lowercase strings (section 4) |
| `prepare_sim_config`, `PreparedSimConfig`, `RuntimeOperators`, the low-level `propagate` forms | private; `propagate` gains `potential_operator`, `linear_step_operator`; `PulseSpec` gains `wt_axis`, `kx_axis`, `ky_axis` |
| `linear_factor_fn` | `linear_operator` |
| `nonlinear_fn` | `nonlinear_operator` |
| `potential_fn` | `potential_operator` |
| `linear_fn` | `linear_step_operator` |
| `dispersion_factor_fn`, `dispersion_fn` | removed; the binding picks the C context |
| `auto_capture_constants`, raw `constants` | removed; capture is always on, a name in `constant_bindings` wins |
| `tensor_layout` | removed; the C API has one layout |
| `meta` keys `output`, `records`, `message`, `storage_enabled`, `records_returned` | removed |
| `meta.step_history`, `meta.storage_result` | `result.step_history`, `result.storage_result` |
| `query_runtime_limits(config, ...)` | `query_runtime_limits(pulse, exec_options)` |
| a dict, mapping or NamedTuple as `pulse` | `PulseSpec` (a struct in MATLAB) |

### Python

| Old | New |
|---|---|
| `NLolib.set_log_file` | `set_log_file` |
| `t_span`, `first_step`, `max_step`, `min_step`, `rtol`, `dense_output`, `events` | removed |
| `sqlite_path`, `run_id`, `sqlite_max_bytes`, `chunk_records`, `cap_policy`, `log_final_output_field_to_db` keywords | `storage=StorageOptions(...)` |
| `PulseSpec` fields repeated as `propagate` keywords | set them on the pulse |
| `OperatorSpec` | removed; pass the function, use `constant_bindings` |
| `PropagateResult` (`status`, `message`, `t_events`, `y_events`, `sol`) | `PropagationResult`; status in `meta.status` |
| `NloExecutionOptions`, `default_execution_options` | `ExecutionOptions` |
| `NloStorageOptions`, `default_storage_options` | `StorageOptions` |
| `NloRuntimeLimits`, `NloPerfProfileSnapshot`, `NloStorageResult` | `RuntimeLimits`, `PerfProfileSnapshot`, `StorageResult` |
| `NloComplex`, `NloTensor3dDesc`, `NloVkBackendConfig`, `make_complex_array`, `complex_array_to_list` | private |
| `NLO_VECTOR_BACKEND_*`, `NLO_FFT_BACKEND_*` | removed |
| `RuntimeError` on a status failure | `NlolibError` |
| (new) | `loaded_library_path`, `tensor_record_view`, `StepHistory` |

### MATLAB

| Old | New |
|---|---|
| `nlolib.NLolib`, `api.propagate(...)`, `api.<method>(...)` | `nlolib.propagate(...)`, `nlolib.<function>(...)`, `nlolib.load`, `nlolib.unload`, `nlolib.loaded_library_path` |
| `propagate(pulse, lin, nonlin, options)` | `propagate(pulse, lin, nonlin, name=value...)` |
| `nlolib.constants` | removed |
| `prepare_sim_config`, `pack_complex_array`, `unpack_records`, `debug_probe_complex_ptr` | private |
| `tensor_record_view(records, index, nt, nx, ny)` | `tensor_record_view(record, nt, nx, ny)` |
| `exec_options.capture_step_history`, `.step_history_capacity` | top-level options |
| `storage.return_records` | top-level `return_records` |
| `exec_options.matlab_live_progress`, `matlab_progress_style`, `matlab_stream_logs` | options `live_progress`, `progress_style`, `stream_logs` |
| `matlab_debug`, `matlab_log_buffer_bytes`, `matlab_progress_stream`, `options.execOptions`, `tail_logs` | removed |
| `meta.step_history_dropped` | `step_history.dropped` |
| camelCase argument names (`capacityBytes`, `streamMode`, ...) | the shared snake_case names |
| `nlolib:stringOperatorRemoved`, `invalidOperatorSpec` | `nlolib:invalidOperator` |
| `nlolib:invalidPulseSpec`, `invalidTensorShape`, `invalidPotentialGridShape`, `invalidSpatialFrequencyGridShape`, `invalidInputField`, `inputFieldLengthMismatch` | `nlolib:invalidPulse` |
| `nlolib:invalidPropagateCall`, `invalidSimulateOptions`, `invalidSimulatePreset`, `invalidStorageOptions`, `invalidRecordZ`, `invalidStepHistoryCapacity`, `invalidLogBufferRead`, `invalidTensorRecordView`, `tooManyRuntimeConstants` | `nlolib:invalidOption` |
| `nlolib:propagateFailed`, `runtimeLimitsFailed`, `perfProfileFailed`, `logFileFailed`, `logBufferFailed`, `logBufferReadFailed`, `logLevelFailed`, `progressOptionsFailed`, `progressStreamFailed` | `nlolib:statusFailed`; an abort is `nlolib:propagationAborted` |
| `nlolib:logUnavailable`, `perfProfileUnavailable`, `runtimeLimitsUnavailable`, `storageUnavailable` | `nlolib:unavailable` |
| `nlolib:libraryNotFound`, `loadlibraryNotFound`, `loadlibraryWarnings`, `headerNotFound`, `simConfigTypeUnavailable`, `notLoaded` | `nlolib:libraryLoadFailed` |
| `nlolib:legacyApiRemoved` and its checks | removed |
| (later) | `progress_callback` |

### Julia

| Old | New |
|---|---|
| `NLolib` struct and every `f(::NLolib, ...)` method | removed |
| `propagate(pulse, [lin], [nonlin]; kwargs...)` with default operators | both operators required; every option a named keyword, so a misspelled option is an error |
| `propagate(config, input; ...)`, `propagate(sim, phys, input)`, `propagate!` | private |
| `t_span`, `first_step`, `max_step`, `min_step`, `rtol`, `dense_output`, `events` | removed |
| six storage keywords | `storage = StorageOptions(...)`, a keyword constructor; the C-layout struct becomes private |
| `default_execution_options(...)` | `ExecutionOptions(; ...)` |
| `storage_options`, `default_storage_options`, `physics_config`, `PreparedValue`, `wrap_records`, `final_record` | private |
| `OperatorSpec` | removed |
| `PropagateResult` | `PropagationResult` |
| `NLOComplex`, `PropagationParams`, `TimeGrid`, `FrequencyGrid`, `SpatialGrid`, `Tensor3DDesc`, `RuntimeOperatorParams`, `SimulationConfig`, `PhysicsConfig`, `VulkanBackendConfig`, `StepEvent`, `CProgressInfo` | private |
| `tensor_record_view(records, index, nt, nx, ny)` | `tensor_record_view(record, nt, nx, ny)` only |
| `set_log_file(path, append)` | `set_log_file(path; append = false)` |
| `set_progress_stream(; stream_mode)` | `set_progress_stream(stream_mode)` |
| `error(...)` on a status failure | `throw(NLOLibError(status, context))` |
| (new) | `perf_profile_set_enabled`, `perf_profile_is_enabled`, `perf_profile_reset`, `perf_profile_read`, `PerfProfileSnapshot`, `StepHistory` |

## 9. Examples

Each shared example is `examples/<name>.py`, `.jl` and `.m`, with one problem
definition. The numbers live in a parameter block the checker reads: plain
`name = number` lines between a comment containing
`nlolib-example-parameters: begin` and one containing
`nlolib-example-parameters: end`. The values are in `tests/api_surface.json`
under `examples.shared`. All examples use backend `"auto"` and print the same
metric labels.

| Example | Grid and records | Solver | Basis |
|---|---|---|---|
| `quickstart` | n=1024, 2 records | `balanced` | MATLAB `runtime_callable_operator_rk4ip` |
| `linear_drift` | n=1024, 180 records | `accuracy` | Python |
| `spm` | n=1024, 100 records | `accuracy` | Python |
| `second_order_soliton` | n=4096, 160 records | explicit, tolerance 1e-10 | Python |
| `conservation_checks` | n=1024, 100 records | explicit, tolerance 1e-6 | Python |
| `raman_scattering` | n=4096, dt=0.002, 80 records | explicit, tolerance 2e-6 | Julia grid, Python physics |
| `tensor_dispersion_3d` | 128 x 64 x 64, 24 records (201 MB) | explicit, tolerance 1e-7 | Python physics |
| `transverse_grin_beam` | 128 x 128, 96 records (25 MB) | `accuracy` | MATLAB `standalone_transverse_grin_beam` |
| `spatiotemporal_kerr_bullet` | 32 x 64 x 64, 48 records (101 MB) | `accuracy` | MATLAB `standalone_spatiotemporal_kerr_bullet` |
| `grin_soliton_fission` | 1024 x 24 x 24, 48 records (453 MB) | `balanced` with overrides | MATLAB `grin_fiber_soliton_fission` |
| `high_order_grin_soliton` | 512 x 64 x 64, 24 records (805 MB) | explicit, tolerance 1e-5 | Python physics |
| `sqlite_storage` | n=128, 24 records to SQLite | `balanced` | Python `sqlite_snapshot_chunking_demo` |

The reduced grids of `tensor_dispersion_3d` and `high_order_grin_soliton` and
the `raman_scattering` grid are sized by arithmetic and have not been run. The
Python implementation runs each once and may correct the numbers in the
manifest before the other two bindings copy them.

Kept on demand in one binding, not run by default: `tensor_backend_scaling*`,
`fixed_step_soliton_convergence`, `two_mode_linear_beating` (Python);
`mmtools_*` (MATLAB). Dropped: `spm_dispersion_comparison`,
`coupled_dispersion_nonlinearity_diffraction`, `grin_fiber_xy`,
`unsaturable_grin_filamentation_fission`, `runtime_temporal_demo`. The
`_rk4ip` suffix is gone.

## 10. Known gaps

- MATLAB has no `progress_callback`: `progress_mex` cannot forward to a
  function handle yet. `ProgressInfo` and the `"accepted"`, `"rejected"`,
  `"finish"` event strings reach a caller only through that callback, so they
  are part of the same gap and are not counted as MATLAB differences.

## 11. Order of work

1. Each binding moves to this surface on its own branch: API and tests, then
   its own docs and README.
2. Examples follow, Python first, since it fixes the grids and records the
   reference metric values.
3. The superproject then moves the three submodule pins, switches the checker
   to `--surface enforce`, removes `tests/api_parity.json` and the constants
   check, and rewrites `docs/doxygen/`.

Every binding breaks its public API, so each takes a major version.
