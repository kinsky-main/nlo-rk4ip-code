/**
 * @file matlab_api.h
 * @brief Docs-only MATLAB binding API shim for Doxygen.
 *
 * This header exists purely so Doxygen can render the MATLAB package surface;
 * it is never compiled or installed. Each entity below mirrors a function of
 * the `nlolib` package, whose implementation lives in the `nlolib-matlab`
 * repository (`matlab/+nlolib/`).
 *
 * The surface is the one all three bindings share, described in
 * `docs/binding_api_vocabulary.md` and held as data in
 * `tests/api_surface.json`: package functions, no client class, the same
 * fields, and lowercase strings for every enum-style value.
 *
 * MATLAB is untyped, so the types here describe intent rather than a
 * signature: `StructValue` stands for a MATLAB struct, `Handle` for a
 * `function_handle`, and parameters marked "name=value" are name=value
 * arguments. The data types below are plain structs with the listed fields.
 *
 * The narrative documentation is the MATLAB User Guide
 * (`matlab/docs/user_guide.md`), listed under the Guides tab.
 *
 * @code{.m}
 * result = nlolib.propagate( ...
 *     struct('samples', field0, 'delta_time', dt), ...
 *     @(A, w) 1i * (0.5 * beta2) * w .* w, ...
 *     @(A, I) 1i * gamma * A .* I, ...
 *     propagation_distance=1.0, ...
 *     num_records=2, ...
 *     preset="balanced", ...
 *     exec_options=struct('backend_type', "auto"));
 * finalPower = sum(abs(result.final) .^ 2);
 * @endcode
 *
 * @par Enum strings
 * - `exec_options.backend_type`: `"cpu"`, `"vulkan"`, `"auto"`
 * - `exec_options.fft_backend`: `"auto"`, `"fftw"`, `"vkfft"`
 * - `nonlinear_model`: `"expr"`, `"kerr_raman"`
 * - `storage.cap_policy`: `"stop_writes"`, `"fail"`
 * - `set_log_level`: `"error"`, `"warn"`, `"info"`, `"debug"`
 * - `set_progress_stream`: `"stderr"`, `"stdout"`, `"both"`
 * - status: `"ok"`, `"invalid_argument"`, `"allocation_failed"`,
 *   `"not_implemented"`, `"aborted"`
 *
 * An unknown string is `nlolib:invalidOption`.
 *
 * @par Error identifiers
 * The package raises these seven identifiers and no other `nlolib:` one.
 * - `nlolib:invalidOperator` — an operator is not a `function_handle`
 *   (strings included), cannot be translated, or uses a symbol its context
 *   does not allow.
 * - `nlolib:invalidPulse` — bad pulse struct.
 * - `nlolib:invalidOption` — bad option value.
 * - `nlolib:libraryLoadFailed` — the shared library is not found or cannot be
 *   loaded.
 * - `nlolib:unavailable` — the feature is not in this build.
 * - `nlolib:statusFailed` — a C call returned a non-OK status; the message
 *   names the status string.
 * - `nlolib:propagationAborted` — the run was aborted.
 *
 * An unknown option name is MATLAB's own name=value error.
 *
 * @par Known gap
 * There is no `progress_callback` yet, and so no `ProgressInfo`. Live
 * progress in the Command Window is the `live_progress` option.
 */

#pragma once

namespace matlab {
namespace nlolib {

/** @brief Placeholder for an arbitrary MATLAB struct value. */
struct StructValue {};

/** @brief Placeholder for a MATLAB `function_handle`. */
struct Handle {};

/**
 * @ingroup matlab_binding
 * @brief Launch field and its grid: the `pulse` struct.
 *
 * MATLAB fields: `samples`, `delta_time`, `pulse_period`, `frequency_grid`,
 * `tensor_nt`, `tensor_nx`, `tensor_ny`, `delta_x` (1.0), `delta_y` (1.0),
 * `spatial_frequency_grid`, `potential_grid`, `wt_axis`, `kx_axis`, `ky_axis`.
 *
 * Only `samples` and `delta_time` are required. Setting `tensor_nt`,
 * `tensor_nx` and `tensor_ny` makes the run a tensor (t, y, x) run, with
 * `samples` holding `nt*nx*ny` values, t fastest, then y, then x.
 */
struct PulseSpec {};

/**
 * @ingroup matlab_binding
 * @brief Backend selection and device memory budget: the `exec_options`
 *        struct.
 *
 * MATLAB fields: `backend_type` (`"auto"`), `fft_backend` (`"auto"`),
 * `device_heap_fraction` (0.70), `record_ring_target` (0),
 * `forced_device_budget_bytes` (0). Any field may be left out.
 */
struct ExecutionOptions {};

/**
 * @ingroup matlab_binding
 * @brief SQLite snapshot storage for a run: the `storage` struct.
 *
 * MATLAB fields: `sqlite_path` (required), `run_id`, `sqlite_max_bytes` (0),
 * `chunk_records` (0), `cap_policy` (`"stop_writes"`),
 * `log_final_output_field_to_db` (true), `record_mode` (`"reduced"`, `"raw"`
 * or `"reduced_and_raw"`), `raw_stride_t`, `raw_stride_x`, `raw_stride_y` (1),
 * `defer_merge` (false).
 */
struct StorageOptions {};

/**
 * @ingroup matlab_binding
 * @brief Result of propagate().
 *
 * MATLAB fields:
 * - `records` — `numRecords x numSamples` complex matrix, one row per
 *   recorded \f$z\f$ position.
 * - `z_axis` — the recorded \f$z\f$ positions.
 * - `final` — the last record.
 * - `step_history` — a StepHistory struct, or `[]` unless
 *   `capture_step_history`.
 * - `storage_result` — a StorageResult struct, or `[]` unless `storage` was
 *   given.
 * - `meta` — `preset`, `records_requested`, `records_written`,
 *   `backend_requested`, `coupled`, `status`.
 *
 * `records_written` may be smaller than `records_requested`: the library
 * reduces the count for fixed-step runs and aborted runs.
 */
struct PropagationResult {};

/**
 * @ingroup matlab_binding
 * @brief Per-step solver telemetry.
 *
 * MATLAB fields: `step_index`, `z`, `step_size`, `next_step_size`, `error`
 * (column vectors), plus `dropped` and `capacity`. A non-zero `dropped` means
 * the history was truncated; raise `step_history_capacity`.
 */
struct StepHistory {};

/**
 * @ingroup matlab_binding
 * @brief Outcome of a storage-backed run.
 *
 * MATLAB fields: `run_id`, `records_captured`, `records_spilled`,
 * `chunks_written`, `db_size_bytes`, `truncated` (logical), `key` (the run
 * key), `merged` and `deferred` (logical).
 */
struct StorageResult {};

/**
 * @ingroup matlab_binding
 * @brief What `storage_recover_staging` returns.
 *
 * MATLAB fields: `merged`, `discarded`, `skipped`.
 */
struct StagingRecovery {};

/**
 * @ingroup matlab_binding
 * @brief Limits returned by query_runtime_limits().
 *
 * MATLAB fields: `max_num_time_samples_runtime`,
 * `max_num_recorded_samples_in_memory`,
 * `max_num_recorded_samples_with_storage`,
 * `estimated_required_working_set_bytes`, `estimated_device_budget_bytes`,
 * `storage_available` (logical).
 */
struct RuntimeLimits {};

/**
 * @ingroup matlab_binding
 * @brief Performance counters returned by perf_profile_read().
 *
 * MATLAB fields: `dispersion_ms`, `nonlinear_ms`, `dispersion_calls`,
 * `nonlinear_calls`, `gpu_dispatch_count`, `gpu_copy_count`,
 * `gpu_device_copy_count`, `gpu_device_copy_bytes`,
 * `gpu_host_transfer_copy_count`, `gpu_host_transfer_copy_bytes`,
 * `gpu_memory_pass_count`, `gpu_memory_pass_bytes`, `gpu_upload_count`,
 * `gpu_download_count`, `gpu_upload_bytes`, `gpu_download_bytes`,
 * `operator_jit_programs`, `operator_jit_fallbacks`.
 */
struct PerfProfileSnapshot {};

/**
 * @ingroup matlab_binding
 * @brief Load the nlolib shared library.
 *
 * Native MATLAB function name: \c %nlolib.load. Every other package function
 * loads the library on first use, so calling this is needed only to pick a
 * specific file.
 *
 * @param path Library file to load. When omitted, `NLOLIB_LIBRARY` and then
 *             the toolbox and build-tree locations are searched.
 * @throws nlolib:libraryLoadFailed the library is not found or cannot be loaded.
 */
void load(const char* path = "");

/**
 * @ingroup matlab_binding
 * @brief Unload the shared library.
 *
 * Native MATLAB function name: \c %nlolib.unload. MATLAB keeps the library
 * loaded until told otherwise; unload it before rebuilding it in a session.
 */
void unload();

/**
 * @ingroup matlab_binding
 * @brief Path of the loaded shared library.
 *
 * Native MATLAB function name: \c %nlolib.loaded_library_path.
 */
const char* loaded_library_path();

/**
 * @ingroup matlab_binding
 * @brief Propagate a pulse. The one entry point of the package.
 *
 * Native MATLAB call:
 * `result = nlolib.propagate(pulse, linear_operator, nonlinear_operator, propagation_distance=z, name=value...)`.
 *
 * Operators are function handles whose body is one expression. Scalars the
 * body uses from the workspace become runtime constants; a name in
 * `constant_bindings` takes that value instead. A string is
 * `nlolib:invalidOperator`.
 *
 * @par Operator contexts
 * The binding lowers each handle to the expression text the C core compiles,
 * in the context the run type selects:
 * - temporal run (`pulse.tensor_nt` unset): `linear_operator` is `@(A, w)`;
 *   `linear_step_operator` is `@(A, D, h, w, V)`.
 * - tensor run: `linear_operator` is `@(A, wt, kx, ky, x, y, t)`;
 *   `linear_step_operator` is `@(A, D, h, wt, kx, ky, x, y, t, V)`;
 *   `potential_operator` is `@(x, y, t)`.
 * - `nonlinear_operator` is `@(A, I, V)` in both.
 *
 * @par Parameter binding
 * An argument named after one of the context's symbols binds to it, wherever
 * it stands. The remaining arguments take the next unused symbol in the
 * context's order. An argument named after a symbol the context does not
 * provide, or two arguments resolving to one symbol, is
 * `nlolib:invalidOperator`. `w` and `wt` are the same symbol, so `@(A, w)`
 * works for both run types.
 *
 * @par Which records come back
 * `record_z` given: those positions. `num_records == 1` without `record_z`:
 * the final field only. Otherwise `num_records` uniformly spaced positions
 * from 0 to `propagation_distance`.
 *
 * @param pulse                 PulseSpec struct.
 * @param linear_operator       Handle giving the linear factor \f$D\f$,
 *                              applied as \f$\exp(hD)\f$.
 * @param nonlinear_operator    Handle giving the nonlinear right-hand side.
 * @param propagation_distance  name=value, required. Total distance in z.
 * @param preset                name=value. `"fast"`, `"balanced"` (default)
 *                              or `"accuracy"`: supplies the step sizes,
 *                              tolerance and record count left unset.
 * @param starting_step_size    name=value. First trial step, in z units.
 * @param max_step_size         name=value. Upper step bound.
 * @param min_step_size         name=value. Lower step bound.
 * @param error_tolerance       name=value. Adaptive-step local error target.
 * @param num_records           name=value. Number of recorded z positions.
 * @param record_z              name=value. Explicit record positions.
 * @param return_records        name=value, default true. False keeps records
 *                              out of the result, for storage-only runs.
 * @param potential_operator    name=value. Handle giving \f$V(x, y, t)\f$;
 *                              tensor runs only, and not together with
 *                              `pulse.potential_grid`.
 * @param linear_step_operator  name=value. Handle replacing the default
 *                              per-step linear operator \f$\exp(hD)\f$.
 * @param constant_bindings     name=value. Struct or `containers.Map` of name
 *                              to real scalar.
 * @param nonlinear_model       name=value. `"expr"` (default) or
 *                              `"kerr_raman"`.
 * @param nonlinear_gamma       name=value. Kerr coefficient of the built-in
 *                              model.
 * @param raman_fraction        name=value. Raman fraction \f$f_R\f$.
 * @param raman_tau1            name=value, default 0.0122.
 * @param raman_tau2            name=value, default 0.0320.
 * @param shock_omega0          name=value. Self-steepening carrier frequency;
 *                              0 disables.
 * @param raman_response_time   name=value. Sampled Raman response, replacing
 *                              the analytic one.
 * @param exec_options          name=value. ExecutionOptions struct.
 * @param storage               name=value. StorageOptions struct.
 * @param capture_step_history  name=value. True to fill `result.step_history`.
 * @param step_history_capacity name=value, default 200000.
 * @param live_progress         name=value, MATLAB only. Print progress and
 *                              logs in the Command Window while running;
 *                              Ctrl+C aborts.
 * @param progress_style        name=value, MATLAB only. `"inline"` or
 *                              `"lines"`.
 * @param stream_logs           name=value, MATLAB only. Print the log buffer
 *                              after the call.
 * @return PropagationResult struct.
 * @throws nlolib:invalidOperator    an operator is not a handle or cannot be
 *                                   translated.
 * @throws nlolib:invalidPulse       bad pulse struct.
 * @throws nlolib:invalidOption      bad option value.
 * @throws nlolib:statusFailed       the C call failed.
 * @throws nlolib:propagationAborted the run was aborted.
 */
PropagationResult propagate(const PulseSpec& pulse,
                            Handle linear_operator,
                            Handle nonlinear_operator,
                            double propagation_distance,
                            const char* preset = "balanced",
                            StructValue starting_step_size = StructValue(),
                            StructValue max_step_size = StructValue(),
                            StructValue min_step_size = StructValue(),
                            StructValue error_tolerance = StructValue(),
                            StructValue num_records = StructValue(),
                            StructValue record_z = StructValue(),
                            bool return_records = true,
                            Handle potential_operator = Handle(),
                            Handle linear_step_operator = Handle(),
                            StructValue constant_bindings = StructValue(),
                            const char* nonlinear_model = "expr",
                            double nonlinear_gamma = 0.0,
                            double raman_fraction = 0.0,
                            double raman_tau1 = 0.0122,
                            double raman_tau2 = 0.0320,
                            double shock_omega0 = 0.0,
                            StructValue raman_response_time = StructValue(),
                            const ExecutionOptions& exec_options = ExecutionOptions(),
                            const StorageOptions& storage = StorageOptions(),
                            bool capture_step_history = false,
                            int step_history_capacity = 200000,
                            bool live_progress = true,
                            const char* progress_style = "inline",
                            bool stream_logs = false);

/**
 * @ingroup matlab_binding
 * @brief Query the limits the runtime derives for a pulse and execution
 *        options.
 *
 * Native MATLAB call:
 * `limits = nlolib.query_runtime_limits(pulse=pulse, exec_options=opts)`.
 *
 * @param pulse        name=value. PulseSpec struct; omitted for a default run.
 * @param exec_options name=value. ExecutionOptions struct; omitted for the
 *                     defaults.
 * @return RuntimeLimits struct.
 */
RuntimeLimits query_runtime_limits(const PulseSpec& pulse = PulseSpec(),
                                   const ExecutionOptions& exec_options = ExecutionOptions());

/**
 * @ingroup matlab_binding
 * @brief True when this build has SQLite snapshot storage.
 *
 * Native MATLAB function name: \c %nlolib.storage_is_available.
 */
bool storage_is_available();

/**
 * @ingroup matlab_binding
 * @brief Run key under which `propagate` with the same arguments stores the run.
 *
 * Native MATLAB function name: \c %nlolib.storage_run_key. Takes
 * `propagate`'s arguments that enter the key, as name=value options.
 *
 * @return 16 lowercase hex digits (string).
 */
const char* storage_run_key(const PulseSpec& pulse, Any linear_operator, Any nonlinear_operator);

/**
 * @ingroup matlab_binding
 * @brief True when the shared run database `sqlite_path` has a run with `key`.
 *
 * Native MATLAB function name: \c %nlolib.storage_has_run.
 */
bool storage_has_run(const char* sqlite_path, const char* key);

/**
 * @ingroup matlab_binding
 * @brief Merge the staging files left for the shared run database `sqlite_path`.
 *
 * Native MATLAB function name: \c %nlolib.storage_recover_staging; options
 * `discard_incomplete` (false) and `min_age_seconds` (86400).
 *
 * @return StagingRecovery struct.
 */
StagingRecovery storage_recover_staging(const char* sqlite_path,
                                        bool discard_incomplete = false,
                                        double min_age_seconds = 86400.0);

/**
 * @ingroup matlab_binding
 * @brief Reshape one flat tensor record into an `nt x ny x nx` volume.
 *
 * Native MATLAB function name: \c %nlolib.tensor_record_view.
 *
 * @code{.m}
 * final = nlolib.tensor_record_view(result.final, nt, nx, ny);
 * imagesc(abs(squeeze(final(ceil(nt / 2), :, :))).^2);
 * @endcode
 *
 * @param record One record of a tensor run (a row of `result.records`), of
 *               `nt*nx*ny` samples.
 * @param nt     Samples along t.
 * @param nx     Samples along x.
 * @param ny     Samples along y.
 * @return A 3-D array indexed `volume(it, iy, ix)`.
 */
StructValue tensor_record_view(StructValue record, int nt, int nx, int ny);

/**
 * @ingroup matlab_binding
 * @brief Write the library log to a file.
 *
 * Native MATLAB call: `nlolib.set_log_file(path, append=false)`.
 */
void set_log_file(const char* path, bool append = false);

/**
 * @ingroup matlab_binding
 * @brief Keep the library log in an in-memory ring buffer.
 *
 * Native MATLAB function name: \c %nlolib.set_log_buffer.
 */
void set_log_buffer(int capacity_bytes = 262144);

/**
 * @ingroup matlab_binding
 * @brief Empty the log buffer.
 *
 * Native MATLAB function name: \c %nlolib.clear_log_buffer.
 */
void clear_log_buffer();

/**
 * @ingroup matlab_binding
 * @brief Read the log buffer as text.
 *
 * Native MATLAB call: `text = nlolib.read_log_buffer(consume=true, max_bytes=262144)`.
 */
const char* read_log_buffer(bool consume = true, int max_bytes = 262144);

/**
 * @ingroup matlab_binding
 * @brief Set the log level: `"error"`, `"warn"`, `"info"` or `"debug"`.
 *
 * Native MATLAB function name: \c %nlolib.set_log_level.
 */
void set_log_level(const char* level = "info");

/**
 * @ingroup matlab_binding
 * @brief Configure the library's own progress output.
 *
 * Native MATLAB call:
 * `nlolib.set_progress_options(enabled=true, milestone_percent=5, emit_on_step_adjust=false)`.
 */
void set_progress_options(bool enabled = true, int milestone_percent = 5,
                          bool emit_on_step_adjust = false);

/**
 * @ingroup matlab_binding
 * @brief Send progress output to `"stderr"`, `"stdout"` or `"both"`.
 *
 * Native MATLAB function name: \c %nlolib.set_progress_stream.
 */
void set_progress_stream(const char* stream_mode = "stderr");

/**
 * @ingroup matlab_binding
 * @brief Turn the performance counters on or off.
 *
 * Native MATLAB function name: \c %nlolib.perf_profile_set_enabled.
 */
void perf_profile_set_enabled(bool enabled = true);

/**
 * @ingroup matlab_binding
 * @brief True when the performance counters are on.
 *
 * Native MATLAB function name: \c %nlolib.perf_profile_is_enabled.
 */
bool perf_profile_is_enabled();

/**
 * @ingroup matlab_binding
 * @brief Zero the performance counters.
 *
 * Native MATLAB function name: \c %nlolib.perf_profile_reset.
 */
void perf_profile_reset();

/**
 * @ingroup matlab_binding
 * @brief Read the performance counters.
 *
 * Native MATLAB function name: \c %nlolib.perf_profile_read.
 */
PerfProfileSnapshot perf_profile_read();

} // namespace nlolib
} // namespace matlab
