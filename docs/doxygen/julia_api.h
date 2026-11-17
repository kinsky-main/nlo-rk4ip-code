/**
 * @file julia_api.h
 * @brief Docs-only Julia binding API shim for Doxygen.
 *
 * This header exists purely so Doxygen can render the exported `NLOLib` module
 * surface; it is never compiled or installed. The implementation lives in the
 * `NLOLib.jl` repository (`julia/src/`).
 *
 * The surface is the one all three bindings share, described in
 * `docs/binding_api_vocabulary.md` and held as data in
 * `tests/api_surface.json`: module functions, no client type, the same types
 * and fields, and lowercase strings for every enum-style value.
 *
 * Julia is dynamically typed, so the C++ types below describe intent rather
 * than a signature. Arguments after `propagation_distance`-style names marked
 * "keyword" are Julia keyword arguments, written after `;`.
 *
 * @code{.jl}
 * using NLOLib
 *
 * result = propagate(
 *     PulseSpec(samples = field0, delta_time = dt),
 *     (A, w) -> im * (0.5 * beta2) * w^2,
 *     (A, I) -> im * gamma * A * I;
 *     propagation_distance = 1.0,
 *     num_records = 2,
 *     preset = "balanced",
 *     exec_options = ExecutionOptions(backend_type = "auto"),
 * )
 * final_power = sum(abs2, result.final)
 * @endcode
 *
 * @par Enum strings
 * - `ExecutionOptions.backend_type`: `"cpu"`, `"vulkan"`, `"auto"`
 * - `ExecutionOptions.fft_backend`: `"auto"`, `"fftw"`, `"vkfft"`
 * - `nonlinear_model`: `"expr"`, `"kerr_raman"`
 * - `StorageOptions.cap_policy`: `"stop_writes"`, `"fail"`
 * - `set_log_level`: `"error"`, `"warn"`, `"info"`, `"debug"`
 * - `set_progress_stream`: `"stderr"`, `"stdout"`, `"both"`
 * - `ProgressInfo.event_type`: `"accepted"`, `"rejected"`, `"finish"`
 * - status: `"ok"`, `"invalid_argument"`, `"allocation_failed"`,
 *   `"not_implemented"`, `"aborted"`
 *
 * An unknown string throws `ArgumentError`.
 *
 * @par Errors
 * - operator is not a function, or cannot be translated: `ArgumentError`
 * - bad pulse or bad option value: `ArgumentError`
 * - unknown keyword: `MethodError`
 * - library not found or not loadable: NLOLibError with `status === nothing`
 * - feature not in this build: NLOLibError with `status == "not_implemented"`
 * - a C call returned a non-OK status: NLOLibError
 * - aborted by the progress callback: PropagationAbortedError
 */

#pragma once

namespace julia {
namespace NLOLib {

/** @brief Placeholder for an arbitrary Julia value. */
struct Any {};

/**
 * @ingroup julia_binding
 * @brief Launch field and its grid.
 *
 * Fields, in order: `samples`, `delta_time`, `pulse_period`, `frequency_grid`,
 * `tensor_nt`, `tensor_nx`, `tensor_ny`, `delta_x` (1.0), `delta_y` (1.0),
 * `spatial_frequency_grid`, `potential_grid`, `wt_axis`, `kx_axis`, `ky_axis`.
 *
 * Built with keywords: `PulseSpec(samples = field0, delta_time = dt)`. Setting
 * `tensor_nt`, `tensor_nx` and `tensor_ny` makes the run a tensor (t, y, x)
 * run, with `samples` flattened t-fastest. `propagate` accepts only this type
 * as its pulse; a NamedTuple is an `ArgumentError`.
 */
struct PulseSpec {};

/**
 * @ingroup julia_binding
 * @brief Backend selection and device memory budget.
 *
 * Fields: `backend_type` (`"auto"`), `fft_backend` (`"auto"`),
 * `device_heap_fraction` (0.70), `record_ring_target` (0),
 * `forced_device_budget_bytes` (0). Built with keywords.
 */
struct ExecutionOptions {};

/**
 * @ingroup julia_binding
 * @brief SQLite snapshot storage for a run.
 *
 * Fields: `sqlite_path` (required), `run_id`, `sqlite_max_bytes` (0),
 * `chunk_records` (0), `cap_policy` (`"stop_writes"`),
 * `log_final_output_field_to_db` (false). Built with keywords.
 */
struct StorageOptions {};

/**
 * @ingroup julia_binding
 * @brief Result of propagate().
 *
 * Fields: `records` (samples x records matrix, one column per recorded
 * \f$z\f$), `z_axis`, `final`, `step_history` (a StepHistory, or `nothing`
 * unless `capture_step_history`), `storage_result` (a StorageResult, or
 * `nothing` unless `storage` was given), `meta`.
 *
 * `meta` holds `preset`, `records_requested`, `records_written`,
 * `backend_requested`, `coupled` and `status`.
 */
struct PropagationResult {};

/**
 * @ingroup julia_binding
 * @brief Per-step solver telemetry.
 *
 * Fields: `step_index`, `z`, `step_size`, `next_step_size`, `error`,
 * `dropped`, `capacity`. A non-zero `dropped` means the history was
 * truncated; raise `step_history_capacity`.
 */
struct StepHistory {};

/**
 * @ingroup julia_binding
 * @brief Outcome of a storage-backed run.
 *
 * Fields: `run_id`, `records_captured`, `records_spilled`, `chunks_written`,
 * `db_size_bytes`, `truncated` (Bool).
 */
struct StorageResult {};

/**
 * @ingroup julia_binding
 * @brief Payload passed to `progress_callback`.
 *
 * Fields: `event_type` (`"accepted"`, `"rejected"` or `"finish"`),
 * `step_index`, `reject_attempt`, `z`, `z_end`, `percent`, `step_size`,
 * `next_step_size`, `error`, `elapsed_seconds`, `eta_seconds`.
 */
struct ProgressInfo {};

/**
 * @ingroup julia_binding
 * @brief Limits returned by query_runtime_limits().
 *
 * Fields: `max_num_time_samples_runtime`,
 * `max_num_recorded_samples_in_memory`,
 * `max_num_recorded_samples_with_storage`,
 * `estimated_required_working_set_bytes`, `estimated_device_budget_bytes`,
 * `storage_available` (Bool).
 */
struct RuntimeLimits {};

/**
 * @ingroup julia_binding
 * @brief Performance counters returned by perf_profile_read().
 *
 * Fields: `dispersion_ms`, `nonlinear_ms`, `dispersion_calls`,
 * `nonlinear_calls`, `gpu_dispatch_count`, `gpu_copy_count`,
 * `gpu_device_copy_count`, `gpu_device_copy_bytes`,
 * `gpu_host_transfer_copy_count`, `gpu_host_transfer_copy_bytes`,
 * `gpu_memory_pass_count`, `gpu_memory_pass_bytes`, `gpu_upload_count`,
 * `gpu_download_count`, `gpu_upload_bytes`, `gpu_download_bytes`,
 * `operator_jit_programs`, `operator_jit_fallbacks`.
 */
struct PerfProfileSnapshot {};

/**
 * @ingroup julia_binding
 * @brief A C call returned a non-OK status, a feature is not in this build,
 *        or the library cannot be loaded.
 *
 * Fields: `status` and `context`. `status` is the status string, or `nothing`
 * when the library could not be loaded and no C call was made.
 */
struct NLOLibError {};

/**
 * @ingroup julia_binding
 * @brief The progress callback aborted the run.
 *
 * `result` is the PropagationResult holding what was recorded before the
 * abort.
 */
struct PropagationAbortedError {};

/**
 * @ingroup julia_binding
 * @brief Load the nlolib shared library.
 *
 * @param path Library file to load. When omitted, the library staged beside
 *             the package is used.
 * @throws NLOLibError the library is not found or cannot be loaded.
 */
void load(const char* path = nullptr);

/**
 * @ingroup julia_binding
 * @brief Path of the loaded shared library.
 */
const char* loaded_library_path();

/**
 * @ingroup julia_binding
 * @brief Propagate a pulse. The one entry point of the module.
 *
 * Julia signature:
 * `propagate(pulse, linear_operator, nonlinear_operator; propagation_distance, kwargs...)`,
 * where every keyword is named explicitly, so a misspelled one is a
 * `MethodError`.
 *
 * Operators are plain functions. They are traced once at setup, so ordinary
 * Julia in the body works and captured numbers become constants;
 * `constant_bindings` names values explicitly. A string is an
 * `ArgumentError`.
 *
 * @par Operator contexts
 * The binding lowers each operator to the expression text the C core
 * compiles, in the context the run type selects:
 * - temporal run (`pulse.tensor_nt` unset): `linear_operator` takes `(A, w)`;
 *   `linear_step_operator` takes `(A, D, h, w, V)`.
 * - tensor run: `linear_operator` takes `(A, wt, kx, ky, x, y, t)`;
 *   `linear_step_operator` takes `(A, D, h, wt, kx, ky, x, y, t, V)`;
 *   `potential_operator` takes `(x, y, t)`.
 * - `nonlinear_operator` takes `(A, I, V)` in both.
 *
 * @par Parameter binding
 * A parameter named after one of the context's symbols binds to it, wherever
 * it stands. The remaining parameters take the next unused symbol in the
 * context's order. A parameter named after a symbol the context does not
 * provide, or two parameters resolving to one symbol, is an `ArgumentError`.
 * `w` and `wt` are the same symbol, so `(A, w) -> ...` works for both run
 * types.
 *
 * @par Which records come back
 * `record_z` given: those positions. `num_records == 1` without `record_z`:
 * the final field only. Otherwise `num_records` uniformly spaced positions
 * from 0 to `propagation_distance`.
 *
 * @param pulse                 PulseSpec.
 * @param linear_operator       Function giving the linear factor \f$D\f$,
 *                              applied as \f$\exp(hD)\f$.
 * @param nonlinear_operator    Function giving the nonlinear right-hand side.
 * @param propagation_distance  Keyword, required. Total distance in z.
 * @param preset                Keyword. `"fast"`, `"balanced"` (default) or
 *                              `"accuracy"`: supplies the step sizes,
 *                              tolerance and record count left unset.
 * @param starting_step_size    Keyword. First trial step, in z units.
 * @param max_step_size         Keyword. Upper step bound.
 * @param min_step_size         Keyword. Lower step bound.
 * @param error_tolerance       Keyword. Adaptive-step local error target.
 * @param num_records           Keyword. Number of recorded z positions.
 * @param record_z              Keyword. Explicit record positions.
 * @param return_records        Keyword, default true. False keeps records out
 *                              of the result, for storage-only runs.
 * @param potential_operator    Keyword. Function giving \f$V(x, y, t)\f$;
 *                              tensor runs only, and not together with
 *                              `pulse.potential_grid`.
 * @param linear_step_operator  Keyword. Function replacing the default
 *                              per-step linear operator \f$\exp(hD)\f$.
 * @param constant_bindings     Keyword. Name to real value, for names used in
 *                              the operators.
 * @param nonlinear_model       Keyword. `"expr"` (default) or `"kerr_raman"`.
 * @param nonlinear_gamma       Keyword. Kerr coefficient of the built-in model.
 * @param raman_fraction        Keyword. Raman fraction \f$f_R\f$.
 * @param raman_tau1            Keyword, default 0.0122.
 * @param raman_tau2            Keyword, default 0.0320.
 * @param shock_omega0          Keyword. Self-steepening carrier frequency; 0
 *                              disables.
 * @param raman_response_time   Keyword. Sampled Raman response, replacing the
 *                              analytic one.
 * @param exec_options          Keyword. ExecutionOptions.
 * @param storage               Keyword. StorageOptions.
 * @param capture_step_history  Keyword. True to fill `result.step_history`.
 * @param step_history_capacity Keyword, default 200000.
 * @param progress_callback     Keyword. Called with a ProgressInfo; return
 *                              `false` to abort.
 * @return PropagationResult.
 * @throws ArgumentError           bad pulse, bad option value, or an operator
 *                                 that is not a function or cannot be
 *                                 translated.
 * @throws NLOLibError             the C call failed.
 * @throws PropagationAbortedError the progress callback aborted the run.
 */
PropagationResult propagate(const PulseSpec& pulse,
                            Any linear_operator,
                            Any nonlinear_operator,
                            double propagation_distance,
                            const char* preset = "balanced",
                            Any starting_step_size = Any(),
                            Any max_step_size = Any(),
                            Any min_step_size = Any(),
                            Any error_tolerance = Any(),
                            Any num_records = Any(),
                            Any record_z = Any(),
                            bool return_records = true,
                            Any potential_operator = Any(),
                            Any linear_step_operator = Any(),
                            Any constant_bindings = Any(),
                            const char* nonlinear_model = "expr",
                            double nonlinear_gamma = 0.0,
                            double raman_fraction = 0.0,
                            double raman_tau1 = 0.0122,
                            double raman_tau2 = 0.0320,
                            double shock_omega0 = 0.0,
                            Any raman_response_time = Any(),
                            Any exec_options = Any(),
                            Any storage = Any(),
                            bool capture_step_history = false,
                            int step_history_capacity = 200000,
                            Any progress_callback = Any());

/**
 * @ingroup julia_binding
 * @brief Query the limits the runtime derives for a pulse and execution
 *        options.
 *
 * Julia signature: `query_runtime_limits(; pulse = nothing, exec_options = nothing)`.
 *
 * @param pulse        Keyword. PulseSpec, or `nothing` for a default run.
 * @param exec_options Keyword. ExecutionOptions, or `nothing` for the defaults.
 * @return RuntimeLimits.
 */
RuntimeLimits query_runtime_limits(Any pulse = Any(), Any exec_options = Any());

/**
 * @ingroup julia_binding
 * @brief True when this build has SQLite snapshot storage.
 */
bool storage_is_available();

/**
 * @ingroup julia_binding
 * @brief Reshape one flat tensor record to (t, y, x), t fastest.
 *
 * @param record One record of a tensor run, such as `result.records[:, k]`,
 *               of length `nt * nx * ny`.
 * @param nt     Samples along t.
 * @param nx     Samples along x.
 * @param ny     Samples along y.
 * @return An `nt x ny x nx` array sharing the record's memory.
 */
Any tensor_record_view(Any record, int nt, int nx, int ny);

/**
 * @ingroup julia_binding
 * @brief Write the library log to a file.
 *
 * Julia signature: `set_log_file(path; append = false)`.
 */
void set_log_file(const char* path, bool append = false);

/**
 * @ingroup julia_binding
 * @brief Keep the library log in an in-memory ring buffer.
 */
void set_log_buffer(int capacity_bytes = 262144);

/**
 * @ingroup julia_binding
 * @brief Empty the log buffer.
 */
void clear_log_buffer();

/**
 * @ingroup julia_binding
 * @brief Read the log buffer as text.
 *
 * Julia signature: `read_log_buffer(; consume = true, max_bytes = 262144)`.
 */
const char* read_log_buffer(bool consume = true, int max_bytes = 262144);

/**
 * @ingroup julia_binding
 * @brief Set the log level: `"error"`, `"warn"`, `"info"` or `"debug"`.
 */
void set_log_level(const char* level = "info");

/**
 * @ingroup julia_binding
 * @brief Configure the library's own progress output.
 *
 * Julia signature:
 * `set_progress_options(; enabled = true, milestone_percent = 5, emit_on_step_adjust = false)`.
 */
void set_progress_options(bool enabled = true, int milestone_percent = 5,
                          bool emit_on_step_adjust = false);

/**
 * @ingroup julia_binding
 * @brief Send progress output to `"stderr"`, `"stdout"` or `"both"`.
 */
void set_progress_stream(const char* stream_mode = "stderr");

/**
 * @ingroup julia_binding
 * @brief Turn the performance counters on or off.
 */
void perf_profile_set_enabled(bool enabled = true);

/**
 * @ingroup julia_binding
 * @brief True when the performance counters are on.
 */
bool perf_profile_is_enabled();

/**
 * @ingroup julia_binding
 * @brief Zero the performance counters.
 */
void perf_profile_reset();

/**
 * @ingroup julia_binding
 * @brief Read the performance counters.
 */
PerfProfileSnapshot perf_profile_read();

} // namespace NLOLib
} // namespace julia
