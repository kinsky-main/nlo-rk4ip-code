/**
 * @file matlab_api.h
 * @brief Docs-only MATLAB binding API shim for Doxygen.
 *
 * This header exists purely so Doxygen can render the MATLAB package surface;
 * it is never compiled or installed. Each entity below mirrors a native MATLAB
 * class, method, or function whose real implementation lives in
 * `matlab/+nlolib/`.
 *
 * MATLAB is untyped, so the parameter and return types here describe intent
 * rather than a signature: `StructValue` stands for a MATLAB struct, and
 * optional trailing arguments follow MATLAB's `nargin` conventions.
 *
 * The narrative documentation is the MATLAB User Guide, rendered from
 * `docs/matlab_user_guide.md` and listed under the Guides tab.
 */

#pragma once

namespace matlab {
namespace nlolib {

/** @brief Placeholder for an arbitrary MATLAB struct value. */
struct StructValue {};

/**
 * @ingroup matlab_binding
 * @brief Result of a propagation, as returned by NLolib::propagate.
 *
 * MATLAB fields:
 * - `records` — `numRecords x numSamples` complex matrix, one row per recorded
 *   \f$z\f$ position.
 * - `final` — the last row of `records`, or `[]` when no records were returned.
 * - `z_axis` — recorded \f$z\f$ positions, `linspace(0, L, numRecords)`.
 * - `step_history` — solver telemetry, see @ref StepHistory.
 * - `meta` — run metadata: `output`, `records`, `records_requested`,
 *   `records_written`, `records_returned`, `storage_enabled`, `coupled`,
 *   `step_history_dropped`, plus `preset` (high-level form),
 *   `backend_requested` (low-level form), and `storage_result` when storage is
 *   enabled.
 *
 * `records_written` may be smaller than `records_requested`: the library
 * reduces the count for fixed-step runs, explicit-\f$z\f$ schedules, and
 * callback-aborted runs.
 */
struct PropagateResult {};

/**
 * @ingroup matlab_binding
 * @brief Per-step solver telemetry, populated when step-history capture is on.
 *
 * MATLAB fields: `step_index`, `z`, `step_size`, `next_step_size`, `error`
 * (column vectors), plus `dropped` and `capacity`. A non-zero `dropped` means
 * the history was truncated — raise `exec_options.step_history_capacity`.
 *
 * Enable capture with `exec_options.capture_step_history = true`.
 */
struct StepHistory {};

/**
 * @ingroup matlab_binding
 * @brief Outcome of a storage-backed run.
 *
 * MATLAB fields: `run_id`, `records_captured`, `records_spilled`,
 * `chunks_written`, `db_size_bytes`, `truncated`. Reported as
 * `result.meta.storage_result`.
 */
struct StorageResult {};

/**
 * @ingroup matlab_binding
 * @brief Runtime-derived solver limits, as returned by
 *        NLolib::query_runtime_limits.
 *
 * MATLAB fields: `max_num_time_samples_runtime`,
 * `max_num_recorded_samples_in_memory`,
 * `max_num_recorded_samples_with_storage`,
 * `estimated_required_working_set_bytes`, `estimated_device_budget_bytes`,
 * `storage_available`.
 */
struct RuntimeLimits {};

/**
 * @ingroup matlab_binding
 * @brief Runtime performance counters, as returned by
 *        NLolib::perf_profile_read.
 *
 * MATLAB fields: `dispersion_ms`, `nonlinear_ms`, `dispersion_calls`,
 * `nonlinear_calls`, `gpu_dispatch_count`, `gpu_copy_count`,
 * `gpu_device_copy_count`, `gpu_device_copy_bytes`,
 * `gpu_host_transfer_copy_count`, `gpu_host_transfer_copy_bytes`,
 * `gpu_memory_pass_count`, `gpu_memory_pass_bytes`, `gpu_upload_count`,
 * `gpu_download_count`, `gpu_upload_bytes`, `gpu_download_bytes`.
 */
struct PerfProfileSnapshot {};

/**
 * @ingroup matlab_binding
 * @brief High-level MATLAB wrapper around the nlolib shared library.
 *
 * Native MATLAB class name: \c %nlolib.NLolib. The class is a `handle` class
 * that loads the shared library with `loadlibrary` and dispatches through
 * `calllib`; no Python or MEX layer is involved.
 *
 * @code{.m}
 * api    = nlolib.NLolib();
 * pulse  = struct('samples', field0, 'delta_time', dt);
 * result = api.propagate(pulse, "gvd", "kerr", ...
 *                        struct('propagation_distance', 1.0));
 * @endcode
 *
 * @see The MATLAB User Guide (`docs/matlab_user_guide.md`) for the full
 *      walkthrough of pulse specs, operators, options, and telemetry.
 */
class NLolib {
public:
    /**
     * @brief Construct the wrapper and load the shared library.
     *
     * The library is located by, in order: @p libraryPath, the
     * `NLOLIB_LIBRARY` environment variable, then staged and build-tree
     * locations relative to the package (newest file first). Loading is a
     * no-op when the library is already loaded, so several wrapper objects can
     * coexist.
     *
     * @param libraryPath Explicit path to `nlolib.dll` / `libnlolib.so` /
     *                    `libnlolib.dylib`; empty to auto-discover.
     * @throws nlolib:libraryNotFound     no candidate library was found.
     * @throws nlolib:headerNotFound      `nlolib_matlab.h` is not next to the library.
     * @throws nlolib:libraryLoadFailed   every candidate failed to load.
     */
    NLolib(const char* libraryPath = "");

    /**
     * @brief Unified propagation entry point.
     *
     * Two calling conventions are accepted; the form is chosen by inspecting
     * @p primary. A struct carrying `samples` and `delta_time` but no
     * `num_time_samples` selects the high-level form.
     *
     * High-level — `propagate(pulse, linearOp, nonlinearOp, options)`.
     * The operators are preset names (`"gvd"`, `"kerr"`, `"none"`), structs
     * with `expr` + `params`, or structs with a function handle `fn`.
     * `options.propagation_distance` is required; `preset`
     * (`"fast"`/`"balanced"`/`"accuracy"`), `records`, `output`
     * (`"dense"`/`"final"`), `exec_options`, and `storage` are optional.
     *
     * Low-level — `propagate(cfg, field, numRecords[, execOptions][, storageOptions])`.
     * @p cfg is a flat struct matching the C API; `cfg.num_time_samples` must
     * equal `numel(field)`, which for a tensor run is the whole volume.
     *
     * @code{.m}
     * % high-level, physical coefficients
     * linearOp    = struct('expr', "i*beta2*w*w - loss", ...
     *                      'params', struct('beta2', 0.5 * beta2, 'loss', 0.0));
     * nonlinearOp = struct('expr', "i*gamma*A*I", 'params', struct('gamma', gamma));
     * options     = struct('propagation_distance', 0.5, 'records', 160);
     * result      = api.propagate(pulse, linearOp, nonlinearOp, options);
     *
     * % low-level, explicit solver schedule
     * result = api.propagate(cfg, field0, 128, struct('backend_type', 2));
     * @endcode
     *
     * @param primary  Pulse spec (high-level) or config struct (low-level).
     * @param varargin Remaining arguments for the selected form.
     * @return A @ref PropagateResult struct.
     * @throws nlolib:invalidPropagateCall        wrong argument count for the form.
     * @throws nlolib:invalidPulseSpec            bad pulse shape or missing fields.
     * @throws nlolib:invalidOperatorSpec         `expr` and `fn` both set, or an unusable operator.
     * @throws nlolib:inputFieldLengthMismatch    `cfg.num_time_samples ~= numel(field)`.
     * @throws nlolib:propagateFailed             the native call returned a non-OK status.
     */
    PropagateResult propagate(const StructValue& primary,
                              const StructValue& varargin = StructValue());

    /**
     * @brief Query runtime-derived solver limits without running a simulation.
     *
     * Use this to size a run — in particular to clamp the record count or
     * decide whether storage is needed — before committing to a large grid.
     *
     * @code{.m}
     * limits = api.query_runtime_limits(cfg, execOptions);
     * records = min(requestedRecords, limits.max_num_recorded_samples_in_memory);
     * @endcode
     *
     * @param config      Config struct; a trivial default is used when omitted.
     * @param execOptions Execution options; defaults to auto backend selection.
     * @return A @ref RuntimeLimits struct.
     * @throws nlolib:runtimeLimitsUnavailable this build lacks the entry point.
     * @throws nlolib:runtimeLimitsFailed      the native call returned a non-OK status.
     */
    RuntimeLimits query_runtime_limits(const StructValue& config = StructValue(),
                                       const StructValue& execOptions = StructValue()) const;

    /** @brief Return true when SQLite storage is compiled into this build. */
    int storage_is_available() const;

    /**
     * @brief Mirror runtime logs to a file.
     * @param path   Destination path; empty disables file logging.
     * @param append Non-zero appends instead of truncating.
     * @throws nlolib:logUnavailable this build lacks the entry point.
     */
    void set_log_file(const char* path, int append = 0);

    /**
     * @brief Enable and size the in-memory log ring buffer.
     * @param capacityBytes Buffer capacity in bytes.
     */
    void set_log_buffer(unsigned long long capacityBytes = 262144u);

    /** @brief Discard everything currently in the log ring buffer. */
    void clear_log_buffer();

    /**
     * @brief Read buffered runtime logs and return them as a string.
     * @param consume  Non-zero drains the buffer as it is read.
     * @param maxBytes Maximum bytes to read; must be >= 2.
     */
    const char* read_log_buffer(int consume = 1,
                                unsigned long long maxBytes = 262144u) const;

    /**
     * @brief Read buffered logs and also print them to the Command Window.
     *
     * Equivalent to read_log_buffer() followed by `fprintf`.
     */
    const char* tail_logs(int consume = 1,
                          unsigned long long maxBytes = 262144u) const;

    /**
     * @brief Set the runtime log threshold.
     * @param level 0 ERROR, 1 WARN, 2 INFO, 3 DEBUG.
     */
    void set_log_level(int level);

    /**
     * @brief Configure the runtime progress display.
     * @param enabled           Non-zero enables progress output.
     * @param milestonePercent  Emit a line every N percent.
     * @param emitOnStepAdjust  Non-zero also emits on each step-size change.
     */
    void set_progress_options(int enabled = 1, int milestonePercent = 5,
                              int emitOnStepAdjust = 0);

    /**
     * @brief Select the progress output stream.
     * @param streamMode 0 stderr, 1 stdout, 2 both. The wrapper defaults to
     *                   *both* when MATLAB runs without a desktop (`-batch`).
     */
    void set_progress_stream(int streamMode);

    /** @brief Enable or disable runtime performance counters. */
    void perf_profile_set_enabled(int enabled = 1);

    /** @brief Return true when performance counters are enabled. */
    int perf_profile_is_enabled() const;

    /** @brief Reset all performance counters to zero. */
    void perf_profile_reset();

    /**
     * @brief Read the current performance counters.
     *
     * @code{.m}
     * api.perf_profile_set_enabled(true);
     * api.perf_profile_reset();
     * result   = api.propagate(pulse, linearOp, nonlinearOp, options);
     * snapshot = api.perf_profile_read();
     * @endcode
     *
     * @return A @ref PerfProfileSnapshot struct.
     * @throws nlolib:perfProfileUnavailable this build lacks the entry point.
     */
    PerfProfileSnapshot perf_profile_read() const;

    /**
     * @brief Unload the shared library (static method).
     *
     * The destructor deliberately does *not* unload, since several wrapper
     * objects may share one loaded library. Call this, followed by
     * `clear classes`, to recover from a stale parsed type table.
     */
    static void unload();
};

/**
 * @ingroup matlab_binding
 * @brief Translate a MATLAB function handle into a runtime expression plus constants.
 *
 * Native MATLAB function name: \c %nlolib.translate_runtime_handle.
 *
 * Argument positions bind runtime symbols by context: `dispersion_factor`
 * takes `@(A, w)`, `dispersion` takes `@(A, D, h, w)`, and `nonlinear` takes
 * `@(A, I, V)`, each accepting a prefix of that list. Element-wise operators
 * are normalised (`.*` → `*`) and `1i`/`1j` literals become `i`.
 *
 * Identifiers that are not reserved symbols are captured from the handle's
 * closure as real scalar constants, unless bound explicitly through
 * `runtime.constant_bindings` or disabled with
 * `runtime.auto_capture_constants = false`.
 *
 * @code{.m}
 * beta2 = 0.05;
 * [expr, constants] = nlolib.translate_runtime_handle( ...
 *     @(A, w) 1i * (beta2 / 2.0) * (w .* w), "dispersion_factor");
 * @endcode
 *
 * @param fn      A `function_handle`.
 * @param context `"dispersion_factor"`, `"dispersion"`, or `"nonlinear"`.
 * @param runtime Optional runtime struct supplying `constant_bindings` and
 *                `auto_capture_constants`.
 * @return `[expression, constants]` — the compiled expression text and the
 *         captured constant values, in `cN` order.
 */
void translate_runtime_handle();

/**
 * @ingroup matlab_binding
 * @brief Prepare split simulation and physics structs for the native entry points.
 *
 * Native MATLAB function name: \c %nlolib.prepare_sim_config.
 *
 * Required config fields: `num_time_samples`, `propagation_distance`,
 * `starting_step_size`, `max_step_size`, `min_step_size`, `error_tolerance`,
 * `pulse_period`, `delta_time`, `frequency_grid`. Optional: the `tensor_*`
 * shape fields, `delta_x`/`delta_y`, `spatial_frequency_grid`,
 * `kx_axis`/`ky_axis`, `potential_grid`, `wt_axis`, and the `runtime`
 * sub-struct.
 *
 * @code{.m}
 * [simCfgPtr, physicsCfgPtr, keepalive] = nlolib.prepare_sim_config(cfg);
 * @endcode
 *
 * @param cfg Flat MATLAB config struct.
 * @return `[simCfgPtr, physicsCfgPtr, keepalive]`. The `keepalive` cell array
 *         holds the `libpointer` objects backing the pointer fields and must
 *         stay in scope for as long as the config is used, or MATLAB will free
 *         the buffers underneath the library.
 * @throws nlolib:notLoaded                create an `nlolib.NLolib` instance first.
 * @throws nlolib:tooManyRuntimeConstants  more than `RUNTIME_OPERATOR_CONSTANTS_MAX` constants.
 * @throws nlolib:legacyApiRemoved         `time_nt`/`spatial_nx`/`spatial_ny` or `transverse_*` fields.
 */
void prepare_sim_config();

/**
 * @ingroup matlab_binding
 * @brief Pack a MATLAB complex array into the native buffer layout.
 *
 * Native MATLAB function name: \c %nlolib.pack_complex_array.
 *
 * @param values Complex vector, in any orientation.
 * @return A `libpointer('nlo_complexPtr', ...)` whose payload is a struct array
 *         with `.re` and `.im` fields. `setdatatype()` cannot bind an element
 *         count to a struct pointer, so read-back through `.Value` yields only
 *         the first element — see unpack_records(), which walks the buffer with
 *         pointer arithmetic instead.
 */
void pack_complex_array();

/**
 * @ingroup matlab_binding
 * @brief Unpack propagated records into MATLAB arrays.
 *
 * Native MATLAB function name: \c %nlolib.unpack_records.
 *
 * @param outPtr         `libpointer('nlo_complexPtr', ...)` in record-major order.
 * @param numRecords     Records the library actually wrote (`records_written`),
 *                       which may be fewer than the allocated capacity;
 *                       trailing unwritten capacity is ignored.
 * @param numTimeSamples Samples per record.
 * @param debugContext   Optional struct enriching failure diagnostics.
 * @return A `numRecords x numTimeSamples` complex matrix.
 * @throws nlolib:invalidComplexBuffer       unsupported buffer representation.
 * @throws nlolib:invalidComplexBufferLength the buffer is shorter than required.
 */
void unpack_records();

} // namespace nlolib
} // namespace matlab
