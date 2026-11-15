/**
 * @file nlolib.h
 * @dir src
 * @brief Main header file for the Nonlinear Optics Library (NLOLib).
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#pragma once

// MARK: Includes

#include <stddef.h>
#include <stdint.h>

#include "nlo_progress.h"
#include "core/state.h"
#include "core/init.h"
#include "backend/nlo_complex.h"

#ifdef __cplusplus
extern "C" {
#endif

// MARK: Const & Macros

// DLL export/import for Windows builds.
#ifndef NLOLIB_API
  #if defined(_WIN32)
    #if defined(NLOLIB_EXPORTS)
      #define NLOLIB_API __declspec(dllexport)
    #else
      #define NLOLIB_API __declspec(dllimport)
    #endif
  #else
    #define NLOLIB_API
  #endif
#endif

// MARK: Typedefs

typedef enum {
    /** Operation completed successfully. */
    NLOLIB_STATUS_OK = 0,
    /** Input parameters were invalid or inconsistent. */
    NLOLIB_STATUS_INVALID_ARGUMENT = 1,
    /** Required allocation failed. */
    NLOLIB_STATUS_ALLOCATION_FAILED = 2,
    /** Requested behavior is not available in current build/runtime. */
    NLOLIB_STATUS_NOT_IMPLEMENTED = 3,
    /** Operation was aborted by a caller-provided progress callback. */
    NLOLIB_STATUS_ABORTED = 4
} nlolib_status;

/**
 * @ingroup c_api
 * @brief Public log-level thresholds for nlolib runtime logging.
 */
typedef enum {
    /** Emit only error-level log entries. */
    NLOLIB_LOG_LEVEL_ERROR = 0,
    /** Emit warning and error-level log entries. */
    NLOLIB_LOG_LEVEL_WARN = 1,
    /** Emit informational, warning, and error-level log entries. */
    NLOLIB_LOG_LEVEL_INFO = 2,
    /** Emit debug, informational, warning, and error-level log entries. */
    NLOLIB_LOG_LEVEL_DEBUG = 3
} nlolib_log_level;

/**
 * @ingroup c_api
 * @brief Output stream selection for runtime progress TUI rendering.
 */
typedef enum {
    /** Render progress updates to stderr only. */
    NLOLIB_PROGRESS_STREAM_STDERR = 0,
    /** Render progress updates to stdout only. */
    NLOLIB_PROGRESS_STREAM_STDOUT = 1,
    /** Render progress updates to both stderr and stdout. */
    NLOLIB_PROGRESS_STREAM_BOTH = 2
} nlolib_progress_stream_mode;

/**
 * @brief Snapshot of accumulated runtime performance counters.
 */
#ifndef NLO_PERF_PROFILE_SNAPSHOT_DEFINED
#define NLO_PERF_PROFILE_SNAPSHOT_DEFINED 1
typedef struct {
    double dispersion_ms;
    double nonlinear_ms;
    uint64_t dispersion_calls;
    uint64_t nonlinear_calls;
    uint64_t gpu_dispatch_count;
    uint64_t gpu_copy_count;
    uint64_t gpu_device_copy_count;
    uint64_t gpu_device_copy_bytes;
    uint64_t gpu_host_transfer_copy_count;
    uint64_t gpu_host_transfer_copy_bytes;
    uint64_t gpu_memory_pass_count;
    uint64_t gpu_memory_pass_bytes;
    uint64_t gpu_upload_count;
    uint64_t gpu_download_count;
    uint64_t gpu_upload_bytes;
    uint64_t gpu_download_bytes;
} nlo_perf_profile_snapshot;
#endif

/**
 * @brief Enable or disable runtime performance counter accumulation.
 *
 * @param enabled Nonzero enables profiling; zero disables profiling.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_perf_profile_set_enabled(int enabled);

/**
 * @brief Query whether runtime performance counters are enabled.
 *
 * @return int Nonzero when profiling is enabled.
 */
NLOLIB_API int nlolib_perf_profile_is_enabled(void);

/**
 * @brief Reset all runtime performance counters.
 *
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_perf_profile_reset(void);

/**
 * @brief Read current runtime performance counters into caller memory.
 *
 * @param out_snapshot Destination snapshot.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_perf_profile_read(nlo_perf_profile_snapshot* out_snapshot);

/**
 * @brief Query runtime-derived limits for current backend/config selection.
 *
 * @param simulation_config Optional simulation configuration used to estimate
 *        required working-set bytes and in-memory record capacity.
 * @param physics_config Optional physics/operator configuration.
 * @param exec_options Optional runtime backend selection/options.
 * @param out_limits Destination limits descriptor.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_query_runtime_limits(
    const simulation_config* simulation_config,
    const physics_config* physics_config,
    const execution_options* exec_options,
    runtime_limits* out_limits
);

// MARK: Function Declarations

typedef enum {
    /** Return all requested records. */
    PROPAGATE_OUTPUT_DENSE = 0,
    /** Return only the final output field record. */
    PROPAGATE_OUTPUT_FINAL_ONLY = 1
} propagate_output_mode;

/**
 * @ingroup c_api
 * @brief Unified propagation request options.
 *
 * @param num_recorded_samples Number of records to compute.
 * @param output_mode Record output mode.
 * @param return_records Nonzero to write records to output buffer.
 * @param exec_options Optional runtime backend selection/options.
 * @param storage_options Optional storage configuration.
 * @param explicit_record_z Optional monotonic z-sample locations for exact
 *        record capture. When provided with explicit_record_z_count > 0, these
 *        override uniform record spacing.
 * @param explicit_record_z_count Number of entries in explicit_record_z.
 * @param progress_callback Optional per-event progress callback.
 *        Returning zero aborts propagation.
 * @param progress_user_data Caller-owned opaque pointer passed to
 *        progress_callback.
 */
typedef struct {
    size_t num_recorded_samples;
    propagate_output_mode output_mode;
    int return_records;
    const execution_options* exec_options;
    const storage_options* storage_options;
    const double* explicit_record_z;
    size_t explicit_record_z_count;
    progress_callback progress_callback;
    void* progress_user_data;
} propagate_options;

/**
 * @ingroup c_api
 * @brief Unified propagation output metadata and buffers.
 *
 * @param output_records Optional destination records buffer.
 *        When omitted, records may still be captured through storage.
 * @param output_record_capacity Maximum records available in output_records.
 * @param records_written Number of records actually written.
 * @param storage_result Optional storage summary output.
 * @param output_step_events Optional destination step-event buffer.
 * @param output_step_event_capacity Maximum events available in output_step_events.
 * @param step_events_written Number of step events actually written.
 * @param step_events_dropped Number of events dropped due to capacity limits.
 */
typedef struct {
    nlo_complex* output_records;
    size_t output_record_capacity;
    size_t* records_written;
    storage_result* storage_result;
    step_event* output_step_events;
    size_t output_step_event_capacity;
    size_t* step_events_written;
    size_t* step_events_dropped;
} propagate_output;

/**
 * @ingroup c_api
 * @brief Build default propagation options.
 *
 * Defaults:
 * - dense output mode
 * - 2 recorded samples
 * - return records enabled
 * - AUTO backend exec options
 * - storage disabled
 *
 * @return propagate_options Initialized options.
 */
NLOLIB_API propagate_options nlolib_propagate_options_default(void);

/**
 * @ingroup c_api
 * @brief Build default propagation output descriptor.
 *
 * @return propagate_output Initialized output descriptor.
 */
NLOLIB_API propagate_output nlolib_propagate_output_default(void);

/**
 * @ingroup c_api
 * @brief Propagate an input field using split simulation/physics configs.
 *
 * @param simulation_config Simulation configuration parameters.
 * @param physics_config Runtime physics/operator expression parameters.
 * @param num_time_samples Total number of input field samples.
 * @param input_field Pointer to input field buffer.
 * @param options Optional propagation options.
 * @param output Optional propagation output descriptor.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_propagate(
    const simulation_config* simulation_config,
    const physics_config* physics_config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    const propagate_options* options,
    propagate_output* output
);

/**
 * @ingroup c_api
 * @brief Returns nonzero when SQLite storage support is available.
 *
 * @return int Nonzero when storage backend support is compiled in.
 */
NLOLIB_API int nlolib_storage_is_available(void);

/**
 * @ingroup c_api
 * @brief Compute the run key under which nlolib_propagate() would store a run.
 *
 * The key is 16 lowercase hex digits: the start of a SHA-256 over a canonical
 * text beginning with `kind=nlolib` that covers grids and steps, propagation
 * distance, solver tolerances and step bounds, every operator expression and
 * constant, the nonlinear model and Raman settings, digests of the input
 * field, the potential grid and any other caller-supplied grid, and the
 * record specification (record count or explicit z schedule, record mode,
 * raw strides, final-field flag). Execution backend and device settings are
 * not part of it, so the same run on CPU and Vulkan has one key.
 *
 * @param simulation_config Simulation configuration, as passed to nlolib_propagate().
 * @param physics_config Physics configuration, as passed to nlolib_propagate().
 * @param num_time_samples Total number of input field samples.
 * @param input_field Input field buffer.
 * @param options Propagation options including storage_options (NULL: defaults).
 * @param out_key Destination of at least @ref STORAGE_KEY_MAX bytes; receives
 *        the NUL-terminated key.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_storage_run_key(
    const simulation_config* simulation_config,
    const physics_config* physics_config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    const propagate_options* options,
    char* out_key
);

/**
 * @ingroup c_api
 * @brief Ask whether a shared run database already holds a run.
 *
 * Opens @p sqlite_path read-only. A missing file holds no runs.
 *
 * @param sqlite_path UTF-8 path of the shared run database.
 * @param key Run key from nlolib_storage_run_key().
 * @param out_present Receives 1 when a `runs` row with @p key exists, else 0.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_storage_has_run(const char* sqlite_path, const char* key, int* out_present);

/**
 * @ingroup c_api
 * @brief Merge or discard staging databases left by interrupted or unmerged runs.
 *
 * Scans `<dir of sqlite_path>/staging/` for this database's staging files.
 * Complete files (with a `runs` row) are merged under the lock and removed.
 * Incomplete files are removed when @p discard_incomplete is nonzero and the
 * file is at least @p min_age_seconds old; incomplete files of live processes
 * on this host are always kept. With @p discard_incomplete, the data of
 * interrupted merges whose staging file is gone (io_runs rows with a key but no
 * `runs` row) is deleted too and counted in @p out_discarded.
 *
 * @param sqlite_path UTF-8 path of the shared run database.
 * @param discard_incomplete Nonzero deletes incomplete staging files.
 * @param min_age_seconds Minimum file age before an incomplete file is deleted.
 * @param out_merged Optional count of merged files.
 * @param out_discarded Optional count of deleted files.
 * @param out_skipped Optional count of files left in place.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_storage_recover_staging(
    const char* sqlite_path,
    int discard_incomplete,
    double min_age_seconds,
    size_t* out_merged,
    size_t* out_discarded,
    size_t* out_skipped
);

/**
 * @ingroup c_api
 * @brief Read the grid and record counts of a stored run.
 *
 * @param sqlite_path UTF-8 path of a run database (shared or staging).
 * @param key Run key.
 * @param out_layout Destination layout.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_storage_read_run_layout(
    const char* sqlite_path,
    const char* key,
    storage_run_layout* out_layout
);

/**
 * @ingroup c_api
 * @brief Read one stored array of a run as float64 values.
 *
 * @p name is one of `z`, `t`, `omega`, `x`, `y` (axes), `power`, `spectrum`
 * (num_records x nt, record-major), `fluence_x` (num_records x nx),
 * `fluence_y` (num_records x ny), `final_field` (nt*nx*ny complex values as
 * interleaved re, im) or `raw` (raw_records x raw_nt*raw_nx*raw_ny complex
 * values as interleaved re, im). Layout and units: docs/nlolib_run_database.md.
 *
 * @param sqlite_path UTF-8 path of a run database.
 * @param key Run key.
 * @param name Array name.
 * @param out_values Destination, or NULL to query the size.
 * @param capacity Capacity of @p out_values in doubles.
 * @param out_count Receives the number of doubles in the array; nothing is
 *        copied when it exceeds @p capacity.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_storage_read_run_array(
    const char* sqlite_path,
    const char* key,
    const char* name,
    double* out_values,
    size_t capacity,
    size_t* out_count
);

/**
 * @brief Configure optional runtime log file output.
 *
 * @param path_utf8 UTF-8 path to output log file.
 *        Pass NULL or empty string to disable file sink.
 * @param append Nonzero appends to existing file; zero truncates file.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_set_log_file(const char* path_utf8, int append);

/**
 * @brief Configure optional in-memory runtime log ring buffer.
 *
 * @param capacity_bytes Ring capacity in bytes.
 *        Pass zero to disable in-memory buffering.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_set_log_buffer(size_t capacity_bytes);

/**
 * @brief Clear buffered in-memory runtime logs.
 *
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_clear_log_buffer(void);

/**
 * @brief Read buffered runtime logs into caller memory.
 *
 * @param dst Destination text buffer.
 * @param dst_bytes Capacity of @p dst in bytes.
 * @param out_written Number of bytes written to @p dst (excluding null terminator).
 * @param consume Nonzero consumes copied bytes from ring buffer.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_read_log_buffer(
    char* dst,
    size_t dst_bytes,
    size_t* out_written,
    int consume
);

/**
 * @brief Set global runtime logging level threshold.
 *
 * @param level Log-level threshold (see @ref nlolib_log_level).
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_set_log_level(int level);

/**
 * @brief Configure runtime progress TUI behavior.
 *
 * @param enabled Nonzero enables progress rendering.
 * @param milestone_percent Refresh milestone cadence in [1, 100] percent.
 * @param emit_on_step_adjust Nonzero forces immediate refresh on adaptive
 *        step-size adjustment events.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_set_progress_options(
    int enabled,
    int milestone_percent,
    int emit_on_step_adjust
);

/**
 * @brief Configure output stream selection for runtime progress TUI lines.
 *
 * @param stream_mode Stream mode in @ref nlolib_progress_stream_mode range.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_set_progress_stream(int stream_mode);

#ifdef __cplusplus
}
#endif
