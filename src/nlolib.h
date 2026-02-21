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
    NLOLIB_STATUS_NOT_IMPLEMENTED = 3
} nlolib_status;

/**
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
 * @brief Query runtime-derived limits for current backend/config selection.
 *
 * @param config Optional simulation configuration used to estimate
 *        required working-set bytes and in-memory record capacity.
 * @param exec_options Optional runtime backend selection/options.
 * @param out_limits Destination limits descriptor.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_query_runtime_limits(
    const sim_config* config,
    const nlo_execution_options* exec_options,
    nlo_runtime_limits* out_limits
);

// MARK: Function Declarations

/**
 * @brief Propagate an input field and return recorded envelopes across z.
 *
 * @param config Simulation configuration parameters.
 * @param num_time_samples Number of samples in the flattened input/output arrays.
 *        For legacy 1D use this is the temporal sample count.
 *        If config->time.nt == 0 and spatial nx/ny are provided, this must
 *        equal config->spatial.nx * config->spatial.ny (legacy flattened mode).
 *        If config->time.nt > 0, this must equal
 *        config->time.nt * config->spatial.nx * config->spatial.ny.
 * @param input_field Pointer to input field buffer (length: num_time_samples),
 *        flattened in row-major order with x as the fastest index.
 * @param num_recorded_samples Number of envelope records to return.
 * @param output_records Pointer to output record buffer (length:
 *        num_recorded_samples * num_time_samples). The layout is record-major:
 *        output_records[record_index * num_time_samples + sample_index].
 *        For num_recorded_samples == 1, record 0 is the final field at z_end.
 *        For num_recorded_samples > 1, records are evenly distributed over [0, z_end].
 * @param exec_options Runtime backend selection/options
 *        (NULL uses AUTO hardware-detected defaults).
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_propagate(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options
);

/**
 * @brief MATLAB/FFI convenience API using interleaved complex doubles.
 *
 * @param config Simulation configuration parameters.
 * @param num_time_samples Number of complex samples in input/output records.
 * @param input_field_interleaved Pointer to interleaved complex input of
 *        length 2 * num_time_samples:
 *        [re0, im0, re1, im1, ...].
 * @param num_recorded_samples Number of envelope records to return.
 * @param output_records_interleaved Pointer to interleaved complex output of
 *        length 2 * num_recorded_samples * num_time_samples.
 * @param exec_options Runtime backend selection/options
 *        (NULL uses AUTO hardware-detected defaults).
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_propagate_interleaved(
    const sim_config* config,
    size_t num_time_samples,
    const double* input_field_interleaved,
    size_t num_recorded_samples,
    double* output_records_interleaved,
    const nlo_execution_options* exec_options
);

/**
 * @brief Propagate while optionally spilling snapshot chunks into SQLite.
 *
 * @param config Simulation configuration parameters.
 * @param num_time_samples Number of complex samples in the flattened field.
 * @param input_field Pointer to input field buffer (length: num_time_samples).
 * @param num_recorded_samples Number of envelope records to capture.
 * @param output_records Optional host output buffer (same layout as
 *        nlolib_propagate). Pass NULL for storage-only capture.
 * @param exec_options Runtime backend selection/options.
 * @param storage_options Optional storage configuration (NULL disables storage).
 *        Set storage_options->log_final_output_field_to_db nonzero to persist
 *        the final output field at z_end in a dedicated DB row.
 * @param storage_result Optional output summary for persisted chunks/run state.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_propagate_with_storage(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options,
    const nlo_storage_options* storage_options,
    nlo_storage_result* storage_result
);

/**
 * @brief Returns nonzero when SQLite storage support is available.
 *
 * @return int Nonzero when storage backend support is compiled in.
 */
NLOLIB_API int nlolib_storage_is_available(void);

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
 * @brief Configure runtime progress logging behavior.
 *
 * @param enabled Nonzero enables progress logging.
 * @param milestone_percent Percent milestone cadence in [1, 100].
 * @param emit_on_step_adjust Nonzero emits step-adjustment entries.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_set_progress_options(
    int enabled,
    int milestone_percent,
    int emit_on_step_adjust
);

#ifdef __cplusplus
}
#endif
