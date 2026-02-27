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
 * @param simulation_config Optional simulation configuration used to estimate
 *        required working-set bytes and in-memory record capacity.
 * @param physics_config Optional physics/operator configuration.
 * @param exec_options Optional runtime backend selection/options.
 * @param out_limits Destination limits descriptor.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_query_runtime_limits(
    const nlo_simulation_config* simulation_config,
    const nlo_physics_config* physics_config,
    const nlo_execution_options* exec_options,
    nlo_runtime_limits* out_limits
);

// MARK: Function Declarations

typedef enum {
    /** Return all requested records. */
    NLO_PROPAGATE_OUTPUT_DENSE = 0,
    /** Return only the final output field record. */
    NLO_PROPAGATE_OUTPUT_FINAL_ONLY = 1
} nlo_propagate_output_mode;

/**
 * @brief Unified propagation request options.
 *
 * @param num_recorded_samples Number of records to compute.
 * @param output_mode Record output mode.
 * @param return_records Nonzero to write records to output buffer.
 * @param exec_options Optional runtime backend selection/options.
 * @param storage_options Optional storage configuration.
 */
typedef struct {
    size_t num_recorded_samples;
    nlo_propagate_output_mode output_mode;
    int return_records;
    const nlo_execution_options* exec_options;
    const nlo_storage_options* storage_options;
} nlo_propagate_options;

/**
 * @brief Unified propagation output metadata and buffers.
 *
 * @param output_records Optional destination records buffer.
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
    nlo_storage_result* storage_result;
    nlo_step_event* output_step_events;
    size_t output_step_event_capacity;
    size_t* step_events_written;
    size_t* step_events_dropped;
} nlo_propagate_output;

/**
 * @brief Build default propagation options.
 *
 * Defaults:
 * - dense output mode
 * - 2 recorded samples
 * - return records enabled
 * - AUTO backend exec options
 * - storage disabled
 *
 * @return nlo_propagate_options Initialized options.
 */
NLOLIB_API nlo_propagate_options nlolib_propagate_options_default(void);

/**
 * @brief Build default propagation output descriptor.
 *
 * @return nlo_propagate_output Initialized output descriptor.
 */
NLOLIB_API nlo_propagate_output nlolib_propagate_output_default(void);

/**
 * @brief Propagate an input field using split simulation/physics configs.
 *
 * @param simulation_config Simulation configuration parameters.
 * @param physics_config Runtime physics/operator expression parameters.
 * @param num_time_samples Number of samples in flattened input.
 * @param input_field Pointer to input field buffer.
 * @param options Optional propagation options.
 * @param output Optional propagation output descriptor.
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_propagate(
    const nlo_simulation_config* simulation_config,
    const nlo_physics_config* physics_config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    const nlo_propagate_options* options,
    nlo_propagate_output* output
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
