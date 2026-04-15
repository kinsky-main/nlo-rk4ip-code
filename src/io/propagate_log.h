/**
 * @file propagate_log.h
 * @dir src/io
 * @brief Structured logging helpers for nlolib propagation entry points.
 */

#pragma once

#include "backend/nlo_complex.h"
#include "core/init.h"

/**
 * @brief Log one structured propagate request summary.
 *
 * @param config Effective simulation/physics configuration.
 * @param num_time_samples Number of time samples in the input field.
 * @param input_field Input field pointer provided by the caller.
 * @param num_recorded_samples Number of requested output records.
 * @param output_records Output record pointer provided by the caller.
 * @param exec_options Optional execution options provided by the caller.
 */
void log_propagate_request(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const execution_options* exec_options
);

/**
 * @brief Log the resolved backend and allocation summary for one propagate call.
 *
 * @param requested_backend Backend requested by the caller.
 * @param actual_backend Backend selected at runtime.
 * @param allocation_summary Allocation summary produced during state init.
 * @param mem_info Backend memory information queried after init.
 */
void log_propagate_allocation_summary(
    vector_backend_type requested_backend,
    vector_backend_type actual_backend,
    const allocation_info* allocation_summary,
    const vec_backend_memory_info* mem_info
);
