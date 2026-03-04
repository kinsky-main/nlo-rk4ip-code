/**
 * @file state_debug.h
 * @brief Optional debug logging helpers for simulation-state initialization.
 */
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Emit a state-initialization failure diagnostic when enabled.
 *
 * Logging is controlled by the `NLO_STATE_DEBUG` environment variable.
 * Any non-empty value except `0`, `false`, `off`, or `no` enables output.
 *
 * @param stage Stable stage identifier for the failure site.
 * @param status Backend/status code associated with the failure.
 */
void nlo_state_debug_log_failure(const char* stage, int status);

/**
 * @brief Emit ring-buffer sizing diagnostics when enabled.
 *
 * @param requested_records Number of requested host-side records.
 * @param per_record_bytes Bytes per record vector.
 * @param active_bytes Estimated active device bytes before ring allocation.
 * @param runtime_stack_slots Number of runtime operator stack vectors.
 * @param budget_bytes Effective device budget used for ring sizing.
 * @param ring_capacity Computed ring capacity.
 */
void nlo_state_debug_log_ring_capacity(
    size_t requested_records,
    size_t per_record_bytes,
    size_t active_bytes,
    size_t runtime_stack_slots,
    size_t budget_bytes,
    size_t ring_capacity
);

/**
 * @brief Emit a backend-memory checkpoint for initialization-stage tracing.
 *
 * @param stage Stable stage identifier for the checkpoint.
 * @param query_status Status returned by nlo_vec_query_memory_info.
 * @param total_device_local_bytes Reported total device-local bytes.
 * @param available_device_local_bytes Reported available device-local bytes.
 * @param max_storage_buffer_range_bytes Reported max storage-buffer range.
 */
void nlo_state_debug_log_memory_checkpoint(
    const char* stage,
    int query_status,
    size_t total_device_local_bytes,
    size_t available_device_local_bytes,
    size_t max_storage_buffer_range_bytes
);

#ifdef __cplusplus
}
#endif
