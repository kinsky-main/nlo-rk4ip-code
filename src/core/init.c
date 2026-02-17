/**
 * @file init.c
 * @brief Initialization helpers for simulation state setup.
 */

#include "core/init.h"
#include <stdint.h>
#include <string.h>

static int checked_mul_size_t(size_t a, size_t b, size_t* out)
{
    if (out == NULL) {
        return -1;
    }

    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }

    if (a > SIZE_MAX / b) {
        return -1;
    }

    *out = a * b;
    return 0;
}

NLOLIB_API int nlo_init_simulation_state(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options,
    nlo_allocation_info* allocation_info,
    simulation_state** out_state
)
{
    if (out_state == NULL) {
        return -1;
    }

    *out_state = NULL;
    if (allocation_info != NULL) {
        *allocation_info = (nlo_allocation_info){0};
    }

    if (config == NULL || num_time_samples == 0 || num_recorded_samples == 0) {
        return -1;
    }

    nlo_execution_options local_options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_AUTO);

    simulation_state* state = create_simulation_state(config,
                                                      num_time_samples,
                                                      num_recorded_samples,
                                                      &local_options);
    if (state == NULL) {
        return -1;
    }

    if (allocation_info != NULL) {
        size_t per_record_bytes = 0u;
        size_t host_snapshot_bytes = 0u;
        size_t working_bytes = 0u;

        (void)checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &per_record_bytes);
        (void)checked_mul_size_t(per_record_bytes, state->num_recorded_samples, &host_snapshot_bytes);
        (void)checked_mul_size_t(per_record_bytes, NLO_WORK_VECTOR_COUNT, &working_bytes);

        allocation_info->requested_records = num_recorded_samples;
        allocation_info->allocated_records = state->num_recorded_samples;
        allocation_info->per_record_bytes = per_record_bytes;
        allocation_info->host_snapshot_bytes = host_snapshot_bytes;
        allocation_info->working_vector_bytes = working_bytes;
        allocation_info->device_ring_capacity = state->record_ring_capacity;
        allocation_info->device_budget_bytes = local_options.forced_device_budget_bytes;
        allocation_info->backend_type = local_options.backend_type;
    }

    *out_state = state;
    return 0;
}
