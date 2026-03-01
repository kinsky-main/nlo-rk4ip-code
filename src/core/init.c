/**
 * @file init.c
 * @brief Initialization entry points that wrap simulation-state builders.
 */

#include "core/init.h"
#include "core/init_internal.h"

static void nlo_fill_allocation_info(
    size_t num_time_samples,
    size_t requested_records,
    const simulation_state* state,
    const nlo_execution_options* exec_options,
    nlo_allocation_info* allocation_info
)
{
    if (allocation_info == NULL || state == NULL || exec_options == NULL) {
        return;
    }

    size_t per_record_bytes = 0u;
    size_t host_snapshot_bytes = 0u;
    size_t working_bytes = 0u;

    (void)nlo_checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &per_record_bytes);
    (void)nlo_checked_mul_size_t(per_record_bytes, state->num_host_records, &host_snapshot_bytes);
    (void)nlo_checked_mul_size_t(per_record_bytes, NLO_WORK_VECTOR_COUNT, &working_bytes);

    allocation_info->requested_records = requested_records;
    allocation_info->allocated_records = state->num_host_records;
    allocation_info->per_record_bytes = per_record_bytes;
    allocation_info->host_snapshot_bytes = host_snapshot_bytes;
    allocation_info->working_vector_bytes = working_bytes;
    allocation_info->device_ring_capacity = state->record_ring_capacity;
    allocation_info->device_budget_bytes = exec_options->forced_device_budget_bytes;
    allocation_info->backend_type = exec_options->backend_type;
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

    const nlo_execution_options local_options =
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

    nlo_fill_allocation_info(num_time_samples,
                             num_recorded_samples,
                             state,
                             &local_options,
                             allocation_info);

    *out_state = state;
    return 0;
}

NLOLIB_API int nlo_init_simulation_state_with_storage(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options,
    const nlo_storage_options* storage_options,
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

    const nlo_execution_options local_options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_AUTO);

    simulation_state* state = create_simulation_state_with_storage(config,
                                                                   num_time_samples,
                                                                   num_recorded_samples,
                                                                   &local_options,
                                                                   storage_options);
    if (state == NULL) {
        return -1;
    }

    nlo_fill_allocation_info(num_time_samples,
                             num_recorded_samples,
                             state,
                             &local_options,
                             allocation_info);

    *out_state = state;
    return 0;
}
