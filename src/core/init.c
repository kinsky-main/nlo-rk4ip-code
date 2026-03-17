/**
 * @file init.c
 * @brief Initialization entry points that wrap simulation-state builders.
 */

#include "core/init.h"
#include "core/init_internal.h"

#ifndef NLO_DEVICE_RING_BUDGET_HEADROOM_NUM
#define NLO_DEVICE_RING_BUDGET_HEADROOM_NUM 9u
#endif

#ifndef NLO_DEVICE_RING_BUDGET_HEADROOM_DEN
#define NLO_DEVICE_RING_BUDGET_HEADROOM_DEN 10u
#endif

static size_t nlo_count_working_full_volume_vectors(const simulation_state* state)
{
    if (state == NULL) {
        return 0u;
    }

    size_t count = 0u;
    if (state->current_field_vec != NULL) {
        count += 1u;
    }
    if (state->frequency_grid_vec != NULL) {
        count += 1u;
    }

    if (state->working_vectors.ip_field_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.field_working_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.field_freq_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.k_final_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.k_temp_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.dispersion_factor_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.dispersion_operator_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.potential_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.previous_field_vec != NULL) {
        count += 1u;
    }

    if (state->working_vectors.raman_intensity_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_delayed_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_spectrum_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_mix_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_polarization_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_derivative_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_response_fft_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_derivative_factor_vec != NULL) {
        count += 1u;
    }

    if (state->working_vectors.wt_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.kx_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.ky_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.t_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.x_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.y_mesh_vec != NULL) {
        count += 1u;
    }

    if (state->runtime_operator_stack_slots > SIZE_MAX - count) {
        return SIZE_MAX;
    }
    count += state->runtime_operator_stack_slots;

    if (state->fft_plan != NULL &&
        nlo_vector_backend_get_type(state->backend) == NLO_VECTOR_BACKEND_VULKAN) {
        if (count == SIZE_MAX) {
            return SIZE_MAX;
        }
        count += 1u;
    }

    return count;
}

static int nlo_estimate_working_vector_bytes(
    size_t per_record_bytes,
    const simulation_state* state,
    size_t* out_working_bytes
)
{
    if (state == NULL || out_working_bytes == NULL) {
        return -1;
    }

    const size_t full_count = nlo_count_working_full_volume_vectors(state);
    if (full_count == SIZE_MAX) {
        return -1;
    }

    size_t working_bytes = 0u;
    size_t full_bytes = 0u;
    if (nlo_checked_mul_size_t(per_record_bytes, full_count, &full_bytes) != 0) {
        return -1;
    }
    working_bytes += full_bytes;

    *out_working_bytes = working_bytes;
    return 0;
}

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
    if (state->snapshot_scratch_record != NULL) {
        host_snapshot_bytes += per_record_bytes;
    }
    if (state->field_buffer != NULL) {
        size_t field_buffer_bytes = 0u;
        if (nlo_checked_mul_size_t(per_record_bytes, state->num_host_records, &field_buffer_bytes) == 0) {
            host_snapshot_bytes += field_buffer_bytes;
        }
    }
    if (nlo_estimate_working_vector_bytes(per_record_bytes, state, &working_bytes) != 0) {
        working_bytes = 0u;
    }

    allocation_info->requested_records = requested_records;
    allocation_info->allocated_records = (state->snapshot_scratch_record != NULL) ? 1u : state->num_host_records;
    allocation_info->per_record_bytes = per_record_bytes;
    allocation_info->host_snapshot_bytes = host_snapshot_bytes;
    allocation_info->working_vector_bytes = working_bytes;
    allocation_info->device_ring_capacity = state->record_ring_capacity;
    allocation_info->device_budget_bytes = exec_options->forced_device_budget_bytes;
    if (nlo_vector_backend_get_type(state->backend) == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vec_backend_memory_info mem_info = {0};
        if (nlo_vec_query_memory_info(state->backend, &mem_info) == NLO_VEC_STATUS_OK) {
            const double frac = (exec_options->device_heap_fraction > 0.0 &&
                                 exec_options->device_heap_fraction <= 1.0)
                                    ? exec_options->device_heap_fraction
                                    : NLO_DEFAULT_DEVICE_HEAP_FRACTION;
            size_t effective_budget = exec_options->forced_device_budget_bytes;
            if (effective_budget > 0u) {
                if (mem_info.device_local_available_bytes > 0u &&
                    effective_budget > mem_info.device_local_available_bytes) {
                    effective_budget = mem_info.device_local_available_bytes;
                }
            } else {
                effective_budget = (size_t)((double)mem_info.device_local_available_bytes * frac);
            }
            effective_budget =
                (effective_budget / NLO_DEVICE_RING_BUDGET_HEADROOM_DEN) *
                NLO_DEVICE_RING_BUDGET_HEADROOM_NUM;
            allocation_info->device_budget_bytes = effective_budget;
        }
    }
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
