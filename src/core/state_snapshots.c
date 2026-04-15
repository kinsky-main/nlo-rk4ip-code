/**
 * @file state_snapshots.c
 * @brief Snapshot capture/upload/download runtime state operations.
 */

#include "core/state.h"
#include "io/snapshot_store.h"
#include <stdbool.h>
#include <string.h>

static vec_status snapshot_emit_record(simulation_state* state, size_t record_index, const nlo_complex* record);
static vec_status simulation_state_flush_oldest_ring_entry(simulation_state* state);
static nlo_complex* simulation_state_resolve_download_target(simulation_state* state, size_t record_index);

vec_status simulation_state_upload_initial_field(simulation_state* state, const nlo_complex* field)
{
    if (state == NULL || field == NULL || state->backend == NULL || state->current_field_vec == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t bytes = state->num_time_samples * sizeof(nlo_complex);
    vec_status status = vec_upload(state->backend, state->current_field_vec, field, bytes);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    int captured_initial_record = 0;
    if (state->num_recorded_samples > 1u) {
        int should_capture_initial = 1;
        if (state->explicit_record_schedule_active != 0) {
            should_capture_initial = 0;
            if (state->explicit_record_z != NULL && state->explicit_record_z_count > 0u) {
                const double z0 = state->explicit_record_z[0];
                if (z0 == 0.0) {
                    should_capture_initial = 1;
                }
            }
        }
        if (should_capture_initial != 0) {
            status = snapshot_emit_record(state, 0u, field);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            captured_initial_record = 1;
        }
    }

    state->current_record_index = (captured_initial_record != 0) ? 1u : 0u;
    state->record_ring_flushed_count = state->current_record_index;

    return VEC_STATUS_OK;
}

vec_status simulation_state_download_current_field(const simulation_state* state, nlo_complex* out_field)
{
    if (state == NULL || out_field == NULL || state->backend == NULL || state->current_field_vec == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    return vec_download(state->backend,
                            state->current_field_vec,
                            out_field,
                            state->num_time_samples * sizeof(nlo_complex));
}

static vec_status snapshot_emit_record(simulation_state* state, size_t record_index, const nlo_complex* record)
{
    if (state == NULL || record == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    int recorded_anywhere = 0;
    nlo_complex* output_record = simulation_state_get_output_record(state, record_index);
    if (output_record != NULL) {
        if (output_record != record) {
            memcpy(output_record, record, state->num_time_samples * sizeof(nlo_complex));
        }
        recorded_anywhere = 1;
    }

    if (state->snapshot_store != NULL) {
        const snapshot_store_status store_status =
            snapshot_store_write_record(state->snapshot_store,
                                           record_index,
                                           record,
                                           state->num_time_samples);
        snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
        if (store_status == SNAPSHOT_STORE_STATUS_ERROR) {
            state->snapshot_status = VEC_STATUS_BACKEND_UNAVAILABLE;
            return state->snapshot_status;
        }
        recorded_anywhere = 1;
    }

    if (!recorded_anywhere) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    return VEC_STATUS_OK;
}

static nlo_complex* simulation_state_resolve_download_target(simulation_state* state, size_t record_index)
{
    nlo_complex* output_record = simulation_state_get_output_record(state, record_index);
    if (output_record != NULL) {
        return output_record;
    }

    return state->snapshot_scratch_record;
}

static vec_status simulation_state_flush_oldest_ring_entry(simulation_state* state)
{
    if (state == NULL || state->record_ring_size == 0u || state->record_ring_capacity == 0u) {
        return VEC_STATUS_OK;
    }

    if (state->record_ring_flushed_count >= state->num_recorded_samples) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_complex* download_target =
        simulation_state_resolve_download_target(state, state->record_ring_flushed_count);
    if (download_target == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t slot = state->record_ring_head;
    vec_buffer* src = state->record_ring_vec[slot];
    if (src == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    const bool resume_simulation = vec_is_in_simulation(state->backend);
    if (resume_simulation) {
        vec_status status = vec_end_simulation(state->backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }
    }

    vec_status status = vec_download(state->backend,
                                             src,
                                             download_target,
                                             state->num_time_samples * sizeof(nlo_complex));

    if (resume_simulation) {
        vec_status begin_status = vec_begin_simulation(state->backend);
        if (status == VEC_STATUS_OK) {
            status = begin_status;
        }
    }

    if (status != VEC_STATUS_OK) {
        return status;
    }

    status = snapshot_emit_record(state, state->record_ring_flushed_count, download_target);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    state->record_ring_head = (state->record_ring_head + 1u) % state->record_ring_capacity;
    state->record_ring_size -= 1u;
    state->record_ring_flushed_count += 1u;
    return VEC_STATUS_OK;
}

vec_status simulation_state_capture_snapshot_from_vec(
    simulation_state* state,
    const vec_buffer* source_vec
)
{
    if (state == NULL || state->backend == NULL || source_vec == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    if (state->current_record_index >= state->num_recorded_samples) {
        return VEC_STATUS_OK;
    }

    if (vector_backend_get_type(state->backend) == VECTOR_BACKEND_CPU) {
        const void* host_src = NULL;
        vec_status status = vec_get_const_host_ptr(state->backend,
                                                           source_vec,
                                                           &host_src);
        if (status != VEC_STATUS_OK || host_src == NULL) {
            return (status == VEC_STATUS_OK) ? VEC_STATUS_BACKEND_UNAVAILABLE : status;
        }

        status = snapshot_emit_record(state,
                                          state->current_record_index,
                                          (const nlo_complex*)host_src);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        state->current_record_index += 1u;
        return VEC_STATUS_OK;
    }

    if (state->record_ring_capacity == 0u) {
        nlo_complex* download_target =
            simulation_state_resolve_download_target(state, state->current_record_index);
        if (download_target == NULL) {
            return VEC_STATUS_INVALID_ARGUMENT;
        }

        const bool resume_simulation = vec_is_in_simulation(state->backend);
        if (resume_simulation) {
            vec_status status = vec_end_simulation(state->backend);
            if (status != VEC_STATUS_OK) {
                return status;
            }
        }

        vec_status status = vec_download(state->backend,
                                                 source_vec,
                                                 download_target,
                                                 state->num_time_samples * sizeof(nlo_complex));

        if (resume_simulation) {
            vec_status begin_status = vec_begin_simulation(state->backend);
            if (status == VEC_STATUS_OK) {
                status = begin_status;
            }
        }

        if (status != VEC_STATUS_OK) {
            return status;
        }

        status = snapshot_emit_record(state, state->current_record_index, download_target);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        state->current_record_index += 1u;
        state->record_ring_flushed_count += 1u;
        return VEC_STATUS_OK;
    }

    if (state->record_ring_size == state->record_ring_capacity) {
        vec_status flush_status = simulation_state_flush_oldest_ring_entry(state);
        if (flush_status != VEC_STATUS_OK) {
            return flush_status;
        }
    }

    const size_t slot = (state->record_ring_head + state->record_ring_size) % state->record_ring_capacity;
    vec_buffer* ring_dst = state->record_ring_vec[slot];
    if (ring_dst == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    vec_status copy_status = vec_complex_copy(state->backend, ring_dst, source_vec);
    if (copy_status != VEC_STATUS_OK) {
        return copy_status;
    }

    state->record_ring_size += 1u;
    state->current_record_index += 1u;
    return VEC_STATUS_OK;
}

vec_status simulation_state_capture_snapshot(simulation_state* state)
{
    if (state == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    return simulation_state_capture_snapshot_from_vec(state, state->current_field_vec);
}

vec_status simulation_state_flush_snapshots(simulation_state* state)
{
    if (state == NULL || state->backend == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    while (state->record_ring_size > 0u) {
        vec_status status = simulation_state_flush_oldest_ring_entry(state);
        if (status != VEC_STATUS_OK) {
            return status;
        }
    }

    if (state->snapshot_store != NULL) {
        const snapshot_store_status store_status = snapshot_store_flush(state->snapshot_store);
        snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
        if (store_status == SNAPSHOT_STORE_STATUS_ERROR) {
            state->snapshot_status = VEC_STATUS_BACKEND_UNAVAILABLE;
            return state->snapshot_status;
        }
    }

    return VEC_STATUS_OK;
}
