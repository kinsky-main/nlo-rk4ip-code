/**
 * @file state_snapshots.c
 * @brief Snapshot capture/upload/download runtime state operations.
 */

#include "core/state.h"
#include "io/snapshot_store.h"
#include <stdbool.h>
#include <string.h>

static nlo_vec_status nlo_snapshot_emit_record(simulation_state* state, size_t record_index, const nlo_complex* record);
static nlo_vec_status simulation_state_flush_oldest_ring_entry(simulation_state* state);

nlo_vec_status simulation_state_upload_initial_field(simulation_state* state, const nlo_complex* field)
{
    if (state == NULL || field == NULL || state->backend == NULL || state->current_field_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t bytes = state->num_time_samples * sizeof(nlo_complex);
    nlo_vec_status status = nlo_vec_upload(state->backend, state->current_field_vec, field, bytes);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (state->num_recorded_samples > 1u) {
        status = nlo_snapshot_emit_record(state, 0u, field);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    state->current_record_index = (state->num_recorded_samples > 1u) ? 1u : 0u;
    state->record_ring_flushed_count = state->current_record_index;

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status simulation_state_download_current_field(const simulation_state* state, nlo_complex* out_field)
{
    if (state == NULL || out_field == NULL || state->backend == NULL || state->current_field_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    return nlo_vec_download(state->backend,
                            state->current_field_vec,
                            out_field,
                            state->num_time_samples * sizeof(nlo_complex));
}

static nlo_vec_status nlo_snapshot_emit_record(simulation_state* state, size_t record_index, const nlo_complex* record)
{
    if (state == NULL || record == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    int recorded_anywhere = 0;
    if (record_index < state->num_host_records) {
        nlo_complex* host_record = simulation_state_get_field_record(state, record_index);
        if (host_record == NULL) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }
        if (host_record != record) {
            memcpy(host_record, record, state->num_time_samples * sizeof(nlo_complex));
        }
        recorded_anywhere = 1;
    }

    if (state->snapshot_store != NULL) {
        const nlo_snapshot_store_status store_status =
            nlo_snapshot_store_write_record(state->snapshot_store,
                                           record_index,
                                           record,
                                           state->num_time_samples);
        nlo_snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
        if (store_status == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            state->snapshot_status = NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
            return state->snapshot_status;
        }
        recorded_anywhere = 1;
    }

    if (!recorded_anywhere) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status simulation_state_flush_oldest_ring_entry(simulation_state* state)
{
    if (state == NULL || state->record_ring_size == 0u || state->record_ring_capacity == 0u) {
        return NLO_VEC_STATUS_OK;
    }

    if (state->record_ring_flushed_count >= state->num_recorded_samples) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_complex* host_record = simulation_state_get_field_record(state, state->record_ring_flushed_count);
    nlo_complex* download_target = host_record;
    if (download_target == NULL) {
        download_target = state->snapshot_scratch_record;
    }
    if (download_target == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t slot = state->record_ring_head;
    nlo_vec_buffer* src = state->record_ring_vec[slot];
    if (src == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const bool resume_simulation = nlo_vec_is_in_simulation(state->backend);
    if (resume_simulation) {
        nlo_vec_status status = nlo_vec_end_simulation(state->backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    nlo_vec_status status = nlo_vec_download(state->backend,
                                             src,
                                             download_target,
                                             state->num_time_samples * sizeof(nlo_complex));

    if (resume_simulation) {
        nlo_vec_status begin_status = nlo_vec_begin_simulation(state->backend);
        if (status == NLO_VEC_STATUS_OK) {
            status = begin_status;
        }
    }

    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_snapshot_emit_record(state, state->record_ring_flushed_count, download_target);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    state->record_ring_head = (state->record_ring_head + 1u) % state->record_ring_capacity;
    state->record_ring_size -= 1u;
    state->record_ring_flushed_count += 1u;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status simulation_state_capture_snapshot(simulation_state* state)
{
    if (state == NULL || state->backend == NULL || state->current_field_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (state->current_record_index >= state->num_recorded_samples) {
        return NLO_VEC_STATUS_OK;
    }

    if (nlo_vector_backend_get_type(state->backend) == NLO_VECTOR_BACKEND_CPU) {
        const void* host_src = NULL;
        nlo_vec_status status = nlo_vec_get_const_host_ptr(state->backend,
                                                           state->current_field_vec,
                                                           &host_src);
        if (status != NLO_VEC_STATUS_OK || host_src == NULL) {
            return (status == NLO_VEC_STATUS_OK) ? NLO_VEC_STATUS_BACKEND_UNAVAILABLE : status;
        }

        status = nlo_snapshot_emit_record(state,
                                          state->current_record_index,
                                          (const nlo_complex*)host_src);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        state->current_record_index += 1u;
        return NLO_VEC_STATUS_OK;
    }

    if (state->record_ring_capacity == 0u) {
        nlo_complex* host_record = simulation_state_get_field_record(state, state->current_record_index);
        nlo_complex* download_target = host_record;
        if (download_target == NULL) {
            download_target = state->snapshot_scratch_record;
        }
        if (download_target == NULL) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        const bool resume_simulation = nlo_vec_is_in_simulation(state->backend);
        if (resume_simulation) {
            nlo_vec_status status = nlo_vec_end_simulation(state->backend);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
        }

        nlo_vec_status status = nlo_vec_download(state->backend,
                                                 state->current_field_vec,
                                                 download_target,
                                                 state->num_time_samples * sizeof(nlo_complex));

        if (resume_simulation) {
            nlo_vec_status begin_status = nlo_vec_begin_simulation(state->backend);
            if (status == NLO_VEC_STATUS_OK) {
                status = begin_status;
            }
        }

        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        status = nlo_snapshot_emit_record(state, state->current_record_index, download_target);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        state->current_record_index += 1u;
        state->record_ring_flushed_count += 1u;
        return NLO_VEC_STATUS_OK;
    }

    if (state->record_ring_size == state->record_ring_capacity) {
        nlo_vec_status flush_status = simulation_state_flush_oldest_ring_entry(state);
        if (flush_status != NLO_VEC_STATUS_OK) {
            return flush_status;
        }
    }

    const size_t slot = (state->record_ring_head + state->record_ring_size) % state->record_ring_capacity;
    nlo_vec_buffer* ring_dst = state->record_ring_vec[slot];
    if (ring_dst == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status copy_status = nlo_vec_complex_copy(state->backend, ring_dst, state->current_field_vec);
    if (copy_status != NLO_VEC_STATUS_OK) {
        return copy_status;
    }

    state->record_ring_size += 1u;
    state->current_record_index += 1u;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status simulation_state_flush_snapshots(simulation_state* state)
{
    if (state == NULL || state->backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    while (state->record_ring_size > 0u) {
        nlo_vec_status status = simulation_state_flush_oldest_ring_entry(state);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    if (state->snapshot_store != NULL) {
        const nlo_snapshot_store_status store_status = nlo_snapshot_store_flush(state->snapshot_store);
        nlo_snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
        if (store_status == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            state->snapshot_status = NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
            return state->snapshot_status;
        }
    }

    return NLO_VEC_STATUS_OK;
}
