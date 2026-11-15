/**
 * @file nlolib.c
 * @dir src
 * @brief Public API entry points for NLOLib.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "nlolib.h"
#include "backend/nlo_complex.h"
#include "core/init_internal.h"
#include "core/sim_dimensions_internal.h"
#include "io/log_sink.h"
#include "io/propagate_log.h"
#include "io/run_store.h"
#include "io/snapshot_store.h"
#include "numerics/rk4_kernel.h"
#include "utility/perf_profile.h"
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#if defined(NLOLIB_HAVE_GIT_REVISION)
#include "nlolib_git_revision.h"
#endif
#ifndef NLOLIB_VERSION_STRING
#define NLOLIB_VERSION_STRING "unknown"
#endif
#ifndef NLOLIB_GIT_COMMIT
#define NLOLIB_GIT_COMMIT "nlolib-" NLOLIB_VERSION_STRING
#endif

static nlolib_status propagate_fail(const char* stage, nlolib_status status)
{
    log_emit(LOG_LEVEL_ERROR,
                 "[nlolib] propagate failed:\n"
                 "  - stage: %s\n"
                 "  - status: %d",
                 (stage != NULL) ? stage : "unknown",
                 (int)status);
    return status;
}

static nlolib_status map_log_status(int status)
{
    if (status == 0) {
        return NLOLIB_STATUS_OK;
    }
    if (status == 1) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    if (status == 2) {
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }
    if (status == 3) {
        return NLOLIB_STATUS_NOT_IMPLEMENTED;
    }

    return NLOLIB_STATUS_INVALID_ARGUMENT;
}

static size_t compute_input_bytes(size_t count, size_t stride)
{
    if (stride == 0u || count > (SIZE_MAX / stride)) {
        return 0u;
    }

    return count * stride;
}

static size_t compute_record_bytes(size_t num_recorded_samples, size_t num_time_samples)
{
    const size_t per_record_bytes = compute_input_bytes(num_time_samples, sizeof(nlo_complex));
    if (per_record_bytes == 0u || num_recorded_samples > (SIZE_MAX / per_record_bytes)) {
        return 0u;
    }

    return per_record_bytes * num_recorded_samples;
}

static int values_near(double lhs, double rhs)
{
    const double scale = fmax(1.0, fmax(fabs(lhs), fabs(rhs)));
    const double eps = 64.0 * DBL_EPSILON * scale;
    return (fabs(lhs - rhs) <= eps) ? 1 : 0;
}

static size_t compute_fixed_step_sample_count(double z_end, double step)
{
    if (!(z_end > 0.0) || !(step > 0.0) || !isfinite(z_end) || !isfinite(step)) {
        return 0u;
    }

    const double ratio = z_end / step;
    if (!isfinite(ratio) || ratio < 0.0) {
        return 0u;
    }

    const double ratio_floor = floor(ratio);
    size_t full_steps = 0u;
    if (ratio_floor > (double)(SIZE_MAX - 2u)) {
        return SIZE_MAX;
    }
    full_steps = (size_t)ratio_floor;

    size_t samples = full_steps + 1u;
    const double covered = (double)full_steps * step;
    if (!values_near(covered, z_end)) {
        if (samples == SIZE_MAX) {
            return SIZE_MAX;
        }
        samples += 1u;
    }

    return samples;
}

static int fixed_step_requested(const sim_config* config, double* out_step)
{
    if (config == NULL) {
        return 0;
    }

    const double start_step = config->propagation.starting_step_size;
    const double min_step = config->propagation.min_step_size;
    const double max_step = config->propagation.max_step_size;
    if (!(start_step > 0.0) || !(min_step > 0.0) || !(max_step > 0.0)) {
        return 0;
    }
    if (!(values_near(start_step, min_step) && values_near(start_step, max_step))) {
        return 0;
    }

    if (out_step != NULL) {
        *out_step = start_step;
    }
    return 1;
}

static size_t resolve_effective_record_count(
    const sim_config* config,
    size_t requested_records
)
{
    if (config == NULL || requested_records == 0u) {
        return requested_records;
    }

    double start_step = 0.0;
    if (!fixed_step_requested(config, &start_step)) {
        return requested_records;
    }

    const size_t available_samples =
        compute_fixed_step_sample_count(config->propagation.propagation_distance, start_step);
    if (available_samples == 0u || requested_records <= available_samples) {
        return requested_records;
    }

    log_emit(
        LOG_LEVEL_WARN,
        "[nlolib] fixed-step run requested %zu records, but only %zu step-aligned samples are available; "
        "reducing record count to %zu.",
        requested_records,
        available_samples,
        available_samples);
    return available_samples;
}

static int storage_enabled(const storage_options* storage_options)
{
    return (storage_options != NULL &&
            storage_options->sqlite_path != NULL &&
            storage_options->sqlite_path[0] != '\0');
}

static int storage_log_final_output_enabled(const storage_options* storage_options)
{
    return (storage_options != NULL && storage_options->log_final_output_field_to_db != 0);
}

/* A stored run streams into a staging database and is merged into the shared one at the end. */
typedef struct {
    int active;
    storage_options options;
    run_store_key_info key_info;
    char shared_path[RUN_STORE_PATH_MAX];
    char path[RUN_STORE_PATH_MAX];
    char run_id[STORAGE_RUN_ID_MAX];
} staging_run;

static void staging_discard(staging_run* staging)
{
    if (staging->active) {
        run_store_discard(staging->path);
        staging->active = 0;
    }
}

static int staging_begin(
    staging_run* staging,
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    const propagate_options* options
)
{
    memset(staging, 0, sizeof(*staging));
    if (!storage_enabled(options->storage_options)) {
        return 0;
    }
    staging->options = *options->storage_options;
    if (strlen(staging->options.sqlite_path) >= sizeof(staging->shared_path) ||
        run_store_compute_key(config,
                                  num_time_samples,
                                  input_field,
                                  num_recorded_samples,
                                  options->explicit_record_z,
                                  options->explicit_record_z_count,
                                  &staging->options,
                                  &staging->key_info) != 0) {
        return -1;
    }
    (void)snprintf(staging->shared_path, sizeof(staging->shared_path), "%s", staging->options.sqlite_path);
    if (run_store_make_staging_path(staging->shared_path,
                                        staging->key_info.key,
                                        staging->path,
                                        sizeof(staging->path)) != 0) {
        log_emit(LOG_LEVEL_ERROR,
                 "[nlolib][store] cannot create the staging folder next to %s",
                 staging->shared_path);
        return -1;
    }
    if (staging->options.run_id == NULL || staging->options.run_id[0] == '\0') {
        (void)snprintf(staging->run_id, sizeof(staging->run_id), "%s", staging->key_info.key);
        staging->options.run_id = staging->run_id;
    }
    staging->options.sqlite_path = staging->path;
    staging->active = 1;
    return 0;
}

static int staging_finalize(staging_run* staging, const simulation_state* state)
{
    char index_file[RUN_STORE_PATH_MAX];
    run_store_index_file(staging->shared_path, index_file, sizeof(index_file));
    const vector_backend_type used = vector_backend_get_type(state->backend);
    snapshot_store_finalize_params params;
    memset(&params, 0, sizeof(params));
    params.key = staging->key_info.key;
    params.index_file = index_file;
    params.git_commit = NLOLIB_GIT_COMMIT;
    params.binding = staging->options.binding;
    params.version = NLOLIB_VERSION_STRING;
    params.backend = (used == VECTOR_BACKEND_VULKAN) ? "vulkan" : "cpu";
    params.input_sha256 = staging->key_info.input_sha256;
    params.potential_sha256 = staging->key_info.potential_sha256;
    return (snapshot_store_finalize(state->snapshot_store, &params) == SNAPSHOT_STORE_STATUS_ERROR) ? -1 : 0;
}

static nlolib_status propagate_fail_state(
    simulation_state* state,
    staging_run* staging,
    const char* stage,
    nlolib_status status
)
{
    free_simulation_state(state);
    staging_discard(staging);
    return propagate_fail(stage, status);
}

static int validate_explicit_record_schedule(
    const double* z_values,
    size_t count,
    double z_end
)
{
    if (z_values == NULL) {
        return (count == 0u) ? 0 : -1;
    }
    if (count == 0u) {
        return 0;
    }

    double prev = z_values[0];
    if (!isfinite(prev) || prev < 0.0 || prev > z_end) {
        return -1;
    }
    for (size_t i = 1u; i < count; ++i) {
        const double current = z_values[i];
        if (!isfinite(current) || current < 0.0 || current > z_end) {
            return -1;
        }
        if (current < prev) {
            return -1;
        }
        prev = current;
    }
    return 0;
}

static sim_config merge_simulation_and_physics(
    const simulation_config* simulation_config,
    const physics_config* physics_config
)
{
    sim_config merged;
    memset(&merged, 0, sizeof(merged));
    if (simulation_config != NULL) {
        merged.propagation = simulation_config->propagation;
        merged.tensor = simulation_config->tensor;
        merged.time = simulation_config->time;
        merged.frequency = simulation_config->frequency;
        merged.spatial = simulation_config->spatial;
    }
    if (physics_config != NULL) {
        merged.runtime = *physics_config;
    }
    return merged;
}

NLOLIB_API nlolib_status nlolib_perf_profile_set_enabled(int enabled)
{
    perf_profile_set_enabled(enabled);
    return NLOLIB_STATUS_OK;
}

NLOLIB_API int nlolib_perf_profile_is_enabled(void)
{
    return perf_profile_is_enabled();
}

NLOLIB_API nlolib_status nlolib_perf_profile_reset(void)
{
    perf_profile_reset();
    return NLOLIB_STATUS_OK;
}

NLOLIB_API nlolib_status nlolib_perf_profile_read(nlo_perf_profile_snapshot* out_snapshot)
{
    if (out_snapshot == NULL) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    perf_profile_snapshot_read(out_snapshot);
    return NLOLIB_STATUS_OK;
}

NLOLIB_API nlolib_status nlolib_perf_profile_read_gpu_phases(
    double* out_phase_ms,
    uint64_t* out_phase_intervals,
    double* out_host_slots
)
{
    perf_profile_gpu_phases_read(out_phase_ms, out_phase_intervals, out_host_slots);
    return NLOLIB_STATUS_OK;
}

NLOLIB_API nlolib_status nlolib_query_runtime_limits(
    const simulation_config* simulation_config,
    const physics_config* physics_config,
    const execution_options* exec_options,
    runtime_limits* out_limits
)
{
    if (out_limits == NULL) {
        return propagate_fail("validate.runtime_limits.null_out", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    const sim_config merged = merge_simulation_and_physics(simulation_config, physics_config);
    if (query_runtime_limits_internal(&merged, exec_options, out_limits) != 0) {
        return propagate_fail("query_runtime_limits_internal", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    return NLOLIB_STATUS_OK;
}

NLOLIB_API propagate_options nlolib_propagate_options_default(void)
{
    propagate_options options;
    options.num_recorded_samples = 2u;
    options.output_mode = PROPAGATE_OUTPUT_DENSE;
    options.return_records = 1;
    options.exec_options = NULL;
    options.storage_options = NULL;
    options.explicit_record_z = NULL;
    options.explicit_record_z_count = 0u;
    options.progress_callback = NULL;
    options.progress_user_data = NULL;
    return options;
}

NLOLIB_API propagate_output nlolib_propagate_output_default(void)
{
    propagate_output output;
    output.output_records = NULL;
    output.output_record_capacity = 0u;
    output.records_written = NULL;
    output.storage_result = NULL;
    output.output_step_events = NULL;
    output.output_step_event_capacity = 0u;
    output.step_events_written = NULL;
    output.step_events_dropped = NULL;
    return output;
}

NLOLIB_API nlolib_status nlolib_propagate(
    const simulation_config* simulation_config,
    const physics_config* physics_config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    const propagate_options* options,
    propagate_output* output
)
{
    const sim_config merged = merge_simulation_and_physics(simulation_config, physics_config);
    const sim_config* config = &merged;
    const propagate_options local_options =
        (options != NULL) ? *options : nlolib_propagate_options_default();
    propagate_output local_output = (output != NULL) ? *output : nlolib_propagate_output_default();

    size_t num_recorded_samples = local_options.num_recorded_samples;
    if (local_options.output_mode == PROPAGATE_OUTPUT_FINAL_ONLY) {
        num_recorded_samples = 1u;
    }
    if (local_options.explicit_record_z != NULL && local_options.explicit_record_z_count > 0u) {
        num_recorded_samples = local_options.explicit_record_z_count;
    }
    num_recorded_samples = resolve_effective_record_count(config, num_recorded_samples);
    if (num_recorded_samples == 0u) {
        return propagate_fail("validate.num_recorded_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (local_options.explicit_record_z != NULL && local_options.explicit_record_z_count > 0u) {
        if (validate_explicit_record_schedule(local_options.explicit_record_z,
                                                  local_options.explicit_record_z_count,
                                                  config->propagation.propagation_distance) != 0) {
            return propagate_fail("validate.explicit_record_z", NLOLIB_STATUS_INVALID_ARGUMENT);
        }
    }

    if (local_options.return_records == 0) {
        local_output.output_records = NULL;
        local_output.output_record_capacity = 0u;
    }
    if (local_output.output_step_events == NULL) {
        local_output.output_step_event_capacity = 0u;
    }

    if (local_output.records_written != NULL) {
        *local_output.records_written = 0u;
    }
    if (local_output.storage_result != NULL) {
        *local_output.storage_result = (storage_result){0};
    }
    if (local_output.step_events_written != NULL) {
        *local_output.step_events_written = 0u;
    }
    if (local_output.step_events_dropped != NULL) {
        *local_output.step_events_dropped = 0u;
    }

    log_propagate_request(config,
                              num_time_samples,
                              input_field,
                              num_recorded_samples,
                              local_output.output_records,
                              local_options.exec_options);

    if (config == NULL || input_field == NULL) {
        return propagate_fail("validate.null_pointer", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (local_options.return_records == 0 && !storage_enabled(local_options.storage_options)) {
        return propagate_fail("validate.output_or_storage", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (local_options.return_records != 0 &&
        local_output.output_records == NULL &&
        !storage_enabled(local_options.storage_options)) {
        return propagate_fail("validate.output_or_storage", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (local_output.output_step_events == NULL && local_output.output_step_event_capacity > 0u) {
        return propagate_fail("validate.output_step_event_capacity", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    if (num_time_samples == 0u) {
        return propagate_fail("validate.num_time_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (num_recorded_samples == 0u) {
        return propagate_fail("validate.num_recorded_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (num_recorded_samples >
        runtime_limits_default().max_num_recorded_samples_with_storage) {
        return propagate_fail("validate.num_recorded_samples_precision", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (local_options.return_records != 0) {
        if (local_output.output_records != NULL &&
            local_output.output_record_capacity < num_recorded_samples) {
            return propagate_fail("validate.output_record_capacity", NLOLIB_STATUS_INVALID_ARGUMENT);
        }
        if (local_output.output_records != NULL) {
            const size_t record_bytes = compute_record_bytes(num_recorded_samples, num_time_samples);
            if (record_bytes == 0u) {
                return propagate_fail("validate.record_bytes", NLOLIB_STATUS_INVALID_ARGUMENT);
            }
            /* The output buffer and the input field are allocations the caller
             * has to make between querying the limits and this call. Whatever
             * part of them is already resident has left the available-memory
             * reading, so it is credited back; otherwise a count the query
             * just reported could be refused here. */
            const size_t output_resident =
                query_resident_bytes(local_output.output_records, record_bytes);
            const size_t input_resident =
                query_resident_bytes(input_field, compute_input_bytes(num_time_samples, sizeof(nlo_complex)));
            dense_record_limit dense_limit;
            if (resolve_dense_record_limit(config,
                                           num_time_samples,
                                           (output_resident > SIZE_MAX - input_resident)
                                               ? SIZE_MAX
                                               : (output_resident + input_resident),
                                           &dense_limit) != 0) {
                return propagate_fail("validate.record_bytes", NLOLIB_STATUS_INVALID_ARGUMENT);
            }
            if (dense_limit.derivable != 0 && num_recorded_samples > dense_limit.max_records) {
                log_emit(
                    LOG_LEVEL_ERROR,
                    "[nlolib] dense output request requires %zu bytes (%zu records of %zu bytes), above the "
                    "current limit of %zu bytes (%zu records): %zu bytes of host memory are budgeted and "
                    "%zu of them are reserved for the solver working set. "
                    "Use storage_options with return_records=0 for large recorded outputs.",
                    record_bytes,
                    num_recorded_samples,
                    dense_limit.bytes_per_record,
                    dense_limit.max_records * dense_limit.bytes_per_record,
                    dense_limit.max_records,
                    dense_limit.host_budget_bytes,
                    dense_limit.working_set_bytes);
                return propagate_fail("validate.dense_output_bytes", NLOLIB_STATUS_INVALID_ARGUMENT);
            }
        }
    }
    if (storage_enabled(local_options.storage_options) && !snapshot_store_is_available()) {
        return propagate_fail("validate.storage_unavailable", NLOLIB_STATUS_NOT_IMPLEMENTED);
    }
    {
        size_t nt = 0u;
        size_t nx = 0u;
        size_t ny = 0u;
        int explicit_nd = 0;
        if (resolve_sim_dimensions_internal(config,
                                                num_time_samples,
                                                &nt,
                                                &nx,
                                                &ny,
                                                &explicit_nd) != 0) {
            return propagate_fail("validate.spatial_dimensions", NLOLIB_STATUS_INVALID_ARGUMENT);
        }
    }

    execution_options local_exec_options =
        (local_options.exec_options != NULL)
            ? *local_options.exec_options
            : execution_options_default(VECTOR_BACKEND_AUTO);
    staging_run staging;
    if (staging_begin(&staging,
                      config,
                      num_time_samples,
                      input_field,
                      num_recorded_samples,
                      &local_options) != 0) {
        return propagate_fail("storage.staging", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    simulation_state* state = NULL;
    allocation_info allocation_info = {0};
    const double assess_init_start_ms = perf_profile_now_ms();
    const int init_status =
        storage_enabled(local_options.storage_options)
            ? init_simulation_state_with_storage(config,
                                                     num_time_samples,
                                                     num_recorded_samples,
                                                     &local_exec_options,
                                                     &staging.options,
                                                     &allocation_info,
                                                     &state)
            : init_simulation_state(config,
                                        num_time_samples,
                                        num_recorded_samples,
                                        &local_exec_options,
                                        &allocation_info,
                                        &state);
    if (init_status != 0 || state == NULL) {
        staging_discard(&staging);
        return propagate_fail("init_simulation_state", NLOLIB_STATUS_ALLOCATION_FAILED);
    }
    perf_profile_add_host_slot(PERF_HOST_SLOT_INIT_MS, perf_profile_now_ms() - assess_init_start_ms);
    if (state->snapshot_store != NULL) {
        snapshot_store_set_record_schedule(state->snapshot_store,
                                               local_options.explicit_record_z,
                                               local_options.explicit_record_z_count);
    }
    state->output_records = local_output.output_records;
    state->output_record_capacity = local_output.output_record_capacity;
    state->explicit_record_z = local_options.explicit_record_z;
    state->explicit_record_z_count = local_options.explicit_record_z_count;
    state->explicit_record_schedule_active =
        (local_options.explicit_record_z != NULL && local_options.explicit_record_z_count > 0u) ? 1 : 0;

    vec_backend_memory_info mem_info = {0};
    (void)vec_query_memory_info(state->backend, &mem_info);
    log_propagate_allocation_summary(local_exec_options.backend_type,
                                         vector_backend_get_type(state->backend),
                                         &allocation_info,
                                         &mem_info);

    if (simulation_state_upload_initial_field(state, input_field) != VEC_STATUS_OK) {
        return propagate_fail_state(state, &staging, "upload_initial_field", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    state->step_event_buffer = local_output.output_step_events;
    state->step_event_capacity = local_output.output_step_event_capacity;
    state->step_events_written = 0u;
    state->step_events_dropped = 0u;

    (void)log_set_progress_callback(local_options.progress_callback, local_options.progress_user_data);
    const double assess_solve_start_ms = perf_profile_now_ms();
    const int assess_prev_phase = perf_profile_set_gpu_phase(PERF_GPU_PHASE_RK_ALGEBRA);
    solve_rk4(state);
    (void)perf_profile_set_gpu_phase(assess_prev_phase);
    perf_profile_add_host_slot(PERF_HOST_SLOT_SOLVE_MS, perf_profile_now_ms() - assess_solve_start_ms);
    const int progress_aborted = log_progress_abort_requested();
    (void)log_set_progress_callback(NULL, NULL);

    if (state->snapshot_status != VEC_STATUS_OK) {
        return propagate_fail_state(state, &staging, "snapshot_capture", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    size_t records_available = num_recorded_samples;
    if (num_recorded_samples > 1u) {
        if (progress_aborted != 0 ||
            state->explicit_record_schedule_active != 0 ||
            fixed_step_requested(config, NULL)) {
            records_available = state->current_record_index;
            if (records_available > num_recorded_samples) {
                records_available = num_recorded_samples;
            }
            if (progress_aborted == 0 && records_available < num_recorded_samples) {
                log_emit(
                    LOG_LEVEL_WARN,
                    "[nlolib] record capture completed with %zu/%zu records.",
                    records_available,
                    num_recorded_samples);
            }
        }
    }

    int final_downloaded = 0;
    const nlo_complex* final_output_field_cached = NULL;
    if (num_recorded_samples == 1u && state->snapshot_store != NULL) {
        nlo_complex* final_record = local_output.output_records;
        if (final_record == NULL) {
            final_record = state->snapshot_scratch_record;
        }
        if (final_record == NULL) {
            return propagate_fail_state(state, &staging, "storage.final_record_buffer", NLOLIB_STATUS_ALLOCATION_FAILED);
        }

        if (simulation_state_download_current_field(state, final_record) != VEC_STATUS_OK) {
            return propagate_fail_state(state, &staging, "storage.download_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        final_output_field_cached = final_record;
        final_downloaded = (final_record == local_output.output_records) ? 1 : 0;

        const snapshot_store_status write_status =
            snapshot_store_write_record(state->snapshot_store,
                                            0u,
                                            final_record,
                                            state->num_time_samples);
        if (write_status == SNAPSHOT_STORE_STATUS_ERROR) {
            return propagate_fail_state(state, &staging, "storage.write_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        if (snapshot_store_flush(state->snapshot_store) == SNAPSHOT_STORE_STATUS_ERROR) {
            return propagate_fail_state(state, &staging, "storage.flush_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
    }

    if (local_options.return_records != 0 &&
        local_output.output_records != NULL &&
        num_recorded_samples == 1u) {
        if (!final_downloaded &&
            simulation_state_download_current_field(state, local_output.output_records) != VEC_STATUS_OK) {
            return propagate_fail_state(state, &staging, "download_current_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        final_output_field_cached = local_output.output_records;
    } else if (local_options.return_records != 0 && local_output.output_records != NULL) {
        if (records_available > 0u) {
            final_output_field_cached =
                local_output.output_records + ((records_available - 1u) * state->num_time_samples);
        }
    }

    if (state->snapshot_store != NULL &&
        storage_log_final_output_enabled(local_options.storage_options)) {
        const nlo_complex* final_output_field = final_output_field_cached;

        nlo_complex* final_output_scratch = NULL;
        if (final_output_field == NULL) {
            final_output_scratch = state->snapshot_scratch_record;
            if (final_output_scratch == NULL) {
                return propagate_fail_state(state, &staging, "storage.final_output_buffer", NLOLIB_STATUS_ALLOCATION_FAILED);
            }
            if (simulation_state_download_current_field(state, final_output_scratch) != VEC_STATUS_OK) {
                return propagate_fail_state(state, &staging, "storage.download_output_field", NLOLIB_STATUS_ALLOCATION_FAILED);
            }
            final_output_field = final_output_scratch;
        }

        if (snapshot_store_write_final_output_field(state->snapshot_store,
                                                        final_output_field,
                                                        state->num_time_samples) ==
            SNAPSHOT_STORE_STATUS_ERROR) {
            return propagate_fail_state(state, &staging, "storage.write_output_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
    }

    storage_result stored = {0};
    if (state->snapshot_store != NULL) {
        if (progress_aborted == 0 && staging_finalize(&staging, state) != 0) {
            return propagate_fail_state(state, &staging, "storage.finalize", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        snapshot_store_get_result(state->snapshot_store, &stored);
    }

    const size_t step_events_written = state->step_events_written;
    const size_t step_events_dropped = state->step_events_dropped;
    size_t records_written_actual = 0u;
    if (local_options.return_records != 0) {
        records_written_actual = records_available;
    }
    free_simulation_state(state);

    if (staging.active) {
        if (progress_aborted != 0) {
            /* An interrupted run never reaches the shared database. */
            staging_discard(&staging);
        } else {
            stored.merged = (run_store_merge(staging.path, staging.shared_path) == RUN_STORE_OK) ? 1 : 0;
            if (!stored.merged) {
                log_emit(LOG_LEVEL_WARN,
                         "[nlolib][store] run %s is complete in %s but not merged into %s; "
                         "merge it later with nlolib_storage_recover_staging().",
                         stored.key,
                         staging.path,
                         staging.shared_path);
            }
        }
    }
    if (local_output.storage_result != NULL) {
        *local_output.storage_result = stored;
    }
    if (local_output.records_written != NULL) {
        *local_output.records_written = records_written_actual;
    }
    if (local_output.step_events_written != NULL) {
        *local_output.step_events_written = step_events_written;
    }
    if (local_output.step_events_dropped != NULL) {
        *local_output.step_events_dropped = step_events_dropped;
    }
    const nlolib_status final_status = (progress_aborted != 0) ? NLOLIB_STATUS_ABORTED : NLOLIB_STATUS_OK;
    const char* final_message = (progress_aborted != 0) ? "propagate aborted by progress callback"
                                                        : "propagate completed";
    log_emit(LOG_LEVEL_INFO,
                 "[nlolib] solver_status:\n"
                 "  - status: %d\n"
                 "  - message: %s",
                 (int)final_status,
                 final_message);
    return final_status;
}

NLOLIB_API int nlolib_storage_is_available(void)
{
    return snapshot_store_is_available();
}

NLOLIB_API nlolib_status nlolib_set_log_file(const char* path_utf8, int append)
{
    return map_log_status(log_set_file(path_utf8, append));
}

NLOLIB_API nlolib_status nlolib_set_log_buffer(size_t capacity_bytes)
{
    return map_log_status(log_set_buffer(capacity_bytes));
}

NLOLIB_API nlolib_status nlolib_clear_log_buffer(void)
{
    return map_log_status(log_clear_buffer());
}

NLOLIB_API nlolib_status nlolib_read_log_buffer(
    char* dst,
    size_t dst_bytes,
    size_t* out_written,
    int consume
)
{
    return map_log_status(log_read_buffer(dst, dst_bytes, out_written, consume));
}

NLOLIB_API nlolib_status nlolib_set_log_level(int level)
{
    return map_log_status(log_set_level(level));
}

NLOLIB_API nlolib_status nlolib_set_progress_options(
    int enabled,
    int milestone_percent,
    int emit_on_step_adjust
)
{
    return map_log_status(
        log_set_progress_options(enabled, milestone_percent, emit_on_step_adjust));
}

NLOLIB_API nlolib_status nlolib_set_progress_stream(int stream_mode)
{
    return map_log_status(log_set_progress_stream(stream_mode));
}

static nlolib_status map_store_status(run_store_status status)
{
    switch (status) {
        case RUN_STORE_OK:
            return NLOLIB_STATUS_OK;
        case RUN_STORE_LOCK_TIMEOUT:
        case RUN_STORE_INCOMPLETE:
        case RUN_STORE_ERROR:
        default:
            return snapshot_store_is_available() ? NLOLIB_STATUS_ALLOCATION_FAILED : NLOLIB_STATUS_NOT_IMPLEMENTED;
    }
}

static int is_run_key(const char* key)
{
    if (key == NULL || strlen(key) != 16u) {
        return 0;
    }
    for (size_t i = 0u; i < 16u; ++i) {
        const char c = key[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return 0;
        }
    }
    return 1;
}

NLOLIB_API nlolib_status nlolib_storage_run_key(
    const simulation_config* simulation_config,
    const physics_config* physics_config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    const propagate_options* options,
    char* out_key
)
{
    if (simulation_config == NULL || input_field == NULL || out_key == NULL || num_time_samples == 0u) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    out_key[0] = '\0';
    const sim_config merged = merge_simulation_and_physics(simulation_config, physics_config);
    const propagate_options local_options =
        (options != NULL) ? *options : nlolib_propagate_options_default();
    size_t num_recorded_samples = local_options.num_recorded_samples;
    if (local_options.output_mode == PROPAGATE_OUTPUT_FINAL_ONLY) {
        num_recorded_samples = 1u;
    }
    if (local_options.explicit_record_z != NULL && local_options.explicit_record_z_count > 0u) {
        num_recorded_samples = local_options.explicit_record_z_count;
    }
    num_recorded_samples = resolve_effective_record_count(&merged, num_recorded_samples);
    if (num_recorded_samples == 0u) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    run_store_key_info info;
    if (run_store_compute_key(&merged,
                                  num_time_samples,
                                  input_field,
                                  num_recorded_samples,
                                  local_options.explicit_record_z,
                                  local_options.explicit_record_z_count,
                                  local_options.storage_options,
                                  &info) != 0) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    memcpy(out_key, info.key, STORAGE_KEY_MAX);
    return NLOLIB_STATUS_OK;
}

NLOLIB_API nlolib_status nlolib_storage_has_run(const char* sqlite_path, const char* key, int* out_present)
{
    if (out_present != NULL) {
        *out_present = 0;
    }
    if (sqlite_path == NULL || sqlite_path[0] == '\0' || out_present == NULL || !is_run_key(key)) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    return map_store_status(run_store_has_key(sqlite_path, key, out_present));
}

NLOLIB_API nlolib_status nlolib_storage_recover_staging(
    const char* sqlite_path,
    int discard_incomplete,
    double min_age_seconds,
    size_t* out_merged,
    size_t* out_discarded,
    size_t* out_skipped
)
{
    if (sqlite_path == NULL || sqlite_path[0] == '\0') {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    return map_store_status(run_store_recover(sqlite_path,
                                                  discard_incomplete,
                                                  min_age_seconds,
                                                  out_merged,
                                                  out_discarded,
                                                  out_skipped));
}

NLOLIB_API nlolib_status nlolib_storage_read_run_layout(
    const char* sqlite_path,
    const char* key,
    storage_run_layout* out_layout
)
{
    if (sqlite_path == NULL || out_layout == NULL || !is_run_key(key)) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    return map_store_status(run_store_read_layout(sqlite_path, key, out_layout));
}

NLOLIB_API nlolib_status nlolib_storage_read_run_array(
    const char* sqlite_path,
    const char* key,
    const char* name,
    double* out_values,
    size_t capacity,
    size_t* out_count
)
{
    if (sqlite_path == NULL || name == NULL || out_count == NULL || !is_run_key(key)) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    return map_store_status(run_store_read_array(sqlite_path, key, name, out_values, capacity, out_count));
}
