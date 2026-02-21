/**
 * @file nlolib.c
 * @dir src
 * @brief Public API entry points for NLOLib.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "nlolib.h"
#include "backend/nlo_complex.h"
#include "core/sim_dimensions_internal.h"
#include "io/snapshot_store.h"
#include "numerics/rk4_kernel.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static const char* nlo_backend_type_to_string(nlo_vector_backend_type backend_type)
{
    if (backend_type == NLO_VECTOR_BACKEND_CPU) {
        return "CPU";
    }
    if (backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        return "VULKAN";
    }
    if (backend_type == NLO_VECTOR_BACKEND_AUTO) {
        return "AUTO";
    }

    return "UNKNOWN";
}

static nlolib_status nlo_propagate_fail(const char* stage, nlolib_status status)
{
    fprintf(stderr,
            "[nlolib] nlse propagate failed stage=%s status=%d\n",
            (stage != NULL) ? stage : "unknown",
            (int)status);
    return status;
}

static size_t nlo_compute_input_bytes(size_t count, size_t stride)
{
    if (stride == 0u || count > (SIZE_MAX / stride)) {
        return 0u;
    }

    return count * stride;
}

static size_t nlo_compute_record_bytes(size_t num_recorded_samples, size_t num_time_samples)
{
    const size_t per_record_bytes = nlo_compute_input_bytes(num_time_samples, sizeof(nlo_complex));
    if (per_record_bytes == 0u || num_recorded_samples > (SIZE_MAX / per_record_bytes)) {
        return 0u;
    }

    return per_record_bytes * num_recorded_samples;
}

static int nlo_storage_enabled(const nlo_storage_options* storage_options)
{
    return (storage_options != NULL &&
            storage_options->sqlite_path != NULL &&
            storage_options->sqlite_path[0] != '\0');
}

NLOLIB_API nlolib_status nlolib_query_runtime_limits(
    const sim_config* config,
    const nlo_execution_options* exec_options,
    nlo_runtime_limits* out_limits
)
{
    if (out_limits == NULL) {
        return nlo_propagate_fail("validate.runtime_limits.null_out", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    if (nlo_query_runtime_limits_internal(config, exec_options, out_limits) != 0) {
        return nlo_propagate_fail("query_runtime_limits_internal", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    return NLOLIB_STATUS_OK;
}

static void nlo_log_nlse_propagate_call(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options
)
{
    const nlo_execution_options local_exec_options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_AUTO);

    const size_t field_bytes = nlo_compute_input_bytes(num_time_samples, sizeof(nlo_complex));
    const size_t records_bytes = nlo_compute_record_bytes(num_recorded_samples, num_time_samples);
    const size_t frequency_grid_bytes = nlo_compute_input_bytes(num_time_samples, sizeof(nlo_complex));
    size_t nt = 0u;
    size_t nx = 0u;
    size_t ny = 0u;
    int explicit_nd = 0;
    const int has_spatial_shape =
        (nlo_resolve_sim_dimensions_internal(config, num_time_samples, &nt, &nx, &ny, &explicit_nd) == 0);

    fprintf(stderr,
            "[nlolib] nlse propagate backend=%s | "
            "num_time_samples=%zu num_recorded_samples=%zu field_bytes=%zu record_bytes=%zu | "
            "config=%p(size=%zu) input_field=%p(size=%zu) output_records=%p(size=%zu) "
            "exec_options=%p(size=%zu) | "
            "frequency_grid=%p(size=%zu) runtime(constants=%zu df=%p d=%p tf=%p t=%p n=%p) | "
            "spatial(nt=%zu nx=%zu ny=%zu explicit_nd=%d valid=%d dx=%.9e dy=%.9e spatial_frequency_grid=%p potential_grid=%p)\n",
            nlo_backend_type_to_string(local_exec_options.backend_type),
            num_time_samples,
            num_recorded_samples,
            field_bytes,
            records_bytes,
            (const void*)config,
            sizeof(sim_config),
            (const void*)input_field,
            field_bytes,
            (const void*)output_records,
            records_bytes,
            (const void*)exec_options,
            sizeof(nlo_execution_options),
            (config != NULL) ? (const void*)config->frequency.frequency_grid : NULL,
            frequency_grid_bytes,
            (config != NULL) ? config->runtime.num_constants : 0u,
            (config != NULL) ? (const void*)config->runtime.dispersion_factor_expr : NULL,
            (config != NULL) ? (const void*)config->runtime.dispersion_expr : NULL,
            (config != NULL) ? (const void*)config->runtime.transverse_factor_expr : NULL,
            (config != NULL) ? (const void*)config->runtime.transverse_expr : NULL,
            (config != NULL) ? (const void*)config->runtime.nonlinear_expr : NULL,
            nt,
            nx,
            ny,
            explicit_nd,
            has_spatial_shape,
            (config != NULL) ? config->spatial.delta_x : 0.0,
            (config != NULL) ? config->spatial.delta_y : 0.0,
            (config != NULL) ? (const void*)config->spatial.spatial_frequency_grid : NULL,
            (config != NULL) ? (const void*)config->spatial.potential_grid : NULL);
}

NLOLIB_API nlolib_status nlolib_propagate(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options
)
{
    return nlolib_propagate_with_storage(config,
                                         num_time_samples,
                                         input_field,
                                         num_recorded_samples,
                                         output_records,
                                         exec_options,
                                         NULL,
                                         NULL);
}

static nlolib_status nlo_propagate_impl(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options,
    const nlo_storage_options* storage_options,
    nlo_storage_result* storage_result
)
{
    nlo_log_nlse_propagate_call(config,
                                num_time_samples,
                                input_field,
                                num_recorded_samples,
                                output_records,
                                exec_options);

    if (storage_result != NULL) {
        *storage_result = (nlo_storage_result){0};
    }

    if (config == NULL || input_field == NULL) {
        return nlo_propagate_fail("validate.null_pointer", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (output_records == NULL && !nlo_storage_enabled(storage_options)) {
        return nlo_propagate_fail("validate.output_or_storage", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    if (num_time_samples == 0u) {
        return nlo_propagate_fail("validate.num_time_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (num_recorded_samples == 0u) {
        return nlo_propagate_fail("validate.num_recorded_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (num_recorded_samples >
        nlo_runtime_limits_default().max_num_recorded_samples_with_storage) {
        return nlo_propagate_fail("validate.num_recorded_samples_precision", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (output_records != NULL &&
        nlo_compute_record_bytes(num_recorded_samples, num_time_samples) == 0u) {
        return nlo_propagate_fail("validate.record_bytes", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (nlo_storage_enabled(storage_options) && !nlo_snapshot_store_is_available()) {
        return nlo_propagate_fail("validate.storage_unavailable", NLOLIB_STATUS_NOT_IMPLEMENTED);
    }
    {
        size_t nt = 0u;
        size_t nx = 0u;
        size_t ny = 0u;
        int explicit_nd = 0;
        if (nlo_resolve_sim_dimensions_internal(config,
                                                num_time_samples,
                                                &nt,
                                                &nx,
                                                &ny,
                                                &explicit_nd) != 0) {
            return nlo_propagate_fail("validate.spatial_dimensions", NLOLIB_STATUS_INVALID_ARGUMENT);
        }
    }

    nlo_execution_options local_exec_options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_AUTO);
    simulation_state* state = NULL;
    const int init_status =
        nlo_storage_enabled(storage_options)
            ? nlo_init_simulation_state_with_storage(config,
                                                     num_time_samples,
                                                     num_recorded_samples,
                                                     &local_exec_options,
                                                     storage_options,
                                                     NULL,
                                                     &state)
            : nlo_init_simulation_state(config,
                                        num_time_samples,
                                        num_recorded_samples,
                                        &local_exec_options,
                                        NULL,
                                        &state);
    if (init_status != 0 || state == NULL) {
        return nlo_propagate_fail("init_simulation_state", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    fprintf(stderr,
            "[nlolib] nlse backend resolved requested=%s actual=%s\n",
            nlo_backend_type_to_string(local_exec_options.backend_type),
            nlo_backend_type_to_string(nlo_vector_backend_get_type(state->backend)));

    if (simulation_state_upload_initial_field(state, input_field) != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return nlo_propagate_fail("upload_initial_field", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    solve_rk4(state);

    if (state->snapshot_status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return nlo_propagate_fail("snapshot_capture", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    int final_downloaded = 0;
    if (num_recorded_samples == 1u && state->snapshot_store != NULL) {
        nlo_complex* final_record = output_records;
        if (final_record == NULL) {
            final_record = simulation_state_get_field_record(state, 0u);
            if (final_record == NULL) {
                final_record = state->snapshot_scratch_record;
            }
        }
        if (final_record == NULL) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.final_record_buffer", NLOLIB_STATUS_ALLOCATION_FAILED);
        }

        if (simulation_state_download_current_field(state, final_record) != NLO_VEC_STATUS_OK) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.download_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        final_downloaded = (final_record == output_records) ? 1 : 0;

        const nlo_snapshot_store_status write_status =
            nlo_snapshot_store_write_record(state->snapshot_store,
                                           0u,
                                           final_record,
                                           state->num_time_samples);
        if (write_status == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.write_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        if (nlo_snapshot_store_flush(state->snapshot_store) == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.flush_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        nlo_snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
    }

    if (output_records != NULL && num_recorded_samples == 1u) {
        if (!final_downloaded &&
            simulation_state_download_current_field(state, output_records) != NLO_VEC_STATUS_OK) {
            free_simulation_state(state);
            return nlo_propagate_fail("download_current_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
    } else if (output_records != NULL) {
        if (state->num_host_records != num_recorded_samples || state->field_buffer == NULL) {
            free_simulation_state(state);
            return nlo_propagate_fail("validate.host_record_buffer", NLOLIB_STATUS_ALLOCATION_FAILED);
        }

        const size_t records_bytes = nlo_compute_record_bytes(num_recorded_samples, num_time_samples);
        if (records_bytes == 0u) {
            free_simulation_state(state);
            return nlo_propagate_fail("validate.output_record_bytes", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        memcpy(output_records, state->field_buffer, records_bytes);
    }

    if (storage_result != NULL) {
        if (state->snapshot_store != NULL) {
            nlo_snapshot_store_get_result(state->snapshot_store, storage_result);
        } else {
            *storage_result = (nlo_storage_result){0};
        }
    }

    free_simulation_state(state);
    return NLOLIB_STATUS_OK;
}

NLOLIB_API nlolib_status nlolib_propagate_with_storage(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options,
    const nlo_storage_options* storage_options,
    nlo_storage_result* storage_result
)
{
    return nlo_propagate_impl(config,
                              num_time_samples,
                              input_field,
                              num_recorded_samples,
                              output_records,
                              exec_options,
                              storage_options,
                              storage_result);
}

NLOLIB_API nlolib_status nlolib_propagate_interleaved(
    const sim_config* config,
    size_t num_time_samples,
    const double* input_field_interleaved,
    size_t num_recorded_samples,
    double* output_records_interleaved,
    const nlo_execution_options* exec_options
)
{
    if (sizeof(nlo_complex) != (2u * sizeof(double))) {
        return nlo_propagate_fail("validate.nlo_complex_layout", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (input_field_interleaved == NULL || output_records_interleaved == NULL) {
        return nlo_propagate_fail("validate.interleaved.null_pointer", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    return nlo_propagate_impl(config,
                              num_time_samples,
                              (const nlo_complex*)input_field_interleaved,
                              num_recorded_samples,
                              (nlo_complex*)output_records_interleaved,
                              exec_options,
                              NULL,
                              NULL);
}

NLOLIB_API int nlolib_storage_is_available(void)
{
    return nlo_snapshot_store_is_available();
}
