/**
 * @file nlolib.c
 * @dir src
 * @brief Public API entry points for NLOLib.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "nlolib.h"
#include "backend/nlo_complex.h"
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

static int nlo_resolve_spatial_dimensions(
    const sim_config* config,
    size_t num_time_samples,
    size_t* out_nx,
    size_t* out_ny
)
{
    if (config == NULL || out_nx == NULL || out_ny == NULL || num_time_samples == 0u) {
        return -1;
    }

    size_t nx = config->spatial.nx;
    size_t ny = config->spatial.ny;
    if (nx == 0u && ny == 0u) {
        nx = num_time_samples;
        ny = 1u;
    } else if (nx == 0u || ny == 0u) {
        return -1;
    }

    if (nx > (SIZE_MAX / ny)) {
        return -1;
    }
    if ((nx * ny) != num_time_samples) {
        return -1;
    }

    *out_nx = nx;
    *out_ny = ny;
    return 0;
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
    size_t nx = 0u;
    size_t ny = 0u;
    const int has_spatial_shape =
        (nlo_resolve_spatial_dimensions(config, num_time_samples, &nx, &ny) == 0);

    fprintf(stderr,
            "[nlolib] nlse propagate backend=%s | "
            "num_time_samples=%zu num_recorded_samples=%zu field_bytes=%zu record_bytes=%zu | "
            "config=%p(size=%zu) input_field=%p(size=%zu) output_records=%p(size=%zu) "
            "exec_options=%p(size=%zu) | "
            "frequency_grid=%p(size=%zu) runtime(constants=%zu df=%p d=%p n=%p) | "
            "spatial(nx=%zu ny=%zu valid=%d dx=%.9e dy=%.9e spatial_frequency_grid=%p potential_grid=%p)\n",
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
            (config != NULL) ? (const void*)config->runtime.nonlinear_expr : NULL,
            nx,
            ny,
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
    nlo_log_nlse_propagate_call(config,
                                num_time_samples,
                                input_field,
                                num_recorded_samples,
                                output_records,
                                exec_options);

    if (config == NULL || input_field == NULL || output_records == NULL) {
        return nlo_propagate_fail("validate.null_pointer", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    if (num_time_samples == 0 || num_time_samples > NT_MAX) {
        return nlo_propagate_fail("validate.num_time_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (num_recorded_samples == 0u || num_recorded_samples > NT_MAX) {
        return nlo_propagate_fail("validate.num_recorded_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (nlo_compute_record_bytes(num_recorded_samples, num_time_samples) == 0u) {
        return nlo_propagate_fail("validate.record_bytes", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    {
        size_t nx = 0u;
        size_t ny = 0u;
        if (nlo_resolve_spatial_dimensions(config, num_time_samples, &nx, &ny) != 0) {
            return nlo_propagate_fail("validate.spatial_dimensions", NLOLIB_STATUS_INVALID_ARGUMENT);
        }
    }

    nlo_execution_options local_exec_options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_AUTO);
    simulation_state* state = NULL;
    if (nlo_init_simulation_state(config,
                                  num_time_samples,
                                  num_recorded_samples,
                                  &local_exec_options,
                                  NULL,
                                  &state) != 0 || state == NULL) {
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

    if (num_recorded_samples == 1u) {
        if (simulation_state_download_current_field(state, output_records) != NLO_VEC_STATUS_OK) {
            free_simulation_state(state);
            return nlo_propagate_fail("download_current_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
    } else {
        if (state->num_recorded_samples != num_recorded_samples || state->field_buffer == NULL) {
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

    free_simulation_state(state);
    return NLOLIB_STATUS_OK;
}
