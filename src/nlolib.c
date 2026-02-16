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

static const char* nlo_backend_type_to_string(nlo_vector_backend_type backend_type)
{
    if (backend_type == NLO_VECTOR_BACKEND_CPU) {
        return "CPU";
    }
    if (backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        return "VULKAN";
    }

    return "UNKNOWN";
}

static size_t nlo_compute_input_bytes(size_t count, size_t stride)
{
    if (stride == 0u || count > (SIZE_MAX / stride)) {
        return 0u;
    }

    return count * stride;
}

static void nlo_log_nlse_propagate_call(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    nlo_complex* output_field,
    const nlo_execution_options* exec_options
)
{
    const nlo_execution_options local_exec_options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);

    const size_t field_bytes = nlo_compute_input_bytes(num_time_samples, sizeof(nlo_complex));
    const size_t dispersion_count =
        (config != NULL)
            ? config->dispersion.num_dispersion_terms
            : 0u;
    const size_t dispersion_bytes = nlo_compute_input_bytes(dispersion_count, sizeof(double));
    const size_t frequency_grid_bytes = nlo_compute_input_bytes(num_time_samples, sizeof(nlo_complex));

    fprintf(stderr,
            "[nlolib] nlse propagate backend=%s | "
            "num_time_samples=%zu field_bytes=%zu | "
            "config=%p(size=%zu) input_field=%p(size=%zu) output_field=%p(size=%zu) "
            "exec_options=%p(size=%zu) | "
            "dispersion_terms=%zu betas_bytes=%zu frequency_grid=%p(size=%zu)\n",
            nlo_backend_type_to_string(local_exec_options.backend_type),
            num_time_samples,
            field_bytes,
            (const void*)config,
            sizeof(sim_config),
            (const void*)input_field,
            field_bytes,
            (const void*)output_field,
            field_bytes,
            (const void*)exec_options,
            sizeof(nlo_execution_options),
            dispersion_count,
            dispersion_bytes,
            (config != NULL) ? (const void*)config->frequency.frequency_grid : NULL,
            frequency_grid_bytes);
}

NLOLIB_API nlolib_status nlolib_propagate(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    nlo_complex* output_field,
    const nlo_execution_options* exec_options
)
{
    nlo_log_nlse_propagate_call(config,
                                num_time_samples,
                                input_field,
                                output_field,
                                exec_options);

    if (config == NULL || input_field == NULL || output_field == NULL) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    if (num_time_samples == 0 || num_time_samples > NT_MAX) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    nlo_execution_options local_exec_options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    simulation_state* state = NULL;
    if (nlo_init_simulation_state(config,
                                  num_time_samples,
                                  1u,
                                  &local_exec_options,
                                  NULL,
                                  &state) != 0 || state == NULL) {
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }

    if (simulation_state_upload_initial_field(state, input_field) != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }

    solve_rk4(state);
    if (simulation_state_download_current_field(state, output_field) != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }

    free_simulation_state(state);
    return NLOLIB_STATUS_OK;
}
