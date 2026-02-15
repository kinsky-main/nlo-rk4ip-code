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

NLOLIB_API nlolib_status nlolib_propagate(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    nlo_complex* output_field
)
{
    if (config == NULL || input_field == NULL || output_field == NULL) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    if (num_time_samples == 0 || num_time_samples > NT_MAX) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    simulation_state* state = NULL;
    if (nlo_init_simulation_state(config,
                                  num_time_samples,
                                  1u,
                                  &exec_options,
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
