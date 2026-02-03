/**
 * @file nlolib.c
 * @dir src
 * @brief Public API entry points for NLOLib.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "nlolib.h"
#include "backend/nlo_complex.h"
#include <string.h>
#include <stddef.h>

NLOLIB_API nlolib_status nlolib_propagate(const sim_config* config,
                                         size_t num_time_samples,
                                         const nlo_complex* input_field,
                                         nlo_complex* output_field)
{
    if (config == NULL || input_field == NULL || output_field == NULL) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    if (num_time_samples == 0 || num_time_samples > NT_MAX) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    // TODO: simulation state num recorded samples should be derived from config or passed separately
    simulation_state* state = NULL;
    if (nlo_init_simulation_state(config, num_time_samples, 1u, NULL, &state) != 0 || state == NULL) {
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }

    nlo_complex* initial_field = simulation_state_current_field(state);
    if (initial_field == NULL) {
        free_simulation_state(state);
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }

    memcpy(initial_field, input_field, num_time_samples * sizeof(nlo_complex));

    if (input_field != output_field) {
        memcpy(output_field, input_field, num_time_samples * sizeof(nlo_complex));
    }

    free_simulation_state(state);
    return NLOLIB_STATUS_NOT_IMPLEMENTED;
}
