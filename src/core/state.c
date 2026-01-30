/**
 * @brief State management for nonlinear optics solver
 * @file state.c
 * @author Wenzel Kinsky
 * @date 2026-01-27
 */
#include "core/state.h"
#include <stdlib.h>

// MARK: Simulation State Management

simulation_state* create_simulation_state(const sim_config* config, size_t num_time_samples)
{
    if (num_time_samples == 0 || num_time_samples > NT_MAX || config == NULL) {
        return NULL;
    }

    simulation_state* state = (simulation_state*)calloc(1, sizeof(simulation_state));
    if (state == NULL) {
        return NULL;
    }

    state->config = config;
    state->num_time_samples = num_time_samples;
    state->current_z = 0.0;
    state->current_step_size = config->propagation.starting_step_size;

    state->field_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    state->ip_field_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    state->field_magnitude_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    state->field_working_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    state->current_dispersion_factor = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));

    return state;
}

void free_simulation_state(simulation_state* state)
{
    if (state != NULL) {
        free(state->field_buffer);
        free(state->ip_field_buffer);
        free(state->current_dispersion_factor);
        free(state);
    }
}
