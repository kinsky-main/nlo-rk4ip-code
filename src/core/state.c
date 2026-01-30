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
    {
        const size_t num_betas = config->dispersion.num_dispersion_terms;

        free(state->current_dispersion_factors);
        state->current_dispersion_factors = (config->dispersion.num_dispersion_terms > 0)
            ? (nlo_complex*)calloc(config->dispersion.num_dispersion_terms, sizeof(nlo_complex))
            : nlo_create(0.0, 0.0);
    }
    if (state->ip_field_buffer == NULL ||
        state->field_buffer == NULL ||
        state->current_dispersion_factors == NULL ||
        state->current_step_size == 0.0 ||
        state->num_time_samples == 0
        ) {
        free_simulation_state(state);
        return NULL;
    }

    return state;
}

void free_simulation_state(simulation_state* state)
{
    if (state != NULL) {
        free(state->field_buffer);
        free(state->ip_field_buffer);
        free(state->current_dispersion_factors);
        free(state);
    }
}
