/**
 * @brief State management for nonlinear optics solver
 * @file state.c
 * @author Wenzel Kinsky
 * @date 2026-01-27
 */
#include "core/state.h"
#include "fft/fft.h"
#include <stdlib.h>

// MARK: Simulation Config Management

sim_config* create_sim_config(size_t num_dispersion_terms, size_t num_time_samples)
{
    if (num_dispersion_terms > NT_MAX || num_time_samples == 0 || num_time_samples > NT_MAX) {
        return NULL;
    }

    sim_config* config = (sim_config*)calloc(1, sizeof(sim_config));
    if (config == NULL) {
        return NULL;
    }

    config->dispersion.num_dispersion_terms = num_dispersion_terms;
    config->frequency.frequency_grid = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    if (config->frequency.frequency_grid == NULL) {
        free(config);
        return NULL;
    }

    return config;
}

void free_sim_config(sim_config* config)
{
    if (config == NULL) {
        return;
    }

    free(config->frequency.frequency_grid);
    free(config);
}

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
    state->field_freq_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    state->k_1_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    state->k_2_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    state->k_3_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    state->k_4_buffer = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));

    if (state->field_buffer == NULL ||
        state->ip_field_buffer == NULL ||
        state->field_magnitude_buffer == NULL ||
        state->field_working_buffer == NULL ||
        state->current_dispersion_factor == NULL ||
        state->field_freq_buffer == NULL ||
        state->k_1_buffer == NULL ||
        state->k_2_buffer == NULL ||
        state->k_3_buffer == NULL ||
        state->k_4_buffer == NULL ||
        fft_init(num_time_samples) != 0) {
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
        free(state->field_magnitude_buffer);
        free(state->field_working_buffer);
        free(state->field_freq_buffer);
        free(state->k_1_buffer);
        free(state->k_2_buffer);
        free(state->k_3_buffer);
        free(state->k_4_buffer);
        free(state->current_dispersion_factor);
        free(state);
    }
}
