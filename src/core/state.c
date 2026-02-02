/**
 * @brief State management for nonlinear optics solver
 * @file state.c
 * @author Wenzel Kinsky
 * @date 2026-01-27
 */
#include "core/state.h"
#include "fft/fft.h"
#include <stdlib.h>

static int allocate_nlo_complex_buffer(nlo_complex **buffer, size_t num_elements);

// MARK: Simulation Config Management

sim_config *create_sim_config(size_t num_dispersion_terms, size_t num_time_samples)
{
    if (num_dispersion_terms > NT_MAX || num_time_samples == 0 || num_time_samples > NT_MAX)
    {
        return NULL;
    }

    sim_config *config = (sim_config *)calloc(1, sizeof(sim_config));
    if (config == NULL)
    {
        return NULL;
    }

    config->dispersion.num_dispersion_terms = num_dispersion_terms;
    config->frequency.frequency_grid = (nlo_complex *)calloc(num_time_samples, sizeof(nlo_complex));
    if (config->frequency.frequency_grid == NULL)
    {
        free(config);
        return NULL;
    }

    return config;
}

void free_sim_config(sim_config *config)
{
    if (config == NULL)
    {
        return;
    }

    free(config->frequency.frequency_grid);
    free(config);
}

// MARK: Simulation State Management

simulation_state *create_simulation_state(const sim_config *config, size_t num_time_samples, size_t num_recorded_samples)
{
    if (num_time_samples == 0 ||
        num_time_samples > NT_MAX ||
        num_recorded_samples == 0 ||
        num_recorded_samples > NT_MAX ||
        config == NULL)
    {
        return NULL;
    }

    simulation_state *state = (simulation_state *)calloc(1, sizeof(simulation_state));
    if (state == NULL)
    {
        return NULL;
    }

    state->config = config;
    state->num_time_samples = num_time_samples;
    state->num_recorded_samples = num_recorded_samples;
    state->current_record_index = 0u;
    state->current_z = 0.0;
    state->current_step_size = config->propagation.starting_step_size;

    if (allocate_nlo_complex_buffer(&state->field_buffer, num_time_samples * num_recorded_samples) != 0)
    {
        free_simulation_state(state);
        return NULL;
    }

    state->current_field = state->field_buffer;

    nlo_complex **work_buffers[] = {
        &state->ip_field_buffer,
        &state->field_magnitude_buffer,
        &state->field_working_buffer,
        &state->field_freq_buffer,
        &state->k_1_buffer,
        &state->k_2_buffer,
        &state->k_3_buffer,
        &state->k_4_buffer,
        &state->current_dispersion_factor};

    const size_t num_work_buffers = sizeof(work_buffers) / sizeof(work_buffers[0]);
    for (size_t i = 0; i < num_work_buffers; ++i)
    {
        if (allocate_nlo_complex_buffer(work_buffers[i], num_time_samples) != 0)
        {
            free_simulation_state(state);
            return NULL;
        }
    }

    return state;
}

void free_simulation_state(simulation_state *state)
{
    if (state != NULL)
    {
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

static int allocate_nlo_complex_buffer(nlo_complex **buffer, size_t num_elements)
{
    *buffer = (nlo_complex *)calloc(num_elements, sizeof(nlo_complex));
    return (*buffer == NULL) ? -1 : 0;
}
