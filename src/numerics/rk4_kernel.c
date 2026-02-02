/**
 * @file rk4_kernel.c
 * @dir src/numerics
 * @brief Implementation of RK4 solver kernel for nonlinear optics propagation.
 * @author Wenzel Kinsky
 */

#include "numerics/rk4_kernel.h"
#include "numerics/vector_ops.h"
#include "core/state.h"
#include "fft/nlo_complex.h"
#include "fft/fft.h"
#include "physics/operators.h"
#include <stddef.h>
#include <string.h>

static simulation_working_buffers bind_working_buffers(const simulation_state *state)
{
    simulation_working_buffers buffers = {0};
    if (state == NULL) {
        return buffers;
    }

    buffers.ip_field_buffer = state->ip_field_buffer;
    buffers.field_magnitude_buffer = state->field_magnitude_buffer;
    buffers.field_working_buffer = state->field_working_buffer;
    buffers.field_freq_buffer = state->field_freq_buffer;
    buffers.k_1_buffer = state->k_1_buffer;
    buffers.k_2_buffer = state->k_2_buffer;
    buffers.k_3_buffer = state->k_3_buffer;
    buffers.k_4_buffer = state->k_4_buffer;
    buffers.current_dispersion_factor = state->current_dispersion_factor;

    return buffers;
}

static void step_rk4_with_buffers(simulation_state *state,
                                  const simulation_working_buffers *work)
{
    if (state == NULL || work == NULL) {
        return;
    }

    const size_t num_time_samples = state->num_time_samples;
    const nlo_complex *dispersion_factor = work->current_dispersion_factor;
    const double *gamma = &state->config->nonlinear.gamma;
    double step_size = state->current_step_size;
    nlo_complex *field = simulation_state_current_field(state);
    nlo_complex *field_freq = work->field_freq_buffer;
    nlo_complex *ip_field = work->ip_field_buffer;
    nlo_complex *field_magnitude = work->field_magnitude_buffer;
    nlo_complex *field_working = work->field_working_buffer;
    nlo_complex *k_1 = work->k_1_buffer;
    nlo_complex *k_2 = work->k_2_buffer;

    // Calculate Interaction Picture Field
    forward_fft(field, field_freq, num_time_samples);
    dispersion_operator(dispersion_factor, field_freq, num_time_samples);
    inverse_fft(field_freq, ip_field, num_time_samples);

    // Calculate k1
    calculate_magnitude_squared(
        field,
        field_magnitude,
        num_time_samples);
    nonlinear_operator(
        gamma,
        field_working, // Figure out if nonlinear operator can be inplace, I suspect not
        field_magnitude,
        num_time_samples);
    forward_fft(field_working, field_freq, num_time_samples);
    dispersion_operator(dispersion_factor, field_freq, num_time_samples);
    inverse_fft(field_freq, k_1, num_time_samples);

    // Calculate k2
    nlo_complex_scalar_mul_inplace(k_1, nlo_make(step_size / 2.0, 0.0), num_time_samples);
    nlo_complex_add_inplace(k_2, ip_field, num_time_samples);
    nlo_complex_add_inplace(k_2, k_1, num_time_samples);
    calculate_magnitude_squared(
        k_2,
        field_magnitude,
        num_time_samples);
    nonlinear_operator(
        gamma,
        field_working,
        field_magnitude,
        num_time_samples);
}

void solve_rk4(simulation_state *state) {
    
    double z_end = state->config->propagation.propagation_distance;
    double max_step = state->config->propagation.max_step_size;
    double min_step = state->config->propagation.min_step_size;
    state->current_step_size = state->config->propagation.starting_step_size;
    const size_t num_time_samples = state->num_time_samples;

    simulation_working_buffers work = bind_working_buffers(state);

    calculate_dispersion_factor(
        &state->config->dispersion.num_dispersion_terms,
        state->config->dispersion.betas,
        state->current_step_size,
        work.current_dispersion_factor,
        state->config->frequency.frequency_grid,
        num_time_samples);

    while (state->current_z < z_end) {
        if (state->current_z + state->current_step_size > z_end) {
            state->current_step_size = z_end - state->current_z;
        }

        step_rk4_with_buffers(state, &work);

        // Persist the updated field into the next record slot if capacity allows.
        if (state->current_record_index + 1u < state->num_recorded_samples) {
            nlo_complex* next_record = simulation_state_get_field_record(
                state, state->current_record_index + 1u);
            if (next_record != NULL) {
                memcpy(next_record,
                       simulation_state_current_field(state),
                       num_time_samples * sizeof(nlo_complex));
                state->current_record_index += 1u;
                state->current_field = next_record;
            }
        }

        state->current_z += state->current_step_size;

    }
};

void step_rk4(simulation_state *state)
{
    simulation_working_buffers work = bind_working_buffers(state);
    step_rk4_with_buffers(state, &work);
};
