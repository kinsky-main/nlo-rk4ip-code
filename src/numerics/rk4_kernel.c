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

void solve_rk4(simulation_state *state) {
    
    double z_end = state->config->propagation.propagation_distance;
    double max_step = state->config->propagation.max_step_size;
    double min_step = state->config->propagation.min_step_size;
    state->current_step_size = state->config->propagation.starting_step_size;

    calculate_dispersion_factor(
        &state->config->dispersion.num_dispersion_terms,
        state->config->dispersion.betas,
        state->current_step_size,
        state->current_dispersion_factor,
        state->config->frequency.frequency_grid,
        state->num_time_samples);

    while (state->current_z < z_end) {
        if (state->current_z + state->current_step_size > z_end) {
            state->current_step_size = z_end - state->current_z;
        }

        step_rk4(state);

        state->current_z += state->current_step_size;

    }
};

void step_rk4(simulation_state *state)
{
    const size_t num_time_samples = state->num_time_samples;
    const nlo_complex *dispersion_factor = state->current_dispersion_factor;
    const double *gamma = &state->config->nonlinear.gamma;
    double step_size = state->current_step_size;
    nlo_complex *field = state->field_buffer;
    nlo_complex *field_freq = state->field_freq_buffer;
    nlo_complex *ip_field = state->ip_field_buffer;
    nlo_complex *field_magnitude = state->field_magnitude_buffer;
    nlo_complex *field_working = state->field_working_buffer;
    nlo_complex *k_1 = state->k_1_buffer;
    nlo_complex *k_2 = state->k_2_buffer;

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
    
};
