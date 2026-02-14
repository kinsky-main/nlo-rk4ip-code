/**
 * @file rk4_kernel.c
 * @dir src/numerics
 * @brief Implementation of RK4 solver kernel for nonlinear optics propagation.
 * @author Wenzel Kinsky
 */

#include "numerics/rk4_kernel.h"
#include "numerics/vector_ops.h"
#include "core/state.h"
#include "backend/nlo_complex.h"
#include "fft/fft.h"
#include "physics/operators.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

#ifndef NLO_RK4_ERROR_TOL
#define NLO_RK4_ERROR_TOL 1e-6
#endif

#ifndef NLO_RK4_ERROR_SAFETY
#define NLO_RK4_ERROR_SAFETY 0.9
#endif

#ifndef NLO_RK4_STEP_SHRINK_MIN
#define NLO_RK4_STEP_SHRINK_MIN 0.2
#endif

#ifndef NLO_RK4_STEP_GROW_MAX
#define NLO_RK4_STEP_GROW_MAX 2.0
#endif

#ifndef NLO_RK4_ERROR_EPS
#define NLO_RK4_ERROR_EPS 1e-12
#endif

static simulation_working_buffers bind_working_buffers(const simulation_state *state)
{
    simulation_working_buffers buffers = {0};
    if (state == NULL) {
        return buffers;
    }

    return state->working_buffers;
}

static void step_rk4_with_buffers(simulation_state *state,
                                  const simulation_working_buffers *work);

static double rk4_record_error(const nlo_complex *current,
                               const nlo_complex *previous,
                               size_t n)
{
    if (current == NULL || previous == NULL || n == 0u) {
        return 0.0;
    }

    double max_sq = 0.0;
    double max_prev_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double prev_re = NLO_RE(previous[i]);
        const double prev_im = NLO_IM(previous[i]);
        const double dre = NLO_RE(current[i]) - prev_re;
        const double dim = NLO_IM(current[i]) - prev_im;
        const double diff_sq = dre * dre + dim * dim;
        const double prev_sq = prev_re * prev_re + prev_im * prev_im;

        if (diff_sq > max_sq) {
            max_sq = diff_sq;
        }
        if (prev_sq > max_prev_sq) {
            max_prev_sq = prev_sq;
        }
    }

    if (max_prev_sq < NLO_RK4_ERROR_EPS) {
        max_prev_sq = NLO_RK4_ERROR_EPS;
    }

    return sqrt(max_sq / max_prev_sq);
}

static double rk4_step_check(simulation_state *state,
                             const simulation_working_buffers *work,
                             double record_spacing,
                             double next_record_z,
                             double min_step,
                             double max_step)
{
    if (state == NULL || work == NULL) {
        return 0.0;
    }

    double step_size = state->current_step_size;
    if (max_step > 0.0 && step_size > max_step) {
        step_size = max_step;
    }
    if (min_step > 0.0 && step_size < min_step) {
        step_size = min_step;
    }

    if (record_spacing > 0.0 &&
        state->current_record_index + 1u < state->num_recorded_samples) {
        const double distance_to_next = next_record_z - state->current_z;
        if (distance_to_next > 0.0 && step_size > distance_to_next) {
            step_size = distance_to_next;
        }
    }

    if (step_size <= 0.0) {
        return 0.0;
    }

    if (step_size != state->current_step_size) {
        state->current_step_size = step_size;
    }

    const size_t num_time_samples = state->num_time_samples;
    nlo_complex *field = simulation_state_current_field(state);
    if (field == NULL) {
        step_rk4_with_buffers(state, work);
        return 0.0;
    }

    calculate_dispersion_factor(
        &state->config->dispersion.num_dispersion_terms,
        state->config->dispersion.betas,
        step_size,
        work->current_dispersion_factor,
        state->config->frequency.frequency_grid,
        num_time_samples);

    step_rk4_with_buffers(state, work);

    double error = 0.0;
    if (state->current_record_index > 0u) {
        nlo_complex *previous_record = simulation_state_get_field_record(
            state, state->current_record_index - 1u);
        if (previous_record != NULL) {
            error = rk4_record_error(field, previous_record, num_time_samples);
        }
    }

    double tol = state->config->propagation.error_tolerance;
    if (tol <= 0.0) {
        tol = NLO_RK4_ERROR_TOL;
    }

    if (error > 0.0) {
        double scale = NLO_RK4_ERROR_SAFETY *
            pow(tol / error, 0.2);
        if (scale < NLO_RK4_STEP_SHRINK_MIN) {
            scale = NLO_RK4_STEP_SHRINK_MIN;
        } else if (scale > NLO_RK4_STEP_GROW_MAX) {
            scale = NLO_RK4_STEP_GROW_MAX;
        }
        step_size *= scale;
    } else {
        step_size *= NLO_RK4_STEP_GROW_MAX;
    }

    if (max_step > 0.0 && step_size > max_step) {
        step_size = max_step;
    }
    if (min_step > 0.0 && step_size < min_step) {
        step_size = min_step;
    }

    if (record_spacing > 0.0 &&
        state->current_record_index + 1u < state->num_recorded_samples) {
        const double distance_to_next = next_record_z - state->current_z;
        if (distance_to_next > 0.0 && step_size > distance_to_next) {
            step_size = distance_to_next;
        }
    }

    state->current_step_size = step_size;
    return error;
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
    double half_step_exp = state->current_half_step_exp;
    nlo_complex *field = simulation_state_current_field(state);
    nlo_complex *field_freq = work->field_freq_buffer;
    nlo_complex *ip_field = work->ip_field_buffer;
    nlo_complex *field_magnitude = work->field_magnitude_buffer;
    nlo_complex *field_working = work->field_working_buffer;
    nlo_complex *k_1 = work->k_1_buffer;
    nlo_complex *k_2 = work->k_2_buffer;
    nlo_complex *k_3 = work->k_3_buffer;
    nlo_complex *k_4 = work->k_4_buffer;

    // Calculate Interaction Picture Field
    forward_fft(field, field_freq, num_time_samples);
    dispersion_operator(dispersion_factor, field_freq, num_time_samples, half_step_exp);
    inverse_fft(field_freq, ip_field, num_time_samples);

    // Calculate k1
    calculate_magnitude_squared(
        field,
        field_magnitude,
        num_time_samples);
    nonlinear_operator(
        gamma,
        field_working,
        field_magnitude,
        num_time_samples);
    nlo_complex_scalar_mul_inplace(
        field_working,
        nlo_make(step_size, 0.0),
        num_time_samples);
    forward_fft(field_working, field_freq, num_time_samples);
    dispersion_operator(dispersion_factor, field_freq, num_time_samples, half_step_exp);
    inverse_fft(field_freq, k_1, num_time_samples);
    nlo_complex_mul_inplace(k_1, field, num_time_samples);

    // Calculate k2
    nlo_complex_add_vec(
        field_working,
        ip_field,
        k_1,
        num_time_samples);
}

void solve_rk4(simulation_state *state) {
    if (state == NULL) {
        return;
    }

    double z_end = state->config->propagation.propagation_distance;
    double max_step = state->config->propagation.max_step_size;
    double min_step = state->config->propagation.min_step_size;
    state->current_step_size = state->config->propagation.starting_step_size;
    const size_t num_time_samples = state->num_time_samples;
    double record_spacing = 0.0;
    double next_record_z = 0.0;

    simulation_working_buffers work = bind_working_buffers(state);

    if (state->num_recorded_samples > 1u && z_end > 0.0) {
        record_spacing = z_end / (double)state->num_recorded_samples;
        next_record_z = record_spacing;
    }

    while (state->current_z < z_end) {
        if (state->current_z + state->current_step_size > z_end) {
            state->current_step_size = z_end - state->current_z;
        }

        double step_error = rk4_step_check(state, &work, record_spacing, next_record_z, min_step, max_step);
        (void)step_error;

        state->current_z += state->current_step_size;

        // Persist the updated field only when crossing the next record position.
        if (record_spacing > 0.0 &&
            state->current_record_index + 1u < state->num_recorded_samples &&
            state->current_z >= next_record_z) {
            nlo_complex* next_record = simulation_state_get_field_record(
                state, state->current_record_index + 1u);
            if (next_record != NULL) {
                memcpy(next_record,
                       simulation_state_current_field(state),
                       num_time_samples * sizeof(nlo_complex));
                state->current_record_index += 1u;
                state->current_field = next_record;
            }

            if (record_spacing > 0.0) {
                while (next_record_z <= state->current_z) {
                    next_record_z += record_spacing;
                }
                if (state->current_record_index + 1u < state->num_recorded_samples) {
                    const double distance_to_next = next_record_z - state->current_z;
                    if (distance_to_next > 0.0 && state->current_step_size > distance_to_next) {
                        state->current_step_size = distance_to_next;
                    }
                }
            }
        }

    }

};

void step_rk4(simulation_state *state)
{
    if (state == NULL) {
        return;
    }

    simulation_working_buffers work = bind_working_buffers(state);
    step_rk4_with_buffers(state, &work);
};
