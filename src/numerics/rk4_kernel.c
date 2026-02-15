/**
 * @file rk4_kernel.c
 * @brief Backend-resident RK4 integration kernel.
 */

#include "numerics/rk4_kernel.h"
#include "fft/fft.h"
#include "physics/operators.h"
#include <assert.h>
#include <math.h>

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

#ifndef NLO_RK4_STRICT_STATUS_CHECKS
#if defined(NDEBUG)
#define NLO_RK4_STRICT_STATUS_CHECKS 0
#else
#define NLO_RK4_STRICT_STATUS_CHECKS 1
#endif
#endif

#if NLO_RK4_STRICT_STATUS_CHECKS
#define NLO_RK4_CALL(expr)                          \
    do                                              \
    {                                               \
        const nlo_vec_status call_status_ = (expr); \
        if (call_status_ != NLO_VEC_STATUS_OK)      \
        {                                           \
            return call_status_;                    \
        }                                           \
    } while (0)
#else
#define NLO_RK4_CALL(expr)                          \
    do                                              \
    {                                               \
        const nlo_vec_status call_status_ = (expr); \
        assert(call_status_ == NLO_VEC_STATUS_OK);  \
        (void)call_status_;                         \
    } while (0)
#endif

static double nlo_clamp_step(double value, double min_step, double max_step)
{
    double out = value;
    if (max_step > 0.0 && out > max_step)
    {
        out = max_step;
    }
    if (min_step > 0.0 && out < min_step)
    {
        out = min_step;
    }
    return out;
}

static nlo_vec_status nlo_rk4_step_device(simulation_state *state)
{
    if (state == NULL || state->backend == NULL || state->fft_plan == NULL)
    {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vector_backend *backend = state->backend;
    simulation_working_vectors *work = &state->working_vectors;
    const double step = state->current_step_size;

    // Calculate Interaction Picture Field
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, state->current_field_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_vec(backend,
                                                   work->dispersion_factor_vec,
                                                   work->field_freq_vec,
                                                   state->current_half_step_exp));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->ip_field_vec));

    // Calculate k1
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_vec(
        backend,
        state->config->nonlinear.gamma,
        work->ip_field_vec,
        work->field_magnitude_vec,
        work->field_working_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(step, 0.0)));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_working_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_vec(backend,
                                                   work->dispersion_factor_vec,
                                                   work->field_working_vec,
                                                   state->current_half_step_exp));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_working_vec, work->k_1_vec));
    NLO_RK4_CALL(nlo_vec_complex_mul_inplace(backend, work->k_1_vec, state->current_field_vec));
    
    // Calculate working field term for k2 nonlinearity
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_1_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(0.5, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    // Calculate k2
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_vec(backend,
                                                  state->config->nonlinear.gamma,
                                                  work->field_working_vec,
                                                  work->field_magnitude_vec,
                                                  work->k_2_vec));
    NLO_RK4_CALL(nlo_vec_complex_mul_inplace(backend, work->k_2_vec, work->field_working_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_2_vec, nlo_make(step, 0.0)));
    
    // Calculate Magnitude squared term for k3 nonlinearity
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_2_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(0.5, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    // Calculate k3
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_vec(backend,
                                                  state->config->nonlinear.gamma,
                                                  work->field_working_vec,
                                                  work->field_magnitude_vec,
                                                  work->k_3_vec));
    NLO_RK4_CALL(nlo_vec_complex_mul_inplace(backend, work->k_3_vec, work->field_working_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_3_vec, nlo_make(step, 0.0)));

    // Calculate working field term for k4 nonlinearity
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_3_vec));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_vec(backend,
                                                   work->dispersion_factor_vec,
                                                   work->field_freq_vec,
                                                   state->current_half_step_exp));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->field_working_vec));

    // Calculate k4
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_vec(backend,
                                                  state->config->nonlinear.gamma,
                                                  work->field_working_vec,
                                                  work->field_magnitude_vec,
                                                  work->k_4_vec));
    NLO_RK4_CALL(nlo_vec_complex_mul_inplace(backend, work->k_4_vec, work->field_working_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_4_vec, nlo_make(step, 0.0)));

    // Scale and combine k terms to update field
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_1_vec, nlo_make(1.0 / 6.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_2_vec, nlo_make(1.0 / 3.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_3_vec, nlo_make(1.0 / 3.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_4_vec, nlo_make(1.0 / 6.0, 0.0)));

    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_1_vec, work->k_2_vec));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_1_vec, work->k_3_vec));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_1_vec, work->ip_field_vec));

    NLO_RK4_CALL(nlo_apply_dispersion_operator_vec(backend,
                                                   work->dispersion_factor_vec,
                                                   work->k_1_vec,
                                                   state->current_half_step_exp));

    NLO_RK4_add_inplace(backend, work->k_1_vec, work->k_4_vec);

    NLO_RK4_CALL(nlo_vec_complex_copy(backend, state->current_field_vec, work->k_1_vec));

    return NLO_VEC_STATUS_OK;
}

void solve_rk4(simulation_state *state)
{
    if (state == NULL || state->backend == NULL || state->config == NULL)
    {
        return;
    }
    if (!state->dispersion_valid)
    {
        return;
    }

    const double z_end = state->config->propagation.propagation_distance;
    if (z_end <= 0.0)
    {
        return;
    }

    const double max_step = state->config->propagation.max_step_size;
    const double min_step = state->config->propagation.min_step_size;
    const double tol = (state->config->propagation.error_tolerance > 0.0)
                           ? state->config->propagation.error_tolerance
                           : NLO_RK4_ERROR_TOL;

    if (state->current_step_size <= 0.0)
    {
        state->current_step_size = state->config->propagation.starting_step_size;
    }

    if (nlo_vec_begin_simulation(state->backend) != NLO_VEC_STATUS_OK)
    {
        return;
    }

    if (state->current_record_index == 0u && state->num_recorded_samples > 0u)
    {
        if (simulation_state_capture_snapshot(state) != NLO_VEC_STATUS_OK)
        {
            (void)nlo_vec_end_simulation(state->backend);
            return;
        }
    }

    double record_spacing = 0.0;
    double next_record_z = z_end;
    if (state->num_recorded_samples > 1u)
    {
        record_spacing = z_end / (double)(state->num_recorded_samples - 1u);
        next_record_z = record_spacing * (double)state->current_record_index;
    }

    while (state->current_z < z_end)
    {
        double step = nlo_clamp_step(state->current_step_size, min_step, max_step);
        if (step <= 0.0)
        {
            break;
        }

        const double remaining = z_end - state->current_z;
        if (step > remaining)
        {
            step = remaining;
        }

        if (record_spacing > 0.0 &&
            state->current_record_index < state->num_recorded_samples &&
            next_record_z > state->current_z)
        {
            const double to_next_record = next_record_z - state->current_z;
            if (to_next_record > 0.0 && step > to_next_record)
            {
                step = to_next_record;
            }
        }

        if (step <= 0.0)
        {
            break;
        }

        state->current_step_size = step;
        state->current_half_step_exp = exp(0.5 * step);

        if (nlo_vec_complex_copy(state->backend,
                                 state->working_vectors.previous_field_vec,
                                 state->current_field_vec) != NLO_VEC_STATUS_OK)
        {
            break;
        }

        if (nlo_rk4_step_device(state) != NLO_VEC_STATUS_OK)
        {
            break;
        }

        double error = 0.0;
        if (nlo_vec_complex_relative_error(state->backend,
                                           state->current_field_vec,
                                           state->working_vectors.previous_field_vec,
                                           NLO_RK4_ERROR_EPS,
                                           &error) != NLO_VEC_STATUS_OK)
        {
            error = 0.0;
        }

        state->current_z += step;

        double scale = NLO_RK4_STEP_GROW_MAX;
        if (error > 0.0)
        {
            scale = NLO_RK4_ERROR_SAFETY * pow(tol / error, 0.2);
            if (scale < NLO_RK4_STEP_SHRINK_MIN)
            {
                scale = NLO_RK4_STEP_SHRINK_MIN;
            }
            else if (scale > NLO_RK4_STEP_GROW_MAX)
            {
                scale = NLO_RK4_STEP_GROW_MAX;
            }
        }

        state->current_step_size = nlo_clamp_step(step * scale, min_step, max_step);

        while (record_spacing > 0.0 &&
               state->current_record_index < state->num_recorded_samples &&
               state->current_z + 1e-15 >= next_record_z)
        {
            if (simulation_state_capture_snapshot(state) != NLO_VEC_STATUS_OK)
            {
                break;
            }
            next_record_z += record_spacing;
        }
    }

    (void)simulation_state_flush_snapshots(state);
    (void)nlo_vec_end_simulation(state->backend);
}

void step_rk4(simulation_state *state)
{
    if (state == NULL || state->backend == NULL || state->config == NULL)
    {
        return;
    }
    if (!state->dispersion_valid)
    {
        return;
    }

    if (nlo_vec_begin_simulation(state->backend) != NLO_VEC_STATUS_OK)
    {
        return;
    }

    state->current_half_step_exp = exp(0.5 * state->current_step_size);
    if (nlo_rk4_step_device(state) == NLO_VEC_STATUS_OK)
    {
        state->current_z += state->current_step_size;
    }

    (void)nlo_vec_end_simulation(state->backend);
}
