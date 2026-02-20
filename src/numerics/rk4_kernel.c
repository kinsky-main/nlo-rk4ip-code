/**
 * @file rk4_kernel.c
 * @brief Backend-resident RK4 integration kernel.
 */

#include "numerics/rk4_kernel.h"
#include "fft/fft.h"
#include "physics/operators.h"
#include "utility/perf_profile.h"
#include "utility/rk4_debug.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

#ifndef NLO_RK4_ATOL_SCALE
#define NLO_RK4_ATOL_SCALE 1e-3
#endif

#ifndef NLO_RK4_ATOL_MIN
#define NLO_RK4_ATOL_MIN 1e-14
#endif

#ifndef NLO_RK4_MAX_REJECTION_ATTEMPTS
#define NLO_RK4_MAX_REJECTION_ATTEMPTS 32u
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

static int nlo_rk4_debug_enabled_runtime(void)
{
    const char* env = getenv("NLO_RK4_DEBUG");
    return (env != NULL && *env != '\0' && *env != '0') ? 1 : 0;
}

static double nlo_record_capture_tolerance(double z_end)
{
    const double scale = fmax(1.0, fabs(z_end));
    return 64.0 * DBL_EPSILON * scale;
}

static void nlo_rk4_debug_log_solver_config(const simulation_state* state, double tol)
{
    if (!nlo_rk4_debug_enabled_runtime() || state == NULL || state->config == NULL) {
        return;
    }

    const sim_config* config = state->config;
    const double c0 = (config->runtime.num_constants > 0u) ? config->runtime.constants[0] : -0.5;
    const double c1 = (config->runtime.num_constants > 1u) ? config->runtime.constants[1] : 0.0;
    const double c2 = (config->runtime.num_constants > 2u) ? config->runtime.constants[2] : 1.0;

    fprintf(stderr,
            "[NLO_RK4_DEBUG] config c0=%.9e c1=%.9e c2=%.9e dt=%.9e z_end=%.9e h0=%.9e h_min=%.9e h_max=%.9e tol=%.9e\n",
            c0,
            c1,
            c2,
            config->time.delta_time,
            config->propagation.propagation_distance,
            config->propagation.starting_step_size,
            config->propagation.min_step_size,
            config->propagation.max_step_size,
            tol);
}

static nlo_vec_status nlo_apply_nonlinear_operator_stage(
    simulation_state* state,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* out_field
)
{
    if (state == NULL || state->backend == NULL || state->config == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const nlo_operator_eval_context eval_ctx = {
        .frequency_grid = state->frequency_grid_vec,
        .field = field,
        .dispersion_factor = state->working_vectors.dispersion_factor_vec,
        .potential = state->working_vectors.potential_vec,
        .half_step_size = state->current_half_step_exp
    };

    const double start_ms = nlo_perf_profile_now_ms();
    const nlo_vec_status status = nlo_apply_nonlinear_operator_program_vec(state->backend,
                                                                           &state->nonlinear_operator_program,
                                                                           &eval_ctx,
                                                                           field,
                                                                           state->working_vectors.nonlinear_multiplier_vec,
                                                                           out_field,
                                                                           state->runtime_operator_stack_vec,
                                                                           state->runtime_operator_stack_slots);
    const double end_ms = nlo_perf_profile_now_ms();
    if (status == NLO_VEC_STATUS_OK) {
        nlo_perf_profile_add_nonlinear_time(end_ms - start_ms);
    }
    return status;
}

static nlo_vec_status nlo_apply_dispersion_operator_stage(
    simulation_state* state,
    nlo_vec_buffer* freq_domain_envelope
)
{
    if (state == NULL || state->backend == NULL || state->config == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const nlo_operator_eval_context eval_ctx = {
        .frequency_grid = state->frequency_grid_vec,
        .field = freq_domain_envelope,
        .dispersion_factor = state->working_vectors.dispersion_factor_vec,
        .potential = state->working_vectors.potential_vec,
        .half_step_size = state->current_half_step_exp
    };

    const double start_ms = nlo_perf_profile_now_ms();
    nlo_vec_status status = nlo_apply_dispersion_operator_program_vec(state->backend,
                                                                      &state->dispersion_operator_program,
                                                                      &eval_ctx,
                                                                      freq_domain_envelope,
                                                                      state->working_vectors.dispersion_operator_vec,
                                                                      state->runtime_operator_stack_vec,
                                                                      state->runtime_operator_stack_slots);
    if (status == NLO_VEC_STATUS_OK && state->transverse_active) {
        const nlo_operator_eval_context transverse_eval_ctx = {
            .frequency_grid = state->spatial_frequency_grid_vec,
            .field = freq_domain_envelope,
            .dispersion_factor = state->transverse_factor_vec,
            .potential = state->working_vectors.potential_vec,
            .half_step_size = state->current_half_step_exp
        };
        status = nlo_apply_dispersion_operator_program_vec(state->backend,
                                                           &state->transverse_operator_program,
                                                           &transverse_eval_ctx,
                                                           freq_domain_envelope,
                                                           state->transverse_operator_vec,
                                                           state->runtime_operator_stack_vec,
                                                           state->runtime_operator_stack_slots);
    }
    const double end_ms = nlo_perf_profile_now_ms();
    if (status == NLO_VEC_STATUS_OK) {
        nlo_perf_profile_add_dispersion_time(end_ms - start_ms);
    }
    return status;
}

static nlo_vec_status nlo_rk4_step_device(simulation_state *state, size_t step_index)
{
    if (state == NULL || state->backend == NULL || state->fft_plan == NULL)
    {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vector_backend *backend = state->backend;
    simulation_working_vectors *work = &state->working_vectors;
    const double step = state->current_step_size;

    nlo_rk4_debug_log_vec_stats(state,
                                state->current_field_vec,
                                "current_field_start",
                                step_index,
                                state->current_z,
                                step);

    // Calculate Interaction Picture Field
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, state->current_field_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->ip_field_vec));
    nlo_rk4_debug_log_vec_stats(state, work->ip_field_vec, "ip_field", step_index, state->current_z, step);

    // Calculate k1
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->ip_field_vec,
                                                    work->field_working_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(step, 0.0)));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_1_vec));
    nlo_rk4_debug_log_vec_stats(state, work->k_1_vec, "k1", step_index, state->current_z, step);
    
    // Calculate working field term for k2 nonlinearity
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_1_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(0.5, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    // Calculate k2
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_2_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_2_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_2_vec, "k2", step_index, state->current_z, step);
    
    // Calculate Magnitude squared term for k3 nonlinearity
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_2_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(0.5, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    // Calculate k3
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_3_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_3_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_3_vec, "k3", step_index, state->current_z, step);

    // Calculate working field term for k4 nonlinearity
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_3_vec));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->field_working_vec));

    // Calculate k4
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_4_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_4_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_4_vec, "k4", step_index, state->current_z, step);

    // Scale and combine k terms to update field
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_1_vec, nlo_make(1.0 / 6.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_2_vec, nlo_make(1.0 / 3.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_3_vec, nlo_make(1.0 / 3.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_4_vec, nlo_make(1.0 / 6.0, 0.0)));

    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_1_vec, work->k_2_vec));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_1_vec, work->k_3_vec));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_1_vec, work->ip_field_vec));
    
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->k_1_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_1_vec));

    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_1_vec, work->k_4_vec));

    NLO_RK4_CALL(nlo_vec_complex_copy(backend, state->current_field_vec, work->k_1_vec));
    nlo_rk4_debug_log_vec_stats(state,
                                state->current_field_vec,
                                "current_field_end",
                                step_index,
                                state->current_z + step,
                                step);

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_rk4_attempt_step_doubling(
    simulation_state* state,
    double step,
    size_t step_index,
    double atol,
    double rtol,
    double* out_error
)
{
    if (state == NULL || state->backend == NULL || out_error == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    simulation_working_vectors* work = &state->working_vectors;
    const double half_step = 0.5 * step;

    nlo_vec_status status = nlo_vec_complex_copy(state->backend,
                                                 work->previous_field_vec,
                                                 state->current_field_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    state->current_step_size = step;
    state->current_half_step_exp = 0.5 * step;
    status = nlo_rk4_step_device(state, step_index);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_copy(state->backend,
                                  work->field_magnitude_vec,
                                  state->current_field_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_copy(state->backend,
                                  state->current_field_vec,
                                  work->previous_field_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    state->current_step_size = half_step;
    state->current_half_step_exp = 0.5 * half_step;
    status = nlo_rk4_step_device(state, step_index);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_rk4_step_device(state, step_index);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_weighted_rms_error(state->backend,
                                                state->current_field_vec,
                                                work->field_magnitude_vec,
                                                atol,
                                                rtol,
                                                out_error);
    return status;
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

    nlo_rk4_debug_reset_run();
    nlo_rk4_debug_log_solver_config(state, tol);

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
    const double record_capture_eps = nlo_record_capture_tolerance(z_end);
    const double rtol = tol;
    const double atol = fmax(tol * NLO_RK4_ATOL_SCALE, NLO_RK4_ATOL_MIN);
    if (state->num_recorded_samples > 1u)
    {
        record_spacing = z_end / (double)(state->num_recorded_samples - 1u);
        next_record_z = record_spacing * (double)state->current_record_index;
    }

    size_t rk4_step_index = 0u;
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

        int step_accepted = 0;
        size_t reject_attempt = 0u;
        double error = 0.0;
        double scale = NLO_RK4_STEP_SHRINK_MIN;
        while (!step_accepted)
        {
            nlo_vec_status status = nlo_rk4_attempt_step_doubling(state,
                                                                  step,
                                                                  rk4_step_index,
                                                                  atol,
                                                                  rtol,
                                                                  &error);
            if (status != NLO_VEC_STATUS_OK) {
                step_accepted = -1;
                break;
            }

            if (!isfinite(error) || error < 0.0) {
                error = DBL_MAX;
            }

            if (error > 0.0 && isfinite(error)) {
                scale = NLO_RK4_ERROR_SAFETY * pow(1.0 / error, 0.2);
            } else {
                scale = NLO_RK4_STEP_GROW_MAX;
            }
            if (scale < NLO_RK4_STEP_SHRINK_MIN) {
                scale = NLO_RK4_STEP_SHRINK_MIN;
            } else if (scale > NLO_RK4_STEP_GROW_MAX) {
                scale = NLO_RK4_STEP_GROW_MAX;
            }

            if (error <= 1.0 || step <= (min_step * (1.0 + 1e-12))) {
                state->current_z += step;
                state->current_step_size = nlo_clamp_step(step * scale, min_step, max_step);
                step_accepted = 1;
                break;
            }

            if (nlo_vec_complex_copy(state->backend,
                                     state->current_field_vec,
                                     state->working_vectors.previous_field_vec) != NLO_VEC_STATUS_OK) {
                step_accepted = -1;
                break;
            }

            step = nlo_clamp_step(step * scale, min_step, max_step);
            const double remaining_retry = z_end - state->current_z;
            if (step > remaining_retry) {
                step = remaining_retry;
            }
            if (record_spacing > 0.0 &&
                state->current_record_index < state->num_recorded_samples &&
                next_record_z > state->current_z) {
                const double to_next_record = next_record_z - state->current_z;
                if (to_next_record > 0.0 && step > to_next_record) {
                    step = to_next_record;
                }
            }
            if (step <= 0.0) {
                step_accepted = -1;
                break;
            }

            reject_attempt += 1u;
            if (reject_attempt >= NLO_RK4_MAX_REJECTION_ATTEMPTS) {
                step_accepted = -1;
                break;
            }
        }

        if (step_accepted < 0) {
            break;
        }

        nlo_rk4_debug_log_error_control(rk4_step_index,
                                        state->current_z,
                                        step,
                                        error,
                                        scale,
                                        state->current_step_size);

        while (record_spacing > 0.0 &&
               state->current_record_index < state->num_recorded_samples &&
               state->current_z + record_capture_eps >= next_record_z)
        {
            if (simulation_state_capture_snapshot(state) != NLO_VEC_STATUS_OK)
            {
                break;
            }
            next_record_z = record_spacing * (double)state->current_record_index;
        }

        rk4_step_index += 1u;
    }

    while (record_spacing > 0.0 &&
           state->current_record_index < state->num_recorded_samples &&
           state->current_z + record_capture_eps >= next_record_z)
    {
        if (simulation_state_capture_snapshot(state) != NLO_VEC_STATUS_OK)
        {
            break;
        }
        next_record_z = record_spacing * (double)state->current_record_index;
    }

    (void)nlo_vec_end_simulation(state->backend);
    (void)simulation_state_flush_snapshots(state);
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

    nlo_rk4_debug_reset_run();

    state->current_half_step_exp = 0.5 * state->current_step_size;
    if (nlo_rk4_step_device(state, 0u) == NLO_VEC_STATUS_OK)
    {
        state->current_z += state->current_step_size;
    }

    (void)nlo_vec_end_simulation(state->backend);
}

