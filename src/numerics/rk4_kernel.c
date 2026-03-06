/**
 * @file rk4_kernel.c
 * @brief Backend-resident RK4 integration kernel.
 */

#include "numerics/rk4_kernel.h"
#include "fft/fft.h"
#include "io/log_sink.h"
#include "physics/operators.h"
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

#ifndef NLO_RK4_STEP_SCALE_MIN
#define NLO_RK4_STEP_SCALE_MIN 0.5
#endif

#ifndef NLO_RK4_STEP_SCALE_MAX
#define NLO_RK4_STEP_SCALE_MAX 2.0
#endif

#ifndef NLO_RK4_REL_ERROR_ATOL_FLOOR
#define NLO_RK4_REL_ERROR_ATOL_FLOOR 1e-14
#endif

#ifndef NLO_RK4_MAX_REJECTION_ATTEMPTS
#define NLO_RK4_MAX_REJECTION_ATTEMPTS 64u
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

static int nlo_rk4_exact_fixed_step_requested(const sim_config* config)
{
    if (config == NULL) {
        return 0;
    }

    const double start_step = config->propagation.starting_step_size;
    const double min_step = config->propagation.min_step_size;
    const double max_step = config->propagation.max_step_size;
    if (!(start_step > 0.0) || !(min_step > 0.0) || !(max_step > 0.0)) {
        return 0;
    }

    const double scale = fmax(1.0, fmax(fabs(start_step), fmax(fabs(min_step), fabs(max_step))));
    const double eps = 64.0 * DBL_EPSILON * scale;
    return (fabs(start_step - min_step) <= eps && fabs(start_step - max_step) <= eps) ? 1 : 0;
}

static int nlo_rk4_scalars_near(double lhs, double rhs)
{
    const double scale = fmax(1.0, fmax(fabs(lhs), fabs(rhs)));
    const double eps = 64.0 * DBL_EPSILON * scale;
    return (fabs(lhs - rhs) <= eps) ? 1 : 0;
}

static int nlo_rk4_nonlinear_depends_on_h(const simulation_state* state)
{
    if (state == NULL || state->nonlinear_model != NLO_NONLINEAR_MODEL_EXPR) {
        return 0;
    }

    const nlo_operator_program* program = &state->nonlinear_operator_program;
    if (!program->active) {
        return 0;
    }

    for (size_t i = 0u; i < program->instruction_count; ++i) {
        if (program->instructions[i].opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_H) {
            return 1;
        }
    }

    return 0;
}

static nlo_vec_status nlo_rk4_capture_interpolated_record(
    simulation_state* state,
    double step_start_z,
    double step_end_z,
    double record_z
)
{
    if (state == NULL || state->backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    simulation_working_vectors* work = &state->working_vectors;
    if (work->previous_field_vec == NULL ||
        work->field_working_vec == NULL ||
        work->field_magnitude_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const double dz = step_end_z - step_start_z;
    const double scale = fmax(1.0, fmax(fabs(step_start_z), fabs(step_end_z)));
    const double alpha_eps = 64.0 * DBL_EPSILON * scale;
    double alpha = 1.0;
    if (dz > 0.0) {
        alpha = (record_z - step_start_z) / dz;
    }

    if (alpha <= alpha_eps) {
        return simulation_state_capture_snapshot_from_vec(state, work->previous_field_vec);
    }
    if (alpha >= (1.0 - alpha_eps)) {
        return simulation_state_capture_snapshot_from_vec(state, state->current_field_vec);
    }

    if (alpha < 0.0) {
        alpha = 0.0;
    } else if (alpha > 1.0) {
        alpha = 1.0;
    }

    NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, work->field_working_vec, work->previous_field_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(state->backend, work->field_working_vec, nlo_make(1.0 - alpha, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, work->field_magnitude_vec, state->current_field_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(state->backend, work->field_magnitude_vec, nlo_make(alpha, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(state->backend, work->field_working_vec, work->field_magnitude_vec));

    return simulation_state_capture_snapshot_from_vec(state, work->field_working_vec);
}

static void nlo_rk4_emit_step_event(
    simulation_state* state,
    size_t step_index,
    double z_current,
    double step_size,
    double next_step_size,
    double error
)
{
    if (state == NULL || state->step_event_capacity == 0u || state->step_event_buffer == NULL) {
        return;
    }

    if (state->step_events_written < state->step_event_capacity) {
        nlo_step_event* event = &state->step_event_buffer[state->step_events_written];
        event->step_index = step_index;
        event->z_current = z_current;
        event->step_size = step_size;
        event->next_step_size = next_step_size;
        event->error = error;
        state->step_events_written += 1u;
        return;
    }

    state->step_events_dropped += 1u;
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
    
    // Copy current field to previous field buffer for interpolation
    NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, work->previous_field_vec, state->current_field_vec));

    // Build interaction-picture field: A_I = exp(D*h/2) * A_n.
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, state->current_field_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->ip_field_vec));
    nlo_rk4_debug_log_vec_stats(state, work->ip_field_vec, "ip_field", step_index, state->current_z, step);

    // k1 = h * exp(D*h/2) * N(A_n)
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    state->current_field_vec,
                                                    work->field_working_vec));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_final_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_final_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_final_vec, "k1", step_index, state->current_z, step);
    
    // k2 = h * N(A_I + k1/2)
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_final_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(0.5, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_temp_vec, "k2", step_index, state->current_z, step);
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_temp_vec));

    // k1/6 + k2/3
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_final_vec, nlo_make(1.0 / 6.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(1.0 / 3.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));
    
    // k3 = h * N(A_I + k2/2)
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(0.5, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_temp_vec, "k3", step_index, state->current_z, step);
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_temp_vec));

    // psi = exp(D*h/2) * (A_I + k1/6 + k2/3 + k3/3)
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(1.0 / 3.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->k_final_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_final_vec));

    // k4 = h * N(exp(D*h/2) * (A_I + k3))
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->field_working_vec));

    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_temp_vec, "k4", step_index, state->current_z, step);

    // A_{n+1} = exp(D*h/2) * (A_I + k1/6 + k2/3 + k3/3) + k4/6
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(1.0 / 6.0, 0.0)));  
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));

    NLO_RK4_CALL(nlo_vec_complex_copy(backend, state->current_field_vec, work->k_final_vec));
    nlo_rk4_debug_log_vec_stats(state,
                                state->current_field_vec,
                                "current_field_end",
                                step_index,
                                state->current_z + step,
                                step);

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_rk4_attempt_embedded_erk43(
    simulation_state* state,
    double step,
    size_t step_index,
    double* out_error
)
{
    if (state == NULL || state->backend == NULL || out_error == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (state->fft_plan == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vector_backend* backend = state->backend;
    simulation_working_vectors* work = &state->working_vectors;

    state->current_step_size = step;
    state->current_half_step_exp = 0.5 * step;

    nlo_rk4_debug_log_vec_stats(state,
                                state->current_field_vec,
                                "current_field_start",
                                step_index,
                                state->current_z,
                                step);

    // Copy current field to previous field buffer for interpolation
    NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, work->previous_field_vec, state->current_field_vec));

    // Build interaction-picture field: A_I = exp(D*h/2) * A_n.
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, state->current_field_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->ip_field_vec));
    nlo_rk4_debug_log_vec_stats(state, work->ip_field_vec, "ip_field", step_index, state->current_z, step);

    // k1 = h * exp(D*h/2) * N(A_n) reuse k_5 from previous step
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->k_temp_vec, work->field_working_vec));

    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->k_temp_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_final_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_final_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_final_vec, "k1", step_index, state->current_z, step);
    
    // k2 = h * N(A_I + k1/2)
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_final_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(0.5, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_temp_vec, "k2", step_index, state->current_z, step);
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_temp_vec));

    // k1/6 + k2/3
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_final_vec, nlo_make(1.0 / 6.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(1.0 / 3.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));

    // k3 = h * N(A_I + k2/2)
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(0.5, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_temp_vec, "k3", step_index, state->current_z, step);
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_working_vec, work->k_temp_vec));

    // psi = exp(D*h/2) * (A_I + k1/6 + k2/3 + k3/3)
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(1.0 / 3.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->k_final_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_final_vec));

    // k4 = h * N(exp(D*h/2) * (A_I + k3))
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));
    NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    NLO_RK4_CALL(nlo_apply_dispersion_operator_stage(state, work->field_freq_vec));
    NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->field_working_vec));

    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(step, 0.0)));
    nlo_rk4_debug_log_vec_stats(state, work->k_temp_vec, "k4", step_index, state->current_z, step);

    // psi{k_final} -> {field_freq}
    NLO_RK4_CALL(nlo_vec_complex_copy(backend, work->field_freq_vec, work->k_final_vec));

    // A^[4]{k_final} = psi{k_final} + k4/6{k_temp}
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(1.0 / 6.0, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));

    // k5{field_working_vec} = N(A^[4]{k_final}) (raw nonlinear operator value, saved for optional next-step k1 reuse).
    NLO_RK4_CALL(nlo_apply_nonlinear_operator_stage(state,
                                                    work->k_final_vec,
                                                    work->field_working_vec));
    nlo_rk4_debug_log_vec_stats(state, work->field_working_vec, "k5", step_index, state->current_z + step, step);

    // A^[3]{field_freq} = psi{field_freq} + (2*6*(k4/6){k_temp} + 3*h*k5{field_working_vec})/30
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->field_working_vec, nlo_make(step * 0.1, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, nlo_make(step * 0.4, 0.0)));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_freq_vec, work->field_working_vec));
    NLO_RK4_CALL(nlo_vec_complex_add_inplace(backend, work->field_freq_vec, work->k_temp_vec));

    // Relative ERK4(3)-IP defect:
    //   delta_rel = sqrt(sum(|A^[4]-A^[3]|^2) / sum((a_floor + |A^[4]|)^2))
    // This aligns solver tolerance with Python/MATLAB rtol semantics.
    NLO_RK4_CALL(nlo_vec_complex_weighted_rms_error(backend,
                                                work->k_final_vec,
                                                work->field_freq_vec,
                                                NLO_RK4_REL_ERROR_ATOL_FLOOR,
                                                1.0,
                                                out_error));

    if (!isfinite(*out_error) || *out_error < 0.0) {
        *out_error = DBL_MAX;
    }

    NLO_RK4_CALL(nlo_vec_complex_copy(backend, state->current_field_vec, work->k_final_vec));
    nlo_rk4_debug_log_vec_stats(state,
                                state->current_field_vec,
                                "current_field_end",
                                step_index,
                                state->current_z + step,
                                step);

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
    const int fixed_step_mode = nlo_rk4_exact_fixed_step_requested(state->config);
    const int has_explicit_schedule =
        (state->explicit_record_schedule_active != 0 &&
         state->explicit_record_z != NULL &&
         state->explicit_record_z_count > 0u);
    const int disable_record_interpolation = (fixed_step_mode && !has_explicit_schedule) ? 1 : 0;
    const int nonlinear_depends_on_h = nlo_rk4_nonlinear_depends_on_h(state);

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

    if (!has_explicit_schedule &&
        state->current_record_index == 0u &&
        state->num_recorded_samples > 1u)
    {
        if (simulation_state_capture_snapshot(state) != NLO_VEC_STATUS_OK)
        {
            (void)nlo_vec_end_simulation(state->backend);
            return;
        }
    }

    nlo_log_progress_begin(state->current_z, z_end);

    double record_spacing = 0.0;
    double next_record_z = z_end;
    const double record_capture_eps = nlo_record_capture_tolerance(z_end);
    if (state->num_recorded_samples > 1u)
    {
        if (has_explicit_schedule) {
            if (state->current_record_index < state->explicit_record_z_count) {
                next_record_z = state->explicit_record_z[state->current_record_index];
            } else {
                next_record_z = z_end;
            }
        } else {
            record_spacing = z_end / (double)(state->num_recorded_samples - 1u);
            next_record_z = record_spacing * (double)state->current_record_index;
        }
    }

    size_t rk4_step_index = 0u;
    int terminated_early = 0;
    int min_step_tol_warning_emitted = 0;
    while (state->current_z < z_end)
    {
        double step = nlo_clamp_step(state->current_step_size, min_step, max_step);
        if (step <= 0.0)
        {
            terminated_early = 1;
            break;
        }

        const double remaining = z_end - state->current_z;
        if (step > remaining)
        {
            step = remaining;
        }

        if (step <= 0.0)
        {
            terminated_early = 1;
            break;
        }

        int step_accepted = 0;
        size_t reject_attempt = 0u;
        double error = 0.0;
        double scale = 1.0;

        if (fixed_step_mode) {
            state->current_step_size = step;
            state->current_half_step_exp = 0.5 * step;
            if (nlo_rk4_step_device(state, rk4_step_index) != NLO_VEC_STATUS_OK) {
                step_accepted = -1;
            } else {
                state->current_z += step;
                state->current_step_size = step;
                error = 0.0;
                scale = 1.0;
                nlo_rk4_emit_step_event(state,
                                        rk4_step_index,
                                        state->current_z,
                                        step,
                                        state->current_step_size,
                                        error);
                nlo_log_progress_step_accepted(rk4_step_index,
                                                state->current_z,
                                                z_end,
                                                step,
                                                error,
                                                state->current_step_size);
                step_accepted = 1;
            }
        } else {
            while (!step_accepted)
            {

                NLO_RK4_CALL(nlo_rk4_attempt_embedded_erk43(state, step, rk4_step_index, &error));

                if (!isfinite(error) || error < 0.0) {
                    error = DBL_MAX;
                }

                if (error > 0.0 && isfinite(error)) {
                    scale = pow(tol / error, 0.25);
                } else if (error == 0.0) {
                    scale = NLO_RK4_STEP_SCALE_MAX;
                } else {
                    scale = NLO_RK4_STEP_SCALE_MIN;
                }

                if (scale < NLO_RK4_STEP_SCALE_MIN) {
                    scale = NLO_RK4_STEP_SCALE_MIN;
                } else if (scale > NLO_RK4_STEP_SCALE_MAX) {
                    scale = NLO_RK4_STEP_SCALE_MAX;
                }
                
                const int step_too_small = (step <= (min_step * (1.0 + 1e-12)));
                const int min_step_forced_accept = (step_too_small && error > tol ? 1 : 0);
                if (error <= tol || step_too_small) {
                    state->current_z += step;
                    state->current_step_size = nlo_clamp_step(step * scale, min_step, max_step);
                    if (min_step_forced_accept && min_step_tol_warning_emitted == 0) {
                        nlo_log_emit(NLO_LOG_LEVEL_WARN,
                                     "[nlolib] adaptive solver reached min_step_size while local relative error "
                                     "remains above tolerance; continuing with constrained steps. "
                                     "z=%.9e step=%.9e error=%.9e tol=%.9e",
                                     state->current_z,
                                     step,
                                     error,
                                     tol);
                        min_step_tol_warning_emitted = 1;
                    }
                    nlo_rk4_emit_step_event(state,
                                            rk4_step_index,
                                            state->current_z,
                                            step,
                                            state->current_step_size,
                                            error);
                    nlo_log_progress_step_accepted(rk4_step_index,
                                                   state->current_z,
                                                   z_end,
                                                   step,
                                                   error,
                                                   state->current_step_size);
                    step_accepted = 1;
                    break;
                }

                const double attempted_step = step;
                double retry_step = nlo_clamp_step(step * scale, min_step, max_step);
                const double remaining_retry = z_end - state->current_z;
                if (retry_step > remaining_retry) {
                    retry_step = remaining_retry;
                }
                if (retry_step <= 0.0) {
                    step_accepted = -1;
                    break;
                }

                reject_attempt += 1u;
                nlo_log_progress_step_rejected(rk4_step_index,
                                               state->current_z,
                                               z_end,
                                               attempted_step,
                                               error,
                                               retry_step,
                                               reject_attempt);

                step = retry_step;
                if (reject_attempt >= NLO_RK4_MAX_REJECTION_ATTEMPTS) {
                    step_accepted = -1;
                    break;
                }
            }
        }

        if (step_accepted < 0) {
            terminated_early = 1;
            break;
        }

        nlo_rk4_debug_log_error_control(rk4_step_index,
                                        state->current_z,
                                        step,
                                        error,
                                        scale,
                                        state->current_step_size);

        const double step_end_z = state->current_z;
        const double step_start_z = step_end_z - step;
        while ((has_explicit_schedule || record_spacing > 0.0) &&
               state->current_record_index < state->num_recorded_samples &&
               step_end_z + record_capture_eps >= next_record_z)
        {
            nlo_vec_status capture_status = NLO_VEC_STATUS_OK;
            if (disable_record_interpolation) {
                capture_status = simulation_state_capture_snapshot(state);
            } else {
                capture_status = nlo_rk4_capture_interpolated_record(state,
                                                                     step_start_z,
                                                                     step_end_z,
                                                                     next_record_z);
            }
            if (capture_status != NLO_VEC_STATUS_OK)
            {
                terminated_early = 1;
                break;
            }
            if (has_explicit_schedule) {
                if (state->current_record_index < state->explicit_record_z_count) {
                    next_record_z = state->explicit_record_z[state->current_record_index];
                } else {
                    next_record_z = z_end;
                }
            } else {
                next_record_z = record_spacing * (double)state->current_record_index;
            }
        }
        if (terminated_early != 0) {
            break;
        }

        rk4_step_index += 1u;
    }

    while ((has_explicit_schedule || record_spacing > 0.0) &&
           state->current_record_index < state->num_recorded_samples &&
           state->current_z + record_capture_eps >= next_record_z)
    {
        if (simulation_state_capture_snapshot(state) != NLO_VEC_STATUS_OK)
        {
            break;
        }
        if (has_explicit_schedule) {
            if (state->current_record_index < state->explicit_record_z_count) {
                next_record_z = state->explicit_record_z[state->current_record_index];
            } else {
                next_record_z = z_end;
            }
        } else {
            next_record_z = record_spacing * (double)state->current_record_index;
        }
    }

    (void)nlo_vec_end_simulation(state->backend);
    (void)simulation_state_flush_snapshots(state);
    nlo_log_progress_finish(state->current_z,
                            z_end,
                            (terminated_early == 0 && state->current_z + record_capture_eps >= z_end) ? 1 : 0);
}
