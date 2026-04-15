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

#ifndef RK4_ERROR_TOL
#define RK4_ERROR_TOL 1e-6
#endif

#ifndef RK4_STEP_SCALE_MIN
#define RK4_STEP_SCALE_MIN 0.5
#endif

#ifndef RK4_STEP_SCALE_MAX
#define RK4_STEP_SCALE_MAX 2.0
#endif

#ifndef RK4_REL_ERROR_ATOL_FLOOR
#define RK4_REL_ERROR_ATOL_FLOOR 1e-14
#endif

#ifndef RK4_MAX_REJECTION_ATTEMPTS
#define RK4_MAX_REJECTION_ATTEMPTS 64u
#endif

#ifndef RK4_STRICT_STATUS_CHECKS
#if defined(NDEBUG)
#define RK4_STRICT_STATUS_CHECKS 0
#else
#define RK4_STRICT_STATUS_CHECKS 1
#endif
#endif

#if RK4_STRICT_STATUS_CHECKS
#define RK4_CALL(expr)                          \
    do                                              \
    {                                               \
        const vec_status call_status_ = (expr); \
        if (call_status_ != VEC_STATUS_OK)      \
        {                                           \
            return call_status_;                    \
        }                                           \
    } while (0)
#else
#define RK4_CALL(expr)                          \
    do                                              \
    {                                               \
        const vec_status call_status_ = (expr); \
        assert(call_status_ == VEC_STATUS_OK);  \
        (void)call_status_;                         \
    } while (0)
#endif

static double clamp_step(double value, double min_step, double max_step)
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

static int rk4_debug_enabled_runtime(void)
{
    const char* env = getenv("RK4_DEBUG");
    return (env != NULL && *env != '\0' && *env != '0') ? 1 : 0;
}

static double record_capture_tolerance(double z_end)
{
    const double scale = fmax(1.0, fabs(z_end));
    return 64.0 * DBL_EPSILON * scale;
}

static int rk4_exact_fixed_step_requested(const sim_config* config)
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

static int rk4_nonlinear_depends_on_h(const simulation_state* state)
{
    if (state == NULL || state->nonlinear_model != NONLINEAR_MODEL_EXPR) {
        return 0;
    }

    const operator_program* program = &state->nonlinear_operator_program;
    if (!program->active) {
        return 0;
    }

    for (size_t i = 0u; i < program->instruction_count; ++i) {
        if (program->instructions[i].opcode == OPERATOR_OP_PUSH_SYMBOL_H) {
            return 1;
        }
    }

    return 0;
}

static vec_status rk4_capture_interpolated_record(
    simulation_state* state,
    double step_start_z,
    double step_end_z,
    double record_z
)
{
    if (state == NULL || state->backend == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    simulation_working_vectors* work = &state->working_vectors;
    if (work->previous_field_vec == NULL ||
        work->field_freq_vec == NULL ||
        work->k_temp_vec == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
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

    RK4_CALL(vec_complex_copy(state->backend, work->field_freq_vec, work->previous_field_vec));
    RK4_CALL(vec_complex_scalar_mul_inplace(state->backend,
                                                    work->field_freq_vec,
                                                    make(1.0 - alpha, 0.0)));
    RK4_CALL(vec_complex_copy(state->backend, work->k_temp_vec, state->current_field_vec));
    RK4_CALL(vec_complex_scalar_mul_inplace(state->backend,
                                                    work->k_temp_vec,
                                                    make(alpha, 0.0)));
    RK4_CALL(vec_complex_add_inplace(state->backend, work->field_freq_vec, work->k_temp_vec));

    return simulation_state_capture_snapshot_from_vec(state, work->field_freq_vec);
}

static void rk4_emit_step_event(
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
        step_event* event = &state->step_event_buffer[state->step_events_written];
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

static void rk4_debug_log_solver_config(const simulation_state* state, double tol)
{
    if (!rk4_debug_enabled_runtime() || state == NULL || state->config == NULL) {
        return;
    }

    const sim_config* config = state->config;
    const double c0 = (config->runtime.num_constants > 0u) ? config->runtime.constants[0] : -0.5;
    const double c1 = (config->runtime.num_constants > 1u) ? config->runtime.constants[1] : 0.0;
    const double c2 = (config->runtime.num_constants > 2u) ? config->runtime.constants[2] : 1.0;

    fprintf(stderr,
            "[RK4_DEBUG] config c0=%.9e c1=%.9e c2=%.9e dt=%.9e z_end=%.9e h0=%.9e h_min=%.9e h_max=%.9e tol=%.9e\n",
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

static vec_status rk4_step_device(simulation_state *state, size_t step_index)
{
    if (state == NULL || state->backend == NULL || state->fft_plan == NULL)
    {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    vector_backend *backend = state->backend;
    simulation_working_vectors *work = &state->working_vectors;
    const double step = state->current_step_size;

    rk4_debug_log_vec_stats(state,
                                state->current_field_vec,
                                "current_field_start",
                                step_index,
                                state->current_z,
                                step);
    
    // Copy current field to previous field buffer for interpolation
    RK4_CALL(vec_complex_copy(state->backend, work->previous_field_vec, state->current_field_vec));

    // Build interaction-picture field: A_I = exp(D*h/2) * A_n.
    RK4_CALL(fft_forward_vec(state->fft_plan, state->current_field_vec, work->field_freq_vec));
    RK4_CALL(apply_dispersion_operator_stage(state, work->field_freq_vec));
    RK4_CALL(fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->ip_field_vec));
    rk4_debug_log_vec_stats(state, work->ip_field_vec, "ip_field", step_index, state->current_z, step);

    // k1 = exp(D*h/2) * N(A_n)
    RK4_CALL(apply_nonlinear_operator_stage(state,
                                                    state->current_field_vec,
                                                    work->field_working_vec));
    RK4_CALL(fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    RK4_CALL(apply_dispersion_operator_stage(state, work->field_freq_vec));
    RK4_CALL(fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_final_vec));
    rk4_debug_log_vec_stats(state, work->k_final_vec, "k1", step_index, state->current_z, step);
    
    // k2 = N(A_I + h*k1/2)
    RK4_CALL(vec_complex_copy(backend, work->field_working_vec, work->k_final_vec));
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->field_working_vec, make(step * 0.5, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    RK4_CALL(apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    rk4_debug_log_vec_stats(state, work->k_temp_vec, "k2", step_index, state->current_z, step);
    RK4_CALL(vec_complex_copy(backend, work->field_working_vec, work->k_temp_vec));

    // k1/6 + k2/3
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_final_vec, make(1.0 / 6.0, 0.0)));
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, make(1.0 / 3.0, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));
    
    // k3 = N(A_I + h*k2/2)
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->field_working_vec, make(step * 0.5, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    RK4_CALL(apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    rk4_debug_log_vec_stats(state, work->k_temp_vec, "k3", step_index, state->current_z, step);
    RK4_CALL(vec_complex_copy(backend, work->field_working_vec, work->k_temp_vec));

    // psi = exp(D*h/2) * (A_I + h*(k1/6 + k2/3 + k3/3))
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, make(1.0 / 3.0, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_final_vec, make(step, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->k_final_vec, work->ip_field_vec));
    RK4_CALL(fft_forward_vec(state->fft_plan, work->k_final_vec, work->field_freq_vec));
    RK4_CALL(apply_dispersion_operator_stage(state, work->field_freq_vec));
    RK4_CALL(fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_final_vec));

    // k4 = N(exp(D*h/2) * (A_I + h*k3))
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->field_working_vec, make(step, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));
    RK4_CALL(fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    RK4_CALL(apply_dispersion_operator_stage(state, work->field_freq_vec));
    RK4_CALL(fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->field_working_vec));

    RK4_CALL(apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    rk4_debug_log_vec_stats(state, work->k_temp_vec, "k4", step_index, state->current_z, step);

    // A_{n+1} = exp(D*h/2) * (A_I + h*(k1/6 + k2/3 + k3/3)) + h*k4/6
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, make(step * 1.0 / 6.0, 0.0)));  
    RK4_CALL(vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));

    RK4_CALL(vec_complex_copy(backend, state->current_field_vec, work->k_final_vec));
    rk4_debug_log_vec_stats(state,
                                state->current_field_vec,
                                "current_field_end",
                                step_index,
                                state->current_z + step,
                                step);

    return VEC_STATUS_OK;
}

static vec_status rk4_attempt_embedded_erk43(
    simulation_state* state,
    double step,
    size_t step_index,
    int reuse_cached_k5,
    double* out_error
)
{
    if (state == NULL || state->backend == NULL || out_error == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    if (state->fft_plan == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    vector_backend* backend = state->backend;
    simulation_working_vectors* work = &state->working_vectors;

    state->current_step_size = step;
    state->current_half_step_exp = 0.5 * step;

    rk4_debug_log_vec_stats(state,
                                state->current_field_vec,
                                "current_field_start",
                                step_index,
                                state->current_z,
                                step);

    // Copy current field to previous field buffer for interpolation
    RK4_CALL(vec_complex_copy(state->backend, work->previous_field_vec, state->current_field_vec));

    // Build interaction-picture field: A_I = exp(D*h/2) * A_n.
    RK4_CALL(fft_forward_vec(state->fft_plan, state->current_field_vec, work->field_freq_vec));
    RK4_CALL(apply_dispersion_operator_stage(state, work->field_freq_vec));
    RK4_CALL(fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->ip_field_vec));
    rk4_debug_log_vec_stats(state, work->ip_field_vec, "ip_field", step_index, state->current_z, step);

    // k1 = exp(D*h/2) * N(A_n), reusing raw k5 from the previous accepted step.
    if (reuse_cached_k5) {
        RK4_CALL(vec_complex_copy(backend, work->k_temp_vec, work->field_working_vec));
    } else {
        RK4_CALL(apply_nonlinear_operator_stage(state,
                                                        state->current_field_vec,
                                                        work->k_temp_vec));
    }

    RK4_CALL(fft_forward_vec(state->fft_plan, work->k_temp_vec, work->field_freq_vec));
    RK4_CALL(apply_dispersion_operator_stage(state, work->field_freq_vec));
    RK4_CALL(fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_final_vec));
    rk4_debug_log_vec_stats(state, work->k_final_vec, "k1", step_index, state->current_z, step);
    
    // k2 = N(A_I + h*k1/2)
    RK4_CALL(vec_complex_copy(backend, work->field_working_vec, work->k_final_vec));
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->field_working_vec, make(step * 0.5, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    RK4_CALL(apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    rk4_debug_log_vec_stats(state, work->k_temp_vec, "k2", step_index, state->current_z, step);
    RK4_CALL(vec_complex_copy(backend, work->field_working_vec, work->k_temp_vec));

    // k1/6 + k2/3
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_final_vec, make(1.0 / 6.0, 0.0)));
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, make(1.0 / 3.0, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));

    // k3 = N(A_I + h*k2/2)
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->field_working_vec, make(step * 0.5, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));

    RK4_CALL(apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    rk4_debug_log_vec_stats(state, work->k_temp_vec, "k3", step_index, state->current_z, step);
    RK4_CALL(vec_complex_copy(backend, work->field_working_vec, work->k_temp_vec));

    // psi = exp(D*h/2) * (A_I + h*(k1/6 + k2/3 + k3/3))
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, make(1.0 / 3.0, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_final_vec, make(step, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->k_final_vec, work->ip_field_vec));
    RK4_CALL(fft_forward_vec(state->fft_plan, work->k_final_vec, work->field_freq_vec));
    RK4_CALL(apply_dispersion_operator_stage(state, work->field_freq_vec));
    RK4_CALL(fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->k_final_vec));

    // k4 = N(exp(D*h/2) * (A_I + h*k3))
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->field_working_vec, make(step, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->field_working_vec, work->ip_field_vec));
    RK4_CALL(fft_forward_vec(state->fft_plan, work->field_working_vec, work->field_freq_vec));
    RK4_CALL(apply_dispersion_operator_stage(state, work->field_freq_vec));
    RK4_CALL(fft_inverse_vec(state->fft_plan, work->field_freq_vec, work->field_working_vec));

    RK4_CALL(apply_nonlinear_operator_stage(state,
                                                    work->field_working_vec,
                                                    work->k_temp_vec));
    rk4_debug_log_vec_stats(state, work->k_temp_vec, "k4", step_index, state->current_z, step);

    // psi{k_final} -> {field_freq}
    RK4_CALL(vec_complex_copy(backend, work->field_freq_vec, work->k_final_vec));

    // A^[4]{k_final} = psi{k_final} + h*k4/6{k_temp}
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, make(step * 1.0 / 6.0, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->k_final_vec, work->k_temp_vec));

    // k5{field_working_vec} = N(A^[4]{k_final}) (raw nonlinear operator value, saved for optional next-step k1 reuse).
    RK4_CALL(apply_nonlinear_operator_stage(state,
                                                    work->k_final_vec,
                                                    work->field_working_vec));
    rk4_debug_log_vec_stats(state, work->field_working_vec, "k5", step_index, state->current_z + step, step);

    // A^[3]{field_freq} = psi{field_freq} + (2*6*(h*k4/6){k_temp} + 3*h*k5)/30.
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, make(0.4, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->field_freq_vec, work->k_temp_vec));
    RK4_CALL(vec_complex_copy(backend, work->k_temp_vec, work->field_working_vec));
    RK4_CALL(vec_complex_scalar_mul_inplace(backend, work->k_temp_vec, make(step * 0.1, 0.0)));
    RK4_CALL(vec_complex_add_inplace(backend, work->field_freq_vec, work->k_temp_vec));

    // Relative ERK4(3)-IP defect:
    //   delta_rel = sqrt(sum(|A^[4]-A^[3]|^2) / sum((a_floor + |A^[4]|)^2))
    // This aligns solver tolerance with Python/MATLAB rtol semantics.
    RK4_CALL(vec_complex_weighted_rms_error(backend,
                                                work->k_final_vec,
                                                work->field_freq_vec,
                                                RK4_REL_ERROR_ATOL_FLOOR,
                                                1.0,
                                                out_error));

    if (!isfinite(*out_error) || *out_error < 0.0) {
        *out_error = DBL_MAX;
    }

    rk4_debug_log_vec_stats(state,
                                work->k_final_vec,
                                "current_field_end",
                                step_index,
                                state->current_z + step,
                                step);

    return VEC_STATUS_OK;
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
                           : RK4_ERROR_TOL;
    const int fixed_step_mode = rk4_exact_fixed_step_requested(state->config);
    const int has_explicit_schedule =
        (state->explicit_record_schedule_active != 0 &&
         state->explicit_record_z != NULL &&
         state->explicit_record_z_count > 0u);
    const int disable_record_interpolation = (fixed_step_mode && !has_explicit_schedule) ? 1 : 0;
    const int nonlinear_depends_on_h = rk4_nonlinear_depends_on_h(state);
    if (state->current_step_size <= 0.0)
    {
        state->current_step_size = state->config->propagation.starting_step_size;
    }

    rk4_debug_reset_run();
    rk4_debug_log_solver_config(state, tol);

    if (vec_begin_simulation(state->backend) != VEC_STATUS_OK)
    {
        return;
    }

    if (!has_explicit_schedule &&
        state->current_record_index == 0u &&
        state->num_recorded_samples > 1u)
    {
        if (simulation_state_capture_snapshot(state) != VEC_STATUS_OK)
        {
            (void)vec_end_simulation(state->backend);
            return;
        }
    }

    log_progress_begin(state->current_z, z_end);

    double record_spacing = 0.0;
    double next_record_z = z_end;
    const double record_capture_eps = record_capture_tolerance(z_end);
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
    int cached_k5_valid = 0;
    int terminated_early = 0;
    int min_step_tol_warning_emitted = 0;
    while (state->current_z < z_end)
    {
        double step = clamp_step(state->current_step_size, min_step, max_step);
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
            if (rk4_step_device(state, rk4_step_index) != VEC_STATUS_OK) {
                step_accepted = -1;
            } else {
                state->current_z += step;
                state->current_step_size = step;
                error = 0.0;
                scale = 1.0;
                rk4_emit_step_event(state,
                                        rk4_step_index,
                                        state->current_z,
                                        step,
                                        state->current_step_size,
                                        error);
                log_progress_step_accepted(rk4_step_index,
                                                state->current_z,
                                                z_end,
                                                step,
                                                error,
                                                state->current_step_size);
                if (log_progress_abort_requested() != 0) {
                    terminated_early = 1;
                }
                step_accepted = 1;
            }
        } else {
            while (!step_accepted)
            {
                const int reuse_cached_k5 =
                    (cached_k5_valid != 0 &&
                     reject_attempt == 0u &&
                     !nonlinear_depends_on_h) ? 1 : 0;
                const vec_status status =
                    rk4_attempt_embedded_erk43(state,
                                                   step,
                                                   rk4_step_index,
                                                   reuse_cached_k5,
                                                   &error);
                if (status != VEC_STATUS_OK) {
                    step_accepted = -1;
                    break;
                }

                if (!isfinite(error) || error < 0.0) {
                    error = DBL_MAX;
                }

                if (error > 0.0 && isfinite(error)) {
                    scale = pow(tol / error, 0.25);
                } else if (error == 0.0) {
                    scale = RK4_STEP_SCALE_MAX;
                } else {
                    scale = RK4_STEP_SCALE_MIN;
                }

                if (scale < RK4_STEP_SCALE_MIN) {
                    scale = RK4_STEP_SCALE_MIN;
                } else if (scale > RK4_STEP_SCALE_MAX) {
                    scale = RK4_STEP_SCALE_MAX;
                }
                
                const int step_too_small = (step <= (min_step * (1.0 + 1e-12)));
                const int min_step_forced_accept = (step_too_small && error > tol ? 1 : 0);
                if (error <= tol || step_too_small) {
                    if (vec_complex_copy(state->backend,
                                             state->current_field_vec,
                                             state->working_vectors.k_final_vec) != VEC_STATUS_OK) {
                        step_accepted = -1;
                        break;
                    }
                    state->current_z += step;
                    state->current_step_size = clamp_step(step * scale, min_step, max_step);
                    cached_k5_valid = (!nonlinear_depends_on_h && !min_step_forced_accept && error <= tol) ? 1 : 0;
                    if (min_step_forced_accept && min_step_tol_warning_emitted == 0) {
                        log_emit(LOG_LEVEL_WARN,
                                     "[nlolib] adaptive solver reached min_step_size while local relative error "
                                     "remains above tolerance; continuing with constrained steps. "
                                     "z=%.9e step=%.9e error=%.9e tol=%.9e",
                                     state->current_z,
                                     step,
                                     error,
                                     tol);
                        min_step_tol_warning_emitted = 1;
                    }
                    rk4_emit_step_event(state,
                                            rk4_step_index,
                                            state->current_z,
                                            step,
                                            state->current_step_size,
                                            error);
                    log_progress_step_accepted(rk4_step_index,
                                                   state->current_z,
                                                   z_end,
                                                   step,
                                                   error,
                                                   state->current_step_size);
                    if (log_progress_abort_requested() != 0) {
                        terminated_early = 1;
                    }
                    step_accepted = 1;
                    break;
                }

                if (vec_complex_copy(state->backend,
                                         state->current_field_vec,
                                         state->working_vectors.previous_field_vec) != VEC_STATUS_OK) {
                    step_accepted = -1;
                    break;
                }
                cached_k5_valid = 0;

                const double attempted_step = step;
                double retry_step = clamp_step(step * scale, min_step, max_step);
                const double remaining_retry = z_end - state->current_z;
                if (retry_step > remaining_retry) {
                    retry_step = remaining_retry;
                }
                if (retry_step <= 0.0) {
                    step_accepted = -1;
                    break;
                }

                reject_attempt += 1u;
                log_progress_step_rejected(rk4_step_index,
                                               state->current_z,
                                               z_end,
                                               attempted_step,
                                               error,
                                               retry_step,
                                               reject_attempt);
                if (log_progress_abort_requested() != 0) {
                    step_accepted = -1;
                    break;
                }

                step = retry_step;
                if (reject_attempt >= RK4_MAX_REJECTION_ATTEMPTS) {
                    step_accepted = -1;
                    break;
                }
            }
        }

        if (step_accepted < 0) {
            terminated_early = 1;
            break;
        }

        rk4_debug_log_error_control(rk4_step_index,
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
            vec_status capture_status = VEC_STATUS_OK;
            if (disable_record_interpolation) {
                capture_status = simulation_state_capture_snapshot(state);
            } else {
                capture_status = rk4_capture_interpolated_record(state,
                                                                     step_start_z,
                                                                     step_end_z,
                                                                     next_record_z);
            }
            if (capture_status != VEC_STATUS_OK)
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
        if (log_progress_abort_requested() != 0) {
            terminated_early = 1;
            break;
        }

        rk4_step_index += 1u;
    }

    while ((has_explicit_schedule || record_spacing > 0.0) &&
           state->current_record_index < state->num_recorded_samples &&
           state->current_z + record_capture_eps >= next_record_z)
    {
        if (simulation_state_capture_snapshot(state) != VEC_STATUS_OK)
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

    (void)vec_end_simulation(state->backend);
    (void)simulation_state_flush_snapshots(state);
    log_progress_finish(state->current_z,
                            z_end,
                            (terminated_early == 0 && state->current_z + record_capture_eps >= z_end) ? 1 : 0);
}
