/**
 * @file operators.c
 * @brief Runtime-program-backed vector operator implementation.
 */

#include "physics/operators.h"
#include "backend/vector_backend_internal.h"
#include "fft/fft.h"
#include "utility/perf_profile.h"
#include <assert.h>

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

static nlo_vec_status nlo_apply_nonlinear_operator_stage_raman(
    simulation_state* state,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* out_field
)
{
    if (state == NULL || state->backend == NULL || state->fft_plan == NULL || field == NULL || out_field == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    simulation_working_vectors* work = &state->working_vectors;

    const double f_r = state->raman_fraction;
    nlo_vec_status status = nlo_vec_complex_magnitude_squared(state->backend, field, work->raman_intensity_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (f_r > 0.0) {
        NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, work->raman_delayed_vec, work->raman_intensity_vec));
        NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->raman_delayed_vec, work->raman_spectrum_vec));
        NLO_RK4_CALL(nlo_vec_complex_mul_inplace(state->backend, work->raman_spectrum_vec, work->raman_response_fft_vec));
        NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->raman_spectrum_vec, work->raman_delayed_vec));
        NLO_RK4_CALL(nlo_vec_complex_affine_comb2_real(state->backend,
                                                       work->raman_mix_vec,
                                                       work->raman_intensity_vec,
                                                       1.0 - f_r,
                                                       work->raman_delayed_vec,
                                                       f_r));
    } else {
        NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, work->raman_mix_vec, work->raman_intensity_vec));
    }

    NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, work->raman_polarization_vec, field));
    NLO_RK4_CALL(nlo_vec_complex_mul_inplace(state->backend, work->raman_polarization_vec, work->raman_mix_vec));

    NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, out_field, work->raman_polarization_vec));
    NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(state->backend, out_field, nlo_make(0.0, state->nonlinear_gamma)));

    if (state->nonlinear_shock_active) {
        NLO_RK4_CALL(nlo_vec_complex_copy(state->backend, work->raman_derivative_vec, work->raman_polarization_vec));
        NLO_RK4_CALL(nlo_fft_forward_vec(state->fft_plan, work->raman_derivative_vec, work->raman_spectrum_vec));
        NLO_RK4_CALL(nlo_vec_complex_mul_inplace(state->backend, work->raman_spectrum_vec, work->raman_derivative_factor_vec));
        NLO_RK4_CALL(nlo_fft_inverse_vec(state->fft_plan, work->raman_spectrum_vec, work->raman_derivative_vec));
        NLO_RK4_CALL(nlo_vec_complex_scalar_mul_inplace(
            state->backend,
            work->raman_derivative_vec,
            nlo_make(-(state->nonlinear_gamma / state->shock_omega0), 0.0)
        ));
        NLO_RK4_CALL(nlo_vec_complex_add_inplace(state->backend, out_field, work->raman_derivative_vec));
    }

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_apply_dispersion_operator_stage(
    simulation_state* state,
    nlo_vec_buffer* freq_domain_envelope
)
{
    nlo_perf_scope perf_scope = {0.0, 0};
    NLO_PERF_SCOPE_BEGIN(perf_scope);
    nlo_vec_status status = NLO_VEC_STATUS_OK;
    if (state->tensor_mode_active) {
        const nlo_operator_eval_context linear_eval_ctx = {
            .frequency_grid = state->frequency_grid_vec,
            .wt_grid = nlo_state_operator_wt_grid(state),
            .kx_grid = nlo_state_operator_kx_grid(state),
            .ky_grid = nlo_state_operator_ky_grid(state),
            .t_grid = nlo_state_operator_t_grid(state),
            .x_grid = nlo_state_operator_x_grid(state),
            .y_grid = nlo_state_operator_y_grid(state),
            .field = freq_domain_envelope,
            .dispersion_factor = state->working_vectors.dispersion_factor_vec,
            .potential = state->working_vectors.potential_vec,
            .half_step_size = state->current_half_step_exp
        };
        status = nlo_operator_program_execute(state->backend,
                                              &state->linear_operator_program,
                                              &linear_eval_ctx,
                                              state->runtime_operator_stack_vec,
                                              state->runtime_operator_stack_slots,
                                              state->working_vectors.dispersion_operator_vec);
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_mul_inplace(state->backend,
                                                 freq_domain_envelope,
                                                 state->working_vectors.dispersion_operator_vec);
        }
    } else {
        const nlo_operator_eval_context eval_ctx = {
            .frequency_grid = state->frequency_grid_vec,
            .field = freq_domain_envelope,
            .dispersion_factor = state->working_vectors.dispersion_factor_vec,
            .potential = state->working_vectors.potential_vec,
            .half_step_size = state->current_half_step_exp
        };
        status = nlo_operator_program_execute(state->backend,
                                              &state->dispersion_operator_program,
                                              &eval_ctx,
                                              state->runtime_operator_stack_vec,
                                              state->runtime_operator_stack_slots,
                                              state->working_vectors.dispersion_operator_vec);
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_mul_inplace(state->backend,
                                                 freq_domain_envelope,
                                                 state->working_vectors.dispersion_operator_vec);
        }
    }
    if (status == NLO_VEC_STATUS_OK) {
        NLO_PERF_SCOPE_END(perf_scope, NLO_PERF_EVENT_DISPERSION_APPLY, 0u);
    }
    return status;
}

nlo_vec_status nlo_apply_nonlinear_operator_stage(
    simulation_state* state,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* out_field
)
{
    if (state == NULL ||
        state->backend == NULL ||
        state->config == NULL ||
        field == NULL ||
        out_field == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_perf_scope perf_scope = {0.0, 0};
    NLO_PERF_SCOPE_BEGIN(perf_scope);
    nlo_vec_status status = NLO_VEC_STATUS_OK;
    if (state->nonlinear_model == NLO_NONLINEAR_MODEL_KERR_RAMAN) {
        status = nlo_apply_nonlinear_operator_stage_raman(state, field, out_field);
    } else {
        const nlo_operator_eval_context eval_ctx = {
            .frequency_grid = state->frequency_grid_vec,
            .wt_grid = state->tensor_mode_active ? nlo_state_operator_wt_grid(state) : NULL,
            .kx_grid = state->tensor_mode_active ? nlo_state_operator_kx_grid(state) : NULL,
            .ky_grid = state->tensor_mode_active ? nlo_state_operator_ky_grid(state) : NULL,
            .t_grid = state->tensor_mode_active ? nlo_state_operator_t_grid(state) : NULL,
            .x_grid = state->tensor_mode_active ? nlo_state_operator_x_grid(state) : NULL,
            .y_grid = state->tensor_mode_active ? nlo_state_operator_y_grid(state) : NULL,
            .field = field,
            .dispersion_factor = state->working_vectors.dispersion_factor_vec,
            .potential = state->working_vectors.potential_vec,
            .half_step_size = state->current_half_step_exp
        };
        status = nlo_operator_program_execute(state->backend,
                                              &state->nonlinear_operator_program,
                                              &eval_ctx,
                                              state->runtime_operator_stack_vec,
                                              state->runtime_operator_stack_slots,
                                              out_field);
    }
    if (status == NLO_VEC_STATUS_OK) {
        NLO_PERF_SCOPE_END(perf_scope, NLO_PERF_EVENT_NONLINEAR_APPLY, 0u);
    }
    return status;
}
