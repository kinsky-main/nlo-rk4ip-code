/**
 * @file operators.c
 * @brief Runtime-program-backed vector operator implementation.
 */

#include "physics/operators.h"
#include "fft/fft.h"
#include "utility/perf_profile.h"

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
    if (work->raman_intensity_vec == NULL ||
        work->raman_delayed_vec == NULL ||
        work->raman_spectrum_vec == NULL ||
        work->raman_mix_vec == NULL ||
        work->raman_polarization_vec == NULL ||
        work->raman_derivative_vec == NULL ||
        work->raman_response_fft_vec == NULL ||
        work->raman_derivative_factor_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const double f_r = state->raman_fraction;
    nlo_vec_status status = nlo_vec_complex_magnitude_squared(state->backend, field, work->raman_intensity_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (f_r > 0.0) {
        status = nlo_vec_complex_copy(state->backend, work->raman_delayed_vec, work->raman_intensity_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_fft_forward_vec(state->fft_plan, work->raman_delayed_vec, work->raman_spectrum_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_complex_mul_inplace(state->backend, work->raman_spectrum_vec, work->raman_response_fft_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_fft_inverse_vec(state->fft_plan, work->raman_spectrum_vec, work->raman_delayed_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        status = nlo_vec_complex_copy(state->backend, work->raman_mix_vec, work->raman_intensity_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_complex_scalar_mul_inplace(state->backend, work->raman_mix_vec, nlo_make(1.0 - f_r, 0.0));
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_complex_copy(state->backend, work->raman_polarization_vec, work->raman_delayed_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_complex_scalar_mul_inplace(state->backend, work->raman_polarization_vec, nlo_make(f_r, 0.0));
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_complex_add_inplace(state->backend, work->raman_mix_vec, work->raman_polarization_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    } else {
        status = nlo_vec_complex_copy(state->backend, work->raman_mix_vec, work->raman_intensity_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    status = nlo_vec_complex_copy(state->backend, work->raman_polarization_vec, field);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_vec_complex_mul_inplace(state->backend, work->raman_polarization_vec, work->raman_mix_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_copy(state->backend, out_field, work->raman_polarization_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_vec_complex_scalar_mul_inplace(state->backend, out_field, nlo_make(0.0, state->nonlinear_gamma));
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (state->nonlinear_shock_active) {
        status = nlo_vec_complex_copy(state->backend, work->raman_derivative_vec, work->raman_polarization_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_fft_forward_vec(state->fft_plan, work->raman_derivative_vec, work->raman_spectrum_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_complex_mul_inplace(state->backend, work->raman_spectrum_vec, work->raman_derivative_factor_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_fft_inverse_vec(state->fft_plan, work->raman_spectrum_vec, work->raman_derivative_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_complex_scalar_mul_inplace(
            state->backend,
            work->raman_derivative_vec,
            nlo_make(-(state->nonlinear_gamma / state->shock_omega0), 0.0)
        );
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_complex_add_inplace(state->backend, out_field, work->raman_derivative_vec);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_apply_dispersion_operator_stage(
    simulation_state* state,
    nlo_vec_buffer* freq_domain_envelope
)
{
    if (state == NULL || state->backend == NULL || state->config == NULL || freq_domain_envelope == NULL) {
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
    nlo_vec_status status = nlo_operator_program_execute(state->backend,
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
    if (status == NLO_VEC_STATUS_OK && state->transverse_active) {
        const nlo_operator_eval_context transverse_eval_ctx = {
            .frequency_grid = state->spatial_frequency_grid_vec,
            .field = freq_domain_envelope,
            .dispersion_factor = state->transverse_factor_vec,
            .potential = state->working_vectors.potential_vec,
            .half_step_size = state->current_half_step_exp
        };
        status = nlo_operator_program_execute(state->backend,
                                              &state->transverse_operator_program,
                                              &transverse_eval_ctx,
                                              state->runtime_operator_stack_vec,
                                              state->runtime_operator_stack_slots,
                                              state->transverse_operator_vec);
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_mul_inplace(state->backend,
                                                 freq_domain_envelope,
                                                 state->transverse_operator_vec);
        }
    }
    const double end_ms = nlo_perf_profile_now_ms();
    if (status == NLO_VEC_STATUS_OK) {
        nlo_perf_profile_add_dispersion_time(end_ms - start_ms);
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

    const double start_ms = nlo_perf_profile_now_ms();
    nlo_vec_status status = NLO_VEC_STATUS_OK;
    if (state->nonlinear_model == NLO_NONLINEAR_MODEL_KERR_RAMAN) {
        status = nlo_apply_nonlinear_operator_stage_raman(state, field, out_field);
    } else {
        const nlo_operator_eval_context eval_ctx = {
            .frequency_grid = state->frequency_grid_vec,
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
    const double end_ms = nlo_perf_profile_now_ms();
    if (status == NLO_VEC_STATUS_OK) {
        nlo_perf_profile_add_nonlinear_time(end_ms - start_ms);
    }
    return status;
}

