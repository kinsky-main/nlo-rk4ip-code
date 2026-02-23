/**
 * @file operators.c
 * @brief Runtime-program-backed vector operator implementation.
 */

#include "physics/operators.h"
#include "utility/perf_profile.h"

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

    const nlo_operator_eval_context eval_ctx = {
        .frequency_grid = state->frequency_grid_vec,
        .field = field,
        .dispersion_factor = state->working_vectors.dispersion_factor_vec,
        .potential = state->working_vectors.potential_vec,
        .half_step_size = state->current_half_step_exp
    };

    const double start_ms = nlo_perf_profile_now_ms();
    nlo_vec_status status = nlo_operator_program_execute(state->backend,
                                                         &state->nonlinear_operator_program,
                                                         &eval_ctx,
                                                         state->runtime_operator_stack_vec,
                                                         state->runtime_operator_stack_slots,
                                                         state->working_vectors.nonlinear_multiplier_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_copy(state->backend, out_field, field);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_mul_inplace(state->backend,
                                         out_field,
                                         state->working_vectors.nonlinear_multiplier_vec);
    const double end_ms = nlo_perf_profile_now_ms();
    if (status == NLO_VEC_STATUS_OK) {
        nlo_perf_profile_add_nonlinear_time(end_ms - start_ms);
    }
    return status;
}

