/**
 * @file operators.c
 * @brief Runtime-program-backed vector operator implementation.
 */

#include "physics/operators.h"

nlo_vec_status nlo_apply_dispersion_operator_program_vec(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    nlo_vec_buffer* freq_domain_envelope,
    nlo_vec_buffer* multiplier_vec,
    nlo_vec_buffer* const* runtime_stack_vec,
    size_t runtime_stack_count
)
{
    if (backend == NULL ||
        program == NULL ||
        eval_ctx == NULL ||
        freq_domain_envelope == NULL ||
        multiplier_vec == NULL ||
        runtime_stack_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_operator_program_execute(backend,
                                                         program,
                                                         eval_ctx,
                                                         runtime_stack_vec,
                                                         runtime_stack_count,
                                                         multiplier_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vec_complex_mul_inplace(backend, freq_domain_envelope, multiplier_vec);
}

nlo_vec_status nlo_apply_nonlinear_operator_program_vec(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* multiplier_vec,
    nlo_vec_buffer* out_field,
    nlo_vec_buffer* const* runtime_stack_vec,
    size_t runtime_stack_count
)
{
    if (backend == NULL ||
        program == NULL ||
        eval_ctx == NULL ||
        field == NULL ||
        multiplier_vec == NULL ||
        out_field == NULL ||
        runtime_stack_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_operator_program_execute(backend,
                                                         program,
                                                         eval_ctx,
                                                         runtime_stack_vec,
                                                         runtime_stack_count,
                                                         multiplier_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_copy(backend, out_field, field);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vec_complex_mul_inplace(backend, out_field, multiplier_vec);
}

