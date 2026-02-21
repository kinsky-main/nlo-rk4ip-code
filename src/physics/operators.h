/**
 * @file operators.h
 * @brief Backend vector operators for dispersion and nonlinearity.
 */
#pragma once

#include "backend/vector_backend.h"
#include "physics/operator_program.h"
#include <stddef.h>

/**
 * @brief Apply a runtime-compiled dispersion operator in frequency space.
 *
 * @param backend Active vector backend.
 * @param program Compiled dispersion operator program.
 * @param eval_ctx Program evaluation symbols.
 * @param freq_domain_envelope Frequency-domain field updated in place.
 * @param multiplier_vec Scratch/output multiplier vector.
 * @param runtime_stack_vec Runtime scratch stack vectors.
 * @param runtime_stack_count Number of runtime scratch stack vectors.
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_apply_dispersion_operator_program_vec(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    nlo_vec_buffer* freq_domain_envelope,
    nlo_vec_buffer* multiplier_vec,
    nlo_vec_buffer* const* runtime_stack_vec,
    size_t runtime_stack_count
);

/**
 * @brief Apply a runtime-compiled nonlinear multiplier program.
 *
 * @param backend Active vector backend.
 * @param program Compiled nonlinear program.
 * @param eval_ctx Program evaluation symbols.
 * @param field Input field vector.
 * @param multiplier_vec Scratch/output multiplier vector.
 * @param out_field Output field vector.
 * @param runtime_stack_vec Runtime scratch stack vectors.
 * @param runtime_stack_count Number of runtime scratch stack vectors.
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_apply_nonlinear_operator_program_vec(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* multiplier_vec,
    nlo_vec_buffer* out_field,
    nlo_vec_buffer* const* runtime_stack_vec,
    size_t runtime_stack_count
);
