/**
 * @file operator_program_execute.c
 * @brief Runtime execution for compiled operator programs.
 */

#include "physics/operator_program.h"

nlo_vec_status nlo_operator_program_execute(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    nlo_vec_buffer* const* stack_vectors,
    size_t stack_vector_count,
    nlo_vec_buffer* out_vector
)
{
    if (backend == NULL ||
        program == NULL ||
        eval_ctx == NULL ||
        stack_vectors == NULL ||
        out_vector == NULL ||
        !program->active ||
        program->required_stack_slots == 0u ||
        stack_vector_count < program->required_stack_slots) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    size_t stack_depth = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const nlo_operator_instruction instruction = program->instructions[i];
        nlo_vec_status status = NLO_VEC_STATUS_OK;

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_LITERAL) {
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_fill(backend, dst, instruction.literal);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_W) {
            const nlo_vec_buffer* source =
                (eval_ctx->frequency_grid != NULL)
                    ? eval_ctx->frequency_grid
                    : eval_ctx->wt_grid;
            if (source == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, source);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_WT) {
            const nlo_vec_buffer* source =
                (eval_ctx->wt_grid != NULL)
                    ? eval_ctx->wt_grid
                    : eval_ctx->frequency_grid;
            if (source == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, source);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_KX) {
            if (eval_ctx->kx_grid == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->kx_grid);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_KY) {
            if (eval_ctx->ky_grid == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->ky_grid);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_T) {
            if (eval_ctx->t_grid == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->t_grid);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_X) {
            if (eval_ctx->x_grid == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->x_grid);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_Y) {
            if (eval_ctx->y_grid == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->y_grid);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_A) {
            if (eval_ctx->field == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->field);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_I) {
            if (eval_ctx->field == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_magnitude_squared(backend, eval_ctx->field, dst);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_D) {
            if (eval_ctx->dispersion_factor == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->dispersion_factor);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_V) {
            if (eval_ctx->potential == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->potential);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_H) {
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_fill(backend, dst, nlo_make(eval_ctx->half_step_size, 0.0));
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_IMAG_UNIT) {
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_fill(backend, dst, nlo_make(0.0, 1.0));
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (stack_depth == 0u) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_NEGATE) {
            nlo_vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = nlo_vec_complex_scalar_mul_inplace(backend, top, nlo_make(-1.0, 0.0));
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_EXP) {
            nlo_vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = nlo_vec_complex_exp_inplace(backend, top);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_LOG) {
            nlo_vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = nlo_vec_complex_log_inplace(backend, top);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_SQRT) {
            nlo_vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = nlo_vec_complex_real_pow_inplace(backend, top, 0.5);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_SIN) {
            nlo_vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = nlo_vec_complex_sin_inplace(backend, top);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_COS) {
            nlo_vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = nlo_vec_complex_cos_inplace(backend, top);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (stack_depth < 2u) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        nlo_vec_buffer* left = stack_vectors[stack_depth - 2u];
        nlo_vec_buffer* right = stack_vectors[stack_depth - 1u];
        if (instruction.opcode == NLO_OPERATOR_OP_ADD) {
            status = nlo_vec_complex_add_inplace(backend, left, right);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_MUL) {
            status = nlo_vec_complex_mul_inplace(backend, left, right);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_DIV) {
            status = nlo_vec_complex_real_pow_inplace(backend, right, -1.0);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            status = nlo_vec_complex_mul_inplace(backend, left, right);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_POW) {
            status = nlo_vec_complex_pow_elementwise_inplace(backend, left, right);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (stack_depth != 1u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    return nlo_vec_complex_copy(backend, out_vector, stack_vectors[0]);
}
