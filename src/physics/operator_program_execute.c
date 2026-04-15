/**
 * @file operator_program_execute.c
 * @brief Runtime execution for compiled operator programs.
 */

#include "physics/operator_program.h"

vec_status operator_program_execute(
    vector_backend* backend,
    const operator_program* program,
    const operator_eval_context* eval_ctx,
    vec_buffer* const* stack_vectors,
    size_t stack_vector_count,
    vec_buffer* out_vector
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
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    size_t stack_depth = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const operator_instruction instruction = program->instructions[i];
        vec_status status = VEC_STATUS_OK;

        if (instruction.opcode == OPERATOR_OP_PUSH_LITERAL) {
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_fill(backend, dst, instruction.literal);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_W) {
            const vec_buffer* source =
                (eval_ctx->frequency_grid != NULL)
                    ? eval_ctx->frequency_grid
                    : eval_ctx->wt_grid;
            if (source == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, source);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_WT) {
            const vec_buffer* source =
                (eval_ctx->wt_grid != NULL)
                    ? eval_ctx->wt_grid
                    : eval_ctx->frequency_grid;
            if (source == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, source);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_KX) {
            if (eval_ctx->kx_grid == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, eval_ctx->kx_grid);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_KY) {
            if (eval_ctx->ky_grid == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, eval_ctx->ky_grid);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_T) {
            if (eval_ctx->t_grid == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, eval_ctx->t_grid);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_X) {
            if (eval_ctx->x_grid == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, eval_ctx->x_grid);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_Y) {
            if (eval_ctx->y_grid == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, eval_ctx->y_grid);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_A) {
            if (eval_ctx->field == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, eval_ctx->field);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_I) {
            if (eval_ctx->field == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_magnitude_squared(backend, eval_ctx->field, dst);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_D) {
            if (eval_ctx->dispersion_factor == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, eval_ctx->dispersion_factor);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_V) {
            if (eval_ctx->potential == NULL) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_copy(backend, dst, eval_ctx->potential);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_SYMBOL_H) {
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_fill(backend, dst, make(eval_ctx->half_step_size, 0.0));
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_IMAG_UNIT) {
            vec_buffer* dst = stack_vectors[stack_depth];
            status = vec_complex_fill(backend, dst, make(0.0, 1.0));
            if (status != VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (stack_depth == 0u) {
            return VEC_STATUS_INVALID_ARGUMENT;
        }

        if (instruction.opcode == OPERATOR_OP_NEGATE) {
            vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = vec_complex_scalar_mul_inplace(backend, top, make(-1.0, 0.0));
            if (status != VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_EXP) {
            vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = vec_complex_exp_inplace(backend, top);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_LOG) {
            vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = vec_complex_log_inplace(backend, top);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_SQRT) {
            vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = vec_complex_real_pow_inplace(backend, top, 0.5);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_SIN) {
            vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = vec_complex_sin_inplace(backend, top);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_COS) {
            vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = vec_complex_cos_inplace(backend, top);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_POW_REAL_LITERAL) {
            vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = vec_complex_real_pow_inplace(backend, top, RE(instruction.literal));
            if (status != VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (stack_depth < 2u) {
            return VEC_STATUS_INVALID_ARGUMENT;
        }

        vec_buffer* left = stack_vectors[stack_depth - 2u];
        vec_buffer* right = stack_vectors[stack_depth - 1u];
        if (instruction.opcode == OPERATOR_OP_ADD) {
            status = vec_complex_add_inplace(backend, left, right);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_MUL) {
            status = vec_complex_mul_inplace(backend, left, right);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_DIV) {
            status = vec_complex_real_pow_inplace(backend, right, -1.0);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            status = vec_complex_mul_inplace(backend, left, right);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_POW) {
            status = vec_complex_pow_elementwise_inplace(backend, left, right);
            if (status != VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        return VEC_STATUS_INVALID_ARGUMENT;
    }

    if (stack_depth != 1u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    return vec_complex_copy(backend, out_vector, stack_vectors[0]);
}
