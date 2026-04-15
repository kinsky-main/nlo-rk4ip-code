/**
 * @file operator_program.h
 * @brief Runtime-compiled operator expression program.
 */
#pragma once

#include "backend/vector_backend.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OPERATOR_PROGRAM_MAX_INSTRUCTIONS
#define OPERATOR_PROGRAM_MAX_INSTRUCTIONS 128u
#endif

#ifndef OPERATOR_PROGRAM_MAX_STACK_SLOTS
#define OPERATOR_PROGRAM_MAX_STACK_SLOTS 8u
#endif

typedef enum {
    OPERATOR_CONTEXT_DISPERSION_FACTOR = 0,
    OPERATOR_CONTEXT_DISPERSION = 1,
    OPERATOR_CONTEXT_NONLINEAR = 2,
    OPERATOR_CONTEXT_LINEAR_FACTOR = 3,
    OPERATOR_CONTEXT_LINEAR = 4,
    OPERATOR_CONTEXT_POTENTIAL = 5
} operator_program_context;

typedef enum {
    OPERATOR_OP_PUSH_LITERAL = 0,
    OPERATOR_OP_PUSH_SYMBOL_W = 1,
    OPERATOR_OP_PUSH_SYMBOL_A = 2,
    OPERATOR_OP_PUSH_SYMBOL_I = 3,
    OPERATOR_OP_PUSH_SYMBOL_D = 4,
    OPERATOR_OP_PUSH_SYMBOL_V = 5,
    OPERATOR_OP_PUSH_SYMBOL_H = 6,
    OPERATOR_OP_PUSH_IMAG_UNIT = 7,
    OPERATOR_OP_NEGATE = 8,
    OPERATOR_OP_ADD = 9,
    OPERATOR_OP_MUL = 10,
    OPERATOR_OP_EXP = 11,
    OPERATOR_OP_DIV = 12,
    OPERATOR_OP_POW = 13,
    OPERATOR_OP_LOG = 14,
    OPERATOR_OP_SQRT = 15,
    OPERATOR_OP_SIN = 16,
    OPERATOR_OP_COS = 17,
    OPERATOR_OP_PUSH_SYMBOL_WT = 18,
    OPERATOR_OP_PUSH_SYMBOL_KX = 19,
    OPERATOR_OP_PUSH_SYMBOL_KY = 20,
    OPERATOR_OP_PUSH_SYMBOL_T = 21,
    OPERATOR_OP_PUSH_SYMBOL_X = 22,
    OPERATOR_OP_PUSH_SYMBOL_Y = 23,
    OPERATOR_OP_POW_REAL_LITERAL = 24
} operator_opcode;

typedef struct {
    operator_opcode opcode;
    nlo_complex literal;
} operator_instruction;

typedef struct {
    int active;
    operator_program_context context;
    size_t instruction_count;
    size_t required_stack_slots;
    operator_instruction instructions[OPERATOR_PROGRAM_MAX_INSTRUCTIONS];
} operator_program;

typedef struct {
    const vec_buffer* frequency_grid;
    const vec_buffer* wt_grid;
    const vec_buffer* kx_grid;
    const vec_buffer* ky_grid;
    const vec_buffer* t_grid;
    const vec_buffer* x_grid;
    const vec_buffer* y_grid;
    const vec_buffer* field;
    const vec_buffer* dispersion_factor;
    const vec_buffer* potential;
    double half_step_size;
} operator_eval_context;

/**
 * @brief Compile an operator expression into an internal instruction program.
 *
 * @param expression Null-terminated expression string.
 * @param context Allowed symbol set for the program.
 * @param num_constants Number of valid runtime constants in @p constants.
 * @param constants Runtime constants array.
 * @param out_program Compiled program output.
 * @return vec_status compile status.
 */
vec_status operator_program_compile(
    const char* expression,
    operator_program_context context,
    size_t num_constants,
    const double* constants,
    operator_program* out_program
);

/**
 * @brief Execute a compiled operator program into @p out_vector.
 *
 * @param backend Active backend.
 * @param program Compiled program.
 * @param eval_ctx Symbol source buffers for execution.
 * @param stack_vectors Scratch stack vectors used by the executor.
 * @param stack_vector_count Number of stack vectors available.
 * @param out_vector Program result vector.
 * @return vec_status execution status.
 */
vec_status operator_program_execute(
    vector_backend* backend,
    const operator_program* program,
    const operator_eval_context* eval_ctx,
    vec_buffer* const* stack_vectors,
    size_t stack_vector_count,
    vec_buffer* out_vector
);

#ifdef __cplusplus
}
#endif
