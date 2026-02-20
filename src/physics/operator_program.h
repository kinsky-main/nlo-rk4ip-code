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

#ifndef NLO_OPERATOR_PROGRAM_MAX_INSTRUCTIONS
#define NLO_OPERATOR_PROGRAM_MAX_INSTRUCTIONS 128u
#endif

#ifndef NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS
#define NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS 8u
#endif

typedef enum {
    NLO_OPERATOR_CONTEXT_DISPERSION = 0,
    NLO_OPERATOR_CONTEXT_NONLINEAR = 1
} nlo_operator_program_context;

typedef enum {
    NLO_OPERATOR_OP_PUSH_LITERAL = 0,
    NLO_OPERATOR_OP_PUSH_SYMBOL_W = 1,
    NLO_OPERATOR_OP_PUSH_SYMBOL_A = 2,
    NLO_OPERATOR_OP_PUSH_SYMBOL_I = 3,
    NLO_OPERATOR_OP_PUSH_IMAG_UNIT = 4,
    NLO_OPERATOR_OP_NEGATE = 5,
    NLO_OPERATOR_OP_ADD = 6,
    NLO_OPERATOR_OP_MUL = 7,
    NLO_OPERATOR_OP_EXP = 8
} nlo_operator_opcode;

typedef struct {
    nlo_operator_opcode opcode;
    nlo_complex literal;
} nlo_operator_instruction;

typedef struct {
    int active;
    nlo_operator_program_context context;
    size_t instruction_count;
    size_t required_stack_slots;
    nlo_operator_instruction instructions[NLO_OPERATOR_PROGRAM_MAX_INSTRUCTIONS];
} nlo_operator_program;

typedef struct {
    const nlo_vec_buffer* frequency_grid;
    const nlo_vec_buffer* field;
} nlo_operator_eval_context;

/**
 * @brief Compile an operator expression into an internal instruction program.
 *
 * @param expression Null-terminated expression string.
 * @param context Allowed symbol set for the program.
 * @param num_constants Number of valid runtime constants in @p constants.
 * @param constants Runtime constants array.
 * @param out_program Compiled program output.
 * @return nlo_vec_status compile status.
 */
nlo_vec_status nlo_operator_program_compile(
    const char* expression,
    nlo_operator_program_context context,
    size_t num_constants,
    const double* constants,
    nlo_operator_program* out_program
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
 * @return nlo_vec_status execution status.
 */
nlo_vec_status nlo_operator_program_execute(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    nlo_vec_buffer* const* stack_vectors,
    size_t stack_vector_count,
    nlo_vec_buffer* out_vector
);

#ifdef __cplusplus
}
#endif
