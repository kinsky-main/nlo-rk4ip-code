/**
 * @file operator_program.h
 * @brief Runtime-compiled operator expression program.
 */
#pragma once

#include "backend/vector_backend.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NLO_OPERATOR_PROGRAM_MAX_INSTRUCTIONS
#define NLO_OPERATOR_PROGRAM_MAX_INSTRUCTIONS 128u
#endif

#ifndef NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS
#define NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS 8u
#endif

#ifndef NLO_OPERATOR_PROGRAM_MAX_VALUES
#define NLO_OPERATOR_PROGRAM_MAX_VALUES NLO_OPERATOR_PROGRAM_MAX_INSTRUCTIONS
#endif

#ifndef NLO_OPERATOR_VALUE_INVALID
#define NLO_OPERATOR_VALUE_INVALID UINT16_MAX
#endif

typedef enum {
    NLO_OPERATOR_CONTEXT_DISPERSION_FACTOR = 0,
    NLO_OPERATOR_CONTEXT_DISPERSION = 1,
    NLO_OPERATOR_CONTEXT_NONLINEAR = 2,
    NLO_OPERATOR_CONTEXT_LINEAR_FACTOR = 3,
    NLO_OPERATOR_CONTEXT_LINEAR = 4,
    NLO_OPERATOR_CONTEXT_POTENTIAL = 5
} nlo_operator_program_context;

typedef enum {
    NLO_OPERATOR_OP_PUSH_LITERAL = 0,
    NLO_OPERATOR_OP_PUSH_SYMBOL_W = 1,
    NLO_OPERATOR_OP_PUSH_SYMBOL_A = 2,
    NLO_OPERATOR_OP_PUSH_SYMBOL_I = 3,
    NLO_OPERATOR_OP_PUSH_SYMBOL_D = 4,
    NLO_OPERATOR_OP_PUSH_SYMBOL_V = 5,
    NLO_OPERATOR_OP_PUSH_SYMBOL_H = 6,
    NLO_OPERATOR_OP_PUSH_IMAG_UNIT = 7,
    NLO_OPERATOR_OP_NEGATE = 8,
    NLO_OPERATOR_OP_ADD = 9,
    NLO_OPERATOR_OP_MUL = 10,
    NLO_OPERATOR_OP_EXP = 11,
    NLO_OPERATOR_OP_DIV = 12,
    NLO_OPERATOR_OP_POW = 13,
    NLO_OPERATOR_OP_LOG = 14,
    NLO_OPERATOR_OP_SQRT = 15,
    NLO_OPERATOR_OP_SIN = 16,
    NLO_OPERATOR_OP_COS = 17,
    NLO_OPERATOR_OP_PUSH_SYMBOL_WT = 18,
    NLO_OPERATOR_OP_PUSH_SYMBOL_KX = 19,
    NLO_OPERATOR_OP_PUSH_SYMBOL_KY = 20,
    NLO_OPERATOR_OP_PUSH_SYMBOL_T = 21,
    NLO_OPERATOR_OP_PUSH_SYMBOL_X = 22,
    NLO_OPERATOR_OP_PUSH_SYMBOL_Y = 23,
    NLO_OPERATOR_OP_POW_REAL_LITERAL = 24
} nlo_operator_opcode;

typedef struct {
    nlo_operator_opcode opcode;
    nlo_complex literal;
} nlo_operator_instruction;

typedef enum {
    NLO_OPERATOR_SYMBOL_MASK_NONE = 0u,
    NLO_OPERATOR_SYMBOL_MASK_W = (1u << 0),
    NLO_OPERATOR_SYMBOL_MASK_A = (1u << 1),
    NLO_OPERATOR_SYMBOL_MASK_I = (1u << 2),
    NLO_OPERATOR_SYMBOL_MASK_D = (1u << 3),
    NLO_OPERATOR_SYMBOL_MASK_V = (1u << 4),
    NLO_OPERATOR_SYMBOL_MASK_H = (1u << 5),
    NLO_OPERATOR_SYMBOL_MASK_WT = (1u << 6),
    NLO_OPERATOR_SYMBOL_MASK_KX = (1u << 7),
    NLO_OPERATOR_SYMBOL_MASK_KY = (1u << 8),
    NLO_OPERATOR_SYMBOL_MASK_T = (1u << 9),
    NLO_OPERATOR_SYMBOL_MASK_X = (1u << 10),
    NLO_OPERATOR_SYMBOL_MASK_Y = (1u << 11)
} nlo_operator_symbol_mask;

typedef struct {
    nlo_operator_opcode opcode;
    uint16_t left;
    uint16_t right;
    nlo_complex literal;
    uint32_t symbol_mask;
} nlo_operator_value;

typedef struct nlo_vk_operator_jit_entry nlo_vk_operator_jit_entry;

typedef struct {
    int active;
    nlo_operator_program_context context;
    size_t instruction_count;
    size_t required_stack_slots;
    nlo_operator_instruction instructions[NLO_OPERATOR_PROGRAM_MAX_INSTRUCTIONS];
    size_t value_count;
    size_t root_value;
    uint32_t active_symbol_mask;
    uint64_t lowered_hash;
    int vk_jit_eligible;
    int vk_jit_active;
    int vk_jit_compile_attempted;
    int vk_jit_warning_emitted;
    nlo_vk_operator_jit_entry* vk_jit_entry;
    nlo_operator_value values[NLO_OPERATOR_PROGRAM_MAX_VALUES];
} nlo_operator_program;

typedef struct {
    const nlo_vec_buffer* frequency_grid;
    const nlo_vec_buffer* wt_grid;
    const nlo_vec_buffer* kx_grid;
    const nlo_vec_buffer* ky_grid;
    const nlo_vec_buffer* t_grid;
    const nlo_vec_buffer* x_grid;
    const nlo_vec_buffer* y_grid;
    const nlo_vec_buffer* field;
    const nlo_vec_buffer* dispersion_factor;
    const nlo_vec_buffer* potential;
    double half_step_size;
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
