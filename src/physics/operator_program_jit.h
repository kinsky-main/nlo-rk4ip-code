/**
 * @file operator_program_jit.h
 * @brief Internal Vulkan JIT hooks for compiled operator programs.
 */
#pragma once

#include "physics/operator_program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NLO_OPERATOR_JIT_MODE_ON = 0,
    NLO_OPERATOR_JIT_MODE_OFF = 1
} nlo_operator_jit_mode;

/**
 * @brief Override the internal Vulkan operator JIT mode.
 *
 * @param mode Requested JIT mode.
 */
void nlo_operator_program_set_jit_mode(nlo_operator_jit_mode mode);

/**
 * @brief Query the current internal Vulkan operator JIT mode.
 *
 * @return nlo_operator_jit_mode Current JIT mode.
 */
nlo_operator_jit_mode nlo_operator_program_get_jit_mode(void);

/**
 * @brief Attempt to prepare a Vulkan JIT program for one compiled operator.
 *
 * @param backend Backend that will execute the program.
 * @param program Operator program to prepare.
 * @return nlo_vec_status Preparation status. Non-OK results are safe to ignore
 *         when the caller intends to fall back to the interpreter path.
 */
nlo_vec_status nlo_operator_program_prepare_jit(
    nlo_vector_backend* backend,
    nlo_operator_program* program
);

/**
 * @brief Execute a Vulkan JIT program when available.
 *
 * @param backend Backend that will execute the program.
 * @param program Operator program to execute.
 * @param eval_ctx Symbol source buffers for execution.
 * @param out_vector Program result vector.
 * @return nlo_vec_status Operation status, or `NLO_VEC_STATUS_UNSUPPORTED`
 *         when no JIT program is available and callers should fall back.
 */
nlo_vec_status nlo_operator_program_execute_jit(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    nlo_vec_buffer* out_vector
);

#ifdef __cplusplus
}
#endif
