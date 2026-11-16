/**
 * @file operator_program_jit.h
 * @brief Internal Vulkan JIT hooks for compiled operator programs.
 *
 * A program lowered by operator_program_compile (`jit_eligible`) can be turned
 * into one Vulkan compute kernel that evaluates the whole expression per
 * element, instead of one dispatch per instruction. Compiled SPIR-V is cached
 * for the life of the process, keyed by the lowered program, so repeated
 * propagate calls with the same operators do not recompile.
 *
 * Setting the environment variable `NLO_OPERATOR_JIT=0` disables the JIT and
 * keeps every program on the interpreter.
 */
#pragma once

#include "physics/operator_program.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Prepare a Vulkan kernel for one compiled operator program.
 *
 * On success sets `program->vk_jit_active`; the kernel stays owned by
 * @p backend and is released with it.
 *
 * @param backend Backend that will execute the program.
 * @param program Compiled operator program.
 * @return vec_status `VEC_STATUS_OK` when the program will run as a JIT
 *         kernel; `VEC_STATUS_UNSUPPORTED` when JIT does not apply (non-Vulkan
 *         backend, program not lowered, JIT disabled); other codes when
 *         compilation failed. Every non-OK result leaves the program on the
 *         interpreter.
 */
vec_status operator_program_prepare_jit(
    vector_backend* backend,
    operator_program* program
);

/**
 * @brief Execute a prepared JIT kernel into @p out_vector.
 *
 * @param backend Backend that prepared the program.
 * @param program Program with `vk_jit_active` set.
 * @param eval_ctx Symbol source buffers.
 * @param out_vector Result vector, same length as the field.
 * @return vec_status Operation status, or `VEC_STATUS_UNSUPPORTED` when no
 *         kernel is available and the caller should use the interpreter.
 */
vec_status operator_program_execute_jit(
    vector_backend* backend,
    const operator_program* program,
    const operator_eval_context* eval_ctx,
    vec_buffer* out_vector
);

/**
 * @brief Number of programs held in the process-wide SPIR-V cache.
 *
 * @return size_t Cache entry count (0 when the Vulkan backend is not built).
 */
size_t operator_program_jit_cache_entry_count(void);

#ifdef __cplusplus
}
#endif
