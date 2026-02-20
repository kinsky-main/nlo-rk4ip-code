/**
 * @file vk_vector_ops.h
 * @dir src/numerics
 * @brief Internal Vulkan vector operation backend.
 */
#pragma once

#include "backend/vector_backend_internal.h"

nlo_vec_status nlo_vk_backend_init(nlo_vector_backend* backend, const nlo_vk_backend_config* config);
void nlo_vk_backend_shutdown(nlo_vector_backend* backend);

/**
 * @brief Ensure a Vulkan simulation phase command buffer is actively recording.
 */
nlo_vec_status nlo_vk_simulation_phase_begin(nlo_vector_backend* backend);

/**
 * @brief Submit and wait for the active Vulkan simulation phase command buffer.
 */
nlo_vec_status nlo_vk_simulation_phase_flush(nlo_vector_backend* backend);

/**
 * @brief Retrieve the active simulation phase command buffer for command recording.
 */
nlo_vec_status nlo_vk_simulation_phase_command_buffer(
    nlo_vector_backend* backend,
    VkCommandBuffer* out_command_buffer
);

/**
 * @brief Mark that simulation phase commands were recorded and require submission.
 */
void nlo_vk_simulation_phase_mark_commands(nlo_vector_backend* backend);

nlo_vec_status nlo_vk_buffer_create(nlo_vector_backend* backend, nlo_vec_buffer* buffer);
void nlo_vk_buffer_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer);

nlo_vec_status nlo_vk_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
);
nlo_vec_status nlo_vk_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
);

nlo_vec_status nlo_vk_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value);
nlo_vec_status nlo_vk_op_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vk_op_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vk_op_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value);
nlo_vec_status nlo_vk_op_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vk_op_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
);
nlo_vec_status nlo_vk_op_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
);
nlo_vec_status nlo_vk_op_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
);
nlo_vec_status nlo_vk_op_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
);
nlo_vec_status nlo_vk_op_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
nlo_vec_status nlo_vk_op_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
);
nlo_vec_status nlo_vk_op_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
);
nlo_vec_status nlo_vk_op_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
);


