/**
 * @file vk_vector_ops.h
 * @dir src/numerics
 * @brief Internal Vulkan vector operation backend.
 */
#pragma once

#include "backend/vector_backend_internal.h"

/**
 * @brief Initialize Vulkan backend state and kernel resources.
 *
 * @param backend Backend handle to initialize.
 * @param config Optional Vulkan overrides (NULL enables auto-detect path).
 * @return nlo_vec_status Initialization status.
 */
nlo_vec_status nlo_vk_backend_init(nlo_vector_backend* backend, const nlo_vk_backend_config* config);

/**
 * @brief Release Vulkan backend resources.
 *
 * @param backend Backend handle to shut down.
 */
void nlo_vk_backend_shutdown(nlo_vector_backend* backend);

/**
 * @brief Ensure a Vulkan simulation phase command buffer is actively recording.
 *
 * @param backend Backend handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_simulation_phase_begin(nlo_vector_backend* backend);

/**
 * @brief Submit and wait for the active Vulkan simulation phase command buffer.
 *
 * @param backend Backend handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_simulation_phase_flush(nlo_vector_backend* backend);

/**
 * @brief Retrieve the active simulation phase command buffer for command recording.
 *
 * @param backend Backend handle.
 * @param out_command_buffer Destination Vulkan command buffer handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_simulation_phase_command_buffer(
    nlo_vector_backend* backend,
    VkCommandBuffer* out_command_buffer
);

/**
 * @brief Mark that simulation phase commands were recorded and require submission.
 *
 * @param backend Backend handle.
 */
void nlo_vk_simulation_phase_mark_commands(nlo_vector_backend* backend);

/**
 * @brief Allocate Vulkan resources for one logical vector buffer.
 *
 * @param backend Backend handle.
 * @param buffer Buffer descriptor to initialize.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_buffer_create(nlo_vector_backend* backend, nlo_vec_buffer* buffer);

/**
 * @brief Destroy Vulkan resources for one logical vector buffer.
 *
 * @param backend Backend handle.
 * @param buffer Buffer descriptor to destroy/reset.
 */
void nlo_vk_buffer_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer);

/**
 * @brief Upload host data into a Vulkan vector buffer.
 *
 * @param backend Backend handle.
 * @param buffer Destination Vulkan buffer.
 * @param data Source host pointer.
 * @param bytes Number of bytes to upload.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
);

/**
 * @brief Download Vulkan vector buffer data into host memory.
 *
 * @param backend Backend handle.
 * @param buffer Source Vulkan buffer.
 * @param data Destination host pointer.
 * @param bytes Number of bytes to download.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
);

/**
 * @brief Fill a real-valued Vulkan vector with a scalar.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param value Fill value.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value);

/**
 * @brief Copy one real-valued Vulkan vector to another.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param src Source vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);

/**
 * @brief Element-wise real multiply in place.
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);

/**
 * @brief Fill a complex Vulkan vector with a scalar.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param value Fill value.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value);

/**
 * @brief Copy one complex Vulkan vector to another.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param src Source vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);

/**
 * @brief Compute element-wise complex magnitude squared.
 *
 * @param backend Backend handle.
 * @param src Source complex vector.
 * @param dst Destination complex vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
);

/**
 * @brief Multiply complex vector elements by a scalar in place.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param alpha Complex scale factor.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
);

/**
 * @brief Element-wise complex multiply in place.
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
);

/**
 * @brief Element-wise complex add in place.
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
);

/**
 * @brief Apply element-wise complex exponential in place.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);

/**
 * @brief Apply element-wise complex real power in place.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param exponent Real exponent.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
);

/**
 * @brief Compute relative complex error between current and previous vectors.
 *
 * @param backend Backend handle.
 * @param current Current vector.
 * @param previous Previous/reference vector.
 * @param epsilon Stabilizing denominator floor.
 * @param out_error Destination scalar error.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
);

/**
 * @brief Compute weighted RMS complex error between fine and coarse vectors.
 *
 * @param backend Backend handle.
 * @param fine Fine/reference vector.
 * @param coarse Coarse/approximate vector.
 * @param atol Absolute tolerance term.
 * @param rtol Relative tolerance term.
 * @param out_error Destination scalar error.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
);

/**
 * @brief Build one unshifted angular-frequency axis from sample spacing.
 *
 * @param backend Backend handle.
 * @param dst Destination axis vector.
 * @param delta Sample spacing (> 0).
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_axis_unshifted_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
);

/**
 * @brief Build one centered coordinate axis from sample spacing.
 *
 * @param backend Backend handle.
 * @param dst Destination axis vector.
 * @param delta Sample spacing.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_axis_centered_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
);

/**
 * @brief Expand one axis vector into a full 3D mesh for t-fast layout.
 *
 * @param backend Backend handle.
 * @param dst Destination full-volume vector.
 * @param axis Source 1D axis vector.
 * @param nt Temporal sample count.
 * @param ny Y sample count.
 * @param axis_kind Axis selector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vk_op_complex_mesh_from_axis_tfast(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* axis,
    size_t nt,
    size_t ny,
    nlo_vec_mesh_axis axis_kind
);

