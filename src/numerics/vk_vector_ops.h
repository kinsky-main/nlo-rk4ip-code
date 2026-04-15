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
 * @return vec_status Initialization status.
 */
vec_status vk_backend_init(vector_backend* backend, const vk_backend_config* config);

/**
 * @brief Release Vulkan backend resources.
 *
 * @param backend Backend handle to shut down.
 */
void vk_backend_shutdown(vector_backend* backend);

/**
 * @brief Ensure a Vulkan simulation phase command buffer is actively recording.
 *
 * @param backend Backend handle.
 * @return vec_status Operation status.
 */
vec_status vk_simulation_phase_begin(vector_backend* backend);

/**
 * @brief Submit and wait for the active Vulkan simulation phase command buffer.
 *
 * @param backend Backend handle.
 * @return vec_status Operation status.
 */
vec_status vk_simulation_phase_flush(vector_backend* backend);

/**
 * @brief Retrieve the active simulation phase command buffer for command recording.
 *
 * @param backend Backend handle.
 * @param out_command_buffer Destination Vulkan command buffer handle.
 * @return vec_status Operation status.
 */
vec_status vk_simulation_phase_command_buffer(
    vector_backend* backend,
    VkCommandBuffer* out_command_buffer
);

/**
 * @brief Mark that simulation phase commands were recorded and require submission.
 *
 * @param backend Backend handle.
 */
void vk_simulation_phase_mark_commands(vector_backend* backend);

/**
 * @brief Allocate Vulkan resources for one logical vector buffer.
 *
 * @param backend Backend handle.
 * @param buffer Buffer descriptor to initialize.
 * @return vec_status Operation status.
 */
vec_status vk_buffer_create(vector_backend* backend, vec_buffer* buffer);

/**
 * @brief Destroy Vulkan resources for one logical vector buffer.
 *
 * @param backend Backend handle.
 * @param buffer Buffer descriptor to destroy/reset.
 */
void vk_buffer_destroy(vector_backend* backend, vec_buffer* buffer);

/**
 * @brief Upload host data into a Vulkan vector buffer.
 *
 * @param backend Backend handle.
 * @param buffer Destination Vulkan buffer.
 * @param data Source host pointer.
 * @param bytes Number of bytes to upload.
 * @return vec_status Operation status.
 */
vec_status vk_upload(
    vector_backend* backend,
    vec_buffer* buffer,
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
 * @return vec_status Operation status.
 */
vec_status vk_download(
    vector_backend* backend,
    const vec_buffer* buffer,
    void* data,
    size_t bytes
);

/**
 * @brief Fill a real-valued Vulkan vector with a scalar.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param value Fill value.
 * @return vec_status Operation status.
 */
vec_status vk_op_real_fill(vector_backend* backend, vec_buffer* dst, double value);

/**
 * @brief Copy one real-valued Vulkan vector to another.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param src Source vector.
 * @return vec_status Operation status.
 */
vec_status vk_op_real_copy(vector_backend* backend, vec_buffer* dst, const vec_buffer* src);

/**
 * @brief Element-wise real multiply in place.
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @return vec_status Operation status.
 */
vec_status vk_op_real_mul_inplace(vector_backend* backend, vec_buffer* dst, const vec_buffer* src);

/**
 * @brief Fill a complex Vulkan vector with a scalar.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param value Fill value.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_fill(vector_backend* backend, vec_buffer* dst, nlo_complex value);

/**
 * @brief Copy one complex Vulkan vector to another.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param src Source vector.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_copy(vector_backend* backend, vec_buffer* dst, const vec_buffer* src);

/**
 * @brief Compute element-wise complex magnitude squared.
 *
 * @param backend Backend handle.
 * @param src Source complex vector.
 * @param dst Destination complex vector.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_magnitude_squared(
    vector_backend* backend,
    const vec_buffer* src,
    vec_buffer* dst
);

/**
 * @brief Multiply complex vector elements by a scalar in place.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param alpha Complex scale factor.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_scalar_mul_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    nlo_complex alpha
);

/**
 * @brief Element-wise complex multiply in place.
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_mul_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src
);

/**
 * @brief Element-wise complex add in place.
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_add_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src
);

/**
 * @brief Apply element-wise complex exponential in place.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_exp_inplace(vector_backend* backend, vec_buffer* dst);

/**
 * @brief Apply element-wise complex real power in place.
 *
 * @param backend Backend handle.
 * @param dst Destination vector.
 * @param exponent Real exponent.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_real_pow_inplace(
    vector_backend* backend,
    vec_buffer* dst,
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
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_relative_error(
    vector_backend* backend,
    const vec_buffer* current,
    const vec_buffer* previous,
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
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_weighted_rms_error(
    vector_backend* backend,
    const vec_buffer* fine,
    const vec_buffer* coarse,
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
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_axis_unshifted_from_delta(
    vector_backend* backend,
    vec_buffer* dst,
    double delta
);

/**
 * @brief Build one centered coordinate axis from sample spacing.
 *
 * @param backend Backend handle.
 * @param dst Destination axis vector.
 * @param delta Sample spacing.
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_axis_centered_from_delta(
    vector_backend* backend,
    vec_buffer* dst,
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
 * @return vec_status Operation status.
 */
vec_status vk_op_complex_mesh_from_axis_tfast(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* axis,
    size_t nt,
    size_t ny,
    vec_mesh_axis axis_kind
);

