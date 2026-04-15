/**
 * @file vk_backend_internal.h
 * @brief Internal cross-module helpers for Vulkan backend implementation.
 */
#pragma once

#include "numerics/vk_vector_ops.h"

typedef struct {
    double numerator;
    double denominator;
} vk_error_pair;

VkDeviceSize vk_min_size(VkDeviceSize a, VkDeviceSize b);

vec_status vk_create_buffer_raw(
    vector_backend* backend,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer* out_buffer,
    VkDeviceMemory* out_memory
);

void vk_destroy_buffer_raw(
    vector_backend* backend,
    VkBuffer* buffer,
    VkDeviceMemory* memory
);

vec_status vk_ensure_staging_capacity(vector_backend* backend, VkDeviceSize min_bytes);
vec_status vk_ensure_reduction_capacity(vector_backend* backend, VkDeviceSize min_elements);

vec_status vk_begin_commands(vector_backend* backend);
vec_status vk_submit_commands(vector_backend* backend);

void vk_cmd_transfer_to_compute(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

void vk_cmd_compute_to_compute(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

void vk_cmd_compute_to_transfer(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

void vk_cmd_transfer_to_host(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

vec_status vk_dispatch_kernel(
    vector_backend* backend,
    vk_kernel_id kernel_id,
    vec_buffer* dst,
    const vec_buffer* src,
    size_t elem_size,
    size_t length,
    double scalar0,
    double scalar1
);

vec_status vk_dispatch_complex_relative_error(
    vector_backend* backend,
    const vec_buffer* current,
    const vec_buffer* previous,
    double epsilon,
    double* out_error
);

vec_status vk_dispatch_complex_weighted_rms_error(
    vector_backend* backend,
    const vec_buffer* fine,
    const vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
);

vec_status vk_copy_buffer_chunked(
    vector_backend* backend,
    VkBuffer src_buffer,
    VkBuffer dst_buffer,
    size_t bytes
);
