/**
 * @file vk_backend_internal.h
 * @brief Internal cross-module helpers for Vulkan backend implementation.
 */
#pragma once

#include "numerics/vk_vector_ops.h"

typedef struct {
    double numerator;
    double denominator;
} nlo_vk_error_pair;

VkDeviceSize nlo_vk_min_size(VkDeviceSize a, VkDeviceSize b);

nlo_vec_status nlo_vk_create_buffer_raw(
    nlo_vector_backend* backend,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer* out_buffer,
    VkDeviceMemory* out_memory
);

void nlo_vk_destroy_buffer_raw(
    nlo_vector_backend* backend,
    VkBuffer* buffer,
    VkDeviceMemory* memory
);

nlo_vec_status nlo_vk_ensure_staging_capacity(nlo_vector_backend* backend, VkDeviceSize min_bytes);
nlo_vec_status nlo_vk_ensure_reduction_capacity(nlo_vector_backend* backend, VkDeviceSize min_elements);

nlo_vec_status nlo_vk_begin_commands(nlo_vector_backend* backend);
nlo_vec_status nlo_vk_submit_commands(nlo_vector_backend* backend);

void nlo_vk_cmd_transfer_to_compute(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

void nlo_vk_cmd_compute_to_compute(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

void nlo_vk_cmd_compute_to_transfer(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

void nlo_vk_cmd_transfer_to_host(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

nlo_vec_status nlo_vk_dispatch_kernel(
    nlo_vector_backend* backend,
    nlo_vk_kernel_id kernel_id,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    size_t elem_size,
    size_t length,
    double scalar0,
    double scalar1
);

nlo_vec_status nlo_vk_dispatch_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
);

nlo_vec_status nlo_vk_dispatch_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
);

nlo_vec_status nlo_vk_copy_buffer_chunked(
    nlo_vector_backend* backend,
    VkBuffer src_buffer,
    VkBuffer dst_buffer,
    size_t bytes
);
