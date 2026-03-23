/**
 * @file vk_backend_internal.h
 * @brief Internal cross-module helpers for Vulkan backend implementation.
 */
#pragma once

#include "numerics/vk_vector_ops.h"
#include "utility/perf_profile.h"

typedef struct {
    double numerator;
    double denominator;
} nlo_vk_error_pair;

typedef struct {
    uint32_t event_id;
    uint32_t span_index;
    uint32_t start_query;
    uint32_t end_query;
    int active;
} nlo_vk_timestamp_ticket;

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
nlo_vec_status nlo_vk_wait_for_pending_submit(nlo_vector_backend* backend);
nlo_vec_status nlo_vk_acquire_descriptor_set(
    nlo_vector_backend* backend,
    VkDescriptorSet* out_descriptor_set
);
void nlo_vk_operator_jit_destroy_all(nlo_vector_backend* backend);
int nlo_vk_timestamps_supported(const nlo_vector_backend* backend);
void nlo_vk_timestamp_begin_command_buffer(nlo_vector_backend* backend, VkCommandBuffer cmd);
void nlo_vk_timestamp_write_begin(
    nlo_vector_backend* backend,
    VkCommandBuffer cmd,
    uint32_t event_id,
    nlo_vk_timestamp_ticket* out_ticket
);
void nlo_vk_timestamp_write_end(
    nlo_vector_backend* backend,
    VkCommandBuffer cmd,
    const nlo_vk_timestamp_ticket* ticket
);
void nlo_vk_timestamp_collect(nlo_vector_backend* backend);

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
    nlo_perf_event_id perf_event,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    size_t elem_size,
    size_t length,
    double scalar0,
    double scalar1
);

nlo_vec_status nlo_vk_dispatch_complex_axpy_inplace_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    double alpha
);

nlo_vec_status nlo_vk_dispatch_complex_affine_comb2_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta
);

nlo_vec_status nlo_vk_dispatch_complex_affine_comb3_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta,
    const nlo_vec_buffer* c,
    double gamma
);

nlo_vec_status nlo_vk_dispatch_complex_affine_comb4_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta,
    const nlo_vec_buffer* c,
    double gamma,
    const nlo_vec_buffer* d,
    double delta
);

nlo_vec_status nlo_vk_dispatch_complex_embedded_error_pair_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* fine_out,
    nlo_vec_buffer* coarse_out,
    const nlo_vec_buffer* base,
    const nlo_vec_buffer* stage_k4,
    double fine_k4_coeff,
    double coarse_k4_coeff,
    const nlo_vec_buffer* stage_k5,
    double coarse_k5_coeff
);

nlo_vec_status nlo_vk_dispatch_complex_lerp(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    double alpha
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
