/**
 * @file vk_backend_dispatch.c
 * @brief Vulkan kernel dispatch and reduction internals.
 */

#include "numerics/vk_backend_internal.h"
#include "utility/perf_profile.h"
#include <limits.h>
#include <math.h>
#include <string.h>

static size_t nlo_vk_max_chunk_elements(const nlo_vector_backend* backend, size_t elem_size)
{
    uint64_t by_storage = (uint64_t)backend->vk.limits.maxStorageBufferRange / (uint64_t)elem_size;
    uint64_t by_dispatch = (uint64_t)backend->vk.limits.maxComputeWorkGroupCount[0] * (uint64_t)NLO_VK_LOCAL_SIZE_X;
    uint64_t max_elems = (by_storage < by_dispatch) ? by_storage : by_dispatch;
    if (max_elems > (uint64_t)UINT32_MAX) {
        max_elems = (uint64_t)UINT32_MAX;
    }
    return (size_t)max_elems;
}

nlo_vec_status nlo_vk_acquire_descriptor_set(
    nlo_vector_backend* backend,
    VkDescriptorSet* out_descriptor_set
)
{
    if (backend == NULL || out_descriptor_set == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->vk.descriptor_sets == NULL || backend->vk.descriptor_set_count == 0u) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (!backend->in_simulation) {
        *out_descriptor_set = backend->vk.descriptor_sets[0];
        return NLO_VEC_STATUS_OK;
    }

    uint32_t index = backend->vk.simulation_descriptor_set_cursor;
    if (index >= backend->vk.descriptor_set_count) {
        nlo_vec_status status = nlo_vk_simulation_phase_flush(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        status = nlo_vk_simulation_phase_begin(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        index = backend->vk.simulation_descriptor_set_cursor;
        if (index >= backend->vk.descriptor_set_count) {
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
    }

    *out_descriptor_set = backend->vk.descriptor_sets[index];
    backend->vk.simulation_descriptor_set_cursor = index + 1u;
    return NLO_VEC_STATUS_OK;
}

static void nlo_vk_update_descriptor_set(
    nlo_vector_backend* backend,
    VkDescriptorSet descriptor_set,
    VkBuffer dst_buffer,
    VkDeviceSize dst_offset,
    VkDeviceSize dst_size,
    VkBuffer src_buffer,
    VkDeviceSize src_offset,
    VkDeviceSize src_size,
    VkBuffer src2_buffer,
    VkDeviceSize src2_offset,
    VkDeviceSize src2_size
)
{
    VkDescriptorBufferInfo dst_info = {
        .buffer = dst_buffer,
        .offset = dst_offset,
        .range = dst_size
    };
    VkDescriptorBufferInfo src_info = {
        .buffer = src_buffer,
        .offset = src_offset,
        .range = src_size
    };
    VkDescriptorBufferInfo src2_info = {
        .buffer = src2_buffer,
        .offset = src2_offset,
        .range = src2_size
    };
    VkWriteDescriptorSet writes[3] = {0};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptor_set;
    writes[0].dstBinding = 0u;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1u;
    writes[0].pBufferInfo = &dst_info;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptor_set;
    writes[1].dstBinding = 1u;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1u;
    writes[1].pBufferInfo = &src_info;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptor_set;
    writes[2].dstBinding = 2u;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1u;
    writes[2].pBufferInfo = &src2_info;

    vkUpdateDescriptorSets(backend->vk.device, 3u, writes, 0u, NULL);
}

static void nlo_vk_update_descriptor_set4(
    nlo_vector_backend* backend,
    VkDescriptorSet descriptor_set,
    VkBuffer dst_buffer,
    VkDeviceSize dst_offset,
    VkDeviceSize dst_size,
    VkBuffer src0_buffer,
    VkDeviceSize src0_offset,
    VkDeviceSize src0_size,
    VkBuffer src1_buffer,
    VkDeviceSize src1_offset,
    VkDeviceSize src1_size,
    VkBuffer src2_buffer,
    VkDeviceSize src2_offset,
    VkDeviceSize src2_size
)
{
    VkDescriptorBufferInfo infos[4] = {
        {
            .buffer = dst_buffer,
            .offset = dst_offset,
            .range = dst_size
        },
        {
            .buffer = src0_buffer,
            .offset = src0_offset,
            .range = src0_size
        },
        {
            .buffer = src1_buffer,
            .offset = src1_offset,
            .range = src1_size
        },
        {
            .buffer = src2_buffer,
            .offset = src2_offset,
            .range = src2_size
        }
    };
    VkWriteDescriptorSet writes[4] = {0};
    for (uint32_t i = 0u; i < 4u; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1u;
        writes[i].pBufferInfo = &infos[i];
    }

    vkUpdateDescriptorSets(backend->vk.device, 4u, writes, 0u, NULL);
}

static void nlo_vk_update_descriptor_set5(
    nlo_vector_backend* backend,
    VkDescriptorSet descriptor_set,
    VkBuffer dst_buffer,
    VkDeviceSize dst_offset,
    VkDeviceSize dst_size,
    VkBuffer src0_buffer,
    VkDeviceSize src0_offset,
    VkDeviceSize src0_size,
    VkBuffer src1_buffer,
    VkDeviceSize src1_offset,
    VkDeviceSize src1_size,
    VkBuffer src2_buffer,
    VkDeviceSize src2_offset,
    VkDeviceSize src2_size,
    VkBuffer src3_buffer,
    VkDeviceSize src3_offset,
    VkDeviceSize src3_size
)
{
    VkDescriptorBufferInfo infos[5] = {
        {.buffer = dst_buffer, .offset = dst_offset, .range = dst_size},
        {.buffer = src0_buffer, .offset = src0_offset, .range = src0_size},
        {.buffer = src1_buffer, .offset = src1_offset, .range = src1_size},
        {.buffer = src2_buffer, .offset = src2_offset, .range = src2_size},
        {.buffer = src3_buffer, .offset = src3_offset, .range = src3_size}
    };
    VkWriteDescriptorSet writes[5] = {0};
    for (uint32_t i = 0u; i < 5u; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1u;
        writes[i].pBufferInfo = &infos[i];
    }

    vkUpdateDescriptorSets(backend->vk.device, 5u, writes, 0u, NULL);
}

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
)
{
    const size_t max_chunk_elems = nlo_vk_max_chunk_elements(backend, elem_size);
    if (max_chunk_elems == 0u) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    size_t offset_elems = 0u;
    while (offset_elems < length) {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        size_t chunk_elems = length - offset_elems;
        if (chunk_elems > max_chunk_elems) {
            chunk_elems = max_chunk_elems;
        }

        nlo_vec_status status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkBuffer src_buffer = (src != NULL) ? src->vk_buffer : dst->vk_buffer;
        VkDeviceSize byte_offset = (VkDeviceSize)(offset_elems * elem_size);
        VkDeviceSize chunk_bytes = (VkDeviceSize)(chunk_elems * elem_size);
        nlo_vk_timestamp_ticket timestamp_ticket = {0};

        nlo_vk_update_descriptor_set(backend,
                                     descriptor_set,
                                     dst->vk_buffer,
                                     byte_offset,
                                     chunk_bytes,
                                     src_buffer,
                                     byte_offset,
                                     chunk_bytes,
                                     src_buffer,
                                     byte_offset,
                                     chunk_bytes);

        status = nlo_vk_begin_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkCommandBuffer cmd = backend->vk.command_buffer;
        nlo_vk_cmd_transfer_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        if (src_buffer != dst->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, src_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, src_buffer, byte_offset, chunk_bytes);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, backend->vk.kernels[kernel_id].pipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                backend->vk.pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                NULL);

        nlo_vk_push_constants push = {
            .count = (uint32_t)chunk_elems,
            .pad = 0u,
            .scalar0 = scalar0,
            .scalar1 = scalar1,
            .scalar2 = 0.0,
            .scalar3 = 0.0
        };
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);

        uint32_t groups = (uint32_t)((chunk_elems + (size_t)NLO_VK_LOCAL_SIZE_X - 1u) / (size_t)NLO_VK_LOCAL_SIZE_X);
        if (perf_event >= 0 && perf_event < NLO_PERF_EVENT_COUNT) {
            nlo_vk_timestamp_write_begin(backend,
                                         cmd,
                                         (uint32_t)perf_event,
                                         &timestamp_ticket);
        }
        vkCmdDispatch(cmd, groups, 1u, 1u);

        nlo_vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_transfer(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        if (timestamp_ticket.active) {
            nlo_vk_timestamp_write_end(backend, cmd, &timestamp_ticket);
        }

        status = nlo_vk_submit_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_perf_profile_add_gpu_dispatch(1u,
                                          2u,
                                          2u * (uint64_t)chunk_bytes);

        offset_elems += chunk_elems;
    }

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_dispatch_complex_affine_kernel(
    nlo_vector_backend* backend,
    nlo_vk_kernel_id kernel_id,
    nlo_perf_event_id perf_event,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    const nlo_vec_buffer* c,
    double scalar0,
    double scalar1,
    double scalar2
)
{
    const size_t elem_size = sizeof(nlo_complex);
    const size_t max_chunk_elems = nlo_vk_max_chunk_elements(backend, elem_size);
    if (max_chunk_elems == 0u) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    size_t offset_elems = 0u;
    while (offset_elems < dst->length) {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        size_t chunk_elems = dst->length - offset_elems;
        if (chunk_elems > max_chunk_elems) {
            chunk_elems = max_chunk_elems;
        }

        nlo_vec_status status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        const VkDeviceSize byte_offset = (VkDeviceSize)(offset_elems * elem_size);
        const VkDeviceSize chunk_bytes = (VkDeviceSize)(chunk_elems * elem_size);
        VkBuffer c_buffer = (c != NULL) ? c->vk_buffer : b->vk_buffer;
        nlo_vk_timestamp_ticket timestamp_ticket = {0};

        nlo_vk_update_descriptor_set4(backend,
                                      descriptor_set,
                                      dst->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      a->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      b->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      c_buffer,
                                      byte_offset,
                                      chunk_bytes);

        status = nlo_vk_begin_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkCommandBuffer cmd = backend->vk.command_buffer;
        nlo_vk_cmd_transfer_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_transfer_to_compute(cmd, a->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_compute(cmd, a->vk_buffer, byte_offset, chunk_bytes);
        if (b->vk_buffer != a->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, b->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, b->vk_buffer, byte_offset, chunk_bytes);
        }
        if (c != NULL && c->vk_buffer != a->vk_buffer && c->vk_buffer != b->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, c->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, c->vk_buffer, byte_offset, chunk_bytes);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, backend->vk.kernels[kernel_id].pipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                backend->vk.pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                NULL);

        nlo_vk_push_constants push = {
            .count = (uint32_t)chunk_elems,
            .pad = 0u,
            .scalar0 = scalar0,
            .scalar1 = scalar1,
            .scalar2 = scalar2,
            .scalar3 = 0.0
        };
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);

        const uint32_t groups =
            (uint32_t)((chunk_elems + (size_t)NLO_VK_LOCAL_SIZE_X - 1u) / (size_t)NLO_VK_LOCAL_SIZE_X);
        if (perf_event >= 0 && perf_event < NLO_PERF_EVENT_COUNT) {
            nlo_vk_timestamp_write_begin(backend, cmd, (uint32_t)perf_event, &timestamp_ticket);
        }
        vkCmdDispatch(cmd, groups, 1u, 1u);
        nlo_vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_transfer(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        if (timestamp_ticket.active) {
            nlo_vk_timestamp_write_end(backend, cmd, &timestamp_ticket);
        }

        status = nlo_vk_submit_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_perf_profile_add_gpu_dispatch(1u, 4u, 4u * (uint64_t)chunk_bytes);

        offset_elems += chunk_elems;
    }

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_dispatch_complex_axpy_inplace_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    double alpha
)
{
    return nlo_vk_dispatch_complex_affine_kernel(backend,
                                                 NLO_VK_KERNEL_COMPLEX_AXPY_INPLACE_REAL,
                                                 NLO_PERF_EVENT_VEC_AXPY,
                                                 dst,
                                                 dst,
                                                 src,
                                                 NULL,
                                                 alpha,
                                                 0.0,
                                                 0.0);
}

nlo_vec_status nlo_vk_dispatch_complex_affine_comb2_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta
)
{
    return nlo_vk_dispatch_complex_affine_kernel(backend,
                                                 NLO_VK_KERNEL_COMPLEX_AFFINE_COMB2_REAL,
                                                 NLO_PERF_EVENT_VEC_AFFINE_COMB2,
                                                 dst,
                                                 a,
                                                 b,
                                                 NULL,
                                                 alpha,
                                                 beta,
                                                 0.0);
}

nlo_vec_status nlo_vk_dispatch_complex_affine_comb3_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta,
    const nlo_vec_buffer* c,
    double gamma
)
{
    return nlo_vk_dispatch_complex_affine_kernel(backend,
                                                 NLO_VK_KERNEL_COMPLEX_AFFINE_COMB3_REAL,
                                                 NLO_PERF_EVENT_VEC_AFFINE_COMB3,
                                                 dst,
                                                 a,
                                                 b,
                                                 c,
                                                 alpha,
                                                 beta,
                                                 gamma);
}

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
)
{
    const size_t elem_size = sizeof(nlo_complex);
    const size_t max_chunk_elems = nlo_vk_max_chunk_elements(backend, elem_size);
    if (max_chunk_elems == 0u) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    size_t offset_elems = 0u;
    while (offset_elems < dst->length) {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        size_t chunk_elems = dst->length - offset_elems;
        if (chunk_elems > max_chunk_elems) {
            chunk_elems = max_chunk_elems;
        }

        nlo_vec_status status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        const VkDeviceSize byte_offset = (VkDeviceSize)(offset_elems * elem_size);
        const VkDeviceSize chunk_bytes = (VkDeviceSize)(chunk_elems * elem_size);
        nlo_vk_timestamp_ticket timestamp_ticket = {0};

        nlo_vk_update_descriptor_set5(backend,
                                      descriptor_set,
                                      dst->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      a->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      b->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      c->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      d->vk_buffer,
                                      byte_offset,
                                      chunk_bytes);

        status = nlo_vk_begin_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkCommandBuffer cmd = backend->vk.command_buffer;
        nlo_vk_cmd_transfer_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_transfer_to_compute(cmd, a->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_compute(cmd, a->vk_buffer, byte_offset, chunk_bytes);
        if (b->vk_buffer != a->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, b->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, b->vk_buffer, byte_offset, chunk_bytes);
        }
        if (c->vk_buffer != a->vk_buffer && c->vk_buffer != b->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, c->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, c->vk_buffer, byte_offset, chunk_bytes);
        }
        if (d->vk_buffer != a->vk_buffer && d->vk_buffer != b->vk_buffer && d->vk_buffer != c->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, d->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, d->vk_buffer, byte_offset, chunk_bytes);
        }

        vkCmdBindPipeline(cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          backend->vk.kernels[NLO_VK_KERNEL_COMPLEX_AFFINE_COMB4_REAL].pipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                backend->vk.pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                NULL);

        nlo_vk_push_constants push = {
            .count = (uint32_t)chunk_elems,
            .pad = 0u,
            .scalar0 = alpha,
            .scalar1 = beta,
            .scalar2 = gamma,
            .scalar3 = delta
        };
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);

        const uint32_t groups =
            (uint32_t)((chunk_elems + (size_t)NLO_VK_LOCAL_SIZE_X - 1u) / (size_t)NLO_VK_LOCAL_SIZE_X);
        nlo_vk_timestamp_write_begin(backend, cmd, (uint32_t)NLO_PERF_EVENT_VEC_AFFINE_COMB4, &timestamp_ticket);
        vkCmdDispatch(cmd, groups, 1u, 1u);
        nlo_vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_transfer(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_timestamp_write_end(backend, cmd, &timestamp_ticket);

        status = nlo_vk_submit_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_perf_profile_add_gpu_dispatch(1u, 5u, 5u * (uint64_t)chunk_bytes);

        offset_elems += chunk_elems;
    }

    return NLO_VEC_STATUS_OK;
}

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
)
{
    const size_t elem_size = sizeof(nlo_complex);
    const size_t max_chunk_elems = nlo_vk_max_chunk_elements(backend, elem_size);
    if (max_chunk_elems == 0u) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    size_t offset_elems = 0u;
    while (offset_elems < fine_out->length) {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        size_t chunk_elems = fine_out->length - offset_elems;
        if (chunk_elems > max_chunk_elems) {
            chunk_elems = max_chunk_elems;
        }

        nlo_vec_status status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        const VkDeviceSize byte_offset = (VkDeviceSize)(offset_elems * elem_size);
        const VkDeviceSize chunk_bytes = (VkDeviceSize)(chunk_elems * elem_size);
        nlo_vk_timestamp_ticket timestamp_ticket = {0};

        nlo_vk_update_descriptor_set5(backend,
                                      descriptor_set,
                                      fine_out->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      coarse_out->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      base->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      stage_k4->vk_buffer,
                                      byte_offset,
                                      chunk_bytes,
                                      stage_k5->vk_buffer,
                                      byte_offset,
                                      chunk_bytes);

        status = nlo_vk_begin_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkCommandBuffer cmd = backend->vk.command_buffer;
        nlo_vk_cmd_transfer_to_compute(cmd, fine_out->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_compute(cmd, fine_out->vk_buffer, byte_offset, chunk_bytes);
        if (coarse_out->vk_buffer != fine_out->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, coarse_out->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, coarse_out->vk_buffer, byte_offset, chunk_bytes);
        }
        nlo_vk_cmd_transfer_to_compute(cmd, base->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_compute(cmd, base->vk_buffer, byte_offset, chunk_bytes);
        if (stage_k4->vk_buffer != base->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, stage_k4->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, stage_k4->vk_buffer, byte_offset, chunk_bytes);
        }
        if (stage_k5->vk_buffer != base->vk_buffer && stage_k5->vk_buffer != stage_k4->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, stage_k5->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, stage_k5->vk_buffer, byte_offset, chunk_bytes);
        }

        vkCmdBindPipeline(cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          backend->vk.kernels[NLO_VK_KERNEL_COMPLEX_EMBEDDED_ERROR_PAIR_REAL].pipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                backend->vk.pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                NULL);

        nlo_vk_push_constants push = {
            .count = (uint32_t)chunk_elems,
            .pad = 0u,
            .scalar0 = fine_k4_coeff,
            .scalar1 = coarse_k4_coeff,
            .scalar2 = coarse_k5_coeff,
            .scalar3 = 0.0
        };
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);

        const uint32_t groups =
            (uint32_t)((chunk_elems + (size_t)NLO_VK_LOCAL_SIZE_X - 1u) / (size_t)NLO_VK_LOCAL_SIZE_X);
        nlo_vk_timestamp_write_begin(backend,
                                     cmd,
                                     (uint32_t)NLO_PERF_EVENT_VEC_EMBEDDED_ERROR_PAIR,
                                     &timestamp_ticket);
        vkCmdDispatch(cmd, groups, 1u, 1u);
        nlo_vk_cmd_compute_to_compute(cmd, fine_out->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_transfer(cmd, fine_out->vk_buffer, byte_offset, chunk_bytes);
        if (coarse_out->vk_buffer != fine_out->vk_buffer) {
            nlo_vk_cmd_compute_to_compute(cmd, coarse_out->vk_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_transfer(cmd, coarse_out->vk_buffer, byte_offset, chunk_bytes);
        }
        nlo_vk_timestamp_write_end(backend, cmd, &timestamp_ticket);

        status = nlo_vk_submit_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_perf_profile_add_gpu_dispatch(1u, 5u, 5u * (uint64_t)chunk_bytes);

        offset_elems += chunk_elems;
    }

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_dispatch_complex_lerp(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    double alpha
)
{
    return nlo_vk_dispatch_complex_affine_kernel(backend,
                                                 NLO_VK_KERNEL_COMPLEX_LERP,
                                                 NLO_PERF_EVENT_VEC_LERP,
                                                 dst,
                                                 a,
                                                 b,
                                                 NULL,
                                                 alpha,
                                                 0.0,
                                                 0.0);
}

static uint32_t nlo_vk_dispatch_groups_for_count(size_t count)
{
    if (count == 0u) {
        return 1u;
    }
    return (uint32_t)((count + (size_t)NLO_VK_LOCAL_SIZE_X - 1u) / (size_t)NLO_VK_LOCAL_SIZE_X);
}

static uint32_t nlo_vk_required_descriptor_sets_for_reduction(uint32_t groups)
{
    uint32_t required_sets = 1u;
    uint32_t reduce_count = groups;
    while (reduce_count > 1u) {
        reduce_count = nlo_vk_dispatch_groups_for_count((size_t)reduce_count);
        required_sets += 1u;
    }
    return required_sets;
}

static nlo_vec_status nlo_vk_reserve_descriptor_sets_for_reduction(
    nlo_vector_backend* backend,
    uint32_t required_sets
)
{
    if (backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    uint32_t available_sets = backend->vk.descriptor_set_count;
    if (backend->in_simulation) {
        const uint32_t used = backend->vk.simulation_descriptor_set_cursor;
        if (used > available_sets) {
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
        available_sets -= used;
    }

    if (required_sets <= available_sets) {
        return NLO_VEC_STATUS_OK;
    }

    if (backend->in_simulation) {
        nlo_vec_status status = nlo_vk_simulation_phase_flush(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        available_sets = backend->vk.descriptor_set_count;
    }

    return (required_sets <= available_sets)
               ? NLO_VEC_STATUS_OK
               : NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_vk_dispatch_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    if (backend == NULL || current == NULL || previous == NULL || out_error == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (current->length != previous->length) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (current->length > (size_t)UINT32_MAX) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    uint32_t groups = nlo_vk_dispatch_groups_for_count(current->length);
    const uint32_t required_sets = nlo_vk_required_descriptor_sets_for_reduction(groups);

    nlo_vec_status status = nlo_vk_reserve_descriptor_sets_for_reduction(backend, required_sets);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vk_ensure_reduction_capacity(backend, (VkDeviceSize)groups);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_vk_ensure_staging_capacity(backend, (VkDeviceSize)sizeof(double));
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vk_begin_commands(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkCommandBuffer cmd = backend->vk.command_buffer;
    VkDeviceSize complex_bytes = (VkDeviceSize)current->bytes;
    VkDeviceSize partial_bytes = (VkDeviceSize)groups * (VkDeviceSize)sizeof(double);
    uint64_t dispatch_count = 0u;
    uint64_t dispatch_pass_count = 0u;
    uint64_t dispatch_pass_bytes = 0u;

    nlo_vk_cmd_compute_to_compute(cmd, current->vk_buffer, 0u, complex_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, previous->vk_buffer, 0u, complex_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_b, 0u, partial_bytes);

    nlo_vk_update_descriptor_set(backend,
                                 descriptor_set,
                                 backend->vk.reduction_buffer_a,
                                 0u,
                                 partial_bytes,
                                 current->vk_buffer,
                                 0u,
                                 complex_bytes,
                                 previous->vk_buffer,
                                 0u,
                                 complex_bytes);

    vkCmdBindPipeline(cmd,
                      VK_PIPELINE_BIND_POINT_COMPUTE,
                      backend->vk.kernels[NLO_VK_KERNEL_COMPLEX_RELATIVE_ERROR_REDUCE].pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            backend->vk.pipeline_layout,
                            0u,
                            1u,
                            &descriptor_set,
                            0u,
                            NULL);

    nlo_vk_push_constants push = {
        .count = (uint32_t)current->length,
        .pad = 0u,
        .scalar0 = epsilon,
        .scalar1 = 0.0
    };
    vkCmdPushConstants(cmd,
                       backend->vk.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0u,
                       (uint32_t)sizeof(push),
                       &push);
    vkCmdDispatch(cmd, groups, 1u, 1u);
    dispatch_count += 1u;
    dispatch_pass_count += 2u;
    dispatch_pass_bytes += (uint64_t)partial_bytes * 2u;
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);

    VkBuffer src_buffer = backend->vk.reduction_buffer_a;
    VkBuffer dst_buffer = backend->vk.reduction_buffer_b;
    uint32_t count = groups;
    while (count > 1u) {
        status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        uint32_t next_groups = nlo_vk_dispatch_groups_for_count((size_t)count);
        VkDeviceSize src_bytes = (VkDeviceSize)count * (VkDeviceSize)sizeof(double);
        VkDeviceSize dst_bytes = (VkDeviceSize)next_groups * (VkDeviceSize)sizeof(double);

        nlo_vk_update_descriptor_set(backend,
                                     descriptor_set,
                                     dst_buffer,
                                     0u,
                                     dst_bytes,
                                     src_buffer,
                                     0u,
                                     src_bytes,
                                     src_buffer,
                                     0u,
                                     src_bytes);

        vkCmdBindPipeline(cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          backend->vk.kernels[NLO_VK_KERNEL_REAL_MAX_REDUCE].pipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                backend->vk.pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                NULL);

        push.count = count;
        push.scalar0 = 0.0;
        push.scalar1 = 0.0;
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);
        vkCmdDispatch(cmd, next_groups, 1u, 1u);
        dispatch_count += 1u;
        dispatch_pass_count += 2u;
        dispatch_pass_bytes += (uint64_t)src_bytes + (uint64_t)dst_bytes;
        nlo_vk_cmd_compute_to_compute(cmd, dst_buffer, 0u, dst_bytes);

        VkBuffer tmp = src_buffer;
        src_buffer = dst_buffer;
        dst_buffer = tmp;
        count = next_groups;
    }

    nlo_vk_cmd_compute_to_transfer(cmd, src_buffer, 0u, (VkDeviceSize)sizeof(double));

    VkBufferCopy copy = {
        .srcOffset = 0u,
        .dstOffset = 0u,
        .size = (VkDeviceSize)sizeof(double)
    };
    vkCmdCopyBuffer(cmd, src_buffer, backend->vk.staging_buffer, 1u, &copy);
    nlo_vk_cmd_transfer_to_host(cmd, backend->vk.staging_buffer, 0u, (VkDeviceSize)sizeof(double));

    status = nlo_vk_submit_commands(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    nlo_perf_profile_add_gpu_dispatch(dispatch_count,
                                      dispatch_pass_count,
                                      dispatch_pass_bytes);
    nlo_perf_profile_add_gpu_host_transfer_copy(1u, (uint64_t)sizeof(double));
    nlo_perf_profile_add_gpu_download(1u, (uint64_t)sizeof(double));
    if (backend->in_simulation) {
        status = nlo_vk_simulation_phase_flush(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    double ratio = 0.0;
    memcpy(&ratio, backend->vk.staging_mapped_ptr, sizeof(double));
    if (ratio < 0.0) {
        ratio = 0.0;
    }
    *out_error = sqrt(ratio);
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_dispatch_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
)
{
    if (backend == NULL || fine == NULL || coarse == NULL || out_error == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (fine->length != coarse->length) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (fine->length > (size_t)UINT32_MAX) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    uint32_t groups = nlo_vk_dispatch_groups_for_count(fine->length);
    const uint32_t required_sets = nlo_vk_required_descriptor_sets_for_reduction(groups);

    nlo_vec_status status = nlo_vk_reserve_descriptor_sets_for_reduction(backend, required_sets);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vk_ensure_reduction_capacity(backend, (VkDeviceSize)(groups * 2u));
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_vk_ensure_staging_capacity(backend, (VkDeviceSize)sizeof(nlo_vk_error_pair));
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vk_begin_commands(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkCommandBuffer cmd = backend->vk.command_buffer;
    VkDeviceSize complex_bytes = (VkDeviceSize)fine->bytes;
    VkDeviceSize partial_bytes = (VkDeviceSize)groups * (VkDeviceSize)sizeof(nlo_vk_error_pair);
    uint64_t dispatch_count = 0u;
    uint64_t dispatch_pass_count = 0u;
    uint64_t dispatch_pass_bytes = 0u;
    nlo_vk_timestamp_ticket timestamp_ticket = {0};

    nlo_vk_cmd_compute_to_compute(cmd, fine->vk_buffer, 0u, complex_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, coarse->vk_buffer, 0u, complex_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_b, 0u, partial_bytes);

    nlo_vk_update_descriptor_set(backend,
                                 descriptor_set,
                                 backend->vk.reduction_buffer_a,
                                 0u,
                                 partial_bytes,
                                 fine->vk_buffer,
                                 0u,
                                 complex_bytes,
                                 coarse->vk_buffer,
                                 0u,
                                 complex_bytes);

    vkCmdBindPipeline(cmd,
                      VK_PIPELINE_BIND_POINT_COMPUTE,
                      backend->vk.kernels[NLO_VK_KERNEL_COMPLEX_WEIGHTED_RMS_REDUCE].pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            backend->vk.pipeline_layout,
                            0u,
                            1u,
                            &descriptor_set,
                            0u,
                            NULL);

    nlo_vk_push_constants push = {
        .count = (uint32_t)fine->length,
        .pad = 0u,
        .scalar0 = atol,
        .scalar1 = rtol
    };
    vkCmdPushConstants(cmd,
                       backend->vk.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0u,
                       (uint32_t)sizeof(push),
                       &push);
    nlo_vk_timestamp_write_begin(backend,
                                 cmd,
                                 (uint32_t)NLO_PERF_EVENT_WEIGHTED_RMS_ERROR,
                                 &timestamp_ticket);
    vkCmdDispatch(cmd, groups, 1u, 1u);
    dispatch_count += 1u;
    dispatch_pass_count += 2u;
    dispatch_pass_bytes += (uint64_t)partial_bytes * 2u;
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);

    VkBuffer src_buffer = backend->vk.reduction_buffer_a;
    VkBuffer dst_buffer = backend->vk.reduction_buffer_b;
    uint32_t count = groups;
    while (count > 1u) {
        status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        uint32_t next_groups = nlo_vk_dispatch_groups_for_count((size_t)count);
        VkDeviceSize src_bytes = (VkDeviceSize)count * (VkDeviceSize)sizeof(nlo_vk_error_pair);
        VkDeviceSize dst_bytes = (VkDeviceSize)next_groups * (VkDeviceSize)sizeof(nlo_vk_error_pair);

        nlo_vk_update_descriptor_set(backend,
                                     descriptor_set,
                                     dst_buffer,
                                     0u,
                                     dst_bytes,
                                     src_buffer,
                                     0u,
                                     src_bytes,
                                     src_buffer,
                                     0u,
                                     src_bytes);

        vkCmdBindPipeline(cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          backend->vk.kernels[NLO_VK_KERNEL_PAIR_SUM_REDUCE].pipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                backend->vk.pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                NULL);

        push.count = count;
        push.scalar0 = 0.0;
        push.scalar1 = 0.0;
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);
        vkCmdDispatch(cmd, next_groups, 1u, 1u);
        dispatch_count += 1u;
        dispatch_pass_count += 2u;
        dispatch_pass_bytes += (uint64_t)src_bytes + (uint64_t)dst_bytes;
        nlo_vk_cmd_compute_to_compute(cmd, dst_buffer, 0u, dst_bytes);

        VkBuffer tmp = src_buffer;
        src_buffer = dst_buffer;
        dst_buffer = tmp;
        count = next_groups;
    }

    nlo_vk_cmd_compute_to_transfer(cmd, src_buffer, 0u, (VkDeviceSize)sizeof(nlo_vk_error_pair));

    VkBufferCopy copy = {
        .srcOffset = 0u,
        .dstOffset = 0u,
        .size = (VkDeviceSize)sizeof(nlo_vk_error_pair)
    };
    vkCmdCopyBuffer(cmd, src_buffer, backend->vk.staging_buffer, 1u, &copy);
    nlo_vk_cmd_transfer_to_host(cmd, backend->vk.staging_buffer, 0u, (VkDeviceSize)sizeof(nlo_vk_error_pair));
    nlo_vk_timestamp_write_end(backend, cmd, &timestamp_ticket);

    status = nlo_vk_submit_commands(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    nlo_perf_profile_add_gpu_dispatch(dispatch_count,
                                      dispatch_pass_count,
                                      dispatch_pass_bytes);
    nlo_perf_profile_add_gpu_host_transfer_copy(1u, (uint64_t)sizeof(nlo_vk_error_pair));
    nlo_perf_profile_add_gpu_download(1u, (uint64_t)sizeof(nlo_vk_error_pair));
    if (backend->in_simulation) {
        status = nlo_vk_simulation_phase_flush(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    nlo_vk_error_pair pair = {0.0, 0.0};
    memcpy(&pair, backend->vk.staging_mapped_ptr, sizeof(pair));
    if (pair.denominator <= 0.0) {
        *out_error = 0.0;
        return NLO_VEC_STATUS_OK;
    }

    double ratio = pair.numerator / pair.denominator;
    if (ratio < 0.0) {
        ratio = 0.0;
    }
    *out_error = sqrt(ratio);
    return NLO_VEC_STATUS_OK;
}
