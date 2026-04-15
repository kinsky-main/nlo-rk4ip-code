/**
 * @file vk_backend_dispatch.c
 * @brief Vulkan kernel dispatch and reduction internals.
 */

#include "numerics/vk_backend_internal.h"
#include "utility/perf_profile.h"
#include <limits.h>
#include <math.h>
#include <string.h>

static size_t vk_max_chunk_elements(const vector_backend* backend, size_t elem_size)
{
    uint64_t by_storage = (uint64_t)backend->vk.limits.maxStorageBufferRange / (uint64_t)elem_size;
    uint64_t by_dispatch = (uint64_t)backend->vk.limits.maxComputeWorkGroupCount[0] * (uint64_t)VK_LOCAL_SIZE_X;
    uint64_t max_elems = (by_storage < by_dispatch) ? by_storage : by_dispatch;
    if (max_elems > (uint64_t)UINT32_MAX) {
        max_elems = (uint64_t)UINT32_MAX;
    }
    return (size_t)max_elems;
}

static vec_status vk_acquire_descriptor_set(
    vector_backend* backend,
    VkDescriptorSet* out_descriptor_set
)
{
    if (backend == NULL || out_descriptor_set == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->vk.descriptor_sets == NULL || backend->vk.descriptor_set_count == 0u) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (!backend->in_simulation) {
        *out_descriptor_set = backend->vk.descriptor_sets[0];
        return VEC_STATUS_OK;
    }

    uint32_t index = backend->vk.simulation_descriptor_set_cursor;
    if (index >= backend->vk.descriptor_set_count) {
        vec_status status = vk_simulation_phase_flush(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        status = vk_simulation_phase_begin(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        index = backend->vk.simulation_descriptor_set_cursor;
        if (index >= backend->vk.descriptor_set_count) {
            return VEC_STATUS_BACKEND_UNAVAILABLE;
        }
    }

    *out_descriptor_set = backend->vk.descriptor_sets[index];
    backend->vk.simulation_descriptor_set_cursor = index + 1u;
    return VEC_STATUS_OK;
}

static void vk_update_descriptor_set(
    vector_backend* backend,
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

vec_status vk_dispatch_kernel(
    vector_backend* backend,
    vk_kernel_id kernel_id,
    vec_buffer* dst,
    const vec_buffer* src,
    size_t elem_size,
    size_t length,
    double scalar0,
    double scalar1
)
{
    const size_t max_chunk_elems = vk_max_chunk_elements(backend, elem_size);
    if (max_chunk_elems == 0u) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    size_t offset_elems = 0u;
    while (offset_elems < length) {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        size_t chunk_elems = length - offset_elems;
        if (chunk_elems > max_chunk_elems) {
            chunk_elems = max_chunk_elems;
        }

        vec_status status = vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        VkBuffer src_buffer = (src != NULL) ? src->vk_buffer : dst->vk_buffer;
        VkDeviceSize byte_offset = (VkDeviceSize)(offset_elems * elem_size);
        VkDeviceSize chunk_bytes = (VkDeviceSize)(chunk_elems * elem_size);

        vk_update_descriptor_set(backend,
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

        status = vk_begin_commands(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        VkCommandBuffer cmd = backend->vk.command_buffer;
        vk_cmd_transfer_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        if (src_buffer != dst->vk_buffer) {
            vk_cmd_transfer_to_compute(cmd, src_buffer, byte_offset, chunk_bytes);
            vk_cmd_compute_to_compute(cmd, src_buffer, byte_offset, chunk_bytes);
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

        vk_push_constants push = {
            .count = (uint32_t)chunk_elems,
            .pad = 0u,
            .scalar0 = scalar0,
            .scalar1 = scalar1
        };
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);

        uint32_t groups = (uint32_t)((chunk_elems + (size_t)VK_LOCAL_SIZE_X - 1u) / (size_t)VK_LOCAL_SIZE_X);
        vkCmdDispatch(cmd, groups, 1u, 1u);

        vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        vk_cmd_compute_to_transfer(cmd, dst->vk_buffer, byte_offset, chunk_bytes);

        status = vk_submit_commands(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }
        perf_profile_add_gpu_dispatch(1u,
                                          2u,
                                          2u * (uint64_t)chunk_bytes);

        offset_elems += chunk_elems;
    }

    return VEC_STATUS_OK;
}

static uint32_t vk_dispatch_groups_for_count(size_t count)
{
    if (count == 0u) {
        return 1u;
    }
    return (uint32_t)((count + (size_t)VK_LOCAL_SIZE_X - 1u) / (size_t)VK_LOCAL_SIZE_X);
}

static uint32_t vk_required_descriptor_sets_for_reduction(uint32_t groups)
{
    uint32_t required_sets = 1u;
    uint32_t reduce_count = groups;
    while (reduce_count > 1u) {
        reduce_count = vk_dispatch_groups_for_count((size_t)reduce_count);
        required_sets += 1u;
    }
    return required_sets;
}

static vec_status vk_reserve_descriptor_sets_for_reduction(
    vector_backend* backend,
    uint32_t required_sets
)
{
    if (backend == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    uint32_t available_sets = backend->vk.descriptor_set_count;
    if (backend->in_simulation) {
        const uint32_t used = backend->vk.simulation_descriptor_set_cursor;
        if (used > available_sets) {
            return VEC_STATUS_BACKEND_UNAVAILABLE;
        }
        available_sets -= used;
    }

    if (required_sets <= available_sets) {
        return VEC_STATUS_OK;
    }

    if (backend->in_simulation) {
        vec_status status = vk_simulation_phase_flush(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }
        available_sets = backend->vk.descriptor_set_count;
    }

    return (required_sets <= available_sets)
               ? VEC_STATUS_OK
               : VEC_STATUS_BACKEND_UNAVAILABLE;
}

vec_status vk_dispatch_complex_relative_error(
    vector_backend* backend,
    const vec_buffer* current,
    const vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    if (backend == NULL || current == NULL || previous == NULL || out_error == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (current->length != previous->length) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (current->length > (size_t)UINT32_MAX) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    uint32_t groups = vk_dispatch_groups_for_count(current->length);
    const uint32_t required_sets = vk_required_descriptor_sets_for_reduction(groups);

    vec_status status = vk_reserve_descriptor_sets_for_reduction(backend, required_sets);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    status = vk_ensure_reduction_capacity(backend, (VkDeviceSize)groups);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    status = vk_ensure_staging_capacity(backend, (VkDeviceSize)sizeof(double));
    if (status != VEC_STATUS_OK) {
        return status;
    }

    status = vk_begin_commands(backend);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    status = vk_acquire_descriptor_set(backend, &descriptor_set);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    VkCommandBuffer cmd = backend->vk.command_buffer;
    VkDeviceSize complex_bytes = (VkDeviceSize)current->bytes;
    VkDeviceSize partial_bytes = (VkDeviceSize)groups * (VkDeviceSize)sizeof(double);
    uint64_t dispatch_count = 0u;
    uint64_t dispatch_pass_count = 0u;
    uint64_t dispatch_pass_bytes = 0u;

    vk_cmd_compute_to_compute(cmd, current->vk_buffer, 0u, complex_bytes);
    vk_cmd_compute_to_compute(cmd, previous->vk_buffer, 0u, complex_bytes);
    vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);
    vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_b, 0u, partial_bytes);

    vk_update_descriptor_set(backend,
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
                      backend->vk.kernels[VK_KERNEL_COMPLEX_RELATIVE_ERROR_REDUCE].pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            backend->vk.pipeline_layout,
                            0u,
                            1u,
                            &descriptor_set,
                            0u,
                            NULL);

    vk_push_constants push = {
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
    vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);

    VkBuffer src_buffer = backend->vk.reduction_buffer_a;
    VkBuffer dst_buffer = backend->vk.reduction_buffer_b;
    uint32_t count = groups;
    while (count > 1u) {
        status = vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        uint32_t next_groups = vk_dispatch_groups_for_count((size_t)count);
        VkDeviceSize src_bytes = (VkDeviceSize)count * (VkDeviceSize)sizeof(double);
        VkDeviceSize dst_bytes = (VkDeviceSize)next_groups * (VkDeviceSize)sizeof(double);

        vk_update_descriptor_set(backend,
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
                          backend->vk.kernels[VK_KERNEL_REAL_MAX_REDUCE].pipeline);
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
        vk_cmd_compute_to_compute(cmd, dst_buffer, 0u, dst_bytes);

        VkBuffer tmp = src_buffer;
        src_buffer = dst_buffer;
        dst_buffer = tmp;
        count = next_groups;
    }

    vk_cmd_compute_to_transfer(cmd, src_buffer, 0u, (VkDeviceSize)sizeof(double));

    VkBufferCopy copy = {
        .srcOffset = 0u,
        .dstOffset = 0u,
        .size = (VkDeviceSize)sizeof(double)
    };
    vkCmdCopyBuffer(cmd, src_buffer, backend->vk.staging_buffer, 1u, &copy);
    vk_cmd_transfer_to_host(cmd, backend->vk.staging_buffer, 0u, (VkDeviceSize)sizeof(double));

    status = vk_submit_commands(backend);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    perf_profile_add_gpu_dispatch(dispatch_count,
                                      dispatch_pass_count,
                                      dispatch_pass_bytes);
    perf_profile_add_gpu_host_transfer_copy(1u, (uint64_t)sizeof(double));
    perf_profile_add_gpu_download(1u, (uint64_t)sizeof(double));
    if (backend->in_simulation) {
        status = vk_simulation_phase_flush(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }
    }

    double ratio = 0.0;
    memcpy(&ratio, backend->vk.staging_mapped_ptr, sizeof(double));
    if (ratio < 0.0) {
        ratio = 0.0;
    }
    *out_error = sqrt(ratio);
    return VEC_STATUS_OK;
}

vec_status vk_dispatch_complex_weighted_rms_error(
    vector_backend* backend,
    const vec_buffer* fine,
    const vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
)
{
    if (backend == NULL || fine == NULL || coarse == NULL || out_error == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (fine->length != coarse->length) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (fine->length > (size_t)UINT32_MAX) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    uint32_t groups = vk_dispatch_groups_for_count(fine->length);
    const uint32_t required_sets = vk_required_descriptor_sets_for_reduction(groups);

    vec_status status = vk_reserve_descriptor_sets_for_reduction(backend, required_sets);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    status = vk_ensure_reduction_capacity(backend, (VkDeviceSize)(groups * 2u));
    if (status != VEC_STATUS_OK) {
        return status;
    }
    status = vk_ensure_staging_capacity(backend, (VkDeviceSize)sizeof(vk_error_pair));
    if (status != VEC_STATUS_OK) {
        return status;
    }

    status = vk_begin_commands(backend);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    status = vk_acquire_descriptor_set(backend, &descriptor_set);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    VkCommandBuffer cmd = backend->vk.command_buffer;
    VkDeviceSize complex_bytes = (VkDeviceSize)fine->bytes;
    VkDeviceSize partial_bytes = (VkDeviceSize)groups * (VkDeviceSize)sizeof(vk_error_pair);
    uint64_t dispatch_count = 0u;
    uint64_t dispatch_pass_count = 0u;
    uint64_t dispatch_pass_bytes = 0u;

    vk_cmd_compute_to_compute(cmd, fine->vk_buffer, 0u, complex_bytes);
    vk_cmd_compute_to_compute(cmd, coarse->vk_buffer, 0u, complex_bytes);
    vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);
    vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_b, 0u, partial_bytes);

    vk_update_descriptor_set(backend,
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
                      backend->vk.kernels[VK_KERNEL_COMPLEX_WEIGHTED_RMS_REDUCE].pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            backend->vk.pipeline_layout,
                            0u,
                            1u,
                            &descriptor_set,
                            0u,
                            NULL);

    vk_push_constants push = {
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
    vkCmdDispatch(cmd, groups, 1u, 1u);
    dispatch_count += 1u;
    dispatch_pass_count += 2u;
    dispatch_pass_bytes += (uint64_t)partial_bytes * 2u;
    vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);

    VkBuffer src_buffer = backend->vk.reduction_buffer_a;
    VkBuffer dst_buffer = backend->vk.reduction_buffer_b;
    uint32_t count = groups;
    while (count > 1u) {
        status = vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        uint32_t next_groups = vk_dispatch_groups_for_count((size_t)count);
        VkDeviceSize src_bytes = (VkDeviceSize)count * (VkDeviceSize)sizeof(vk_error_pair);
        VkDeviceSize dst_bytes = (VkDeviceSize)next_groups * (VkDeviceSize)sizeof(vk_error_pair);

        vk_update_descriptor_set(backend,
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
                          backend->vk.kernels[VK_KERNEL_PAIR_SUM_REDUCE].pipeline);
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
        vk_cmd_compute_to_compute(cmd, dst_buffer, 0u, dst_bytes);

        VkBuffer tmp = src_buffer;
        src_buffer = dst_buffer;
        dst_buffer = tmp;
        count = next_groups;
    }

    vk_cmd_compute_to_transfer(cmd, src_buffer, 0u, (VkDeviceSize)sizeof(vk_error_pair));

    VkBufferCopy copy = {
        .srcOffset = 0u,
        .dstOffset = 0u,
        .size = (VkDeviceSize)sizeof(vk_error_pair)
    };
    vkCmdCopyBuffer(cmd, src_buffer, backend->vk.staging_buffer, 1u, &copy);
    vk_cmd_transfer_to_host(cmd, backend->vk.staging_buffer, 0u, (VkDeviceSize)sizeof(vk_error_pair));

    status = vk_submit_commands(backend);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    perf_profile_add_gpu_dispatch(dispatch_count,
                                      dispatch_pass_count,
                                      dispatch_pass_bytes);
    perf_profile_add_gpu_host_transfer_copy(1u, (uint64_t)sizeof(vk_error_pair));
    perf_profile_add_gpu_download(1u, (uint64_t)sizeof(vk_error_pair));
    if (backend->in_simulation) {
        status = vk_simulation_phase_flush(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }
    }

    vk_error_pair pair = {0.0, 0.0};
    memcpy(&pair, backend->vk.staging_mapped_ptr, sizeof(pair));
    if (pair.denominator <= 0.0) {
        *out_error = 0.0;
        return VEC_STATUS_OK;
    }

    double ratio = pair.numerator / pair.denominator;
    if (ratio < 0.0) {
        ratio = 0.0;
    }
    *out_error = sqrt(ratio);
    return VEC_STATUS_OK;
}
