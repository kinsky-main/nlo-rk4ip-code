/**
 * @file vk_backend_transfer.c
 * @brief Vulkan buffer creation and host/device transfer helpers.
 */

#include "numerics/vk_backend_internal.h"
#include "utility/perf_profile.h"
#include <string.h>

/* Host buffers hold doubles; device buffers hold vk_real. These convert one
 * staged chunk, addressed by its device byte offset and length. */
static void vk_stage_from_host(void* staging, const void* host, VkDeviceSize device_offset, VkDeviceSize device_chunk)
{
#if defined(NLO_VK_FP32) && NLO_VK_FP32
    const double* src = (const double*)host + (size_t)(device_offset / VK_REAL_BYTES);
    vk_real* dst = (vk_real*)staging;
    const size_t count = (size_t)(device_chunk / VK_REAL_BYTES);
    for (size_t i = 0u; i < count; ++i) {
        dst[i] = (vk_real)src[i];
    }
#else
    memcpy(staging, (const uint8_t*)host + (size_t)device_offset, (size_t)device_chunk);
#endif
}

static void vk_stage_to_host(void* host, const void* staging, VkDeviceSize device_offset, VkDeviceSize device_chunk)
{
#if defined(NLO_VK_FP32) && NLO_VK_FP32
    double* dst = (double*)host + (size_t)(device_offset / VK_REAL_BYTES);
    const vk_real* src = (const vk_real*)staging;
    const size_t count = (size_t)(device_chunk / VK_REAL_BYTES);
    for (size_t i = 0u; i < count; ++i) {
        dst[i] = (double)src[i];
    }
#else
    memcpy((uint8_t*)host + (size_t)device_offset, staging, (size_t)device_chunk);
#endif
}

vec_status vk_copy_buffer_chunked(
    vector_backend* backend,
    VkBuffer src_buffer,
    VkBuffer dst_buffer,
    size_t bytes
)
{
    VkDeviceSize remaining = (VkDeviceSize)bytes;
    VkDeviceSize offset = 0u;

    while (remaining > 0u) {
        VkDeviceSize chunk = vk_min_size(remaining, backend->vk.max_kernel_chunk_bytes);
        vec_status status = vk_begin_commands(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        VkCommandBuffer cmd = backend->vk.command_buffer;
        vk_cmd_compute_to_transfer(cmd, src_buffer, offset, chunk);
        vk_cmd_compute_to_transfer(cmd, dst_buffer, offset, chunk);

        VkBufferCopy copy = {
            .srcOffset = offset,
            .dstOffset = offset,
            .size = chunk
        };
        vkCmdCopyBuffer(cmd, src_buffer, dst_buffer, 1u, &copy);
        vk_cmd_transfer_to_compute(cmd, dst_buffer, offset, chunk);

        status = vk_submit_commands(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }
        perf_profile_add_gpu_device_copy(1u, (uint64_t)chunk);

        offset += chunk;
        remaining -= chunk;
    }

    return VEC_STATUS_OK;
}

vec_status vk_buffer_create(vector_backend* backend, vec_buffer* buffer)
{
    return vk_create_buffer_raw(backend,
                                    (VkDeviceSize)vk_device_bytes(buffer->bytes),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    &buffer->vk_buffer,
                                    &buffer->vk_memory);
}

void vk_buffer_destroy(vector_backend* backend, vec_buffer* buffer)
{
    vk_destroy_buffer_raw(backend, &buffer->vk_buffer, &buffer->vk_memory);
}

vec_status vk_upload(
    vector_backend* backend,
    vec_buffer* buffer,
    const void* data,
    size_t bytes
)
{
    vec_status status = vk_ensure_staging_capacity(backend, (VkDeviceSize)VK_DEFAULT_STAGING_BYTES);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    VkDeviceSize remaining = (VkDeviceSize)vk_device_bytes(bytes);
    VkDeviceSize offset = 0u;

    while (remaining > 0u) {
        VkDeviceSize chunk = vk_min_size(remaining, backend->vk.staging_capacity);
        vk_stage_from_host(backend->vk.staging_mapped_ptr, data, offset, chunk);

        status = vk_begin_commands(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        VkBufferCopy copy = {
            .srcOffset = 0u,
            .dstOffset = offset,
            .size = chunk
        };
        vkCmdCopyBuffer(backend->vk.command_buffer, backend->vk.staging_buffer, buffer->vk_buffer, 1u, &copy);
        vk_cmd_transfer_to_compute(backend->vk.command_buffer, buffer->vk_buffer, offset, chunk);

        status = vk_submit_commands(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }
        perf_profile_add_gpu_host_transfer_copy(1u, (uint64_t)chunk);
        perf_profile_add_gpu_upload(1u, (uint64_t)chunk);

        remaining -= chunk;
        offset += chunk;
    }

    return VEC_STATUS_OK;
}

vec_status vk_download(
    vector_backend* backend,
    const vec_buffer* buffer,
    void* data,
    size_t bytes
)
{
    vec_status status = vk_ensure_staging_capacity(backend, (VkDeviceSize)VK_DEFAULT_STAGING_BYTES);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    VkDeviceSize remaining = (VkDeviceSize)vk_device_bytes(bytes);
    VkDeviceSize offset = 0u;

    while (remaining > 0u) {
        VkDeviceSize chunk = vk_min_size(remaining, backend->vk.staging_capacity);
        status = vk_begin_commands(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        vk_cmd_compute_to_transfer(backend->vk.command_buffer, buffer->vk_buffer, offset, chunk);
        VkBufferCopy copy = {
            .srcOffset = offset,
            .dstOffset = 0u,
            .size = chunk
        };
        vkCmdCopyBuffer(backend->vk.command_buffer, buffer->vk_buffer, backend->vk.staging_buffer, 1u, &copy);
        vk_cmd_transfer_to_host(backend->vk.command_buffer, backend->vk.staging_buffer, 0u, chunk);

        status = vk_submit_commands(backend);
        if (status != VEC_STATUS_OK) {
            return status;
        }
        perf_profile_add_gpu_host_transfer_copy(1u, (uint64_t)chunk);
        perf_profile_add_gpu_download(1u, (uint64_t)chunk);

        vk_stage_to_host(data, backend->vk.staging_mapped_ptr, offset, chunk);
        remaining -= chunk;
        offset += chunk;
    }

    return VEC_STATUS_OK;
}
