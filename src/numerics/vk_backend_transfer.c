/**
 * @file vk_backend_transfer.c
 * @brief Vulkan buffer creation and host/device transfer helpers.
 */

#include "numerics/vk_backend_internal.h"
#include "utility/perf_profile.h"
#include <string.h>

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
                                    (VkDeviceSize)buffer->bytes,
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

    const uint8_t* src = (const uint8_t*)data;
    VkDeviceSize remaining = (VkDeviceSize)bytes;
    VkDeviceSize offset = 0u;

    while (remaining > 0u) {
        VkDeviceSize chunk = vk_min_size(remaining, backend->vk.staging_capacity);
        memcpy(backend->vk.staging_mapped_ptr, src + (size_t)offset, (size_t)chunk);

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

    uint8_t* dst = (uint8_t*)data;
    VkDeviceSize remaining = (VkDeviceSize)bytes;
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

        memcpy(dst + (size_t)offset, backend->vk.staging_mapped_ptr, (size_t)chunk);
        remaining -= chunk;
        offset += chunk;
    }

    return VEC_STATUS_OK;
}
