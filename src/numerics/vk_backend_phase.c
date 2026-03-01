/**
 * @file vk_backend_phase.c
 * @brief Vulkan command recording and simulation-phase control.
 */

#include "numerics/vk_backend_internal.h"

static nlo_vec_status nlo_vk_begin_commands_raw(nlo_vector_backend* backend)
{
    if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (vkResetCommandBuffer(backend->vk.command_buffer, 0u) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    if (vkBeginCommandBuffer(backend->vk.command_buffer, &begin_info) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_begin_commands(nlo_vector_backend* backend)
{
    if (backend->in_simulation) {
        return nlo_vk_simulation_phase_begin(backend);
    }
    return nlo_vk_begin_commands_raw(backend);
}

static void nlo_vk_reset_simulation_phase_state(nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return;
    }

    backend->vk.simulation_phase_recording = false;
    backend->vk.simulation_phase_has_commands = false;
    backend->vk.simulation_descriptor_set_cursor = 0u;
}

void nlo_vk_simulation_phase_mark_commands(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return;
    }
    backend->vk.simulation_phase_has_commands = true;
}

nlo_vec_status nlo_vk_simulation_phase_begin(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->vk.simulation_phase_recording) {
        return NLO_VEC_STATUS_OK;
    }

    nlo_vec_status status = nlo_vk_begin_commands_raw(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    backend->vk.simulation_phase_recording = true;
    backend->vk.simulation_phase_has_commands = false;
    backend->vk.simulation_descriptor_set_cursor = 0u;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_simulation_phase_command_buffer(
    nlo_vector_backend* backend,
    VkCommandBuffer* out_command_buffer
)
{
    if (backend == NULL || out_command_buffer == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vk_simulation_phase_begin(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    *out_command_buffer = backend->vk.command_buffer;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_simulation_phase_flush(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (!backend->vk.simulation_phase_recording) {
        return NLO_VEC_STATUS_OK;
    }

    if (vkEndCommandBuffer(backend->vk.command_buffer) != VK_SUCCESS) {
        nlo_vk_reset_simulation_phase_state(backend);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (backend->vk.simulation_phase_has_commands) {
        if (vkResetFences(backend->vk.device, 1u, &backend->vk.submit_fence) != VK_SUCCESS) {
            nlo_vk_reset_simulation_phase_state(backend);
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1u,
            .pCommandBuffers = &backend->vk.command_buffer
        };
        if (vkQueueSubmit(backend->vk.queue, 1u, &submit_info, backend->vk.submit_fence) != VK_SUCCESS) {
            nlo_vk_reset_simulation_phase_state(backend);
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
        if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            nlo_vk_reset_simulation_phase_state(backend);
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
    }

    nlo_vk_reset_simulation_phase_state(backend);
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_submit_commands(nlo_vector_backend* backend)
{
    if (backend->in_simulation) {
        nlo_vk_simulation_phase_mark_commands(backend);
        return NLO_VEC_STATUS_OK;
    }

    if (vkEndCommandBuffer(backend->vk.command_buffer) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (vkResetFences(backend->vk.device, 1u, &backend->vk.submit_fence) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1u,
        .pCommandBuffers = &backend->vk.command_buffer
    };
    if (vkQueueSubmit(backend->vk.queue, 1u, &submit_info, backend->vk.submit_fence) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

static void nlo_vk_cmd_buffer_barrier(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkPipelineStageFlags src_stage,
    VkAccessFlags src_access,
    VkPipelineStageFlags dst_stage,
    VkAccessFlags dst_access
)
{
    VkBufferMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = offset,
        .size = size
    };
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0u, 0u, NULL, 1u, &barrier, 0u, NULL);
}

void nlo_vk_cmd_transfer_to_compute(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
    nlo_vk_cmd_buffer_barrier(cmd,
                              buffer,
                              offset,
                              size,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void nlo_vk_cmd_compute_to_compute(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
    nlo_vk_cmd_buffer_barrier(cmd,
                              buffer,
                              offset,
                              size,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void nlo_vk_cmd_compute_to_transfer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
    nlo_vk_cmd_buffer_barrier(cmd,
                              buffer,
                              offset,
                              size,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

void nlo_vk_cmd_transfer_to_host(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
    nlo_vk_cmd_buffer_barrier(cmd,
                              buffer,
                              offset,
                              size,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_HOST_BIT,
                              VK_ACCESS_HOST_READ_BIT);
}

