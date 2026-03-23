/**
 * @file vk_backend_phase.c
 * @brief Vulkan command recording and simulation-phase control.
 */

#include "numerics/vk_backend_internal.h"
#include "utility/perf_profile.h"
#include <string.h>

int nlo_vk_timestamps_supported(const nlo_vector_backend* backend)
{
    return (backend != NULL &&
            backend->type == NLO_VECTOR_BACKEND_VULKAN &&
            backend->vk.timestamp_queries_supported &&
            backend->vk.timestamp_query_pool != VK_NULL_HANDLE &&
            backend->vk.timestamp_spans != NULL &&
            backend->vk.timestamp_query_capacity >= 2u &&
            backend->vk.timestamp_period_ns > 0.0)
               ? 1
               : 0;
}

nlo_vec_status nlo_vk_wait_for_pending_submit(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (!backend->vk.submit_pending) {
        return NLO_VEC_STATUS_OK;
    }
    if (backend->vk.submit_fence == VK_NULL_HANDLE) {
        backend->vk.submit_pending = false;
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    nlo_vk_timestamp_collect(backend);
    backend->vk.submit_pending = false;
    return NLO_VEC_STATUS_OK;
}

void nlo_vk_timestamp_begin_command_buffer(nlo_vector_backend* backend, VkCommandBuffer cmd)
{
    if (!nlo_vk_timestamps_supported(backend) || cmd == VK_NULL_HANDLE) {
        if (backend != NULL) {
            backend->vk.timestamp_query_cursor = 0u;
            backend->vk.timestamp_span_count = 0u;
        }
        return;
    }

    backend->vk.timestamp_query_cursor = 0u;
    backend->vk.timestamp_span_count = 0u;
    vkCmdResetQueryPool(cmd,
                        backend->vk.timestamp_query_pool,
                        0u,
                        backend->vk.timestamp_query_capacity);
}

void nlo_vk_timestamp_write_begin(
    nlo_vector_backend* backend,
    VkCommandBuffer cmd,
    uint32_t event_id,
    nlo_vk_timestamp_ticket* out_ticket
)
{
    if (out_ticket == NULL) {
        return;
    }
    memset(out_ticket, 0, sizeof(*out_ticket));

    if (!nlo_vk_timestamps_supported(backend) || cmd == VK_NULL_HANDLE) {
        return;
    }

    if (backend->vk.timestamp_query_cursor + 1u >= backend->vk.timestamp_query_capacity ||
        backend->vk.timestamp_span_count >= backend->vk.timestamp_span_capacity) {
        return;
    }

    const uint32_t span_index = backend->vk.timestamp_span_count;
    const uint32_t start_query = backend->vk.timestamp_query_cursor;
    const uint32_t end_query = start_query + 1u;

    backend->vk.timestamp_spans[span_index].event_id = event_id;
    backend->vk.timestamp_spans[span_index].start_query = start_query;
    backend->vk.timestamp_spans[span_index].end_query = end_query;
    backend->vk.timestamp_query_cursor = end_query + 1u;
    backend->vk.timestamp_span_count = span_index + 1u;

    out_ticket->event_id = event_id;
    out_ticket->span_index = span_index;
    out_ticket->start_query = start_query;
    out_ticket->end_query = end_query;
    out_ticket->active = 1;

    vkCmdWriteTimestamp(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        backend->vk.timestamp_query_pool,
                        start_query);
}

void nlo_vk_timestamp_write_end(
    nlo_vector_backend* backend,
    VkCommandBuffer cmd,
    const nlo_vk_timestamp_ticket* ticket
)
{
    if (!nlo_vk_timestamps_supported(backend) || cmd == VK_NULL_HANDLE ||
        ticket == NULL || !ticket->active) {
        return;
    }

    vkCmdWriteTimestamp(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        backend->vk.timestamp_query_pool,
                        ticket->end_query);
}

void nlo_vk_timestamp_collect(nlo_vector_backend* backend)
{
    if (!nlo_vk_timestamps_supported(backend) ||
        backend->vk.timestamp_span_count == 0u ||
        backend->vk.timestamp_query_cursor == 0u) {
        return;
    }

    uint64_t query_values[4096] = {0};
    const uint32_t query_count = backend->vk.timestamp_query_cursor;
    if (query_count > (uint32_t)(sizeof(query_values) / sizeof(query_values[0]))) {
        backend->vk.timestamp_query_cursor = 0u;
        backend->vk.timestamp_span_count = 0u;
        return;
    }

    if (vkGetQueryPoolResults(backend->vk.device,
                              backend->vk.timestamp_query_pool,
                              0u,
                              query_count,
                              (size_t)query_count * sizeof(uint64_t),
                              query_values,
                              sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
        backend->vk.timestamp_query_cursor = 0u;
        backend->vk.timestamp_span_count = 0u;
        return;
    }

    for (uint32_t i = 0u; i < backend->vk.timestamp_span_count; ++i) {
        const nlo_vk_timestamp_span* span = &backend->vk.timestamp_spans[i];
        if (span->end_query >= query_count || span->start_query >= query_count) {
            continue;
        }
        if (query_values[span->end_query] < query_values[span->start_query]) {
            continue;
        }

        const uint64_t ticks = query_values[span->end_query] - query_values[span->start_query];
        const double elapsed_ms =
            ((double)ticks * backend->vk.timestamp_period_ns) / 1000000.0;
        NLO_PERF_ADD_GPU_TIME((nlo_perf_event_id)span->event_id, elapsed_ms);
    }

    NLO_PERF_MARK_GPU_TIMESTAMPS_AVAILABLE(1);
    backend->vk.timestamp_query_cursor = 0u;
    backend->vk.timestamp_span_count = 0u;
}

static nlo_vec_status nlo_vk_begin_commands_raw(nlo_vector_backend* backend)
{
    nlo_perf_scope perf_scope = {0.0, 0};
    NLO_PERF_SCOPE_BEGIN(perf_scope);
    nlo_vec_status status = nlo_vk_wait_for_pending_submit(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
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
    nlo_vk_timestamp_begin_command_buffer(backend, backend->vk.command_buffer);

    NLO_PERF_SCOPE_END(perf_scope, NLO_PERF_EVENT_VK_COMMAND_BEGIN, 0u);
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
    nlo_perf_scope perf_scope = {0.0, 0};
    NLO_PERF_SCOPE_BEGIN(perf_scope);
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
        backend->vk.submit_pending = true;
    }

    nlo_vk_reset_simulation_phase_state(backend);
    NLO_PERF_SCOPE_END(perf_scope, NLO_PERF_EVENT_VK_SIMULATION_FLUSH, 0u);
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_submit_commands(nlo_vector_backend* backend)
{
    nlo_perf_scope perf_scope = {0.0, 0};
    NLO_PERF_SCOPE_BEGIN(perf_scope);
    if (backend->in_simulation) {
        nlo_vk_simulation_phase_mark_commands(backend);
        return NLO_VEC_STATUS_OK;
    }

    nlo_vec_status status = nlo_vk_wait_for_pending_submit(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
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
    nlo_vk_timestamp_collect(backend);
    backend->vk.submit_pending = false;

    NLO_PERF_SCOPE_END(perf_scope, NLO_PERF_EVENT_VK_COMMAND_SUBMIT_WAIT, 0u);
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

