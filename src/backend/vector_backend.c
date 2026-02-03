/**
 * @file vector_backend.c
 * @dir src/backend
 * @brief Backend abstraction for vector operations (CPU or Vulkan).
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */

#include "numerics/vector_backend.h"
#include "numerics/vector_ops.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifdef NLO_VECTOR_BACKEND_VULKAN
#include <vulkan/vulkan.h>
#endif

// MARK: Internal Types

#ifdef NLO_VECTOR_BACKEND_VULKAN
typedef struct {
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkCommandPool command_pool;
    uint32_t queue_family_index;
} nlo_vk_backend;
#endif

struct nlo_vector_backend {
    nlo_vector_backend_type type;
    bool in_simulation;
#ifdef NLO_VECTOR_BACKEND_VULKAN
    nlo_vk_backend vk;
#endif
};

struct nlo_vec_buffer {
    nlo_vector_backend* owner;
    nlo_vec_kind kind;
    size_t length;
    size_t bytes;
    void* host_ptr;
#ifdef NLO_VECTOR_BACKEND_VULKAN
    VkBuffer vk_buffer;
    VkDeviceMemory vk_memory;
#endif
};

// MARK: Internal Helpers

static size_t nlo_vec_element_size(nlo_vec_kind kind)
{
    switch (kind) {
        case NLO_VEC_KIND_REAL64:
            return sizeof(double);
        case NLO_VEC_KIND_COMPLEX64:
            return sizeof(nlo_complex);
        default:
            return 0u;
    }
}

static nlo_vec_status nlo_vec_validate_backend(const nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vec_validate_buffer(const nlo_vector_backend* backend,
                                              const nlo_vec_buffer* buffer,
                                              nlo_vec_kind kind)
{
    if (backend == NULL || buffer == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend || buffer->kind != kind) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vec_validate_pair(const nlo_vector_backend* backend,
                                            const nlo_vec_buffer* a,
                                            const nlo_vec_buffer* b,
                                            nlo_vec_kind kind)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, a, kind);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_vec_validate_buffer(backend, b, kind);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    if (a->length != b->length) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return NLO_VEC_STATUS_OK;
}

#ifdef NLO_VECTOR_BACKEND_VULKAN
static uint32_t nlo_vk_find_memory_type(nlo_vector_backend* backend,
                                        uint32_t type_filter,
                                        VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(backend->vk.physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) != 0u &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

static nlo_vec_status nlo_vk_create_buffer(nlo_vector_backend* backend,
                                           VkDeviceSize size,
                                           VkBufferUsageFlags usage,
                                           VkMemoryPropertyFlags properties,
                                           VkBuffer* buffer,
                                           VkDeviceMemory* memory)
{
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    if (vkCreateBuffer(backend->vk.device, &buffer_info, NULL, buffer) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(backend->vk.device, *buffer, &mem_req);
    uint32_t type_index = nlo_vk_find_memory_type(backend, mem_req.memoryTypeBits, properties);
    if (type_index == UINT32_MAX) {
        vkDestroyBuffer(backend->vk.device, *buffer, NULL);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = type_index
    };
    if (vkAllocateMemory(backend->vk.device, &alloc_info, NULL, memory) != VK_SUCCESS) {
        vkDestroyBuffer(backend->vk.device, *buffer, NULL);
        return NLO_VEC_STATUS_ALLOCATION_FAILED;
    }

    if (vkBindBufferMemory(backend->vk.device, *buffer, *memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(backend->vk.device, *buffer, NULL);
        vkFreeMemory(backend->vk.device, *memory, NULL);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_submit_copy(nlo_vector_backend* backend,
                                         VkBuffer src,
                                         VkBuffer dst,
                                         VkDeviceSize size)
{
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = backend->vk.command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(backend->vk.device, &alloc_info, &cmd) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        vkFreeCommandBuffers(backend->vk.device, backend->vk.command_pool, 1, &cmd);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = size
    };
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy_region);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(backend->vk.device, backend->vk.command_pool, 1, &cmd);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd
    };

    if (vkQueueSubmit(backend->vk.queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(backend->vk.device, backend->vk.command_pool, 1, &cmd);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (vkQueueWaitIdle(backend->vk.queue) != VK_SUCCESS) {
        vkFreeCommandBuffers(backend->vk.device, backend->vk.command_pool, 1, &cmd);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    vkFreeCommandBuffers(backend->vk.device, backend->vk.command_pool, 1, &cmd);
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_upload(nlo_vector_backend* backend,
                                    nlo_vec_buffer* buffer,
                                    const void* data,
                                    size_t bytes)
{
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;

    nlo_vec_status status = nlo_vk_create_buffer(
        backend,
        (VkDeviceSize)bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging_buffer,
        &staging_memory);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    void* mapped = NULL;
    if (vkMapMemory(backend->vk.device, staging_memory, 0, (VkDeviceSize)bytes, 0, &mapped) != VK_SUCCESS) {
        vkDestroyBuffer(backend->vk.device, staging_buffer, NULL);
        vkFreeMemory(backend->vk.device, staging_memory, NULL);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    memcpy(mapped, data, bytes);
    vkUnmapMemory(backend->vk.device, staging_memory);

    status = nlo_vk_submit_copy(backend, staging_buffer, buffer->vk_buffer, (VkDeviceSize)bytes);

    vkDestroyBuffer(backend->vk.device, staging_buffer, NULL);
    vkFreeMemory(backend->vk.device, staging_memory, NULL);

    return status;
}

static nlo_vec_status nlo_vk_download(nlo_vector_backend* backend,
                                      const nlo_vec_buffer* buffer,
                                      void* data,
                                      size_t bytes)
{
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;

    nlo_vec_status status = nlo_vk_create_buffer(
        backend,
        (VkDeviceSize)bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging_buffer,
        &staging_memory);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vk_submit_copy(backend, buffer->vk_buffer, staging_buffer, (VkDeviceSize)bytes);
    if (status != NLO_VEC_STATUS_OK) {
        vkDestroyBuffer(backend->vk.device, staging_buffer, NULL);
        vkFreeMemory(backend->vk.device, staging_memory, NULL);
        return status;
    }

    void* mapped = NULL;
    if (vkMapMemory(backend->vk.device, staging_memory, 0, (VkDeviceSize)bytes, 0, &mapped) != VK_SUCCESS) {
        vkDestroyBuffer(backend->vk.device, staging_buffer, NULL);
        vkFreeMemory(backend->vk.device, staging_memory, NULL);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    memcpy(data, mapped, bytes);
    vkUnmapMemory(backend->vk.device, staging_memory);

    vkDestroyBuffer(backend->vk.device, staging_buffer, NULL);
    vkFreeMemory(backend->vk.device, staging_memory, NULL);
    return NLO_VEC_STATUS_OK;
}
#endif

// MARK: Backend Lifecycle

nlo_vector_backend* nlo_vector_backend_create_cpu(void)
{
    nlo_vector_backend* backend = (nlo_vector_backend*)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        return NULL;
    }
    backend->type = NLO_VECTOR_BACKEND_CPU;
    backend->in_simulation = false;
    return backend;
}

void nlo_vector_backend_destroy(nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return;
    }
    free(backend);
}

nlo_vector_backend_type nlo_vector_backend_get_type(const nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return NLO_VECTOR_BACKEND_CPU;
    }
    return backend->type;
}

#ifdef NLO_VECTOR_BACKEND_VULKAN
nlo_vector_backend* nlo_vector_backend_create_vulkan(const nlo_vk_backend_config* config)
{
    if (config == NULL || config->device == VK_NULL_HANDLE ||
        config->physical_device == VK_NULL_HANDLE ||
        config->queue == VK_NULL_HANDLE ||
        config->command_pool == VK_NULL_HANDLE) {
        return NULL;
    }

    nlo_vector_backend* backend = (nlo_vector_backend*)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        return NULL;
    }

    backend->type = NLO_VECTOR_BACKEND_VULKAN;
    backend->in_simulation = false;
    backend->vk.physical_device = config->physical_device;
    backend->vk.device = config->device;
    backend->vk.queue = config->queue;
    backend->vk.command_pool = config->command_pool;
    backend->vk.queue_family_index = config->queue_family_index;
    return backend;
}
#endif

// MARK: Simulation Guard

nlo_vec_status nlo_vec_begin_simulation(nlo_vector_backend* backend)
{
    nlo_vec_status status = nlo_vec_validate_backend(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    backend->in_simulation = true;
#ifdef NLO_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        if (vkQueueWaitIdle(backend->vk.queue) != VK_SUCCESS) {
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
    }
#endif
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vec_end_simulation(nlo_vector_backend* backend)
{
    nlo_vec_status status = nlo_vec_validate_backend(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    backend->in_simulation = false;
    return NLO_VEC_STATUS_OK;
}

// MARK: Buffer Lifecycle

nlo_vec_status nlo_vec_create(nlo_vector_backend* backend,
                              nlo_vec_kind kind,
                              size_t length,
                              nlo_vec_buffer** out_buffer)
{
    if (backend == NULL || out_buffer == NULL || length == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t elem_size = nlo_vec_element_size(kind);
    if (elem_size == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_buffer* buffer = (nlo_vec_buffer*)calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        return NLO_VEC_STATUS_ALLOCATION_FAILED;
    }

    buffer->owner = backend;
    buffer->kind = kind;
    buffer->length = length;
    buffer->bytes = length * elem_size;

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        buffer->host_ptr = malloc(buffer->bytes);
        if (buffer->host_ptr == NULL) {
            free(buffer);
            return NLO_VEC_STATUS_ALLOCATION_FAILED;
        }
    }
#ifdef NLO_VECTOR_BACKEND_VULKAN
    else if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vec_status status = nlo_vk_create_buffer(
            backend,
            (VkDeviceSize)buffer->bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &buffer->vk_buffer,
            &buffer->vk_memory);
        if (status != NLO_VEC_STATUS_OK) {
            free(buffer);
            return status;
        }
    }
#endif
    else {
        free(buffer);
        return NLO_VEC_STATUS_UNSUPPORTED;
    }

    *out_buffer = buffer;
    return NLO_VEC_STATUS_OK;
}

void nlo_vec_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    if (backend == NULL || buffer == NULL || buffer->owner != backend) {
        return;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        free(buffer->host_ptr);
    }
#ifdef NLO_VECTOR_BACKEND_VULKAN
    else if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        if (buffer->vk_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(backend->vk.device, buffer->vk_buffer, NULL);
        }
        if (buffer->vk_memory != VK_NULL_HANDLE) {
            vkFreeMemory(backend->vk.device, buffer->vk_memory, NULL);
        }
    }
#endif

    free(buffer);
}

// MARK: Host Transfers

nlo_vec_status nlo_vec_upload(nlo_vector_backend* backend,
                              nlo_vec_buffer* buffer,
                              const void* data,
                              size_t bytes)
{
    if (backend == NULL || buffer == NULL || data == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend || bytes != buffer->bytes) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->in_simulation) {
        return NLO_VEC_STATUS_TRANSFER_FORBIDDEN;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        memcpy(buffer->host_ptr, data, bytes);
        return NLO_VEC_STATUS_OK;
    }
#ifdef NLO_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_upload(backend, buffer, data, bytes);
    }
#endif

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_download(nlo_vector_backend* backend,
                                const nlo_vec_buffer* buffer,
                                void* data,
                                size_t bytes)
{
    if (backend == NULL || buffer == NULL || data == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend || bytes != buffer->bytes) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->in_simulation) {
        return NLO_VEC_STATUS_TRANSFER_FORBIDDEN;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        memcpy(data, buffer->host_ptr, bytes);
        return NLO_VEC_STATUS_OK;
    }
#ifdef NLO_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_download(backend, buffer, data, bytes);
    }
#endif

    return NLO_VEC_STATUS_UNSUPPORTED;
}

// MARK: Vector Operations (CPU implemented, Vulkan stubbed)

nlo_vec_status nlo_vec_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_REAL64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_real_fill((double*)dst->host_ptr, dst->length, value);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, dst, src, NLO_VEC_KIND_REAL64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_real_copy((double*)dst->host_ptr, (const double*)src->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, dst, src, NLO_VEC_KIND_REAL64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_real_mul_inplace((double*)dst->host_ptr, (const double*)src->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_real_pow_int(nlo_vector_backend* backend,
                                    const nlo_vec_buffer* base,
                                    nlo_vec_buffer* out,
                                    unsigned int power)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, base, out, NLO_VEC_KIND_REAL64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_real_pow_int((const double*)base->host_ptr, (double*)out->host_ptr, base->length, power);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_fill((nlo_complex*)dst->host_ptr, dst->length, value);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, dst, src, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_copy((nlo_complex*)dst->host_ptr, (const nlo_complex*)src->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_magnitude_squared(nlo_vector_backend* backend,
                                                 const nlo_vec_buffer* src,
                                                 nlo_vec_buffer* dst)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, src, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        calculate_magnitude_squared((const nlo_complex*)src->host_ptr, (nlo_complex*)dst->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_axpy_real(nlo_vector_backend* backend,
                                         nlo_vec_buffer* dst,
                                         const nlo_vec_buffer* src,
                                         nlo_complex alpha)
{
    if (backend == NULL || dst == NULL || src == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (dst->owner != backend || src->owner != backend ||
        dst->kind != NLO_VEC_KIND_COMPLEX64 || src->kind != NLO_VEC_KIND_REAL64) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (dst->length != src->length) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_axpy_real((nlo_complex*)dst->host_ptr, (const double*)src->host_ptr, alpha, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_scalar_mul_inplace(nlo_vector_backend* backend,
                                                  nlo_vec_buffer* dst,
                                                  nlo_complex alpha)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_scalar_mul_inplace((nlo_complex*)dst->host_ptr, alpha, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_mul_inplace(nlo_vector_backend* backend,
                                           nlo_vec_buffer* dst,
                                           const nlo_vec_buffer* src)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, dst, src, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_mul_inplace((nlo_complex*)dst->host_ptr, (const nlo_complex*)src->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_pow(nlo_vector_backend* backend,
                                   const nlo_vec_buffer* base,
                                   nlo_vec_buffer* out,
                                   unsigned int exponent)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, base, out, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_pow((const nlo_complex*)base->host_ptr, (nlo_complex*)out->host_ptr, base->length, exponent);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_pow_inplace(nlo_vector_backend* backend,
                                           nlo_vec_buffer* dst,
                                           unsigned int exponent)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_pow_inplace((nlo_complex*)dst->host_ptr, dst->length, exponent);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_add_inplace(nlo_vector_backend* backend,
                                           nlo_vec_buffer* dst,
                                           const nlo_vec_buffer* src)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, dst, src, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_add_inplace((nlo_complex*)dst->host_ptr, (const nlo_complex*)src->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_exp_inplace((nlo_complex*)dst->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}
