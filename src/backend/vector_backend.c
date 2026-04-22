/**
 * @file vector_backend.c
 * @dir src/backend
 * @brief Backend abstraction for vector operations (CPU or Vulkan).
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */

#include "backend/vector_backend_internal.h"
#include "backend/vk_auto_context.h"
#include "numerics/vector_ops.h"
#include "numerics/vk_vector_ops.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TWO_PI
#define TWO_PI 6.283185307179586476925286766559
#endif

size_t vec_element_size(vec_kind kind)
{
    switch (kind) {
        case VEC_KIND_REAL64:
            return sizeof(double);
        case VEC_KIND_COMPLEX64:
            return sizeof(nlo_complex);
        default:
            return 0u;
    }
}

static bool vec_multiply_size(size_t a, size_t b, size_t* out)
{
    if (out == NULL) {
        return false;
    }
    if (a == 0u || b <= (SIZE_MAX / a)) {
        *out = a * b;
        return true;
    }
    return false;
}

static vec_status vec_validate_mixed_pair(
    const vector_backend* backend,
    const vec_buffer* lhs,
    vec_kind lhs_kind,
    const vec_buffer* rhs,
    vec_kind rhs_kind
)
{
    if (backend == NULL || lhs == NULL || rhs == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (lhs->owner != backend || rhs->owner != backend ||
        lhs->kind != lhs_kind || rhs->kind != rhs_kind) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (lhs->length != rhs->length) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    return VEC_STATUS_OK;
}

static const char* vk_device_type_to_string(VkPhysicalDeviceType device_type)
{
#if ENABLE_VULKAN_BACKEND
    if (device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        return "discrete";
    }
    if (device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        return "integrated";
    }
    if (device_type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
        return "virtual";
    }
    if (device_type == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        return "cpu";
    }
    return "other";
#else
    (void)device_type;
    return "disabled";
#endif
}

static int vk_memory_info_source_logged = 0;

#if ENABLE_VULKAN_BACKEND
static size_t size_saturating_add(size_t lhs, size_t rhs)
{
    if (rhs > SIZE_MAX - lhs) {
        return SIZE_MAX;
    }
    return lhs + rhs;
}

static size_t size_from_u64_saturating(uint64_t value)
{
    if (value > (uint64_t)SIZE_MAX) {
        return SIZE_MAX;
    }
    return (size_t)value;
}

static size_t vk_sum_device_local_heaps(const VkPhysicalDeviceMemoryProperties* memory_properties)
{
    if (memory_properties == NULL) {
        return 0u;
    }

    size_t total_device_local = 0u;
    for (uint32_t i = 0u; i < memory_properties->memoryHeapCount; ++i) {
        if ((memory_properties->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0u) {
            continue;
        }
        total_device_local = size_saturating_add(
            total_device_local,
            size_from_u64_saturating((uint64_t)memory_properties->memoryHeaps[i].size));
    }

    return total_device_local;
}

static bool vk_device_supports_extension(VkPhysicalDevice physical_device, const char* extension_name)
{
    if (physical_device == VK_NULL_HANDLE || extension_name == NULL || extension_name[0] == '\0') {
        return false;
    }

    uint32_t extension_count = 0u;
    if (vkEnumerateDeviceExtensionProperties(physical_device, NULL, &extension_count, NULL) != VK_SUCCESS ||
        extension_count == 0u) {
        return false;
    }

    VkExtensionProperties* extensions =
        (VkExtensionProperties*)calloc((size_t)extension_count, sizeof(*extensions));
    if (extensions == NULL) {
        return false;
    }

    bool supported = false;
    if (vkEnumerateDeviceExtensionProperties(physical_device,
                                             NULL,
                                             &extension_count,
                                             extensions) == VK_SUCCESS) {
        for (uint32_t i = 0u; i < extension_count; ++i) {
            if (strcmp(extensions[i].extensionName, extension_name) == 0) {
                supported = true;
                break;
            }
        }
    }

    free(extensions);
    return supported;
}

static bool vk_try_query_device_local_available_bytes(
    VkPhysicalDevice physical_device,
    size_t* out_available_bytes
)
{
    if (physical_device == VK_NULL_HANDLE || out_available_bytes == NULL) {
        return false;
    }

#if defined(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) && \
    (defined(VK_VERSION_1_1) || defined(VK_KHR_get_physical_device_properties2))
    if (!vk_device_supports_extension(physical_device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        return false;
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_properties = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
    };
    VkPhysicalDeviceMemoryProperties2 memory_properties_2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = &budget_properties
    };

#if defined(VK_VERSION_1_1)
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &memory_properties_2);
#elif defined(VK_KHR_get_physical_device_properties2)
    vkGetPhysicalDeviceMemoryProperties2KHR(physical_device, &memory_properties_2);
#endif

    size_t available_device_local = 0u;
    bool has_budget_data = false;
    for (uint32_t i = 0u; i < memory_properties_2.memoryProperties.memoryHeapCount; ++i) {
        if ((memory_properties_2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0u) {
            continue;
        }

        const uint64_t heap_budget = budget_properties.heapBudget[i];
        const uint64_t heap_usage = budget_properties.heapUsage[i];
        if (heap_budget == 0u && heap_usage == 0u) {
            continue;
        }

        has_budget_data = true;
        const uint64_t heap_available = (heap_budget > heap_usage) ? (heap_budget - heap_usage) : 0u;
        available_device_local = size_saturating_add(
            available_device_local,
            size_from_u64_saturating(heap_available));
    }

    if (!has_budget_data) {
        return false;
    }

    const size_t total_device_local =
        vk_sum_device_local_heaps(&memory_properties_2.memoryProperties);
    if (available_device_local > total_device_local) {
        available_device_local = total_device_local;
    }

    *out_available_bytes = available_device_local;
    return true;
#else
    (void)physical_device;
    (void)out_available_bytes;
    return false;
#endif
}
#endif

static nlo_vector_backend* nlo_vector_backend_create_auto(const nlo_vk_backend_config* config_template);

static int nlo_vk_config_has_explicit_handles(const nlo_vk_backend_config* config)
{
    if (config == NULL) {
        return 0;
    }

    return (config->physical_device != VK_NULL_HANDLE ||
            config->device != VK_NULL_HANDLE ||
            config->queue != VK_NULL_HANDLE)
               ? 1
               : 0;
}

vec_status vec_validate_backend(const vector_backend* backend)
{
    if (backend == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    return VEC_STATUS_OK;
}

vec_status vec_validate_buffer(
    const vector_backend* backend,
    const vec_buffer* buffer,
    vec_kind kind
)
{
    if (backend == NULL || buffer == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend || buffer->kind != kind) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    return VEC_STATUS_OK;
}

vec_status vec_validate_pair(
    const vector_backend* backend,
    const vec_buffer* a,
    const vec_buffer* b,
    vec_kind kind
)
{
    vec_status status = vec_validate_buffer(backend, a, kind);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    status = vec_validate_buffer(backend, b, kind);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    if (a->length != b->length) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    return VEC_STATUS_OK;
}

// MARK: Backend Lifecycle

vector_backend* vector_backend_create_cpu(void)
{
    vector_backend* backend = (vector_backend*)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        return NULL;
    }
    backend->type = VECTOR_BACKEND_CPU;
    backend->in_simulation = false;
    return backend;
}

void vector_backend_destroy(vector_backend* backend)
{
    if (backend == NULL) {
        return;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        vk_backend_shutdown(backend);
    }

    free(backend);
}

vector_backend_type vector_backend_get_type(const vector_backend* backend)
{
    if (backend == NULL) {
        return VECTOR_BACKEND_CPU;
    }
    return backend->type;
}

bool vec_is_in_simulation(const vector_backend* backend)
{
    if (backend == NULL) {
        return false;
    }
    return backend->in_simulation;
}

vector_backend* vector_backend_create_vulkan(const vk_backend_config* config)
{
    if (config == NULL) {
        return vector_backend_create_auto(NULL);
    }

    if (!nlo_vk_config_has_explicit_handles(config)) {
        return nlo_vector_backend_create_auto(config);
    }

    if (config->physical_device == VK_NULL_HANDLE ||
        config->device == VK_NULL_HANDLE ||
        config->queue == VK_NULL_HANDLE) {
        fprintf(stderr,
                "[nlolib] Vulkan backend config is incomplete: physical_device/device/queue must either all be set or all be omitted for auto selection.\n");
        return NULL;
    }

    vector_backend* backend = (vector_backend*)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        return NULL;
    }

    backend->type = VECTOR_BACKEND_VULKAN;
    backend->in_simulation = false;

    if (vk_backend_init(backend, config) != VEC_STATUS_OK) {
        free(backend);
        return NULL;
    }

    return backend;
}

static vector_backend* vector_backend_create_auto(const vk_backend_config* config_template)
{
    char reason[256];
    vk_auto_context auto_ctx;
    if (vk_auto_context_init(&auto_ctx, reason, sizeof(reason)) != 0) {
        fprintf(stderr,
                "[nlolib] auto backend selection failed: %s\n",
                (reason[0] != '\0') ? reason : "unknown Vulkan setup error");
        return NULL;
    }

    vk_backend_config config = {0};
    if (config_template != NULL) {
        config = *config_template;
    }

    config.physical_device = auto_ctx.physical_device;
    config.device = auto_ctx.device;
    config.queue = auto_ctx.queue;
    config.queue_family_index = auto_ctx.queue_family_index;
    config.command_pool = VK_NULL_HANDLE;

    vector_backend* backend = (vector_backend*)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        vk_auto_context_destroy(&auto_ctx);
        return NULL;
    }

    backend->type = VECTOR_BACKEND_VULKAN;
    backend->in_simulation = false;
    if (vk_backend_init(backend, &config) != VEC_STATUS_OK) {
        free(backend);
        vk_auto_context_destroy(&auto_ctx);
        return NULL;
    }

    backend->vk.instance = auto_ctx.instance;
    backend->vk.owns_instance = true;
    backend->vk.owns_device = true;
    backend->vk.device_type = auto_ctx.device_type;
    backend->vk.device_local_bytes = auto_ctx.device_local_bytes;
#if defined(_MSC_VER)
    strncpy_s(backend->vk.device_name,
              sizeof(backend->vk.device_name),
              auto_ctx.device_name,
              _TRUNCATE);
#else
    snprintf(backend->vk.device_name,
             sizeof(backend->vk.device_name),
             "%s",
             auto_ctx.device_name);
#endif

    fprintf(stderr,
            "[nlolib] auto backend selected Vulkan device='%s' type=%s device_local_bytes=%llu\n",
            (backend->vk.device_name[0] != '\0') ? backend->vk.device_name : "unknown",
            vk_device_type_to_string(backend->vk.device_type),
            (unsigned long long)backend->vk.device_local_bytes);
    return backend;
}

// MARK: Simulation Guard

vec_status vec_begin_simulation(vector_backend* backend)
{
    vec_status status = vec_validate_backend(backend);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    backend->in_simulation = true;
    return VEC_STATUS_OK;
}

vec_status vec_end_simulation(vector_backend* backend)
{
    vec_status status = vec_validate_backend(backend);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        status = vk_simulation_phase_flush(backend);
        if (status != VEC_STATUS_OK) {
            backend->in_simulation = false;
            return status;
        }
    }

    backend->in_simulation = false;
    return VEC_STATUS_OK;
}

vec_status vec_query_memory_info(
    const vector_backend* backend,
    vec_backend_memory_info* out_info
)
{
    if (backend == NULL || out_info == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    *out_info = (vec_backend_memory_info){0};
    if (backend->type == VECTOR_BACKEND_CPU) {
        return VEC_STATUS_OK;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
#if ENABLE_VULKAN_BACKEND
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(backend->vk.physical_device, &memory_properties);

        const size_t total_device_local = vk_sum_device_local_heaps(&memory_properties);
        size_t available_device_local = total_device_local;
        const bool used_budget_extension = vk_try_query_device_local_available_bytes(
            backend->vk.physical_device,
            &available_device_local);

        if (vk_memory_info_source_logged == 0) {
            fprintf(stderr,
                    "[vk] memory_info source=%s device_local_total_bytes=%zu device_local_available_bytes=%zu\n",
                    used_budget_extension ? "VK_EXT_memory_budget" : "heap_total_fallback",
                    total_device_local,
                    available_device_local);
            vk_memory_info_source_logged = 1;
        }

        out_info->device_local_total_bytes = total_device_local;
        out_info->device_local_available_bytes = available_device_local;
        out_info->max_storage_buffer_range = (size_t)backend->vk.limits.maxStorageBufferRange;
        out_info->max_compute_workgroups_x = (size_t)backend->vk.limits.maxComputeWorkGroupCount[0];
        out_info->max_kernel_chunk_bytes = (size_t)backend->vk.max_kernel_chunk_bytes;
        return VEC_STATUS_OK;
#else
        return VEC_STATUS_UNSUPPORTED;
#endif
    }

    return VEC_STATUS_UNSUPPORTED;
}

// MARK: Buffer Lifecycle

vec_status vec_create(
    vector_backend* backend,
    vec_kind kind,
    size_t length,
    vec_buffer** out_buffer
)
{
    if (backend == NULL || out_buffer == NULL || length == 0u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t elem_size = vec_element_size(kind);
    if (elem_size == 0u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    size_t bytes = 0u;
    if (!vec_multiply_size(length, elem_size, &bytes) || bytes == 0u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    vec_buffer* buffer = (vec_buffer*)calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        return VEC_STATUS_ALLOCATION_FAILED;
    }

    buffer->owner = backend;
    buffer->kind = kind;
    buffer->length = length;
    buffer->bytes = bytes;

    if (backend->type == VECTOR_BACKEND_CPU) {
        buffer->host_ptr = malloc(buffer->bytes);
        if (buffer->host_ptr == NULL) {
            free(buffer);
            return VEC_STATUS_ALLOCATION_FAILED;
        }
    }
    else if (backend->type == VECTOR_BACKEND_VULKAN) {
        vec_status status = vk_buffer_create(backend, buffer);
        if (status != VEC_STATUS_OK) {
            free(buffer);
            return status;
        }
    }
    else {
        free(buffer);
        return VEC_STATUS_UNSUPPORTED;
    }

    *out_buffer = buffer;
    return VEC_STATUS_OK;
}

void vec_destroy(vector_backend* backend, vec_buffer* buffer)
{
    if (backend == NULL || buffer == NULL || buffer->owner != backend) {
        return;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        free(buffer->host_ptr);
    }
    else if (backend->type == VECTOR_BACKEND_VULKAN) {
        vk_buffer_destroy(backend, buffer);
    }

    free(buffer);
}

// MARK: Host Transfers

vec_status vec_upload(
    vector_backend* backend,
    vec_buffer* buffer,
    const void* data,
    size_t bytes
)
{
    if (backend == NULL || buffer == NULL || data == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend || bytes != buffer->bytes) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->in_simulation) {
        return VEC_STATUS_TRANSFER_FORBIDDEN;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        memcpy(buffer->host_ptr, data, bytes);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_upload(backend, buffer, data, bytes);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_download(
    vector_backend* backend,
    const vec_buffer* buffer,
    void* data,
    size_t bytes
)
{
    if (backend == NULL || buffer == NULL || data == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend || bytes != buffer->bytes) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->in_simulation) {
        return VEC_STATUS_TRANSFER_FORBIDDEN;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        memcpy(data, buffer->host_ptr, bytes);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_download(backend, buffer, data, bytes);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_get_host_ptr(
    vector_backend* backend,
    vec_buffer* buffer,
    void** out_ptr
)
{
    if (backend == NULL || buffer == NULL || out_ptr == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->type != VECTOR_BACKEND_CPU) {
        return VEC_STATUS_UNSUPPORTED;
    }

    *out_ptr = buffer->host_ptr;
    return VEC_STATUS_OK;
}

vec_status vec_get_const_host_ptr(
    const vector_backend* backend,
    const vec_buffer* buffer,
    const void** out_ptr
)
{
    if (backend == NULL || buffer == NULL || out_ptr == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->type != VECTOR_BACKEND_CPU) {
        return VEC_STATUS_UNSUPPORTED;
    }

    *out_ptr = buffer->host_ptr;
    return VEC_STATUS_OK;
}

// MARK: Vector Operations

vec_status vec_real_fill(vector_backend* backend, vec_buffer* dst, double value)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_REAL64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        real_fill((double*)dst->host_ptr, dst->length, value);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_real_fill(backend, dst, value);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_real_copy(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    vec_status status = vec_validate_pair(backend, dst, src, VEC_KIND_REAL64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        real_copy((double*)dst->host_ptr, (const double*)src->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_real_copy(backend, dst, src);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_real_mul_inplace(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    vec_status status = vec_validate_pair(backend, dst, src, VEC_KIND_REAL64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        real_mul_inplace((double*)dst->host_ptr, (const double*)src->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_real_mul_inplace(backend, dst, src);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_real_pow_int(
    vector_backend* backend,
    const vec_buffer* base,
    vec_buffer* out,
    unsigned int power
)
{
    vec_status status = vec_validate_pair(backend, base, out, VEC_KIND_REAL64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        real_pow_int((const double*)base->host_ptr, (double*)out->host_ptr, base->length, power);
        return VEC_STATUS_OK;
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_fill(vector_backend* backend, vec_buffer* dst, nlo_complex value)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_fill((nlo_complex*)dst->host_ptr, dst->length, value);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_fill(backend, dst, value);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_copy(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    vec_status status = vec_validate_pair(backend, dst, src, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_copy((nlo_complex*)dst->host_ptr, (const nlo_complex*)src->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_copy(backend, dst, src);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_magnitude_squared(
    vector_backend* backend,
    const vec_buffer* src,
    vec_buffer* dst
)
{
    vec_status status = vec_validate_pair(backend, src, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        calculate_magnitude_squared((const nlo_complex*)src->host_ptr, (nlo_complex*)dst->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_magnitude_squared(backend, src, dst);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_axpy_real(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src,
    nlo_complex alpha
)
{
    vec_status status = vec_validate_mixed_pair(backend,
                                                        dst,
                                                        VEC_KIND_COMPLEX64,
                                                        src,
                                                        VEC_KIND_REAL64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_axpy_real((nlo_complex*)dst->host_ptr, (const double*)src->host_ptr, alpha, dst->length);
        return VEC_STATUS_OK;
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_scalar_mul_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    nlo_complex alpha
)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_scalar_mul_inplace((nlo_complex*)dst->host_ptr, alpha, dst->length);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_scalar_mul_inplace(backend, dst, alpha);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_mul_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src
)
{
    vec_status status = vec_validate_pair(backend, dst, src, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_mul_inplace((nlo_complex*)dst->host_ptr, (const nlo_complex*)src->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_mul_inplace(backend, dst, src);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_pow(
    vector_backend* backend,
    const vec_buffer* base,
    vec_buffer* out,
    unsigned int exponent
)
{
    vec_status status = vec_validate_pair(backend, base, out, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_pow((const nlo_complex*)base->host_ptr, (nlo_complex*)out->host_ptr, base->length, exponent);
        return VEC_STATUS_OK;
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_pow_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    unsigned int exponent
)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_pow_inplace((nlo_complex*)dst->host_ptr, dst->length, exponent);
        return VEC_STATUS_OK;
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_pow_elementwise_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* exponent
)
{
    vec_status status = vec_validate_pair(backend, dst, exponent, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_pow_elementwise_inplace((nlo_complex*)dst->host_ptr,
                                            (const nlo_complex*)exponent->host_ptr,
                                            dst->length);
        return VEC_STATUS_OK;
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_real_pow_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    double exponent
)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_real_pow_inplace((nlo_complex*)dst->host_ptr, dst->length, exponent);
        return VEC_STATUS_OK;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_real_pow_inplace(backend, dst, exponent);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_add_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src
)
{
    vec_status status = vec_validate_pair(backend, dst, src, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_add_inplace((nlo_complex*)dst->host_ptr, (const nlo_complex*)src->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }
    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_add_inplace(backend, dst, src);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_exp_inplace(vector_backend* backend, vec_buffer* dst)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_exp_inplace((nlo_complex*)dst->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_exp_inplace(backend, dst);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_log_inplace(vector_backend* backend, vec_buffer* dst)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_log_inplace((nlo_complex*)dst->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_sin_inplace(vector_backend* backend, vec_buffer* dst)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_sin_inplace((nlo_complex*)dst->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_cos_inplace(vector_backend* backend, vec_buffer* dst)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        complex_cos_inplace((nlo_complex*)dst->host_ptr, dst->length);
        return VEC_STATUS_OK;
    }

    return VEC_STATUS_UNSUPPORTED;
}

static double expected_unshifted_angular_frequency(size_t i, size_t n, double step)
{
    if (n == 0u) {
        return 0.0;
    }

    const size_t positive_limit = (n - 1u) / 2u;
    if (i <= positive_limit) {
        return (double)i * step;
    }

    return -((double)n - (double)i) * step;
}

vec_status vec_complex_axis_unshifted_from_delta(
    vector_backend* backend,
    vec_buffer* dst,
    double delta
)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    if (!(delta > 0.0)) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        nlo_complex* values = (nlo_complex*)dst->host_ptr;
        const double step = TWO_PI / ((double)dst->length * delta);
        for (size_t i = 0u; i < dst->length; ++i) {
            values[i] = make(expected_unshifted_angular_frequency(i, dst->length, step), 0.0);
        }
        return VEC_STATUS_OK;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_axis_unshifted_from_delta(backend, dst, delta);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_axis_centered_from_delta(
    vector_backend* backend,
    vec_buffer* dst,
    double delta
)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        nlo_complex* values = (nlo_complex*)dst->host_ptr;
        const double center = 0.5 * (double)(dst->length - 1u);
        for (size_t i = 0u; i < dst->length; ++i) {
            values[i] = make(((double)i - center) * delta, 0.0);
        }
        return VEC_STATUS_OK;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_axis_centered_from_delta(backend, dst, delta);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_mesh_from_axis_tfast(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* axis,
    size_t nt,
    size_t ny,
    vec_mesh_axis axis_kind
)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    status = vec_validate_buffer(backend, axis, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    if (nt == 0u || ny == 0u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if ((dst->length % nt) != 0u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    const size_t xy_points = dst->length / nt;
    if ((xy_points % ny) != 0u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    const size_t nx = xy_points / ny;
    if (nx == 0u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    size_t expected_axis_length = 0u;
    if (axis_kind == VEC_MESH_AXIS_T) {
        expected_axis_length = nt;
    } else if (axis_kind == VEC_MESH_AXIS_Y) {
        expected_axis_length = ny;
    } else if (axis_kind == VEC_MESH_AXIS_X) {
        expected_axis_length = nx;
    } else {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (axis->length != expected_axis_length) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        nlo_complex* dst_ptr = (nlo_complex*)dst->host_ptr;
        const nlo_complex* axis_ptr = (const nlo_complex*)axis->host_ptr;
        for (size_t x = 0u; x < nx; ++x) {
            for (size_t y = 0u; y < ny; ++y) {
                const nlo_complex axis_value =
                    (axis_kind == VEC_MESH_AXIS_T)
                        ? make(0.0, 0.0)
                        : ((axis_kind == VEC_MESH_AXIS_Y) ? axis_ptr[y] : axis_ptr[x]);
                const size_t base = ((x * ny) + y) * nt;
                if (axis_kind == VEC_MESH_AXIS_T) {
                    memcpy(dst_ptr + base, axis_ptr, nt * sizeof(nlo_complex));
                } else {
                    for (size_t t = 0u; t < nt; ++t) {
                        dst_ptr[base + t] = axis_value;
                    }
                }
            }
        }
        return VEC_STATUS_OK;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_mesh_from_axis_tfast(backend, dst, axis, nt, ny, axis_kind);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_relative_error(
    vector_backend* backend,
    const vec_buffer* current,
    const vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    if (out_error == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    vec_status status = vec_validate_pair(backend, current, previous, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }
    if (epsilon <= 0.0) {
        epsilon = 1e-12;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        const nlo_complex* curr = (const nlo_complex*)current->host_ptr;
        const nlo_complex* prev = (const nlo_complex*)previous->host_ptr;
        double max_ratio = 0.0;
        for (size_t i = 0; i < current->length; ++i) {
            const double curr_re = RE(curr[i]);
            const double curr_im = IM(curr[i]);
            const double prev_re = RE(prev[i]);
            const double prev_im = IM(prev[i]);

            const double diff_re = curr_re - prev_re;
            const double diff_im = curr_im - prev_im;
            const double diff_sq = diff_re * diff_re + diff_im * diff_im;
            const double prev_sq = prev_re * prev_re + prev_im * prev_im;
            const double denom = (prev_sq > epsilon) ? prev_sq : epsilon;
            const double ratio = diff_sq / denom;
            if (ratio > max_ratio) {
                max_ratio = ratio;
            }
        }

        *out_error = sqrt(max_ratio);
        return VEC_STATUS_OK;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_relative_error(backend, current, previous, epsilon, out_error);
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status vec_complex_weighted_rms_error(
    vector_backend* backend,
    const vec_buffer* fine,
    const vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
)
{
    if (out_error == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    vec_status status = vec_validate_pair(backend, fine, coarse, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    if (atol < 0.0) {
        atol = 0.0;
    }
    if (rtol < 0.0) {
        rtol = 0.0;
    }
    if (atol == 0.0 && rtol == 0.0) {
        rtol = 1e-6;
    }

    if (backend->type == VECTOR_BACKEND_CPU) {
        const nlo_complex* fine_values = (const nlo_complex*)fine->host_ptr;
        const nlo_complex* coarse_values = (const nlo_complex*)coarse->host_ptr;

        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t i = 0u; i < fine->length; ++i) {
            const double fine_re = RE(fine_values[i]);
            const double fine_im = IM(fine_values[i]);
            const double coarse_re = RE(coarse_values[i]);
            const double coarse_im = IM(coarse_values[i]);

            const double diff_re = fine_re - coarse_re;
            const double diff_im = fine_im - coarse_im;
            numerator += (diff_re * diff_re) + (diff_im * diff_im);

            const double fine_abs = sqrt((fine_re * fine_re) + (fine_im * fine_im));
            const double weight = atol + (rtol * fine_abs);
            denominator += weight * weight;
        }

        if (denominator <= 0.0) {
            *out_error = 0.0;
            return VEC_STATUS_OK;
        }

        const double ratio = numerator / denominator;
        *out_error = sqrt((ratio > 0.0) ? ratio : 0.0);
        return VEC_STATUS_OK;
    }

    if (backend->type == VECTOR_BACKEND_VULKAN) {
        return vk_op_complex_weighted_rms_error(backend, fine, coarse, atol, rtol, out_error);
    }

    return VEC_STATUS_UNSUPPORTED;
}

