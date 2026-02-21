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

size_t nlo_vec_element_size(nlo_vec_kind kind)
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

static bool nlo_vec_multiply_size(size_t a, size_t b, size_t* out)
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

static nlo_vec_status nlo_vec_validate_mixed_pair(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* lhs,
    nlo_vec_kind lhs_kind,
    const nlo_vec_buffer* rhs,
    nlo_vec_kind rhs_kind
)
{
    if (backend == NULL || lhs == NULL || rhs == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (lhs->owner != backend || rhs->owner != backend ||
        lhs->kind != lhs_kind || rhs->kind != rhs_kind) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (lhs->length != rhs->length) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return NLO_VEC_STATUS_OK;
}

static const char* nlo_vk_device_type_to_string(VkPhysicalDeviceType device_type)
{
#if NLO_ENABLE_VULKAN_BACKEND
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

static nlo_vector_backend* nlo_vector_backend_create_auto(const nlo_vk_backend_config* config_template);

nlo_vec_status nlo_vec_validate_backend(const nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vec_validate_buffer(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    nlo_vec_kind kind
)
{
    if (backend == NULL || buffer == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend || buffer->kind != kind) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vec_validate_pair(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    nlo_vec_kind kind
)
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

    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vk_backend_shutdown(backend);
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

bool nlo_vec_is_in_simulation(const nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return false;
    }
    return backend->in_simulation;
}

nlo_vector_backend* nlo_vector_backend_create_vulkan(const nlo_vk_backend_config* config)
{
    if (config == NULL) {
        return nlo_vector_backend_create_auto(NULL);
    }
    if (config->physical_device == VK_NULL_HANDLE ||
        config->device == VK_NULL_HANDLE ||
        config->queue == VK_NULL_HANDLE) {
        return NULL;
    }

    nlo_vector_backend* backend = (nlo_vector_backend*)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        return NULL;
    }

    backend->type = NLO_VECTOR_BACKEND_VULKAN;
    backend->in_simulation = false;

    if (nlo_vk_backend_init(backend, config) != NLO_VEC_STATUS_OK) {
        free(backend);
        return NULL;
    }

    return backend;
}

static nlo_vector_backend* nlo_vector_backend_create_auto(const nlo_vk_backend_config* config_template)
{
    char reason[256];
    nlo_vk_auto_context auto_ctx;
    if (nlo_vk_auto_context_init(&auto_ctx, reason, sizeof(reason)) != 0) {
        fprintf(stderr,
                "[nlolib] auto backend selection failed: %s\n",
                (reason[0] != '\0') ? reason : "unknown Vulkan setup error");
        return NULL;
    }

    nlo_vk_backend_config config = {0};
    if (config_template != NULL) {
        config = *config_template;
    }

    config.physical_device = auto_ctx.physical_device;
    config.device = auto_ctx.device;
    config.queue = auto_ctx.queue;
    config.queue_family_index = auto_ctx.queue_family_index;
    config.command_pool = VK_NULL_HANDLE;

    nlo_vector_backend* backend = (nlo_vector_backend*)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        nlo_vk_auto_context_destroy(&auto_ctx);
        return NULL;
    }

    backend->type = NLO_VECTOR_BACKEND_VULKAN;
    backend->in_simulation = false;
    if (nlo_vk_backend_init(backend, &config) != NLO_VEC_STATUS_OK) {
        free(backend);
        nlo_vk_auto_context_destroy(&auto_ctx);
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
            nlo_vk_device_type_to_string(backend->vk.device_type),
            (unsigned long long)backend->vk.device_local_bytes);
    return backend;
}

// MARK: Simulation Guard

nlo_vec_status nlo_vec_begin_simulation(nlo_vector_backend* backend)
{
    nlo_vec_status status = nlo_vec_validate_backend(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    backend->in_simulation = true;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vec_end_simulation(nlo_vector_backend* backend)
{
    nlo_vec_status status = nlo_vec_validate_backend(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        status = nlo_vk_simulation_phase_flush(backend);
        if (status != NLO_VEC_STATUS_OK) {
            backend->in_simulation = false;
            return status;
        }
    }

    backend->in_simulation = false;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vec_query_memory_info(
    const nlo_vector_backend* backend,
    nlo_vec_backend_memory_info* out_info
)
{
    if (backend == NULL || out_info == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    *out_info = (nlo_vec_backend_memory_info){0};
    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        return NLO_VEC_STATUS_OK;
    }

    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
#if NLO_ENABLE_VULKAN_BACKEND
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(backend->vk.physical_device, &memory_properties);

        size_t total_device_local = 0u;
        for (uint32_t i = 0u; i < memory_properties.memoryHeapCount; ++i) {
            if ((memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
                size_t heap_size = (size_t)memory_properties.memoryHeaps[i].size;
                if (heap_size > SIZE_MAX - total_device_local) {
                    total_device_local = SIZE_MAX;
                } else {
                    total_device_local += heap_size;
                }
            }
        }

        out_info->device_local_total_bytes = total_device_local;
        out_info->device_local_available_bytes = total_device_local;
        out_info->max_storage_buffer_range = (size_t)backend->vk.limits.maxStorageBufferRange;
        out_info->max_compute_workgroups_x = (size_t)backend->vk.limits.maxComputeWorkGroupCount[0];
        out_info->max_kernel_chunk_bytes = (size_t)backend->vk.max_kernel_chunk_bytes;
        return NLO_VEC_STATUS_OK;
#else
        return NLO_VEC_STATUS_UNSUPPORTED;
#endif
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

// MARK: Buffer Lifecycle

nlo_vec_status nlo_vec_create(
    nlo_vector_backend* backend,
    nlo_vec_kind kind,
    size_t length,
    nlo_vec_buffer** out_buffer
)
{
    if (backend == NULL || out_buffer == NULL || length == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t elem_size = nlo_vec_element_size(kind);
    if (elem_size == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    size_t bytes = 0u;
    if (!nlo_vec_multiply_size(length, elem_size, &bytes) || bytes == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_buffer* buffer = (nlo_vec_buffer*)calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        return NLO_VEC_STATUS_ALLOCATION_FAILED;
    }

    buffer->owner = backend;
    buffer->kind = kind;
    buffer->length = length;
    buffer->bytes = bytes;

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        buffer->host_ptr = malloc(buffer->bytes);
        if (buffer->host_ptr == NULL) {
            free(buffer);
            return NLO_VEC_STATUS_ALLOCATION_FAILED;
        }
    }
    else if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vec_status status = nlo_vk_buffer_create(backend, buffer);
        if (status != NLO_VEC_STATUS_OK) {
            free(buffer);
            return status;
        }
    }
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
    else if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vk_buffer_destroy(backend, buffer);
    }

    free(buffer);
}

// MARK: Host Transfers

nlo_vec_status nlo_vec_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
)
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
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_upload(backend, buffer, data, bytes);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
)
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
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_download(backend, buffer, data, bytes);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_get_host_ptr(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    void** out_ptr
)
{
    if (backend == NULL || buffer == NULL || out_ptr == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->type != NLO_VECTOR_BACKEND_CPU) {
        return NLO_VEC_STATUS_UNSUPPORTED;
    }

    *out_ptr = buffer->host_ptr;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vec_get_const_host_ptr(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    const void** out_ptr
)
{
    if (backend == NULL || buffer == NULL || out_ptr == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->owner != backend) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->type != NLO_VECTOR_BACKEND_CPU) {
        return NLO_VEC_STATUS_UNSUPPORTED;
    }

    *out_ptr = buffer->host_ptr;
    return NLO_VEC_STATUS_OK;
}

// MARK: Vector Operations

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
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_real_fill(backend, dst, value);
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
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_real_copy(backend, dst, src);
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
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_real_mul_inplace(backend, dst, src);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_real_pow_int(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* base,
    nlo_vec_buffer* out,
    unsigned int power
)
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
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_fill(backend, dst, value);
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
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_copy(backend, dst, src);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, src, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        calculate_magnitude_squared((const nlo_complex*)src->host_ptr, (nlo_complex*)dst->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_magnitude_squared(backend, src, dst);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_axpy_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    nlo_complex alpha
)
{
    nlo_vec_status status = nlo_vec_validate_mixed_pair(backend,
                                                        dst,
                                                        NLO_VEC_KIND_COMPLEX64,
                                                        src,
                                                        NLO_VEC_KIND_REAL64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_axpy_real((nlo_complex*)dst->host_ptr, (const double*)src->host_ptr, alpha, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_scalar_mul_inplace((nlo_complex*)dst->host_ptr, alpha, dst->length);
        return NLO_VEC_STATUS_OK;
    }
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_scalar_mul_inplace(backend, dst, alpha);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, dst, src, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_mul_inplace((nlo_complex*)dst->host_ptr, (const nlo_complex*)src->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_mul_inplace(backend, dst, src);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_pow(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* base,
    nlo_vec_buffer* out,
    unsigned int exponent
)
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

nlo_vec_status nlo_vec_complex_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    unsigned int exponent
)
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

nlo_vec_status nlo_vec_complex_pow_elementwise_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* exponent
)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, dst, exponent, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_pow_elementwise_inplace((nlo_complex*)dst->host_ptr,
                                            (const nlo_complex*)exponent->host_ptr,
                                            dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_real_pow_inplace((nlo_complex*)dst->host_ptr, dst->length, exponent);
        return NLO_VEC_STATUS_OK;
    }

    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_real_pow_inplace(backend, dst, exponent);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    nlo_vec_status status = nlo_vec_validate_pair(backend, dst, src, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_add_inplace((nlo_complex*)dst->host_ptr, (const nlo_complex*)src->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_add_inplace(backend, dst, src);
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

    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_exp_inplace(backend, dst);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_log_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_log_inplace((nlo_complex*)dst->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_sin_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_sin_inplace((nlo_complex*)dst->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_cos_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        nlo_complex_cos_inplace((nlo_complex*)dst->host_ptr, dst->length);
        return NLO_VEC_STATUS_OK;
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    if (out_error == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vec_validate_pair(backend, current, previous, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    if (epsilon <= 0.0) {
        epsilon = 1e-12;
    }

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        const nlo_complex* curr = (const nlo_complex*)current->host_ptr;
        const nlo_complex* prev = (const nlo_complex*)previous->host_ptr;
        double max_ratio = 0.0;
        for (size_t i = 0; i < current->length; ++i) {
            const double curr_re = NLO_RE(curr[i]);
            const double curr_im = NLO_IM(curr[i]);
            const double prev_re = NLO_RE(prev[i]);
            const double prev_im = NLO_IM(prev[i]);

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
        return NLO_VEC_STATUS_OK;
    }

    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_relative_error(backend, current, previous, epsilon, out_error);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_vec_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
)
{
    if (out_error == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vec_validate_pair(backend, fine, coarse, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
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

    if (backend->type == NLO_VECTOR_BACKEND_CPU) {
        const nlo_complex* fine_values = (const nlo_complex*)fine->host_ptr;
        const nlo_complex* coarse_values = (const nlo_complex*)coarse->host_ptr;

        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t i = 0u; i < fine->length; ++i) {
            const double fine_re = NLO_RE(fine_values[i]);
            const double fine_im = NLO_IM(fine_values[i]);
            const double coarse_re = NLO_RE(coarse_values[i]);
            const double coarse_im = NLO_IM(coarse_values[i]);

            const double diff_re = fine_re - coarse_re;
            const double diff_im = fine_im - coarse_im;
            numerator += (diff_re * diff_re) + (diff_im * diff_im);

            const double fine_abs = sqrt((fine_re * fine_re) + (fine_im * fine_im));
            const double weight = atol + (rtol * fine_abs);
            denominator += weight * weight;
        }

        if (denominator <= 0.0) {
            *out_error = 0.0;
            return NLO_VEC_STATUS_OK;
        }

        const double ratio = numerator / denominator;
        *out_error = sqrt((ratio > 0.0) ? ratio : 0.0);
        return NLO_VEC_STATUS_OK;
    }

    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_weighted_rms_error(backend, fine, coarse, atol, rtol, out_error);
    }

    return NLO_VEC_STATUS_UNSUPPORTED;
}

