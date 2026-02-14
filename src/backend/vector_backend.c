/**
 * @file vector_backend.c
 * @dir src/backend
 * @brief Backend abstraction for vector operations (CPU or Vulkan).
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */

#include "backend/vector_backend_internal.h"
#include "numerics/vector_ops.h"
#include "numerics/vk_vector_ops.h"
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

nlo_vec_status nlo_vec_validate_backend(const nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vec_validate_buffer(const nlo_vector_backend* backend,
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

nlo_vec_status nlo_vec_validate_pair(const nlo_vector_backend* backend,
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

#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vk_backend_shutdown(backend);
    }
#endif

    free(backend);
}

nlo_vector_backend_type nlo_vector_backend_get_type(const nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return NLO_VECTOR_BACKEND_CPU;
    }
    return backend->type;
}

#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
nlo_vector_backend* nlo_vector_backend_create_vulkan(const nlo_vk_backend_config* config)
{
    if (config == NULL ||
        config->physical_device == VK_NULL_HANDLE ||
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
#endif

// MARK: Simulation Guard

nlo_vec_status nlo_vec_begin_simulation(nlo_vector_backend* backend)
{
    nlo_vec_status status = nlo_vec_validate_backend(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    backend->in_simulation = true;
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    else if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vec_status status = nlo_vk_buffer_create(backend, buffer);
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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    else if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vk_buffer_destroy(backend, buffer);
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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_download(backend, buffer, data, bytes);
    }
#endif

    return NLO_VEC_STATUS_UNSUPPORTED;
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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_real_fill(backend, dst, value);
    }
#endif

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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_real_copy(backend, dst, src);
    }
#endif

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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_real_mul_inplace(backend, dst, src);
    }
#endif

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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_fill(backend, dst, value);
    }
#endif

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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_copy(backend, dst, src);
    }
#endif

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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_magnitude_squared(backend, src, dst);
    }
#endif

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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_scalar_mul_inplace(backend, dst, alpha);
    }
#endif

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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_mul_inplace(backend, dst, src);
    }
#endif

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
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    if (backend->type == NLO_VECTOR_BACKEND_VULKAN) {
        return nlo_vk_op_complex_add_inplace(backend, dst, src);
    }
#endif

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

