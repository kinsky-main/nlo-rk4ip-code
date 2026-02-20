/**
 * @file vector_backend.h
 * @dir src/numerics
 * @brief Backend abstraction for vector operations (CPU or Vulkan).
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */
#pragma once

// MARK: Includes

#include "backend/nlo_complex.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: Typedefs

typedef enum {
    NLO_VECTOR_BACKEND_CPU = 0,
    NLO_VECTOR_BACKEND_VULKAN = 1,
    NLO_VECTOR_BACKEND_AUTO = 2
} nlo_vector_backend_type;

typedef enum {
    NLO_VEC_KIND_REAL64 = 0,
    NLO_VEC_KIND_COMPLEX64 = 1
} nlo_vec_kind;

typedef enum {
    NLO_VEC_STATUS_OK = 0,
    NLO_VEC_STATUS_INVALID_ARGUMENT = 1,
    NLO_VEC_STATUS_ALLOCATION_FAILED = 2,
    NLO_VEC_STATUS_BACKEND_UNAVAILABLE = 3,
    NLO_VEC_STATUS_TRANSFER_FORBIDDEN = 4,
    NLO_VEC_STATUS_UNSUPPORTED = 5
} nlo_vec_status;

typedef struct nlo_vector_backend nlo_vector_backend;
typedef struct nlo_vec_buffer nlo_vec_buffer;

typedef struct {
    size_t device_local_total_bytes;
    size_t device_local_available_bytes;
    size_t max_storage_buffer_range;
    size_t max_compute_workgroups_x;
    size_t max_kernel_chunk_bytes;
    VkPhysicalDeviceType device_type;
} nlo_vec_backend_memory_info;

// MARK: Backend Lifecycle

/**
 * @brief Create a CPU backend (host-resident buffers).
 */
nlo_vector_backend* nlo_vector_backend_create_cpu(void);

/**
 * @brief Destroy a backend and any internal resources it owns.
 */
void nlo_vector_backend_destroy(nlo_vector_backend* backend);

/**
 * @brief Get the backend type.
 */
nlo_vector_backend_type nlo_vector_backend_get_type(const nlo_vector_backend* backend);

/**
 * @brief Returns true while backend transfers are guarded by simulation mode.
 */
bool nlo_vec_is_in_simulation(const nlo_vector_backend* backend);

/**
 * @brief Vulkan backend configuration (expects externally-created device/queue).
 *        If command_pool is provided, it will be reused; otherwise an internal
 *        pool is created for backend submissions.
 *        Requires shaderFloat64 support for complex<double> kernels.
 */
typedef struct {
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
    /**
     * @brief Optional descriptor-set memory budget used for runtime pool sizing.
     *        Set to 0 to use backend default.
     */
    size_t descriptor_set_budget_bytes;
    /**
     * @brief Optional explicit descriptor-set count override.
     *        Set to 0 to use runtime budget-based sizing.
     */
    uint32_t descriptor_set_count_override;
} nlo_vk_backend_config;

/**
 * @brief Create a Vulkan backend (device-resident buffers).
 *        If config is NULL, Vulkan device/queue are auto-detected from
 *        available hardware.
 */
nlo_vector_backend* nlo_vector_backend_create_vulkan(const nlo_vk_backend_config* config);

// MARK: Simulation Guard

/**
 * @brief Mark the start of a simulation. Transfers are blocked until end.
 */
nlo_vec_status nlo_vec_begin_simulation(nlo_vector_backend* backend);

/**
 * @brief Mark the end of a simulation. Transfers are allowed again.
 */
nlo_vec_status nlo_vec_end_simulation(nlo_vector_backend* backend);

/**
 * @brief Query backend memory and dispatch limits used for chunk planning.
 */
nlo_vec_status nlo_vec_query_memory_info(
    const nlo_vector_backend* backend,
    nlo_vec_backend_memory_info* out_info
);

// MARK: Buffer Lifecycle

/**
 * @brief Create a vector buffer of the requested kind and length.
 */
nlo_vec_status nlo_vec_create(
    nlo_vector_backend* backend,
    nlo_vec_kind kind,
    size_t length,
    nlo_vec_buffer** out_buffer
);

/**
 * @brief Destroy a previously created vector buffer.
 */
void nlo_vec_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer);

// MARK: Host Transfers (blocked during simulation)

/**
 * @brief Upload host data into a backend buffer.
 */
nlo_vec_status nlo_vec_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
);

/**
 * @brief Download backend buffer data into host memory.
 */
nlo_vec_status nlo_vec_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
);

/**
 * @brief Get a direct host pointer for CPU buffers.
 *        Returns NLO_VEC_STATUS_UNSUPPORTED on non-CPU backends.
 */
nlo_vec_status nlo_vec_get_host_ptr(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    void** out_ptr
);

/**
 * @brief Get a direct const host pointer for CPU buffers.
 *        Returns NLO_VEC_STATUS_UNSUPPORTED on non-CPU backends.
 */
nlo_vec_status nlo_vec_get_const_host_ptr(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    const void** out_ptr
);

// MARK: Vector Operations

nlo_vec_status nlo_vec_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value);
nlo_vec_status nlo_vec_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vec_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vec_real_pow_int(nlo_vector_backend* backend, const nlo_vec_buffer* base, nlo_vec_buffer* out, unsigned int power);

nlo_vec_status nlo_vec_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value);
nlo_vec_status nlo_vec_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vec_complex_magnitude_squared(nlo_vector_backend* backend, const nlo_vec_buffer* src, nlo_vec_buffer* dst);
nlo_vec_status nlo_vec_complex_axpy_real(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src, nlo_complex alpha);
nlo_vec_status nlo_vec_complex_scalar_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex alpha);
nlo_vec_status nlo_vec_complex_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vec_complex_pow(nlo_vector_backend* backend, const nlo_vec_buffer* base, nlo_vec_buffer* out, unsigned int exponent);
nlo_vec_status nlo_vec_complex_pow_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, unsigned int exponent);
/**
 * @brief Element-wise complex power with complex exponent inplace:
 *        dst[i] = dst[i] ^ exponent[i].
 */
nlo_vec_status nlo_vec_complex_pow_elementwise_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* exponent
);
/**
 * @brief Element-wise complex real power inplace: dst[i] = dst[i] ^ exponent.
 */
nlo_vec_status nlo_vec_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
);
nlo_vec_status nlo_vec_complex_add_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vec_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
/**
 * @brief Element-wise complex natural logarithm inplace: dst[i] = log(dst[i]).
 */
nlo_vec_status nlo_vec_complex_log_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
/**
 * @brief Element-wise complex sine inplace: dst[i] = sin(dst[i]).
 */
nlo_vec_status nlo_vec_complex_sin_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
/**
 * @brief Element-wise complex cosine inplace: dst[i] = cos(dst[i]).
 */
nlo_vec_status nlo_vec_complex_cos_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);

/**
 * @brief Compute relative L-infinity complex error:
 *        sqrt(max(|current-prev|^2) / max(max(|prev|^2), epsilon)).
 */
nlo_vec_status nlo_vec_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
);

#ifdef __cplusplus
}
#endif

