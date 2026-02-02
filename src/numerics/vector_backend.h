/**
 * @file vector_backend.h
 * @dir src/numerics
 * @brief Backend abstraction for vector operations (CPU or Vulkan).
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */
#pragma once

// MARK: Includes

#include "fft/nlo_complex.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: Typedefs

typedef enum {
    NLO_VECTOR_BACKEND_CPU = 0,
    NLO_VECTOR_BACKEND_VULKAN = 1
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

#ifdef NLO_VECTOR_BACKEND_VULKAN
#include <vulkan/vulkan.h>

/**
 * @brief Vulkan backend configuration (expects externally-created device/queue/pool).
 */
typedef struct {
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
} nlo_vk_backend_config;

/**
 * @brief Create a Vulkan backend (device-resident buffers).
 */
nlo_vector_backend* nlo_vector_backend_create_vulkan(const nlo_vk_backend_config* config);
#endif

// MARK: Simulation Guard

/**
 * @brief Mark the start of a simulation. Transfers are blocked until end.
 */
nlo_vec_status nlo_vec_begin_simulation(nlo_vector_backend* backend);

/**
 * @brief Mark the end of a simulation. Transfers are allowed again.
 */
nlo_vec_status nlo_vec_end_simulation(nlo_vector_backend* backend);

// MARK: Buffer Lifecycle

/**
 * @brief Create a vector buffer of the requested kind and length.
 */
nlo_vec_status nlo_vec_create(nlo_vector_backend* backend,
                              nlo_vec_kind kind,
                              size_t length,
                              nlo_vec_buffer** out_buffer);

/**
 * @brief Destroy a previously created vector buffer.
 */
void nlo_vec_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer);

// MARK: Host Transfers (blocked during simulation)

/**
 * @brief Upload host data into a backend buffer.
 */
nlo_vec_status nlo_vec_upload(nlo_vector_backend* backend,
                              nlo_vec_buffer* buffer,
                              const void* data,
                              size_t bytes);

/**
 * @brief Download backend buffer data into host memory.
 */
nlo_vec_status nlo_vec_download(nlo_vector_backend* backend,
                                const nlo_vec_buffer* buffer,
                                void* data,
                                size_t bytes);

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
nlo_vec_status nlo_vec_complex_add_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vec_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);

#ifdef __cplusplus
}
#endif
