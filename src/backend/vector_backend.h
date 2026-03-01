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

#ifndef NLO_ENABLE_VULKAN_BACKEND
#define NLO_ENABLE_VULKAN_BACKEND 1
#endif

#if NLO_ENABLE_VULKAN_BACKEND
#include <vulkan/vulkan.h>
#else
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkCommandPool;
typedef int VkPhysicalDeviceType;
#ifndef VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// MARK: Typedefs

/**
 * @brief Available vector backend implementations.
 */
typedef enum {
    NLO_VECTOR_BACKEND_CPU = 0,
    NLO_VECTOR_BACKEND_VULKAN = 1,
    NLO_VECTOR_BACKEND_AUTO = 2
} nlo_vector_backend_type;

/**
 * @brief Logical buffer element type.
 */
typedef enum {
    NLO_VEC_KIND_REAL64 = 0,
    NLO_VEC_KIND_COMPLEX64 = 1
} nlo_vec_kind;

/**
 * @brief Status codes returned by vector backend APIs.
 */
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

/**
 * @brief Backend memory/dispatch limits used for chunk planning heuristics.
 */
typedef struct {
    size_t device_local_total_bytes;
    size_t device_local_available_bytes;
    size_t max_storage_buffer_range;
    size_t max_compute_workgroups_x;
    size_t max_kernel_chunk_bytes;
} nlo_vec_backend_memory_info;

/**
 * @brief Axis selector used when expanding one axis vector to a 3D mesh.
 */
typedef enum {
    /** Expand temporal axis (t-fast layout index). */
    NLO_VEC_MESH_AXIS_T = 0,
    /** Expand y axis (t-fast layout index). */
    NLO_VEC_MESH_AXIS_Y = 1,
    /** Expand x axis (t-fast layout index). */
    NLO_VEC_MESH_AXIS_X = 2
} nlo_vec_mesh_axis;

// MARK: Backend Lifecycle

/**
 * @brief Create a CPU backend (host-resident buffers).
 *
 * @return nlo_vector_backend* Backend handle, or NULL on failure.
 */
nlo_vector_backend* nlo_vector_backend_create_cpu(void);

/**
 * @brief Destroy a backend and any internal resources it owns.
 *
 * @param backend Backend handle to destroy (NULL is allowed).
 */
void nlo_vector_backend_destroy(nlo_vector_backend* backend);

/**
 * @brief Get the backend type.
 *
 * @param backend Backend handle.
 * @return nlo_vector_backend_type Backend type enum value.
 */
nlo_vector_backend_type nlo_vector_backend_get_type(const nlo_vector_backend* backend);

/**
 * @brief Returns true while backend transfers are guarded by simulation mode.
 *
 * @param backend Backend handle.
 * @return bool True when simulation mode is active.
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
 *
 * @param config Optional Vulkan configuration overrides.
 * @return nlo_vector_backend* Backend handle, or NULL on failure.
 */
nlo_vector_backend* nlo_vector_backend_create_vulkan(const nlo_vk_backend_config* config);

// MARK: Simulation Guard

/**
 * @brief Mark the start of a simulation. Transfers are blocked until end.
 *
 * @param backend Backend handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_begin_simulation(nlo_vector_backend* backend);

/**
 * @brief Mark the end of a simulation. Transfers are allowed again.
 *
 * @param backend Backend handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_end_simulation(nlo_vector_backend* backend);

/**
 * @brief Query backend memory and dispatch limits used for chunk planning.
 *
 * @param backend Backend handle.
 * @param out_info Destination memory info descriptor.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_query_memory_info(
    const nlo_vector_backend* backend,
    nlo_vec_backend_memory_info* out_info
);

// MARK: Buffer Lifecycle

/**
 * @brief Create a vector buffer of the requested kind and length.
 *
 * @param backend Backend handle.
 * @param kind Logical element type for the buffer.
 * @param length Element count.
 * @param out_buffer Destination buffer handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_create(
    nlo_vector_backend* backend,
    nlo_vec_kind kind,
    size_t length,
    nlo_vec_buffer** out_buffer
);

/**
 * @brief Destroy a previously created vector buffer.
 *
 * @param backend Backend that owns the buffer.
 * @param buffer Buffer handle to destroy (NULL is allowed).
 */
void nlo_vec_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer);

// MARK: Host Transfers (blocked during simulation)

/**
 * @brief Upload host data into a backend buffer.
 *
 * @param backend Backend handle.
 * @param buffer Destination backend buffer.
 * @param data Source host pointer.
 * @param bytes Number of bytes to upload.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
);

/**
 * @brief Download backend buffer data into host memory.
 *
 * @param backend Backend handle.
 * @param buffer Source backend buffer.
 * @param data Destination host pointer.
 * @param bytes Number of bytes to download.
 * @return nlo_vec_status Operation status.
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
 *
 * @param backend Backend handle.
 * @param buffer Buffer handle.
 * @param out_ptr Destination pointer to mapped host memory.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_get_host_ptr(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    void** out_ptr
);

/**
 * @brief Get a direct const host pointer for CPU buffers.
 *        Returns NLO_VEC_STATUS_UNSUPPORTED on non-CPU backends.
 *
 * @param backend Backend handle.
 * @param buffer Buffer handle.
 * @param out_ptr Destination pointer to const mapped host memory.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_get_const_host_ptr(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    const void** out_ptr
);

// MARK: Vector Operations

/**
 * @brief Fill a real-valued backend vector with a constant.
 *
 * @param backend Backend handle.
 * @param dst Destination real vector.
 * @param value Fill value.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value);

/**
 * @brief Copy one real-valued backend vector into another.
 *
 * @param backend Backend handle.
 * @param dst Destination real vector.
 * @param src Source real vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);

/**
 * @brief Element-wise real multiply in place: @p dst[i] *= @p src[i].
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand real vector.
 * @param src Right operand real vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);

/**
 * @brief Raise each real element to an integer power.
 *
 * @param backend Backend handle.
 * @param base Input real vector.
 * @param out Destination real vector.
 * @param power Non-negative integer exponent.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_real_pow_int(nlo_vector_backend* backend, const nlo_vec_buffer* base, nlo_vec_buffer* out, unsigned int power);

/**
 * @brief Fill a complex backend vector with a constant value.
 *
 * @param backend Backend handle.
 * @param dst Destination complex vector.
 * @param value Fill value.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value);

/**
 * @brief Copy one complex backend vector into another.
 *
 * @param backend Backend handle.
 * @param dst Destination complex vector.
 * @param src Source complex vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);

/**
 * @brief Compute element-wise magnitude squared into a complex-valued output.
 *
 * @param backend Backend handle.
 * @param src Source complex vector.
 * @param dst Destination complex vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_magnitude_squared(nlo_vector_backend* backend, const nlo_vec_buffer* src, nlo_vec_buffer* dst);

/**
 * @brief Complex AXPY with real input: @p dst[i] += alpha * src[i].
 *
 * @param backend Backend handle.
 * @param dst Destination complex vector.
 * @param src Source real vector.
 * @param alpha Complex scale factor.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_axpy_real(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src, nlo_complex alpha);

/**
 * @brief Multiply each complex element by a scalar in place.
 *
 * @param backend Backend handle.
 * @param dst Destination complex vector.
 * @param alpha Complex scale factor.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_scalar_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex alpha);

/**
 * @brief Element-wise complex multiply in place: @p dst[i] *= @p src[i].
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand complex vector.
 * @param src Right operand complex vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);

/**
 * @brief Element-wise complex power with integer exponent.
 *
 * @param backend Backend handle.
 * @param base Input complex vector.
 * @param out Destination complex vector.
 * @param exponent Non-negative integer exponent.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_pow(nlo_vector_backend* backend, const nlo_vec_buffer* base, nlo_vec_buffer* out, unsigned int exponent);

/**
 * @brief Element-wise complex power in place with integer exponent.
 *
 * @param backend Backend handle.
 * @param dst Complex vector updated in place.
 * @param exponent Non-negative integer exponent.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_pow_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, unsigned int exponent);
/**
 * @brief Element-wise complex power with complex exponent inplace:
 *        dst[i] = dst[i] ^ exponent[i].
 *
 * @param backend Backend handle.
 * @param dst Complex base vector updated in place.
 * @param exponent Complex exponent vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_pow_elementwise_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* exponent
);
/**
 * @brief Element-wise complex real power inplace: dst[i] = dst[i] ^ exponent.
 *
 * @param backend Backend handle.
 * @param dst Complex vector updated in place.
 * @param exponent Real exponent.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
);

/**
 * @brief Element-wise complex add in place: @p dst[i] += @p src[i].
 *
 * @param backend Backend handle.
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_add_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);

/**
 * @brief Apply element-wise complex exponential in place.
 *
 * @param backend Backend handle.
 * @param dst Complex vector updated in place.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
/**
 * @brief Element-wise complex natural logarithm inplace: dst[i] = log(dst[i]).
 *
 * @param backend Backend handle.
 * @param dst Complex vector updated in place.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_log_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
/**
 * @brief Element-wise complex sine inplace: dst[i] = sin(dst[i]).
 *
 * @param backend Backend handle.
 * @param dst Complex vector updated in place.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_sin_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
/**
 * @brief Element-wise complex cosine inplace: dst[i] = cos(dst[i]).
 *
 * @param backend Backend handle.
 * @param dst Complex vector updated in place.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_cos_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);

/**
 * @brief Build one unshifted angular-frequency axis from sample spacing.
 *
 * Output follows FFT ordering compatible with `fftfreq`-style grids:
 * `[0, 1, ..., floor((n-1)/2), -ceil((n-1)/2), ..., -1] * 2*pi/(n*delta)`.
 *
 * @param backend Backend handle.
 * @param dst Destination complex axis vector.
 * @param delta Sample spacing (> 0).
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_axis_unshifted_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
);

/**
 * @brief Build one centered coordinate axis from sample spacing.
 *
 * Output is `coord[i] = (i - 0.5*(n-1)) * delta`.
 *
 * @param backend Backend handle.
 * @param dst Destination complex axis vector.
 * @param delta Sample spacing.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_axis_centered_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
);

/**
 * @brief Expand one axis vector into a full 3D mesh (t-fast flattening).
 *
 * Uses index mapping `idx = ((x * ny) + y) * nt + t`.
 *
 * @param backend Backend handle.
 * @param dst Destination full-volume complex vector (`nt*nx*ny`).
 * @param axis Source 1D axis vector.
 * @param nt Temporal sample count.
 * @param ny Y sample count.
 * @param axis_kind Axis role for expansion.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_mesh_from_axis_tfast(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* axis,
    size_t nt,
    size_t ny,
    nlo_vec_mesh_axis axis_kind
);

/**
 * @brief Compute relative L-infinity complex error:
 *        sqrt(max(|current-prev|^2) / max(max(|prev|^2), epsilon)).
 *
 * @param backend Backend handle.
 * @param current Current solution vector.
 * @param previous Previous/reference solution vector.
 * @param epsilon Stabilizing floor for denominator.
 * @param out_error Destination scalar error value.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
);

/**
 * @brief Compute weighted RMS complex error:
 *        sqrt(sum(|fine-coarse|^2) / sum((atol + rtol*|fine|)^2)).
 *
 * @param backend Backend handle.
 * @param fine Fine/reference solution vector.
 * @param coarse Coarse/approximate solution vector.
 * @param atol Absolute tolerance term.
 * @param rtol Relative tolerance term.
 * @param out_error Destination scalar error value.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_vec_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
);

#ifdef __cplusplus
}
#endif

