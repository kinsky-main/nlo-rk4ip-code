/**
 * @file cuda_backend.cu
 * @brief CUDA backend implementation for vector operations and single-GPU execution.
 */

#include "backend/cuda_backend_internal.h"

#if NLO_ENABLE_CUDA_BACKEND

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/complex.h>

#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>

#ifndef NLO_CUDA_BLOCK_SIZE
#define NLO_CUDA_BLOCK_SIZE 256u
#endif

#ifndef NLO_CUDA_DEFAULT_PINNED_STAGING_BYTES
#define NLO_CUDA_DEFAULT_PINNED_STAGING_BYTES (8u * 1024u * 1024u)
#endif

#ifndef NLO_CUDA_DEFAULT_REDUCTION_CAPACITY
#define NLO_CUDA_DEFAULT_REDUCTION_CAPACITY (1u << 20u)
#endif

typedef struct {
    double sum_sq_diff;
    double sum_sq_weight;
} nlo_cuda_weighted_pair;

static nlo_vec_status nlo_cuda_status_from_error(cudaError_t error)
{
    if (error == cudaSuccess) {
        return NLO_VEC_STATUS_OK;
    }
    if (error == cudaErrorMemoryAllocation) {
        return NLO_VEC_STATUS_ALLOCATION_FAILED;
    }
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

static nlo_vec_status nlo_cuda_peek_status(void)
{
    return nlo_cuda_status_from_error(cudaPeekAtLastError());
}

static int nlo_cuda_pick_device_ordinal(int requested_ordinal, char* device_name, size_t device_name_size)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return -1;
    }

    int selected = -1;
    size_t best_memory = 0u;
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
            continue;
        }
        if (requested_ordinal >= 0 && device != requested_ordinal) {
            continue;
        }
        if (prop.major < 1 || (prop.major == 1 && prop.minor < 3)) {
            continue;
        }
        if (requested_ordinal >= 0) {
            selected = device;
            best_memory = (size_t)prop.totalGlobalMem;
            if (device_name != NULL && device_name_size > 0u) {
                (void)snprintf(device_name, device_name_size, "%s", prop.name);
            }
            break;
        }
        if ((size_t)prop.totalGlobalMem >= best_memory) {
            selected = device;
            best_memory = (size_t)prop.totalGlobalMem;
            if (device_name != NULL && device_name_size > 0u) {
                (void)snprintf(device_name, device_name_size, "%s", prop.name);
            }
        }
    }

    return selected;
}

static nlo_vec_status nlo_cuda_sync_stream(cudaStream_t stream)
{
    return nlo_cuda_status_from_error(cudaStreamSynchronize(stream));
}

static nlo_vec_status nlo_cuda_prefetch_to_device(
    nlo_vector_backend* backend,
    void* ptr,
    size_t bytes,
    cudaStream_t stream
)
{
    if (backend == NULL || ptr == NULL || bytes == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    (void)backend;
    (void)ptr;
    (void)bytes;
    (void)stream;
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_cuda_prefetch_to_host(void* ptr, size_t bytes, cudaStream_t stream)
{
    if (ptr == NULL || bytes == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    (void)ptr;
    (void)bytes;
    (void)stream;
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_cuda_ensure_reduction_capacity(nlo_vector_backend* backend, size_t required_elements)
{
    if (backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (required_elements <= backend->cuda.reduction_capacity_elements) {
        return NLO_VEC_STATUS_OK;
    }

    size_t new_capacity = backend->cuda.reduction_capacity_elements;
    if (new_capacity == 0u) {
        new_capacity = NLO_CUDA_DEFAULT_REDUCTION_CAPACITY;
    }
    while (new_capacity < required_elements) {
        if (new_capacity > (SIZE_MAX / 2u)) {
            return NLO_VEC_STATUS_ALLOCATION_FAILED;
        }
        new_capacity *= 2u;
    }

    void* new_buffer_a = NULL;
    void* new_buffer_b = NULL;
    cudaError_t error = cudaMallocManaged(&new_buffer_a, new_capacity * sizeof(nlo_cuda_weighted_pair));
    if (error != cudaSuccess) {
        return nlo_cuda_status_from_error(error);
    }
    error = cudaMallocManaged(&new_buffer_b, new_capacity * sizeof(nlo_cuda_weighted_pair));
    if (error != cudaSuccess) {
        cudaFree(new_buffer_a);
        return nlo_cuda_status_from_error(error);
    }

    if (backend->cuda.reduction_buffer_a != NULL) {
        cudaFree(backend->cuda.reduction_buffer_a);
    }
    if (backend->cuda.reduction_buffer_b != NULL) {
        cudaFree(backend->cuda.reduction_buffer_b);
    }

    backend->cuda.reduction_buffer_a = new_buffer_a;
    backend->cuda.reduction_buffer_b = new_buffer_b;
    backend->cuda.reduction_capacity_elements = new_capacity;
    return NLO_VEC_STATUS_OK;
}

extern "C" nlo_vec_status nlo_cuda_backend_init(
    nlo_vector_backend* backend,
    const nlo_cuda_backend_config* config
)
{
    if (backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_cuda_backend_config defaults;
    defaults.device_ordinal = -1;
    defaults.enable_multi_gpu = 1;
    defaults.max_devices = 8u;
    defaults.enable_peer_access = 1;
    defaults.stream_count = 1u;
    defaults.pinned_staging_bytes = 0u;
    defaults.graph_capture_enabled = 1;
    defaults.nvrtc_enabled = 1;
    const nlo_cuda_backend_config* effective = (config != NULL) ? config : &defaults;

    int selected_device = nlo_cuda_pick_device_ordinal(effective->device_ordinal,
                                                       backend->cuda.device_name,
                                                       sizeof(backend->cuda.device_name));
    if (selected_device < 0) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    cudaError_t error = cudaSetDevice(selected_device);
    if (error != cudaSuccess) {
        return nlo_cuda_status_from_error(error);
    }

    backend->type = NLO_VECTOR_BACKEND_CUDA;
    backend->in_simulation = false;
    backend->cuda.device_ordinal = selected_device;
    backend->cuda.active_device_count = 1u;
    backend->cuda.peer_access_enabled = 0;

    error = cudaStreamCreateWithFlags(&backend->cuda.compute_stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
        nlo_cuda_backend_shutdown(backend);
        return nlo_cuda_status_from_error(error);
    }
    error = cudaStreamCreateWithFlags(&backend->cuda.transfer_stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
        nlo_cuda_backend_shutdown(backend);
        return nlo_cuda_status_from_error(error);
    }
    error = cudaEventCreateWithFlags(&backend->cuda.timing_start, cudaEventDefault);
    if (error != cudaSuccess) {
        nlo_cuda_backend_shutdown(backend);
        return nlo_cuda_status_from_error(error);
    }
    error = cudaEventCreateWithFlags(&backend->cuda.timing_stop, cudaEventDefault);
    if (error != cudaSuccess) {
        nlo_cuda_backend_shutdown(backend);
        return nlo_cuda_status_from_error(error);
    }

    size_t free_bytes = 0u;
    size_t total_bytes = 0u;
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        nlo_cuda_backend_shutdown(backend);
        return nlo_cuda_status_from_error(error);
    }
    backend->cuda.device_total_bytes = total_bytes;
    backend->cuda.device_available_bytes = free_bytes;

    backend->cuda.pinned_staging_bytes =
        (effective->pinned_staging_bytes > 0u)
            ? effective->pinned_staging_bytes
            : (size_t)NLO_CUDA_DEFAULT_PINNED_STAGING_BYTES;
    if (backend->cuda.pinned_staging_bytes > 0u) {
        error = cudaMallocHost(&backend->cuda.pinned_staging_ptr, backend->cuda.pinned_staging_bytes);
        if (error != cudaSuccess) {
            nlo_cuda_backend_shutdown(backend);
            return nlo_cuda_status_from_error(error);
        }
    }

    nlo_vec_status status = nlo_cuda_ensure_reduction_capacity(backend, NLO_CUDA_DEFAULT_REDUCTION_CAPACITY);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_cuda_backend_shutdown(backend);
        return status;
    }

    backend->cuda.initialized = 1;
    return NLO_VEC_STATUS_OK;
}

extern "C" void nlo_cuda_backend_shutdown(nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return;
    }

    if (backend->cuda.compute_stream != NULL) {
        (void)cudaStreamSynchronize(backend->cuda.compute_stream);
    }
    if (backend->cuda.transfer_stream != NULL) {
        (void)cudaStreamSynchronize(backend->cuda.transfer_stream);
    }
    if (backend->cuda.reduction_buffer_a != NULL) {
        (void)cudaFree(backend->cuda.reduction_buffer_a);
        backend->cuda.reduction_buffer_a = NULL;
    }
    if (backend->cuda.reduction_buffer_b != NULL) {
        (void)cudaFree(backend->cuda.reduction_buffer_b);
        backend->cuda.reduction_buffer_b = NULL;
    }
    if (backend->cuda.pinned_staging_ptr != NULL) {
        (void)cudaFreeHost(backend->cuda.pinned_staging_ptr);
        backend->cuda.pinned_staging_ptr = NULL;
    }
    if (backend->cuda.timing_start != NULL) {
        (void)cudaEventDestroy(backend->cuda.timing_start);
        backend->cuda.timing_start = NULL;
    }
    if (backend->cuda.timing_stop != NULL) {
        (void)cudaEventDestroy(backend->cuda.timing_stop);
        backend->cuda.timing_stop = NULL;
    }
    if (backend->cuda.compute_stream != NULL) {
        (void)cudaStreamDestroy(backend->cuda.compute_stream);
        backend->cuda.compute_stream = NULL;
    }
    if (backend->cuda.transfer_stream != NULL) {
        (void)cudaStreamDestroy(backend->cuda.transfer_stream);
        backend->cuda.transfer_stream = NULL;
    }

    backend->cuda.reduction_capacity_elements = 0u;
    backend->cuda.pinned_staging_bytes = 0u;
    backend->cuda.device_total_bytes = 0u;
    backend->cuda.device_available_bytes = 0u;
    backend->cuda.initialized = 0;
}

extern "C" nlo_vec_status nlo_cuda_backend_begin_simulation(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_CUDA || !backend->cuda.initialized) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return nlo_cuda_sync_stream(backend->cuda.transfer_stream);
}

extern "C" nlo_vec_status nlo_cuda_backend_end_simulation(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_CUDA || !backend->cuda.initialized) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return nlo_cuda_sync_stream(backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_backend_query_memory_info(
    const nlo_vector_backend* backend,
    nlo_vec_backend_memory_info* out_info
)
{
    if (backend == NULL || out_info == NULL || backend->type != NLO_VECTOR_BACKEND_CUDA) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, backend->cuda.device_ordinal);
    if (error != cudaSuccess) {
        return nlo_cuda_status_from_error(error);
    }

    size_t free_bytes = 0u;
    size_t total_bytes = 0u;
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        return nlo_cuda_status_from_error(error);
    }

    out_info->device_local_total_bytes = total_bytes;
    out_info->device_local_available_bytes = free_bytes;
    out_info->max_storage_buffer_range = total_bytes;
    out_info->max_compute_workgroups_x = (size_t)prop.maxGridSize[0];
    out_info->max_kernel_chunk_bytes = total_bytes;
    return NLO_VEC_STATUS_OK;
}

extern "C" nlo_vec_status nlo_cuda_buffer_create(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    if (backend == NULL || buffer == NULL || buffer->bytes == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    void* ptr = NULL;
    cudaError_t error = cudaMallocManaged(&ptr, buffer->bytes);
    if (error != cudaSuccess) {
        return nlo_cuda_status_from_error(error);
    }

    buffer->host_ptr = ptr;
    return nlo_cuda_prefetch_to_device(backend, ptr, buffer->bytes, backend->cuda.transfer_stream);
}

extern "C" void nlo_cuda_buffer_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    (void)backend;
    if (buffer != NULL && buffer->host_ptr != NULL) {
        (void)cudaFree(buffer->host_ptr);
        buffer->host_ptr = NULL;
    }
}

extern "C" nlo_vec_status nlo_cuda_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
)
{
    if (backend == NULL || buffer == NULL || data == NULL || buffer->host_ptr == NULL || bytes != buffer->bytes) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (backend->cuda.pinned_staging_ptr != NULL && bytes <= backend->cuda.pinned_staging_bytes) {
        memcpy(backend->cuda.pinned_staging_ptr, data, bytes);
        cudaError_t error = cudaMemcpyAsync(buffer->host_ptr,
                                            backend->cuda.pinned_staging_ptr,
                                            bytes,
                                            cudaMemcpyHostToDevice,
                                            backend->cuda.transfer_stream);
        if (error != cudaSuccess) {
            return nlo_cuda_status_from_error(error);
        }
        return nlo_cuda_sync_stream(backend->cuda.transfer_stream);
    }

    memcpy(buffer->host_ptr, data, bytes);
    nlo_vec_status status =
        nlo_cuda_prefetch_to_device(backend, buffer->host_ptr, bytes, backend->cuda.transfer_stream);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    return nlo_cuda_sync_stream(backend->cuda.transfer_stream);
}

extern "C" nlo_vec_status nlo_cuda_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
)
{
    if (backend == NULL || buffer == NULL || data == NULL || buffer->host_ptr == NULL || bytes != buffer->bytes) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_cuda_sync_stream(backend->cuda.compute_stream);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_cuda_prefetch_to_host(buffer->host_ptr, bytes, backend->cuda.transfer_stream);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_cuda_sync_stream(backend->cuda.transfer_stream);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    memcpy(data, buffer->host_ptr, bytes);
    return NLO_VEC_STATUS_OK;
}

static __device__ inline thrust::complex<double> nlo_cuda_load_complex(const nlo_complex& value)
{
    return thrust::complex<double>(value.re, value.im);
}

static __device__ inline nlo_complex nlo_cuda_store_complex(const thrust::complex<double>& value)
{
    nlo_complex out;
    out.re = value.real();
    out.im = value.imag();
    return out;
}

static __global__ void nlo_cuda_real_fill_kernel(double* dst, size_t count, double value)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] = value;
    }
}

static __global__ void nlo_cuda_real_copy_kernel(double* dst, const double* src, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

static __global__ void nlo_cuda_real_mul_kernel(double* dst, const double* src, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] *= src[idx];
    }
}

static __global__ void nlo_cuda_complex_fill_kernel(nlo_complex* dst, size_t count, nlo_complex value)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] = value;
    }
}

static __global__ void nlo_cuda_complex_copy_kernel(nlo_complex* dst, const nlo_complex* src, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

static __global__ void nlo_cuda_complex_magnitude_squared_kernel(
    const nlo_complex* src,
    nlo_complex* dst,
    size_t count
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        const double re = src[idx].re;
        const double im = src[idx].im;
        dst[idx].re = (re * re) + (im * im);
        dst[idx].im = 0.0;
    }
}

static __global__ void nlo_cuda_complex_scalar_mul_kernel(
    nlo_complex* dst,
    size_t count,
    nlo_complex alpha
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        const double re = dst[idx].re;
        const double im = dst[idx].im;
        dst[idx].re = (re * alpha.re) - (im * alpha.im);
        dst[idx].im = (re * alpha.im) + (im * alpha.re);
    }
}

static __global__ void nlo_cuda_complex_mul_kernel(nlo_complex* dst, const nlo_complex* src, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        const double lhs_re = dst[idx].re;
        const double lhs_im = dst[idx].im;
        const double rhs_re = src[idx].re;
        const double rhs_im = src[idx].im;
        dst[idx].re = (lhs_re * rhs_re) - (lhs_im * rhs_im);
        dst[idx].im = (lhs_re * rhs_im) + (lhs_im * rhs_re);
    }
}

static __global__ void nlo_cuda_complex_add_kernel(nlo_complex* dst, const nlo_complex* src, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx].re += src[idx].re;
        dst[idx].im += src[idx].im;
    }
}

static __global__ void nlo_cuda_complex_axpy_real_kernel(
    nlo_complex* dst,
    const double* src,
    size_t count,
    nlo_complex alpha
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx].re += alpha.re * src[idx];
        dst[idx].im += alpha.im * src[idx];
    }
}

static __global__ void nlo_cuda_complex_axpy_inplace_real_kernel(
    nlo_complex* dst,
    const nlo_complex* src,
    size_t count,
    double alpha
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx].re += alpha * src[idx].re;
        dst[idx].im += alpha * src[idx].im;
    }
}

static __global__ void nlo_cuda_complex_affine_comb2_real_kernel(
    nlo_complex* dst,
    const nlo_complex* a,
    double alpha,
    const nlo_complex* b,
    double beta,
    size_t count
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx].re = (alpha * a[idx].re) + (beta * b[idx].re);
        dst[idx].im = (alpha * a[idx].im) + (beta * b[idx].im);
    }
}

static __global__ void nlo_cuda_complex_affine_comb3_real_kernel(
    nlo_complex* dst,
    const nlo_complex* a,
    double alpha,
    const nlo_complex* b,
    double beta,
    const nlo_complex* c,
    double gamma,
    size_t count
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx].re = (alpha * a[idx].re) + (beta * b[idx].re) + (gamma * c[idx].re);
        dst[idx].im = (alpha * a[idx].im) + (beta * b[idx].im) + (gamma * c[idx].im);
    }
}

static __global__ void nlo_cuda_complex_affine_comb4_real_kernel(
    nlo_complex* dst,
    const nlo_complex* a,
    double alpha,
    const nlo_complex* b,
    double beta,
    const nlo_complex* c,
    double gamma,
    const nlo_complex* d,
    double delta,
    size_t count
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx].re = (alpha * a[idx].re) + (beta * b[idx].re) + (gamma * c[idx].re) + (delta * d[idx].re);
        dst[idx].im = (alpha * a[idx].im) + (beta * b[idx].im) + (gamma * c[idx].im) + (delta * d[idx].im);
    }
}

static __global__ void nlo_cuda_complex_embedded_error_pair_real_kernel(
    nlo_complex* fine_out,
    nlo_complex* coarse_out,
    const nlo_complex* base,
    const nlo_complex* stage_k4,
    double fine_k4_coeff,
    double coarse_k4_coeff,
    const nlo_complex* stage_k5,
    double coarse_k5_coeff,
    size_t count
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        fine_out[idx].re = base[idx].re + (fine_k4_coeff * stage_k4[idx].re);
        fine_out[idx].im = base[idx].im + (fine_k4_coeff * stage_k4[idx].im);
        coarse_out[idx].re =
            base[idx].re + (coarse_k4_coeff * stage_k4[idx].re) + (coarse_k5_coeff * stage_k5[idx].re);
        coarse_out[idx].im =
            base[idx].im + (coarse_k4_coeff * stage_k4[idx].im) + (coarse_k5_coeff * stage_k5[idx].im);
    }
}

static __global__ void nlo_cuda_complex_lerp_kernel(
    nlo_complex* dst,
    const nlo_complex* a,
    const nlo_complex* b,
    double alpha,
    size_t count
)
{
    const double beta = 1.0 - alpha;
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx].re = (beta * a[idx].re) + (alpha * b[idx].re);
        dst[idx].im = (beta * a[idx].im) + (alpha * b[idx].im);
    }
}

static nlo_vec_status nlo_cuda_launch_1d(size_t count, cudaStream_t stream)
{
    (void)count;
    (void)stream;
    return nlo_cuda_peek_status();
}

extern "C" nlo_vec_status nlo_cuda_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_real_fill_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (double*)dst->host_ptr,
        dst->length,
        value);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_real_copy(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_real_copy_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (double*)dst->host_ptr,
        (const double*)src->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_real_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_real_mul_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (double*)dst->host_ptr,
        (const double*)src->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_fill(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex value
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_fill_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length,
        value);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_copy(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_copy_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)src->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_magnitude_squared_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (const nlo_complex*)src->host_ptr,
        (nlo_complex*)dst->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_scalar_mul_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length,
        alpha);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_mul_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)src->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_add_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)src->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

static __global__ void nlo_cuda_complex_real_pow_kernel(nlo_complex* dst, size_t count, double exponent)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        const thrust::complex<double> value = nlo_cuda_load_complex(dst[idx]);
        dst[idx] = nlo_cuda_store_complex(thrust::pow(value, exponent));
    }
}

static __global__ void nlo_cuda_complex_pow_elementwise_kernel(
    nlo_complex* dst,
    const nlo_complex* exponent,
    size_t count
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        const thrust::complex<double> value = nlo_cuda_load_complex(dst[idx]);
        const thrust::complex<double> power = nlo_cuda_load_complex(exponent[idx]);
        dst[idx] = nlo_cuda_store_complex(thrust::pow(value, power));
    }
}

static __global__ void nlo_cuda_complex_exp_kernel(nlo_complex* dst, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] = nlo_cuda_store_complex(thrust::exp(nlo_cuda_load_complex(dst[idx])));
    }
}

static __global__ void nlo_cuda_complex_log_kernel(nlo_complex* dst, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] = nlo_cuda_store_complex(thrust::log(nlo_cuda_load_complex(dst[idx])));
    }
}

static __global__ void nlo_cuda_complex_sin_kernel(nlo_complex* dst, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] = nlo_cuda_store_complex(thrust::sin(nlo_cuda_load_complex(dst[idx])));
    }
}

static __global__ void nlo_cuda_complex_cos_kernel(nlo_complex* dst, size_t count)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx] = nlo_cuda_store_complex(thrust::cos(nlo_cuda_load_complex(dst[idx])));
    }
}

static __global__ void nlo_cuda_axis_unshifted_kernel(nlo_complex* dst, size_t count, double step)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        const size_t positive_limit = (count - 1u) / 2u;
        const double value =
            (idx <= positive_limit) ? ((double)idx * step) : (-((double)count - (double)idx) * step);
        dst[idx].re = value;
        dst[idx].im = 0.0;
    }
}

static __global__ void nlo_cuda_axis_centered_kernel(nlo_complex* dst, size_t count, double delta)
{
    const double center = 0.5 * (double)(count - 1u);
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < count) {
        dst[idx].re = ((double)idx - center) * delta;
        dst[idx].im = 0.0;
    }
}

static __global__ void nlo_cuda_mesh_from_axis_tfast_kernel(
    nlo_complex* dst,
    const nlo_complex* axis,
    size_t nt,
    size_t ny,
    size_t nx,
    uint32_t axis_kind
)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t total = nt * ny * nx;
    if (idx >= total) {
        return;
    }

    const size_t t = idx % nt;
    const size_t xy = idx / nt;
    const size_t y = xy % ny;
    const size_t x = xy / ny;

    size_t axis_idx = 0u;
    if (axis_kind == (uint32_t)NLO_VEC_MESH_AXIS_T) {
        axis_idx = t;
    } else if (axis_kind == (uint32_t)NLO_VEC_MESH_AXIS_Y) {
        axis_idx = y;
    } else {
        axis_idx = x;
    }

    dst[idx] = axis[axis_idx];
}

extern "C" nlo_vec_status nlo_cuda_op_complex_axpy_inplace_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    double alpha
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_axpy_inplace_real_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)src->host_ptr,
        dst->length,
        alpha);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_affine_comb2_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_affine_comb2_real_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)a->host_ptr,
        alpha,
        (const nlo_complex*)b->host_ptr,
        beta,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_affine_comb3_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta,
    const nlo_vec_buffer* c,
    double gamma
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_affine_comb3_real_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)a->host_ptr,
        alpha,
        (const nlo_complex*)b->host_ptr,
        beta,
        (const nlo_complex*)c->host_ptr,
        gamma,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_affine_comb4_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta,
    const nlo_vec_buffer* c,
    double gamma,
    const nlo_vec_buffer* d,
    double delta
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_affine_comb4_real_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)a->host_ptr,
        alpha,
        (const nlo_complex*)b->host_ptr,
        beta,
        (const nlo_complex*)c->host_ptr,
        gamma,
        (const nlo_complex*)d->host_ptr,
        delta,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_embedded_error_pair_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* fine_out,
    nlo_vec_buffer* coarse_out,
    const nlo_vec_buffer* base,
    const nlo_vec_buffer* stage_k4,
    double fine_k4_coeff,
    double coarse_k4_coeff,
    const nlo_vec_buffer* stage_k5,
    double coarse_k5_coeff
)
{
    const uint32_t blocks =
        (uint32_t)((fine_out->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_embedded_error_pair_real_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)fine_out->host_ptr,
        (nlo_complex*)coarse_out->host_ptr,
        (const nlo_complex*)base->host_ptr,
        (const nlo_complex*)stage_k4->host_ptr,
        fine_k4_coeff,
        coarse_k4_coeff,
        (const nlo_complex*)stage_k5->host_ptr,
        coarse_k5_coeff,
        fine_out->length);
    return nlo_cuda_launch_1d(fine_out->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_lerp(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    double alpha
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_lerp_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)a->host_ptr,
        (const nlo_complex*)b->host_ptr,
        alpha,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_axpy_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    nlo_complex alpha
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_axpy_real_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const double*)src->host_ptr,
        dst->length,
        alpha);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_real_pow_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length,
        exponent);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_pow_elementwise_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* exponent
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_pow_elementwise_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)exponent->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_exp_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_log_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_log_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_sin_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_sin_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_cos_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_complex_cos_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_axis_unshifted_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
)
{
    const double step = 6.283185307179586476925286766559 / ((double)dst->length * delta);
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_axis_unshifted_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length,
        step);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_axis_centered_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
)
{
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_axis_centered_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        dst->length,
        delta);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

extern "C" nlo_vec_status nlo_cuda_op_complex_mesh_from_axis_tfast(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* axis,
    size_t nt,
    size_t ny,
    nlo_vec_mesh_axis axis_kind
)
{
    const size_t nx = dst->length / (nt * ny);
    const uint32_t blocks = (uint32_t)((dst->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE);
    nlo_cuda_mesh_from_axis_tfast_kernel<<<blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (nlo_complex*)dst->host_ptr,
        (const nlo_complex*)axis->host_ptr,
        nt,
        ny,
        nx,
        (uint32_t)axis_kind);
    return nlo_cuda_launch_1d(dst->length, backend->cuda.compute_stream);
}

static __global__ void nlo_cuda_weighted_rms_reduce_kernel(
    const nlo_complex* fine,
    const nlo_complex* coarse,
    size_t count,
    double atol,
    double rtol,
    nlo_cuda_weighted_pair* partial
)
{
    __shared__ double local_num[NLO_CUDA_BLOCK_SIZE];
    __shared__ double local_den[NLO_CUDA_BLOCK_SIZE];

    const size_t global_idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double numerator = 0.0;
    double denominator = 0.0;
    if (global_idx < count) {
        const double fine_re = fine[global_idx].re;
        const double fine_im = fine[global_idx].im;
        const double coarse_re = coarse[global_idx].re;
        const double coarse_im = coarse[global_idx].im;

        const double diff_re = fine_re - coarse_re;
        const double diff_im = fine_im - coarse_im;
        numerator = (diff_re * diff_re) + (diff_im * diff_im);

        const double fine_abs = sqrt((fine_re * fine_re) + (fine_im * fine_im));
        const double weight = atol + (rtol * fine_abs);
        denominator = weight * weight;
    }

    local_num[threadIdx.x] = numerator;
    local_den[threadIdx.x] = denominator;
    __syncthreads();

    for (uint32_t stride = NLO_CUDA_BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            local_num[threadIdx.x] += local_num[threadIdx.x + stride];
            local_den[threadIdx.x] += local_den[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        partial[blockIdx.x].sum_sq_diff = local_num[0];
        partial[blockIdx.x].sum_sq_weight = local_den[0];
    }
}

static __global__ void nlo_cuda_weighted_pair_reduce_kernel(
    const nlo_cuda_weighted_pair* input,
    size_t count,
    nlo_cuda_weighted_pair* output
)
{
    __shared__ double local_num[NLO_CUDA_BLOCK_SIZE];
    __shared__ double local_den[NLO_CUDA_BLOCK_SIZE];

    const size_t global_idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double numerator = 0.0;
    double denominator = 0.0;
    if (global_idx < count) {
        numerator = input[global_idx].sum_sq_diff;
        denominator = input[global_idx].sum_sq_weight;
    }

    local_num[threadIdx.x] = numerator;
    local_den[threadIdx.x] = denominator;
    __syncthreads();

    for (uint32_t stride = NLO_CUDA_BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            local_num[threadIdx.x] += local_num[threadIdx.x + stride];
            local_den[threadIdx.x] += local_den[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        output[blockIdx.x].sum_sq_diff = local_num[0];
        output[blockIdx.x].sum_sq_weight = local_den[0];
    }
}

static __global__ void nlo_cuda_relative_error_reduce_kernel(
    const nlo_complex* current,
    const nlo_complex* previous,
    size_t count,
    double epsilon,
    double* partial
)
{
    __shared__ double local_max[NLO_CUDA_BLOCK_SIZE];
    const size_t global_idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    double max_ratio = 0.0;
    if (global_idx < count) {
        const double curr_re = current[global_idx].re;
        const double curr_im = current[global_idx].im;
        const double prev_re = previous[global_idx].re;
        const double prev_im = previous[global_idx].im;
        const double diff_re = curr_re - prev_re;
        const double diff_im = curr_im - prev_im;
        const double diff_sq = (diff_re * diff_re) + (diff_im * diff_im);
        const double prev_sq = (prev_re * prev_re) + (prev_im * prev_im);
        const double denom = (prev_sq > epsilon) ? prev_sq : epsilon;
        max_ratio = diff_sq / denom;
    }

    local_max[threadIdx.x] = max_ratio;
    __syncthreads();

    for (uint32_t stride = NLO_CUDA_BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride && local_max[threadIdx.x + stride] > local_max[threadIdx.x]) {
            local_max[threadIdx.x] = local_max[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        partial[blockIdx.x] = local_max[0];
    }
}

static __global__ void nlo_cuda_relative_error_partial_reduce_kernel(
    const double* input,
    size_t count,
    double* output
)
{
    __shared__ double local_max[NLO_CUDA_BLOCK_SIZE];
    const size_t global_idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    double max_ratio = 0.0;
    if (global_idx < count) {
        max_ratio = input[global_idx];
    }
    local_max[threadIdx.x] = max_ratio;
    __syncthreads();

    for (uint32_t stride = NLO_CUDA_BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride && local_max[threadIdx.x + stride] > local_max[threadIdx.x]) {
            local_max[threadIdx.x] = local_max[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        output[blockIdx.x] = local_max[0];
    }
}

extern "C" nlo_vec_status nlo_cuda_op_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
)
{
    const size_t blocks = (fine->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE;
    nlo_vec_status status = nlo_cuda_ensure_reduction_capacity(backend, blocks);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    nlo_cuda_weighted_pair* partial_a = (nlo_cuda_weighted_pair*)backend->cuda.reduction_buffer_a;
    nlo_cuda_weighted_pair* partial_b = (nlo_cuda_weighted_pair*)backend->cuda.reduction_buffer_b;

    nlo_cuda_weighted_rms_reduce_kernel<<<(uint32_t)blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (const nlo_complex*)fine->host_ptr,
        (const nlo_complex*)coarse->host_ptr,
        fine->length,
        atol,
        rtol,
        partial_a);
    status = nlo_cuda_peek_status();
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    size_t current_count = blocks;
    while (current_count > 1u) {
        const size_t next_blocks = (current_count + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE;
        nlo_cuda_weighted_pair_reduce_kernel<<<(uint32_t)next_blocks,
                                               NLO_CUDA_BLOCK_SIZE,
                                               0,
                                               backend->cuda.compute_stream>>>(partial_a,
                                                                              current_count,
                                                                              partial_b);
        status = nlo_cuda_peek_status();
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_cuda_weighted_pair* tmp = partial_a;
        partial_a = partial_b;
        partial_b = tmp;
        current_count = next_blocks;
    }

    status = nlo_cuda_sync_stream(backend->cuda.compute_stream);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    if (partial_a[0].sum_sq_weight <= 0.0) {
        *out_error = 0.0;
        return NLO_VEC_STATUS_OK;
    }

    const double ratio = partial_a[0].sum_sq_diff / partial_a[0].sum_sq_weight;
    *out_error = sqrt((ratio > 0.0) ? ratio : 0.0);
    return NLO_VEC_STATUS_OK;
}

extern "C" nlo_vec_status nlo_cuda_op_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    const size_t blocks = (current->length + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE;
    nlo_vec_status status = nlo_cuda_ensure_reduction_capacity(backend, blocks);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    double* partial_a = (double*)backend->cuda.reduction_buffer_a;
    double* partial_b = (double*)backend->cuda.reduction_buffer_b;
    nlo_cuda_relative_error_reduce_kernel<<<(uint32_t)blocks, NLO_CUDA_BLOCK_SIZE, 0, backend->cuda.compute_stream>>>(
        (const nlo_complex*)current->host_ptr,
        (const nlo_complex*)previous->host_ptr,
        current->length,
        epsilon,
        partial_a);
    status = nlo_cuda_peek_status();
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    size_t current_count = blocks;
    while (current_count > 1u) {
        const size_t next_blocks = (current_count + (NLO_CUDA_BLOCK_SIZE - 1u)) / NLO_CUDA_BLOCK_SIZE;
        nlo_cuda_relative_error_partial_reduce_kernel<<<(uint32_t)next_blocks,
                                                        NLO_CUDA_BLOCK_SIZE,
                                                        0,
                                                        backend->cuda.compute_stream>>>(partial_a,
                                                                                       current_count,
                                                                                       partial_b);
        status = nlo_cuda_peek_status();
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        double* tmp = partial_a;
        partial_a = partial_b;
        partial_b = tmp;
        current_count = next_blocks;
    }

    status = nlo_cuda_sync_stream(backend->cuda.compute_stream);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    *out_error = sqrt((partial_a[0] > 0.0) ? partial_a[0] : 0.0);
    return NLO_VEC_STATUS_OK;
}

#endif
