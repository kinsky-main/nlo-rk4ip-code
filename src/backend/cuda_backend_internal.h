/**
 * @file cuda_backend_internal.h
 * @brief Internal CUDA backend wrappers used by C translation units.
 */
#pragma once

#include "backend/vector_backend_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

nlo_vec_status nlo_cuda_backend_init(nlo_vector_backend* backend, const nlo_cuda_backend_config* config);
void nlo_cuda_backend_shutdown(nlo_vector_backend* backend);
nlo_vec_status nlo_cuda_backend_begin_simulation(nlo_vector_backend* backend);
nlo_vec_status nlo_cuda_backend_end_simulation(nlo_vector_backend* backend);
nlo_vec_status nlo_cuda_backend_query_memory_info(
    const nlo_vector_backend* backend,
    nlo_vec_backend_memory_info* out_info
);
nlo_vec_status nlo_cuda_buffer_create(nlo_vector_backend* backend, nlo_vec_buffer* buffer);
void nlo_cuda_buffer_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer);
nlo_vec_status nlo_cuda_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
);
nlo_vec_status nlo_cuda_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
);
nlo_vec_status nlo_cuda_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value);
nlo_vec_status nlo_cuda_op_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_cuda_op_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_cuda_op_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value);
nlo_vec_status nlo_cuda_op_complex_copy(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
);
nlo_vec_status nlo_cuda_op_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
);
nlo_vec_status nlo_cuda_op_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
);
nlo_vec_status nlo_cuda_op_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
);
nlo_vec_status nlo_cuda_op_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
);
nlo_vec_status nlo_cuda_op_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
);
nlo_vec_status nlo_cuda_op_complex_pow_elementwise_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* exponent
);
nlo_vec_status nlo_cuda_op_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
nlo_vec_status nlo_cuda_op_complex_log_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
nlo_vec_status nlo_cuda_op_complex_sin_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
nlo_vec_status nlo_cuda_op_complex_cos_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst);
nlo_vec_status nlo_cuda_op_complex_axis_unshifted_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
);
nlo_vec_status nlo_cuda_op_complex_axis_centered_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
);
nlo_vec_status nlo_cuda_op_complex_mesh_from_axis_tfast(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* axis,
    size_t nt,
    size_t ny,
    nlo_vec_mesh_axis axis_kind
);
nlo_vec_status nlo_cuda_op_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
);
nlo_vec_status nlo_cuda_op_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
);
nlo_vec_status nlo_cuda_op_complex_axpy_inplace_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    double alpha
);
nlo_vec_status nlo_cuda_op_complex_affine_comb2_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta
);
nlo_vec_status nlo_cuda_op_complex_affine_comb3_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta,
    const nlo_vec_buffer* c,
    double gamma
);
nlo_vec_status nlo_cuda_op_complex_affine_comb4_real(
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
);
nlo_vec_status nlo_cuda_op_complex_embedded_error_pair_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* fine_out,
    nlo_vec_buffer* coarse_out,
    const nlo_vec_buffer* base,
    const nlo_vec_buffer* stage_k4,
    double fine_k4_coeff,
    double coarse_k4_coeff,
    const nlo_vec_buffer* stage_k5,
    double coarse_k5_coeff
);
nlo_vec_status nlo_cuda_op_complex_lerp(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    double alpha
);

#ifdef __cplusplus
}
#endif
