/**
 * @file cuda_backend_stubs.c
 * @brief Non-CUDA stub implementations for CUDA backend wrappers.
 */

#include "backend/cuda_backend_internal.h"

#if !NLO_ENABLE_CUDA_BACKEND

nlo_vec_status nlo_cuda_backend_init(nlo_vector_backend* backend, const nlo_cuda_backend_config* config)
{
    (void)backend;
    (void)config;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

void nlo_cuda_backend_shutdown(nlo_vector_backend* backend)
{
    (void)backend;
}

nlo_vec_status nlo_cuda_backend_begin_simulation(nlo_vector_backend* backend)
{
    (void)backend;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_backend_end_simulation(nlo_vector_backend* backend)
{
    (void)backend;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_backend_query_memory_info(
    const nlo_vector_backend* backend,
    nlo_vec_backend_memory_info* out_info
)
{
    (void)backend;
    (void)out_info;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_buffer_create(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    (void)backend;
    (void)buffer;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

void nlo_cuda_buffer_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    (void)backend;
    (void)buffer;
}

nlo_vec_status nlo_cuda_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
)
{
    (void)backend;
    (void)buffer;
    (void)data;
    (void)bytes;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
)
{
    (void)backend;
    (void)buffer;
    (void)data;
    (void)bytes;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

#define NLO_CUDA_STUB_UNARY(fn_name, arg_type)                     \
    nlo_vec_status fn_name(nlo_vector_backend* backend, arg_type a) \
    {                                                               \
        (void)backend;                                              \
        (void)a;                                                    \
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;                  \
    }

nlo_vec_status nlo_cuda_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value)
{
    (void)backend;
    (void)dst;
    (void)value;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_real_copy(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_real_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_fill(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex value
)
{
    (void)backend;
    (void)dst;
    (void)value;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_copy(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
)
{
    (void)backend;
    (void)src;
    (void)dst;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
)
{
    (void)backend;
    (void)dst;
    (void)alpha;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
)
{
    (void)backend;
    (void)dst;
    (void)exponent;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_pow_elementwise_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* exponent
)
{
    (void)backend;
    (void)dst;
    (void)exponent;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

NLO_CUDA_STUB_UNARY(nlo_cuda_op_complex_exp_inplace, nlo_vec_buffer*);
NLO_CUDA_STUB_UNARY(nlo_cuda_op_complex_log_inplace, nlo_vec_buffer*);
NLO_CUDA_STUB_UNARY(nlo_cuda_op_complex_sin_inplace, nlo_vec_buffer*);
NLO_CUDA_STUB_UNARY(nlo_cuda_op_complex_cos_inplace, nlo_vec_buffer*);

nlo_vec_status nlo_cuda_op_complex_axis_unshifted_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
)
{
    (void)backend;
    (void)dst;
    (void)delta;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_axis_centered_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
)
{
    (void)backend;
    (void)dst;
    (void)delta;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_mesh_from_axis_tfast(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* axis,
    size_t nt,
    size_t ny,
    nlo_vec_mesh_axis axis_kind
)
{
    (void)backend;
    (void)dst;
    (void)axis;
    (void)nt;
    (void)ny;
    (void)axis_kind;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
)
{
    (void)backend;
    (void)fine;
    (void)coarse;
    (void)atol;
    (void)rtol;
    (void)out_error;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    (void)backend;
    (void)current;
    (void)previous;
    (void)epsilon;
    (void)out_error;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_axpy_inplace_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    double alpha
)
{
    (void)backend;
    (void)dst;
    (void)src;
    (void)alpha;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_affine_comb2_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta
)
{
    (void)backend;
    (void)dst;
    (void)a;
    (void)alpha;
    (void)b;
    (void)beta;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_affine_comb3_real(
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
    (void)backend;
    (void)dst;
    (void)a;
    (void)alpha;
    (void)b;
    (void)beta;
    (void)c;
    (void)gamma;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

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
)
{
    (void)backend;
    (void)dst;
    (void)a;
    (void)alpha;
    (void)b;
    (void)beta;
    (void)c;
    (void)gamma;
    (void)d;
    (void)delta;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

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
)
{
    (void)backend;
    (void)fine_out;
    (void)coarse_out;
    (void)base;
    (void)stage_k4;
    (void)fine_k4_coeff;
    (void)coarse_k4_coeff;
    (void)stage_k5;
    (void)coarse_k5_coeff;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_lerp(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    double alpha
)
{
    (void)backend;
    (void)dst;
    (void)a;
    (void)b;
    (void)alpha;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

nlo_vec_status nlo_cuda_op_complex_axpy_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    nlo_complex alpha
)
{
    (void)backend;
    (void)dst;
    (void)src;
    (void)alpha;
    return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
}

#endif
