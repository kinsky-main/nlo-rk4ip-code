/**
 * @file vk_backend_ops.c
 * @brief Public Vulkan vector operation wrappers.
 */

#include "numerics/vk_backend_internal.h"

nlo_vec_status nlo_vk_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_REAL_FILL,
                                  NLO_PERF_EVENT_COUNT,
                                  dst,
                                  NULL,
                                  sizeof(double),
                                  dst->length,
                                  value,
                                  0.0);
}

nlo_vec_status nlo_vk_op_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    return nlo_vk_copy_buffer_chunked(backend, src->vk_buffer, dst->vk_buffer, dst->bytes);
}

nlo_vec_status nlo_vk_op_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_REAL_MUL_INPLACE,
                                  NLO_PERF_EVENT_COUNT,
                                  dst,
                                  src,
                                  sizeof(double),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_FILL,
                                  NLO_PERF_EVENT_COUNT,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  NLO_RE(value),
                                  NLO_IM(value));
}

nlo_vec_status nlo_vk_op_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    return nlo_vk_copy_buffer_chunked(backend, src->vk_buffer, dst->vk_buffer, dst->bytes);
}

nlo_vec_status nlo_vk_op_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_MAGNITUDE_SQUARED,
                                  NLO_PERF_EVENT_VEC_MAGNITUDE_SQUARED,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_SCALAR_MUL_INPLACE,
                                  NLO_PERF_EVENT_VEC_SCALAR_MUL,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  NLO_RE(alpha),
                                  NLO_IM(alpha));
}

nlo_vec_status nlo_vk_op_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_MUL_INPLACE,
                                  NLO_PERF_EVENT_VEC_MUL,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_ADD_INPLACE,
                                  NLO_PERF_EVENT_VEC_ADD,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_axpy_inplace_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    double alpha
)
{
    return nlo_vk_dispatch_complex_axpy_inplace_real(backend, dst, src, alpha);
}

nlo_vec_status nlo_vk_op_complex_affine_comb2_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta
)
{
    return nlo_vk_dispatch_complex_affine_comb2_real(backend, dst, a, alpha, b, beta);
}

nlo_vec_status nlo_vk_op_complex_affine_comb3_real(
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
    return nlo_vk_dispatch_complex_affine_comb3_real(backend, dst, a, alpha, b, beta, c, gamma);
}

nlo_vec_status nlo_vk_op_complex_affine_comb4_real(
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
    return nlo_vk_dispatch_complex_affine_comb4_real(backend, dst, a, alpha, b, beta, c, gamma, d, delta);
}

nlo_vec_status nlo_vk_op_complex_embedded_error_pair_real(
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
    return nlo_vk_dispatch_complex_embedded_error_pair_real(backend,
                                                            fine_out,
                                                            coarse_out,
                                                            base,
                                                            stage_k4,
                                                            fine_k4_coeff,
                                                            coarse_k4_coeff,
                                                            stage_k5,
                                                            coarse_k5_coeff);
}

nlo_vec_status nlo_vk_op_complex_lerp(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    double alpha
)
{
    return nlo_vk_dispatch_complex_lerp(backend, dst, a, b, alpha);
}

nlo_vec_status nlo_vk_op_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_EXP_INPLACE,
                                  NLO_PERF_EVENT_COUNT,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_REAL_POW_INPLACE,
                                  NLO_PERF_EVENT_COUNT,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  exponent,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    if (epsilon <= 0.0) {
        epsilon = 1e-12;
    }
    return nlo_vk_dispatch_complex_relative_error(backend, current, previous, epsilon, out_error);
}

nlo_vec_status nlo_vk_op_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
)
{
    if (atol < 0.0) {
        atol = 0.0;
    }
    if (rtol < 0.0) {
        rtol = 0.0;
    }
    if (atol == 0.0 && rtol == 0.0) {
        rtol = 1e-6;
    }
    return nlo_vk_dispatch_complex_weighted_rms_error(backend, fine, coarse, atol, rtol, out_error);
}

nlo_vec_status nlo_vk_op_complex_axis_unshifted_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
)
{
    if (!(delta > 0.0)) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_AXIS_UNSHIFTED_FROM_DELTA,
                                  NLO_PERF_EVENT_COUNT,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  delta,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_axis_centered_from_delta(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double delta
)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_AXIS_CENTERED_FROM_DELTA,
                                  NLO_PERF_EVENT_COUNT,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  delta,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_mesh_from_axis_tfast(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* axis,
    size_t nt,
    size_t ny,
    nlo_vec_mesh_axis axis_kind
)
{
    nlo_vec_status status = nlo_vec_validate_buffer(backend, dst, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_vec_validate_buffer(backend, axis, NLO_VEC_KIND_COMPLEX64);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    if (nt == 0u || ny == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if ((dst->length % nt) != 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    const size_t xy_points = dst->length / nt;
    if ((xy_points % ny) != 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    const size_t nx = xy_points / ny;
    if (nx == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    size_t expected_axis_length = 0u;
    nlo_vk_kernel_id kernel_id = NLO_VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_T;
    if (axis_kind == NLO_VEC_MESH_AXIS_T) {
        expected_axis_length = nt;
        kernel_id = NLO_VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_T;
    } else if (axis_kind == NLO_VEC_MESH_AXIS_Y) {
        expected_axis_length = ny;
        kernel_id = NLO_VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_Y;
    } else if (axis_kind == NLO_VEC_MESH_AXIS_X) {
        expected_axis_length = nx;
        kernel_id = NLO_VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_X;
    } else {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (axis->length != expected_axis_length) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    return nlo_vk_dispatch_kernel(backend,
                                  kernel_id,
                                  NLO_PERF_EVENT_COUNT,
                                  dst,
                                  axis,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  (double)nt,
                                  (double)ny);
}
