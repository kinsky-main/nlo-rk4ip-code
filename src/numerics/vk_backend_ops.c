/**
 * @file vk_backend_ops.c
 * @brief Public Vulkan vector operation wrappers.
 */

#include "numerics/vk_backend_internal.h"

vec_status vk_op_real_fill(vector_backend* backend, vec_buffer* dst, double value)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_REAL_FILL,
                                  dst,
                                  NULL,
                                  sizeof(double),
                                  dst->length,
                                  value,
                                  0.0);
}

vec_status vk_op_real_copy(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    return vk_copy_buffer_chunked(backend, src->vk_buffer, dst->vk_buffer, dst->bytes);
}

vec_status vk_op_real_mul_inplace(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_REAL_MUL_INPLACE,
                                  dst,
                                  src,
                                  sizeof(double),
                                  dst->length,
                                  0.0,
                                  0.0);
}

vec_status vk_op_complex_fill(vector_backend* backend, vec_buffer* dst, nlo_complex value)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_FILL,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  RE(value),
                                  IM(value));
}

vec_status vk_op_complex_copy(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    return vk_copy_buffer_chunked(backend, src->vk_buffer, dst->vk_buffer, dst->bytes);
}

vec_status vk_op_complex_magnitude_squared(
    vector_backend* backend,
    const vec_buffer* src,
    vec_buffer* dst
)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_MAGNITUDE_SQUARED,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

vec_status vk_op_complex_scalar_mul_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    nlo_complex alpha
)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_SCALAR_MUL_INPLACE,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  RE(alpha),
                                  IM(alpha));
}

vec_status vk_op_complex_mul_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src
)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_MUL_INPLACE,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

vec_status vk_op_complex_add_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src
)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_ADD_INPLACE,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

vec_status vk_op_complex_exp_inplace(vector_backend* backend, vec_buffer* dst)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_EXP_INPLACE,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

vec_status vk_op_complex_real_pow_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    double exponent
)
{
    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_REAL_POW_INPLACE,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  exponent,
                                  0.0);
}

vec_status vk_op_complex_relative_error(
    vector_backend* backend,
    const vec_buffer* current,
    const vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    if (epsilon <= 0.0) {
        epsilon = 1e-12;
    }
    return vk_dispatch_complex_relative_error(backend, current, previous, epsilon, out_error);
}

vec_status vk_op_complex_weighted_rms_error(
    vector_backend* backend,
    const vec_buffer* fine,
    const vec_buffer* coarse,
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
    return vk_dispatch_complex_weighted_rms_error(backend, fine, coarse, atol, rtol, out_error);
}

vec_status vk_op_complex_axis_unshifted_from_delta(
    vector_backend* backend,
    vec_buffer* dst,
    double delta
)
{
    if (!(delta > 0.0)) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_AXIS_UNSHIFTED_FROM_DELTA,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  delta,
                                  0.0);
}

vec_status vk_op_complex_axis_centered_from_delta(
    vector_backend* backend,
    vec_buffer* dst,
    double delta
)
{
    vec_status status = vec_validate_buffer(backend, dst, VEC_KIND_COMPLEX64);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    return vk_dispatch_kernel(backend,
                                  VK_KERNEL_COMPLEX_AXIS_CENTERED_FROM_DELTA,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  delta,
                                  0.0);
}

vec_status vk_op_complex_mesh_from_axis_tfast(
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
    vk_kernel_id kernel_id = VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_T;
    if (axis_kind == VEC_MESH_AXIS_T) {
        expected_axis_length = nt;
        kernel_id = VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_T;
    } else if (axis_kind == VEC_MESH_AXIS_Y) {
        expected_axis_length = ny;
        kernel_id = VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_Y;
    } else if (axis_kind == VEC_MESH_AXIS_X) {
        expected_axis_length = nx;
        kernel_id = VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_X;
    } else {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (axis->length != expected_axis_length) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    return vk_dispatch_kernel(backend,
                                  kernel_id,
                                  dst,
                                  axis,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  (double)nt,
                                  (double)ny);
}
