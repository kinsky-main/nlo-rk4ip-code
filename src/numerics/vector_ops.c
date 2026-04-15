/**
 * @file vector_ops.c
 * @dir src/numerics
 * @brief Vector operations for numerical kernels (CBLAS-backed where available).
 * @author Wenzel Kinsky
 * @date 2026-03-10
 */

#include "numerics/vector_ops.h"
#include <cblas.h>
#include <limits.h>
#include <math.h>
#include <string.h>

static int try_get_blas_length(size_t n, int* out_n)
{
    if (out_n == NULL || n > (size_t)INT_MAX) {
        return 0;
    }

    *out_n = (int)n;
    return 1;
}

void real_fill(double* dst, size_t n, double value)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = value;
    }
}

void real_copy(double* dst, const double* src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    int blas_n = 0;
    if (try_get_blas_length(n, &blas_n)) {
        cblas_dcopy(blas_n, src, 1, dst, 1);
        return;
    }

    memmove(dst, src, n * sizeof(*dst));
}

void real_mul_inplace(double* dst, const double* src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] *= src[i];
    }
}

void real_pow_int(const double* base, double* out, size_t n, unsigned int power)
{
    if (base == NULL || out == NULL) {
        return;
    }

    if (power == 0u) {
        real_fill(out, n, 1.0);
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        double result = 1.0;
        double current = base[i];
        unsigned int exponent = power;

        while (exponent > 0u) {
            if ((exponent & 1u) != 0u) {
                result *= current;
            }
            exponent >>= 1u;
            if (exponent == 0u) {
                break;
            }
            current *= current;
        }

        out[i] = result;
    }
}

void complex_fill(nlo_complex* dst, size_t n, nlo_complex value)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = value;
    }
}

void complex_copy(nlo_complex* dst, const nlo_complex* src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    int blas_n = 0;
    if (try_get_blas_length(n, &blas_n)) {
        cblas_zcopy(blas_n, src, 1, dst, 1);
        return;
    }

    memmove(dst, src, n * sizeof(*dst));
}

void calculate_magnitude_squared(const nlo_complex* src, nlo_complex* dst, size_t n)
{
    if (src == NULL || dst == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double re = RE(src[i]);
        const double im = IM(src[i]);
        dst[i] = make((re * re) + (im * im), 0.0);
    }
}

void complex_axpy_real(nlo_complex* dst, const double* src, nlo_complex alpha, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    int blas_n = 0;
    if (try_get_blas_length(n, &blas_n)) {
        double* dst_lanes = (double*)(void*)dst;
        const double alpha_re = RE(alpha);
        const double alpha_im = IM(alpha);
        cblas_daxpy(blas_n, alpha_re, src, 1, dst_lanes, 2);
        cblas_daxpy(blas_n, alpha_im, src, 1, dst_lanes + 1, 2);
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double term = src[i];
        dst[i] = make(RE(dst[i]) + (RE(alpha) * term),
                          IM(dst[i]) + (IM(alpha) * term));
    }
}

void complex_scalar_mul_inplace(nlo_complex* dst, nlo_complex alpha, size_t n)
{
    if (dst == NULL) {
        return;
    }

    int blas_n = 0;
    if (try_get_blas_length(n, &blas_n)) {
        const double alpha_lanes[2] = { RE(alpha), IM(alpha) };
        cblas_zscal(blas_n, alpha_lanes, dst, 1);
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = mul(dst[i], alpha);
    }
}

void complex_scalar_mul(nlo_complex* dst, const nlo_complex* src, nlo_complex alpha, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    int blas_n = 0;
    if (try_get_blas_length(n, &blas_n)) {
        const double alpha_lanes[2] = { RE(alpha), IM(alpha) };
        cblas_zcopy(blas_n, src, 1, dst, 1);
        cblas_zscal(blas_n, alpha_lanes, dst, 1);
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = mul(src[i], alpha);
    }
}

void complex_mul_inplace(nlo_complex* dst, const nlo_complex* src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = mul(dst[i], src[i]);
    }
}

void complex_mul_vec(nlo_complex* dst, const nlo_complex* a, const nlo_complex* b, size_t n)
{
    if (dst == NULL || a == NULL || b == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = mul(a[i], b[i]);
    }
}

void complex_pow(const nlo_complex* base, nlo_complex* out, size_t n, unsigned int exponent)
{
    if (base == NULL || out == NULL) {
        return;
    }

    if (exponent == 0u) {
        complex_fill(out, n, make(1.0, 0.0));
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        nlo_complex result = make(1.0, 0.0);
        nlo_complex current = base[i];
        unsigned int power = exponent;

        while (power > 0u) {
            if ((power & 1u) != 0u) {
                result = mul(result, current);
            }
            power >>= 1u;
            if (power == 0u) {
                break;
            }
            current = mul(current, current);
        }

        out[i] = result;
    }
}

void complex_pow_inplace(nlo_complex* dst, size_t n, unsigned int exponent)
{
    complex_pow(dst, dst, n, exponent);
}

void complex_pow_elementwise_inplace(
    nlo_complex* dst,
    const nlo_complex* exponent,
    size_t n
)
{
    if (dst == NULL || exponent == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double base_re = RE(dst[i]);
        const double base_im = IM(dst[i]);
        const double exp_re = RE(exponent[i]);
        const double exp_im = IM(exponent[i]);
        const double radius = hypot(base_re, base_im);

        if (radius == 0.0) {
            if (exp_re == 0.0 && exp_im == 0.0) {
                dst[i] = make(1.0, 0.0);
            } else {
                dst[i] = make(0.0, 0.0);
            }
            continue;
        }

        const double theta = atan2(base_im, base_re);
        const double log_radius = log(radius);
        const double scaled_re = (exp_re * log_radius) - (exp_im * theta);
        const double scaled_im = (exp_re * theta) + (exp_im * log_radius);
        const double magnitude = exp(scaled_re);
        dst[i] = make(magnitude * cos(scaled_im),
                          magnitude * sin(scaled_im));
    }
}

static inline nlo_complex complex_real_pow_scalar(nlo_complex value, double exponent)
{
    const double re = RE(value);
    const double im = IM(value);
    const double radius = hypot(re, im);

    if (radius == 0.0) {
        if (exponent == 0.0) {
            return make(1.0, 0.0);
        }
        return make(0.0, 0.0);
    }

    const double angle = atan2(im, re);
    const double scaled_angle = exponent * angle;
    const double scaled_radius = pow(radius, exponent);
    return make(scaled_radius * cos(scaled_angle),
                    scaled_radius * sin(scaled_angle));
}

void complex_real_pow(const nlo_complex* base, nlo_complex* out, size_t n, double exponent)
{
    if (base == NULL || out == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        out[i] = complex_real_pow_scalar(base[i], exponent);
    }
}

void complex_real_pow_inplace(nlo_complex* dst, size_t n, double exponent)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = complex_real_pow_scalar(dst[i], exponent);
    }
}

void complex_add_inplace(nlo_complex* dst, const nlo_complex* src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    int blas_n = 0;
    if (try_get_blas_length(n, &blas_n)) {
        const double one[2] = { 1.0, 0.0 };
        cblas_zaxpy(blas_n, one, src, 1, dst, 1);
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = add(dst[i], src[i]);
    }
}

void complex_add_vec(nlo_complex* dst, const nlo_complex* a, const nlo_complex* b, size_t n)
{
    if (dst == NULL || a == NULL || b == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        dst[i] = add(a[i], b[i]);
    }
}

void complex_exp_inplace(nlo_complex* dst, size_t n)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double re = RE(dst[i]);
        const double im = IM(dst[i]);
        const double exp_re = exp(re);
        dst[i] = make(exp_re * cos(im), exp_re * sin(im));
    }
}

void complex_log_inplace(nlo_complex* dst, size_t n)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double re = RE(dst[i]);
        const double im = IM(dst[i]);
        const double radius = hypot(re, im);
        const double angle = atan2(im, re);
        dst[i] = make(log(radius), angle);
    }
}

void complex_sin_inplace(nlo_complex* dst, size_t n)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double re = RE(dst[i]);
        const double im = IM(dst[i]);
        dst[i] = make(sin(re) * cosh(im),
                          cos(re) * sinh(im));
    }
}

void complex_cos_inplace(nlo_complex* dst, size_t n)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double re = RE(dst[i]);
        const double im = IM(dst[i]);
        dst[i] = make(cos(re) * cosh(im),
                          -sin(re) * sinh(im));
    }
}
