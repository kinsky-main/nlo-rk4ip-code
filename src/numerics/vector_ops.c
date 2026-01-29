/**
 * @file vector_ops.c
 * @dir src/numerics
 * @brief Vector operations for numerical kernels.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "numerics/vector_ops.h"

void nlo_real_fill(double *dst, size_t n, double value)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        dst[i] = value;
    }
}

void nlo_real_copy(double *dst, const double *src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

void nlo_real_mul_inplace(double *dst, const double *src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        dst[i] *= src[i];
    }
}

void nlo_real_pow_int(const double *base, double *out, size_t n, unsigned int power)
{
    if (base == NULL || out == NULL) {
        return;
    }

    if (power == 0U) {
        nlo_real_fill(out, n, 1.0);
        return;
    }

    nlo_real_fill(out, n, 1.0);
    for (unsigned int p = 0; p < power; ++p) {
        for (size_t i = 0; i < n; ++i) {
            out[i] *= base[i];
        }
    }
}

void nlo_complex_fill(nlo_complex *dst, size_t n, nlo_complex value)
{
    if (dst == NULL) {
        return;
    }

    const double value_re = NLO_RE(value);
    const double value_im = NLO_IM(value);

    for (size_t i = 0; i < n; ++i) {
        dst[i] = nlo_make(value_re, value_im);
    }
}

void nlo_complex_axpy_real(nlo_complex *dst, const double *src, nlo_complex alpha, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    const double alpha_re = NLO_RE(alpha);
    const double alpha_im = NLO_IM(alpha);

    for (size_t i = 0; i < n; ++i) {
        const double term = src[i];
        dst[i] = nlo_make(NLO_RE(dst[i]) + alpha_re * term,
                          NLO_IM(dst[i]) + alpha_im * term);
    }
}

void nlo_complex_mul_inplace(nlo_complex *dst, const nlo_complex *src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        dst[i] = nlo_mul(dst[i], src[i]);
    }
}
