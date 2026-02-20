/**
 * @file vector_ops.c
 * @dir src/numerics
 * @brief Vector operations for numerical kernels (SIMD-accelerated).
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "numerics/vector_ops.h"
#include "backend/nlo_complex_simd.h"
#include <math.h>
#include <simde/x86/avx.h>

// SIMD helpers
static inline size_t nlo_simd_aligned_end(size_t n, size_t width)
{
    return n - (n % width);
}

void nlo_real_fill(double *dst, size_t n, double value)
{
    const simde__m256d v = simde_mm256_set1_pd(value);
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 4); i < simd_end; i += 4)
    {
        simde_mm256_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i)
    {
        dst[i] = value;
    }
}

void nlo_real_copy(double *dst, const double *src, size_t n)
{
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 4); i < simd_end; i += 4)
    {
        simde__m256d v = simde_mm256_loadu_pd(src + i);
        simde_mm256_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i)
    {
        dst[i] = src[i];
    }
}

void nlo_real_mul_inplace(double *dst, const double *src, size_t n)
{
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 4); i < simd_end; i += 4)
    {
        const simde__m256d a = simde_mm256_loadu_pd(dst + i);
        const simde__m256d b = simde_mm256_loadu_pd(src + i);
        simde_mm256_storeu_pd(dst + i, simde_mm256_mul_pd(a, b));
    }
    for (; i < n; ++i)
    {
        dst[i] *= src[i];
    }
}

void nlo_real_pow_int(const double *base, double *out, size_t n, unsigned int power)
{
    if (power == 0U)
    {
        nlo_real_fill(out, n, 1.0);
        return;
    }

    const simde__m256d ones = simde_mm256_set1_pd(1.0);

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 4); i < simd_end; i += 4)
    {
        simde__m256d result = ones;
        simde__m256d current = simde_mm256_loadu_pd(base + i);
        unsigned int exp = power;

        while (exp > 0U)
        {
            if ((exp & 1U) != 0U)
            {
                result = simde_mm256_mul_pd(result, current);
            }
            exp >>= 1U;
            if (exp == 0U)
            {
                break;
            }
            current = simde_mm256_mul_pd(current, current);
        }

        simde_mm256_storeu_pd(out + i, result);
    }

    for (; i < n; ++i)
    {
        double result = 1.0;
        double current = base[i];
        unsigned int exp = power;

        while (exp > 0U)
        {
            if ((exp & 1U) != 0U)
            {
                result *= current;
            }
            exp >>= 1U;
            if (exp == 0U)
            {
                break;
            }
            current *= current;
        }

        out[i] = result;
    }
}

void nlo_complex_fill(nlo_complex *dst, size_t n, nlo_complex value)
{
    const nlo_cpack2d fill = nlo_cpack2d_set1(value);

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        nlo_cpack2d_storeu(dst + i, fill);
    }
    for (; i < n; ++i)
    {
        dst[i] = value;
    }
}

void nlo_complex_copy(nlo_complex *dst, const nlo_complex *src, size_t n)
{
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_loadu(src + i));
    }
    for (; i < n; ++i)
    {
        dst[i] = src[i];
    }
}

void calculate_magnitude_squared(const nlo_complex *src, nlo_complex *dst, size_t n)
{
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_mag2_to_complex(nlo_cpack2d_loadu(src + i)));
    }
    for (; i < n; ++i)
    {
        const double re = NLO_RE(src[i]);
        const double im = NLO_IM(src[i]);
        dst[i] = nlo_make(re * re + im * im, 0.0);
    }
}

void nlo_complex_axpy_real(nlo_complex *dst, const double *src, nlo_complex alpha, size_t n)
{
    const nlo_cpack2d alpha_pack = nlo_cpack2d_set1(alpha);

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        const nlo_cpack2d increment = nlo_cpack2d_scale_real2(alpha_pack, src[i], src[i + 1u]);
        const nlo_cpack2d current = nlo_cpack2d_loadu(dst + i);
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_add(current, increment));
    }
    for (; i < n; ++i)
    {
        const double term = src[i];
        dst[i] = nlo_make(NLO_RE(dst[i]) + NLO_RE(alpha) * term,
                          NLO_IM(dst[i]) + NLO_IM(alpha) * term);
    }
}

void nlo_complex_scalar_mul_inplace(nlo_complex *dst, nlo_complex alpha, size_t n)
{
    const nlo_cpack2d alpha_pack = nlo_cpack2d_set1(alpha);

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        const nlo_cpack2d values = nlo_cpack2d_loadu(dst + i);
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_mul(values, alpha_pack));
    }
    for (; i < n; ++i)
    {
        dst[i] = nlo_mul(dst[i], alpha);
    }
}

void nlo_complex_scalar_mul(nlo_complex *dst, const nlo_complex *src, nlo_complex alpha, size_t n)
{
    const nlo_cpack2d alpha_pack = nlo_cpack2d_set1(alpha);

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        const nlo_cpack2d values = nlo_cpack2d_loadu(src + i);
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_mul(values, alpha_pack));
    }
    for (; i < n; ++i)
    {
        dst[i] = nlo_mul(src[i], alpha);
    }
}

void nlo_complex_mul_inplace(nlo_complex *dst, const nlo_complex *src, size_t n)
{
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        const nlo_cpack2d a_pack = nlo_cpack2d_loadu(dst + i);
        const nlo_cpack2d b_pack = nlo_cpack2d_loadu(src + i);
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_mul(a_pack, b_pack));
    }
    for (; i < n; ++i)
    {
        dst[i] = nlo_mul(dst[i], src[i]);
    }
}

void nlo_complex_mul_vec(nlo_complex *dst, const nlo_complex *a, const nlo_complex *b, size_t n)
{
    if (dst == NULL || a == NULL || b == NULL)
    {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        const nlo_cpack2d va = nlo_cpack2d_loadu(a + i);
        const nlo_cpack2d vb = nlo_cpack2d_loadu(b + i);
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_mul(va, vb));
    }
    for (; i < n; ++i)
    {
        dst[i] = nlo_mul(a[i], b[i]);
    }
}

void nlo_complex_pow(const nlo_complex *base, nlo_complex *out, size_t n, unsigned int exponent)
{
    if (base == NULL || out == NULL)
    {
        return;
    }

    if (exponent == 0U)
    {
        nlo_complex_fill(out, n, nlo_make(1.0, 0.0));
        return;
    }

    const nlo_cpack2d ones = nlo_cpack2d_ones();

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        nlo_cpack2d result = ones;
        nlo_cpack2d current = nlo_cpack2d_loadu(base + i);
        unsigned int exp = exponent;

        while (exp > 0U)
        {
            if ((exp & 1U) != 0U)
            {
                result = nlo_cpack2d_mul(result, current);
            }
            exp >>= 1U;
            if (exp == 0U)
            {
                break;
            }
            current = nlo_cpack2d_mul(current, current);
        }

        nlo_cpack2d_storeu(out + i, result);
    }

    for (; i < n; ++i)
    {
        nlo_complex result = nlo_make(1.0, 0.0);
        nlo_complex current = base[i];
        unsigned int exp = exponent;

        while (exp > 0U)
        {
            if ((exp & 1U) != 0U)
            {
                result = nlo_mul(result, current);
            }
            exp >>= 1U;
            if (exp == 0U)
            {
                break;
            }
            current = nlo_mul(current, current);
        }

        out[i] = result;
    }
}

void nlo_complex_pow_inplace(nlo_complex *dst, size_t n, unsigned int exponent)
{
    nlo_complex_pow(dst, dst, n, exponent);
}

void nlo_complex_pow_elementwise_inplace(
    nlo_complex *dst,
    const nlo_complex *exponent,
    size_t n
)
{
    if (dst == NULL || exponent == NULL)
    {
        return;
    }

    for (size_t i = 0; i < n; ++i)
    {
        const double base_re = NLO_RE(dst[i]);
        const double base_im = NLO_IM(dst[i]);
        const double exp_re = NLO_RE(exponent[i]);
        const double exp_im = NLO_IM(exponent[i]);
        const double radius = hypot(base_re, base_im);

        if (radius == 0.0)
        {
            if (exp_re == 0.0 && exp_im == 0.0)
            {
                dst[i] = nlo_make(1.0, 0.0);
            }
            else if (exp_im == 0.0 && exp_re > 0.0)
            {
                dst[i] = nlo_make(0.0, 0.0);
            }
            else
            {
                dst[i] = nlo_make(0.0, 0.0);
            }
            continue;
        }

        const double theta = atan2(base_im, base_re);
        const double log_radius = log(radius);
        const double scaled_re = (exp_re * log_radius) - (exp_im * theta);
        const double scaled_im = (exp_re * theta) + (exp_im * log_radius);
        const double magnitude = exp(scaled_re);
        dst[i] = nlo_make(magnitude * cos(scaled_im),
                          magnitude * sin(scaled_im));
    }
}

static inline nlo_complex nlo_complex_real_pow_scalar(nlo_complex value, double exponent)
{
    const double re = NLO_RE(value);
    const double im = NLO_IM(value);
    const double radius = hypot(re, im);

    if (radius == 0.0) {
        if (exponent == 0.0) {
            return nlo_make(1.0, 0.0);
        }
        return nlo_make(0.0, 0.0);
    }

    const double angle = atan2(im, re);
    const double scaled_angle = exponent * angle;
    const double scaled_radius = pow(radius, exponent);
    return nlo_make(scaled_radius * cos(scaled_angle),
                    scaled_radius * sin(scaled_angle));
}

void nlo_complex_real_pow(const nlo_complex *base, nlo_complex *out, size_t n, double exponent)
{
    if (base == NULL || out == NULL) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        out[i] = nlo_complex_real_pow_scalar(base[i], exponent);
    }
}

void nlo_complex_real_pow_inplace(nlo_complex *dst, size_t n, double exponent)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        dst[i] = nlo_complex_real_pow_scalar(dst[i], exponent);
    }
}

void nlo_complex_add_inplace(nlo_complex *dst, const nlo_complex *src, size_t n)
{
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        const nlo_cpack2d a_pack = nlo_cpack2d_loadu(dst + i);
        const nlo_cpack2d b_pack = nlo_cpack2d_loadu(src + i);
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_add(a_pack, b_pack));
    }
    for (; i < n; ++i)
    {
        dst[i] = nlo_add(dst[i], src[i]);
    }
}

void nlo_complex_add_vec(nlo_complex *dst, const nlo_complex *a, const nlo_complex *b, size_t n)
{
    if (dst == NULL || a == NULL || b == NULL)
    {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2)
    {
        const nlo_cpack2d va = nlo_cpack2d_loadu(a + i);
        const nlo_cpack2d vb = nlo_cpack2d_loadu(b + i);
        nlo_cpack2d_storeu(dst + i, nlo_cpack2d_add(va, vb));
    }
    for (; i < n; ++i)
    {
        dst[i] = nlo_add(a[i], b[i]);
    }
}

void nlo_complex_exp_inplace(nlo_complex *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        const double re = NLO_RE(dst[i]);
        const double im = NLO_IM(dst[i]);
        const double exp_re = exp(re);
        dst[i] = nlo_make(exp_re * cos(im), exp_re * sin(im));
    }
}

void nlo_complex_log_inplace(nlo_complex *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        const double re = NLO_RE(dst[i]);
        const double im = NLO_IM(dst[i]);
        const double radius = hypot(re, im);
        const double angle = atan2(im, re);
        dst[i] = nlo_make(log(radius), angle);
    }
}

void nlo_complex_sin_inplace(nlo_complex *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        const double re = NLO_RE(dst[i]);
        const double im = NLO_IM(dst[i]);
        dst[i] = nlo_make(sin(re) * cosh(im),
                          cos(re) * sinh(im));
    }
}

void nlo_complex_cos_inplace(nlo_complex *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        const double re = NLO_RE(dst[i]);
        const double im = NLO_IM(dst[i]);
        dst[i] = nlo_make(cos(re) * cosh(im),
                          -sin(re) * sinh(im));
    }
}
