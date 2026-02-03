/**
 * @file vector_ops.c
 * @dir src/numerics
 * @brief Vector operations for numerical kernels (SIMD-accelerated).
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "numerics/vector_ops.h"
#include <math.h>
#include <simde/x86/avx.h>
#include "vector_ops.h"

// SIMD helpers
static inline size_t nlo_simd_aligned_end(size_t n, size_t width)
{
    return n - (n % width);
}

// Complex multiplication for two complex numbers packed as [re0, im0, re1, im1].
static inline void nlo_complex_mul_packed(const simde__m256d *a,
                                          const simde__m256d *b,
                                          simde__m256d *out)
{
    // Duplicate real parts of a: [re0, re0, re1, re1]
    const simde__m256d a_re = simde_mm256_movedup_pd(*a);
    // Duplicate imag parts of a: [im0, im0, im1, im1]
    const simde__m256d a_im = simde_mm256_permute_pd(*a, 0xF);
    // b with swapped lanes to pair re with im: [im0, re0, im1, re1]
    const simde__m256d b_swapped = simde_mm256_permute_pd(*b, 0x5);

    // res_even lanes: a_re * b_re - a_im * b_im
    // res_odd  lanes: a_re * b_im + a_im * b_re
    *out = simde_mm256_addsub_pd(simde_mm256_mul_pd(a_re, *b),
                                 simde_mm256_mul_pd(a_im, b_swapped));
}

void nlo_real_fill(double *dst, size_t n, double value)
{
    if (dst == NULL) {
        return;
    }

    const simde__m256d v = simde_mm256_set1_pd(value);
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 4); i < simd_end; i += 4) {
        simde_mm256_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i) {
        dst[i] = value;
    }
}

void nlo_real_copy(double *dst, const double *src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 4); i < simd_end; i += 4) {
        simde__m256d v = simde_mm256_loadu_pd(src + i);
        simde_mm256_storeu_pd(dst + i, v);
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
}

void nlo_real_mul_inplace(double *dst, const double *src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 4); i < simd_end; i += 4) {
        const simde__m256d a = simde_mm256_loadu_pd(dst + i);
        const simde__m256d b = simde_mm256_loadu_pd(src + i);
        simde_mm256_storeu_pd(dst + i, simde_mm256_mul_pd(a, b));
    }
    for (; i < n; ++i) {
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
        size_t i = 0;
        for (size_t simd_end = nlo_simd_aligned_end(n, 4); i < simd_end; i += 4) {
            const simde__m256d a = simde_mm256_loadu_pd(out + i);
            const simde__m256d b = simde_mm256_loadu_pd(base + i);
            simde_mm256_storeu_pd(out + i, simde_mm256_mul_pd(a, b));
        }
        for (; i < n; ++i) {
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
    const simde__m256d v = simde_mm256_set_pd(value_im, value_re, value_im, value_re);

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        simde_mm256_storeu_pd((double *)(dst + i), v);
    }
    for (; i < n; ++i) {
        dst[i] = nlo_make(value_re, value_im);
    }
}

void nlo_complex_copy(nlo_complex *dst, const nlo_complex *src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        simde__m256d v = simde_mm256_loadu_pd((const double *)(src + i));
        simde_mm256_storeu_pd((double *)(dst + i), v);
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
}

void calculate_magnitude_squared(const nlo_complex *src, nlo_complex *dst, size_t n)
{
    if (src == NULL || dst == NULL) {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        const simde__m256d v = simde_mm256_loadu_pd((const double *)(src + i));
        const simde__m256d squared = simde_mm256_mul_pd(v, v);
        // Pairwise sum re^2 + im^2 for each complex.
        const simde__m256d swapped = simde_mm256_permute_pd(squared, 0x5);
        const simde__m256d sums = simde_mm256_add_pd(squared, swapped);
        double mags[4];
        simde_mm256_storeu_pd(mags, sums);
        dst[i] = nlo_make(mags[0], 0.0);
        dst[i + 1u] = nlo_make(mags[2], 0.0);
    }
    for (; i < n; ++i) {
        const double re = NLO_RE(src[i]);
        const double im = NLO_IM(src[i]);
        dst[i] = nlo_make(re * re + im * im, 0.0);
    }
}

void nlo_complex_axpy_real(nlo_complex *dst, const double *src, nlo_complex alpha, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    const simde__m256d alpha_vec = simde_mm256_set_pd(NLO_IM(alpha), NLO_RE(alpha),
                                                      NLO_IM(alpha), NLO_RE(alpha));

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        // Broadcast two real scalars to match complex lane layout
        const simde__m256d terms = simde_mm256_set_pd(src[i + 1u], src[i + 1u],
                                                      src[i], src[i]);
        const simde__m256d increment = simde_mm256_mul_pd(alpha_vec, terms);
        const simde__m256d dst_vec = simde_mm256_loadu_pd((double *)(dst + i));
        simde_mm256_storeu_pd((double *)(dst + i), simde_mm256_add_pd(dst_vec, increment));
    }
    for (; i < n; ++i) {
        const double term = src[i];
        dst[i] = nlo_make(NLO_RE(dst[i]) + NLO_RE(alpha) * term,
                          NLO_IM(dst[i]) + NLO_IM(alpha) * term);
    }
}

void nlo_complex_scalar_mul_inplace(nlo_complex *dst, nlo_complex alpha, size_t n)
{
    if (dst == NULL) {
        return;
    }

    const simde__m256d alpha_vec = simde_mm256_set_pd(NLO_IM(alpha), NLO_RE(alpha),
                                                      NLO_IM(alpha), NLO_RE(alpha));

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        const simde__m256d a = simde_mm256_loadu_pd((double *)(dst + i));
        simde__m256d res;
        nlo_complex_mul_packed(&a, &alpha_vec, &res);
        simde_mm256_storeu_pd((double *)(dst + i), res);
    }
    for (; i < n; ++i) {
        dst[i] = nlo_mul(dst[i], alpha);
    }
}

void nlo_complex_scalar_mul(nlo_complex *dst, const nlo_complex *src, nlo_complex alpha, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    const simde__m256d alpha_vec = simde_mm256_set_pd(NLO_IM(alpha), NLO_RE(alpha),
                                                      NLO_IM(alpha), NLO_RE(alpha));

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        const simde__m256d a = simde_mm256_loadu_pd((const double *)(src + i));
        simde__m256d res;
        nlo_complex_mul_packed(&a, &alpha_vec, &res);
        simde_mm256_storeu_pd((double *)(dst + i), res);
    }
    for (; i < n; ++i) {
        dst[i] = nlo_mul(src[i], alpha);
    }
}

void nlo_complex_mul_inplace(nlo_complex *dst, const nlo_complex *src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        const simde__m256d a = simde_mm256_loadu_pd((double *)(dst + i));
        const simde__m256d b = simde_mm256_loadu_pd((const double *)(src + i));
        simde__m256d res;
        nlo_complex_mul_packed(&a, &b, &res);
        simde_mm256_storeu_pd((double *)(dst + i), res);
    }
    for (; i < n; ++i) {
        dst[i] = nlo_mul(dst[i], src[i]);
    }
}

void nlo_complex_mul_vec(nlo_complex *dst, const nlo_complex *a, const nlo_complex *b, size_t n)
{
    if (dst == NULL || a == NULL || b == NULL) {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        const simde__m256d va = simde_mm256_loadu_pd((const double *)(a + i));
        const simde__m256d vb = simde_mm256_loadu_pd((const double *)(b + i));
        simde__m256d res;
        nlo_complex_mul_packed(&va, &vb, &res);
        simde_mm256_storeu_pd((double *)(dst + i), res);
    }
    for (; i < n; ++i) {
        dst[i] = nlo_mul(a[i], b[i]);
    }
}

void nlo_complex_pow(const nlo_complex *base, nlo_complex *out, size_t n, unsigned int exponent)
{
    if (base == NULL || out == NULL) {
        return;
    }

    if (exponent == 0U) {
        nlo_complex_fill(out, n, nlo_make(1.0, 0.0));
        return;
    }

    nlo_complex_fill(out, n, nlo_make(1.0, 0.0));
    for (unsigned int p = 0; p < exponent; ++p) {
        size_t i = 0;
        for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
            const simde__m256d a = simde_mm256_loadu_pd((double *)(out + i));
            const simde__m256d b = simde_mm256_loadu_pd((const double *)(base + i));
            simde__m256d res;
            nlo_complex_mul_packed(&a, &b, &res);
            simde_mm256_storeu_pd((double *)(out + i), res);
        }
        for (; i < n; ++i) {
            out[i] = nlo_mul(out[i], base[i]);
        }
    }
}

void nlo_complex_pow_inplace(nlo_complex *dst, size_t n, unsigned int exponent)
{
    if (dst == NULL) {
        return;
    }

    if (exponent == 0U) {
        nlo_complex_fill(dst, n, nlo_make(1.0, 0.0));
        return;
    }

    const simde__m256d ones = simde_mm256_setr_pd(1.0, 0.0, 1.0, 0.0);
    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        simde__m256d base = simde_mm256_loadu_pd((double *)(dst + i));
        simde__m256d result = ones;
        unsigned int exp = exponent;

        while (exp > 0U) {
            if ((exp & 1U) != 0U) {
                simde__m256d res;
                nlo_complex_mul_packed(&result, &base, &res);
                result = res;
            }
            exp >>= 1U;
            if (exp == 0U) {
                break;
            }
            {
                simde__m256d base_sq;
                nlo_complex_mul_packed(&base, &base, &base_sq);
                base = base_sq;
            }
        }

        simde_mm256_storeu_pd((double *)(dst + i), result);
    }
    for (; i < n; ++i) {
        nlo_complex base = dst[i];
        nlo_complex result = nlo_make(1.0, 0.0);
        unsigned int exp = exponent;

        while (exp > 0U) {
            if ((exp & 1U) != 0U) {
                result = nlo_mul(result, base);
            }
            exp >>= 1U;
            if (exp == 0U) {
                break;
            }
            base = nlo_mul(base, base);
        }

        dst[i] = result;
    }
}

void nlo_complex_add_inplace(nlo_complex *dst, const nlo_complex *src, size_t n)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        const simde__m256d a = simde_mm256_loadu_pd((double *)(dst + i));
        const simde__m256d b = simde_mm256_loadu_pd((const double *)(src + i));
        simde_mm256_storeu_pd((double *)(dst + i), simde_mm256_add_pd(a, b));
    }
    for (; i < n; ++i) {
        dst[i] = nlo_add(dst[i], src[i]);
    }
}

void nlo_complex_add_vec(nlo_complex *dst, const nlo_complex *a, const nlo_complex *b, size_t n)
{
    if (dst == NULL || a == NULL || b == NULL) {
        return;
    }

    size_t i = 0;
    for (size_t simd_end = nlo_simd_aligned_end(n, 2); i < simd_end; i += 2) {
        const simde__m256d va = simde_mm256_loadu_pd((const double *)(a + i));
        const simde__m256d vb = simde_mm256_loadu_pd((const double *)(b + i));
        simde_mm256_storeu_pd((double *)(dst + i), simde_mm256_add_pd(va, vb));
    }
    for (; i < n; ++i) {
        dst[i] = nlo_add(a[i], b[i]);
    }
}

void nlo_complex_exp_inplace(nlo_complex *dst, size_t n)
{
    if (dst == NULL) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        const double re = NLO_RE(dst[i]);
        const double im = NLO_IM(dst[i]);
        const double exp_re = exp(re);
        dst[i] = nlo_make(exp_re * cos(im), exp_re * sin(im));
    }
}
