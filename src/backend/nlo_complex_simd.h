/**
 * @file nlo_complex_simd.h
 * @dir src/backend
 * @brief SIMD packing helpers for nlo_complex vectors.
 */
#pragma once

#include "backend/nlo_complex_layout.h"
#include <simde/x86/avx.h>

typedef struct {
    simde__m256d lanes;
} nlo_cpack2d;

/**
 * @brief Wrap raw SIMD lanes as a packed two-complex vector.
 *
 * @param lanes Raw SIMD register lanes.
 * @return nlo_cpack2d Packed complex-lane wrapper.
 */
static inline nlo_cpack2d nlo_cpack2d_from_lanes(simde__m256d lanes)
{
    nlo_cpack2d out;
    out.lanes = lanes;
    return out;
}

/**
 * @brief Load two complex numbers from an unaligned host pointer.
 *
 * @param src Source complex pointer (at least two values).
 * @return nlo_cpack2d Loaded packed values.
 */
static inline nlo_cpack2d nlo_cpack2d_loadu(const nlo_complex* src)
{
    return nlo_cpack2d_from_lanes(
        simde_mm256_loadu_pd((const double*)(const void*)src));
}

/**
 * @brief Store two packed complex numbers to an unaligned host pointer.
 *
 * @param dst Destination complex pointer (at least two values).
 * @param values Packed complex values to store.
 */
static inline void nlo_cpack2d_storeu(nlo_complex* dst, nlo_cpack2d values)
{
    simde_mm256_storeu_pd((double*)(void*)dst, values.lanes);
}

/**
 * @brief Broadcast one complex scalar into both packed elements.
 *
 * @param value Complex scalar to broadcast.
 * @return nlo_cpack2d Packed broadcast values.
 */
static inline nlo_cpack2d nlo_cpack2d_set1(nlo_complex value)
{
    return nlo_cpack2d_from_lanes(
        simde_mm256_set_pd(NLO_IM(value), NLO_RE(value), NLO_IM(value), NLO_RE(value)));
}

/**
 * @brief Return packed complex ones: `(1 + 0i, 1 + 0i)`.
 *
 * @return nlo_cpack2d Packed complex-one values.
 */
static inline nlo_cpack2d nlo_cpack2d_ones(void)
{
    return nlo_cpack2d_from_lanes(simde_mm256_setr_pd(1.0, 0.0, 1.0, 0.0));
}

/**
 * @brief Add two packed complex vectors.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return nlo_cpack2d Element-wise complex sum.
 */
static inline nlo_cpack2d nlo_cpack2d_add(nlo_cpack2d a, nlo_cpack2d b)
{
    return nlo_cpack2d_from_lanes(simde_mm256_add_pd(a.lanes, b.lanes));
}

/**
 * @brief Multiply two packed complex vectors.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return nlo_cpack2d Element-wise complex product.
 */
static inline nlo_cpack2d nlo_cpack2d_mul(nlo_cpack2d a, nlo_cpack2d b)
{
    const simde__m256d a_re = simde_mm256_movedup_pd(a.lanes);
    const simde__m256d a_im = simde_mm256_permute_pd(a.lanes, 0xF);
    const simde__m256d b_swapped = simde_mm256_permute_pd(b.lanes, 0x5);

    return nlo_cpack2d_from_lanes(simde_mm256_addsub_pd(simde_mm256_mul_pd(a_re, b.lanes),
                                                        simde_mm256_mul_pd(a_im, b_swapped)));
}

/**
 * @brief Scale packed complex values with two independent real scalars.
 *
 * @param alpha Input packed values.
 * @param s0 Scale for element 0.
 * @param s1 Scale for element 1.
 * @return nlo_cpack2d Scaled packed values.
 */
static inline nlo_cpack2d nlo_cpack2d_scale_real2(nlo_cpack2d alpha, double s0, double s1)
{
    const simde__m256d terms = simde_mm256_set_pd(s1, s1, s0, s0);
    return nlo_cpack2d_from_lanes(simde_mm256_mul_pd(alpha.lanes, terms));
}

/**
 * @brief Convert packed complex values to packed complex magnitudes squared.
 *
 * Output imaginary lanes are zeroed.
 *
 * @param values Input packed complex values.
 * @return nlo_cpack2d Packed `|z|^2 + 0i` values.
 */
static inline nlo_cpack2d nlo_cpack2d_mag2_to_complex(nlo_cpack2d values)
{
    const simde__m256d squared = simde_mm256_mul_pd(values.lanes, values.lanes);
    const simde__m256d swapped = simde_mm256_permute_pd(squared, 0x5);
    const simde__m256d sums = simde_mm256_add_pd(squared, swapped);
    const simde__m256d mask = simde_mm256_setr_pd(1.0, 0.0, 1.0, 0.0);
    return nlo_cpack2d_from_lanes(simde_mm256_mul_pd(sums, mask));
}

