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

static inline nlo_cpack2d nlo_cpack2d_from_lanes(simde__m256d lanes)
{
    nlo_cpack2d out;
    out.lanes = lanes;
    return out;
}

static inline nlo_cpack2d nlo_cpack2d_loadu(const nlo_complex* src)
{
    return nlo_cpack2d_from_lanes(
        simde_mm256_loadu_pd((const double*)(const void*)src));
}

static inline void nlo_cpack2d_storeu(nlo_complex* dst, nlo_cpack2d values)
{
    simde_mm256_storeu_pd((double*)(void*)dst, values.lanes);
}

static inline nlo_cpack2d nlo_cpack2d_set1(nlo_complex value)
{
    return nlo_cpack2d_from_lanes(
        simde_mm256_set_pd(NLO_IM(value), NLO_RE(value), NLO_IM(value), NLO_RE(value)));
}

static inline nlo_cpack2d nlo_cpack2d_ones(void)
{
    return nlo_cpack2d_from_lanes(simde_mm256_setr_pd(1.0, 0.0, 1.0, 0.0));
}

static inline nlo_cpack2d nlo_cpack2d_add(nlo_cpack2d a, nlo_cpack2d b)
{
    return nlo_cpack2d_from_lanes(simde_mm256_add_pd(a.lanes, b.lanes));
}

static inline nlo_cpack2d nlo_cpack2d_mul(nlo_cpack2d a, nlo_cpack2d b)
{
    const simde__m256d a_re = simde_mm256_movedup_pd(a.lanes);
    const simde__m256d a_im = simde_mm256_permute_pd(a.lanes, 0xF);
    const simde__m256d b_swapped = simde_mm256_permute_pd(b.lanes, 0x5);

    return nlo_cpack2d_from_lanes(simde_mm256_addsub_pd(simde_mm256_mul_pd(a_re, b.lanes),
                                                        simde_mm256_mul_pd(a_im, b_swapped)));
}

static inline nlo_cpack2d nlo_cpack2d_scale_real2(nlo_cpack2d alpha, double s0, double s1)
{
    const simde__m256d terms = simde_mm256_set_pd(s1, s1, s0, s0);
    return nlo_cpack2d_from_lanes(simde_mm256_mul_pd(alpha.lanes, terms));
}

static inline nlo_cpack2d nlo_cpack2d_mag2_to_complex(nlo_cpack2d values)
{
    const simde__m256d squared = simde_mm256_mul_pd(values.lanes, values.lanes);
    const simde__m256d swapped = simde_mm256_permute_pd(squared, 0x5);
    const simde__m256d sums = simde_mm256_add_pd(squared, swapped);
    const simde__m256d mask = simde_mm256_setr_pd(1.0, 0.0, 1.0, 0.0);
    return nlo_cpack2d_from_lanes(simde_mm256_mul_pd(sums, mask));
}

