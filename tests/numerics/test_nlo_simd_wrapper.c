/**
 * @file test_nlo_simd_wrapper.c
 * @dir tests/numerics
 * @brief Unit tests for nlo_complex SIMD wrappers.
 */

#include "backend/nlo_complex_layout.h"
#include "numerics/nlo_complex_simd.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifndef NLO_TEST_EPS
#define NLO_TEST_EPS 1e-12
#endif

static void test_layout_contract(void)
{
    assert(sizeof(nlo_complex) == 2u * sizeof(double));
    assert(NLO_COMPLEX_ALIGN >= sizeof(double));
    printf("test_layout_contract: validates nlo_complex layout contract.\n");
}

static void test_mul_pack_matches_scalar(void)
{
    const nlo_complex a[2] = {
        nlo_make(1.0, -2.0),
        nlo_make(-0.5, 3.0)
    };
    const nlo_complex b[2] = {
        nlo_make(4.0, 0.5),
        nlo_make(2.0, -1.0)
    };
    nlo_complex out[2] = {0};

    nlo_cpack2d_storeu(out, nlo_cpack2d_mul(nlo_cpack2d_loadu(a), nlo_cpack2d_loadu(b)));

    for (size_t i = 0; i < 2; ++i) {
        const nlo_complex expected = nlo_mul(a[i], b[i]);
        assert(fabs(NLO_RE(out[i]) - NLO_RE(expected)) < NLO_TEST_EPS);
        assert(fabs(NLO_IM(out[i]) - NLO_IM(expected)) < NLO_TEST_EPS);
    }
    printf("test_mul_pack_matches_scalar: validates packed complex multiply.\n");
}

static void test_mag2_pack_matches_scalar(void)
{
    const nlo_complex src[2] = {
        nlo_make(3.0, 4.0),
        nlo_make(-2.5, 1.5)
    };
    nlo_complex out[2] = {0};

    nlo_cpack2d_storeu(out, nlo_cpack2d_mag2_to_complex(nlo_cpack2d_loadu(src)));

    for (size_t i = 0; i < 2; ++i) {
        const double expected = NLO_RE(src[i]) * NLO_RE(src[i]) +
            NLO_IM(src[i]) * NLO_IM(src[i]);
        assert(fabs(NLO_RE(out[i]) - expected) < NLO_TEST_EPS);
        assert(fabs(NLO_IM(out[i])) < NLO_TEST_EPS);
    }
    printf("test_mag2_pack_matches_scalar: validates packed magnitude squared.\n");
}

int main(void)
{
    test_layout_contract();
    test_mul_pack_matches_scalar();
    test_mag2_pack_matches_scalar();
    printf("test_nlo_simd_wrapper: all subtests completed.\n");
    return 0;
}

