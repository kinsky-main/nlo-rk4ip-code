/**
 * @file test_nlo_numerics.c
 * @dir tests/numerics
 * @brief Unit tests for numerics operators in active use.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "fft/nlo_complex.h"
#include "numerics/math_ops.h"
#include "numerics/vector_ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifndef NLO_TEST_EPS
#define NLO_TEST_EPS 1e-12
#endif

static void test_nlo_real_factorial(void)
{
    assert(nlo_real_factorial(0) == 1);
    assert(nlo_real_factorial(1) == 1);
    assert(nlo_real_factorial(2) == 2);
    assert(nlo_real_factorial(3) == 6);
    assert(nlo_real_factorial(4) == 24);
    assert(nlo_real_factorial(6) == 720);
    assert(nlo_real_factorial(10) == 3628800);
    printf("test_nlo_real_factorial: validates factorial lookup.\n");
}

static void test_nlo_complex_fill(void)
{
    nlo_complex values[3];
    nlo_complex fill = nlo_make(1.5, -2.5);

    nlo_complex_fill(values, 3, fill);

    for (size_t i = 0; i < 3; ++i) {
        assert(NLO_RE(values[i]) == NLO_RE(fill));
        assert(NLO_IM(values[i]) == NLO_IM(fill));
    }

    printf("test_nlo_complex_fill: validates complex fill helper.\n");
}

static void test_nlo_complex_mul_inplace(void)
{
    nlo_complex dst[3] = {
        nlo_make(1.0, 2.0),
        nlo_make(-3.0, 4.0),
        nlo_make(0.5, -1.5)
    };
    const nlo_complex src[3] = {
        nlo_make(2.0, 0.0),
        nlo_make(0.0, -1.0),
        nlo_make(-1.0, 2.0)
    };

    nlo_complex_mul_inplace(dst, src, 3);

    assert(NLO_RE(dst[0]) == 2.0);
    assert(NLO_IM(dst[0]) == 4.0);

    assert(NLO_RE(dst[1]) == 4.0);
    assert(NLO_IM(dst[1]) == 3.0);

    assert(NLO_RE(dst[2]) == 2.5);
    assert(NLO_IM(dst[2]) == 2.5);

    printf("test_nlo_complex_mul_inplace: validates elementwise complex multiply.\n");
}

static void test_calculate_magnitude_squared(void)
{
    const nlo_complex src[4] = {
        nlo_make(3.0, 4.0),
        nlo_make(-2.0, 5.0),
        nlo_make(0.0, -7.0),
        nlo_make(-1.5, -2.5)
    };
    nlo_complex dst[4];

    calculate_magnitude_squared(src, dst, 4);

    assert(fabs(NLO_RE(dst[0]) - 25.0) < NLO_TEST_EPS);
    assert(fabs(NLO_IM(dst[0])) < NLO_TEST_EPS);
    assert(fabs(NLO_RE(dst[1]) - 29.0) < NLO_TEST_EPS);
    assert(fabs(NLO_IM(dst[1])) < NLO_TEST_EPS);
    assert(fabs(NLO_RE(dst[2]) - 49.0) < NLO_TEST_EPS);
    assert(fabs(NLO_IM(dst[2])) < NLO_TEST_EPS);
    assert(fabs(NLO_RE(dst[3]) - (1.5 * 1.5 + 2.5 * 2.5)) < NLO_TEST_EPS);
    assert(fabs(NLO_IM(dst[3])) < NLO_TEST_EPS);

    printf("test_calculate_magnitude_squared: validates |z|^2 helper.\n");
}

static void test_nlo_complex_exp_inplace(void)
{
    const double pi = acos(-1.0);
    nlo_complex values[3] = {
        nlo_make(0.0, 0.0),
        nlo_make(log(2.0), 0.0),
        nlo_make(0.0, pi)
    };

    nlo_complex_exp_inplace(values, 3);

    assert(fabs(NLO_RE(values[0]) - 1.0) < NLO_TEST_EPS);
    assert(fabs(NLO_IM(values[0])) < NLO_TEST_EPS);
    assert(fabs(NLO_RE(values[1]) - 2.0) < NLO_TEST_EPS);
    assert(fabs(NLO_IM(values[1])) < NLO_TEST_EPS);
    assert(fabs(NLO_RE(values[2]) + 1.0) < 1e-10);
    assert(fabs(NLO_IM(values[2])) < 1e-10);

    printf("test_nlo_complex_exp_inplace: validates complex exp implementation.\n");
}

int main(void)
{
    test_nlo_real_factorial();
    test_nlo_complex_fill();
    test_nlo_complex_mul_inplace();
    test_calculate_magnitude_squared();
    test_nlo_complex_exp_inplace();
    printf("test_nlo_numerics: all subtests completed.\n");
    return 0;
}
