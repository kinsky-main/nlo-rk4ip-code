/**
 * @file test_nlo_numerics.c
 * @dir tests/numerics
 * @brief Unit tests for numerics operators in active use.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "backend/nlo_complex.h"
#include "numerics/math_ops.h"
#include "numerics/vector_ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifndef TEST_EPS
#define TEST_EPS 1e-12
#endif

static void assert_complex_close(nlo_complex value, nlo_complex expected, double eps)
{
    assert(fabs(RE(value) - RE(expected)) < eps);
    assert(fabs(IM(value) - IM(expected)) < eps);
}

static void test_nlo_real_factorial(void)
{
    assert(real_factorial(0) == 1);
    assert(real_factorial(1) == 1);
    assert(real_factorial(2) == 2);
    assert(real_factorial(3) == 6);
    assert(real_factorial(4) == 24);
    assert(real_factorial(6) == 720);
    assert(real_factorial(10) == 3628800);
    printf("test_nlo_real_factorial: validates factorial lookup.\n");
}

static void test_nlo_real_fill_copy_mul_pow(void)
{
    double values[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    double copy[5] = {0.0};
    double out[5] = {0.0};
    const double factors[5] = {2.0, -1.0, 0.5, 4.0, -3.0};

    real_fill(values, 5, -2.5);
    for (size_t i = 0; i < 5; ++i) {
        assert(fabs(values[i] + 2.5) < TEST_EPS);
    }

    real_copy(copy, values, 5);
    for (size_t i = 0; i < 5; ++i) {
        assert(fabs(copy[i] - values[i]) < TEST_EPS);
    }

    real_copy(values, factors, 5);
    real_mul_inplace(values, factors, 5);
    assert(fabs(values[0] - 4.0) < TEST_EPS);
    assert(fabs(values[1] - 1.0) < TEST_EPS);
    assert(fabs(values[2] - 0.25) < TEST_EPS);
    assert(fabs(values[3] - 16.0) < TEST_EPS);
    assert(fabs(values[4] - 9.0) < TEST_EPS);

    real_pow_int(factors, out, 5, 3);
    assert(fabs(out[0] - 8.0) < TEST_EPS);
    assert(fabs(out[1] + 1.0) < TEST_EPS);
    assert(fabs(out[2] - 0.125) < TEST_EPS);
    assert(fabs(out[3] - 64.0) < TEST_EPS);
    assert(fabs(out[4] + 27.0) < TEST_EPS);

    real_pow_int(factors, out, 5, 0);
    for (size_t i = 0; i < 5; ++i) {
        assert(fabs(out[i] - 1.0) < TEST_EPS);
    }

    printf("test_nlo_real_fill_copy_mul_pow: validates real vector ops.\n");
}

static void test_nlo_complex_fill(void)
{
    nlo_complex values[3];
    nlo_complex fill = make(1.5, -2.5);

    complex_fill(values, 3, fill);

    for (size_t i = 0; i < 3; ++i) {
        assert(RE(values[i]) == RE(fill));
        assert(IM(values[i]) == IM(fill));
    }

    printf("test_nlo_complex_fill: validates complex fill helper.\n");
}

static void test_nlo_complex_copy_add_axpy_scalar_mul(void)
{
    nlo_complex values[3] = {
        make(1.0, -2.0),
        make(-3.5, 4.5),
        make(0.0, -1.0)
    };
    nlo_complex copy[3] = {0};
    nlo_complex scaled[3] = {0};
    nlo_complex increment[3] = {
        make(0.5, 1.0),
        make(-1.5, 2.0),
        make(3.0, -4.0)
    };
    const double real_terms[3] = {2.0, -1.0, 0.5};
    const nlo_complex alpha = make(-0.5, 1.5);

    complex_copy(copy, values, 3);
    for (size_t i = 0; i < 3; ++i) {
        assert_complex_close(copy[i], values[i], TEST_EPS);
    }

    complex_add_inplace(copy, increment, 3);
    assert_complex_close(copy[0], make(1.5, -1.0), TEST_EPS);
    assert_complex_close(copy[1], make(-5.0, 6.5), TEST_EPS);
    assert_complex_close(copy[2], make(3.0, -5.0), TEST_EPS);

    complex_axpy_real(copy, real_terms, alpha, 3);
    assert_complex_close(copy[0], make(0.5, 2.0), TEST_EPS);
    assert_complex_close(copy[1], make(-4.5, 5.0), TEST_EPS);
    assert_complex_close(copy[2], make(2.75, -4.25), TEST_EPS);

    complex_scalar_mul(scaled, copy, make(2.0, -1.0), 3);
    assert_complex_close(scaled[0], make(3.0, 3.5), TEST_EPS);
    assert_complex_close(scaled[1], make(-4.0, 14.5), TEST_EPS);
    assert_complex_close(scaled[2], make(1.25, -11.25), TEST_EPS);

    complex_scalar_mul_inplace(copy, make(0.0, 2.0), 3);
    assert_complex_close(copy[0], make(-4.0, 1.0), TEST_EPS);
    assert_complex_close(copy[1], make(-10.0, -9.0), TEST_EPS);
    assert_complex_close(copy[2], make(8.5, 5.5), TEST_EPS);

    printf("test_nlo_complex_copy_add_axpy_scalar_mul: validates complex combine ops.\n");
}

static void test_nlo_complex_mul_inplace(void)
{
    nlo_complex dst[3] = {
        make(1.0, 2.0),
        make(-3.0, 4.0),
        make(0.5, -1.5)
    };
    const nlo_complex src[3] = {
        make(2.0, 0.0),
        make(0.0, -1.0),
        make(-1.0, 2.0)
    };

    complex_mul_inplace(dst, src, 3);

    assert(RE(dst[0]) == 2.0);
    assert(IM(dst[0]) == 4.0);

    assert(RE(dst[1]) == 4.0);
    assert(IM(dst[1]) == 3.0);

    assert(RE(dst[2]) == 2.5);
    assert(IM(dst[2]) == 2.5);

    printf("test_nlo_complex_mul_inplace: validates elementwise complex multiply.\n");
}

static void test_nlo_complex_pow_variants(void)
{
    const nlo_complex base[3] = {
        make(2.0, 0.0),
        make(0.0, 1.0),
        make(1.0, 1.0)
    };
    nlo_complex out[3];
    nlo_complex inplace[3];

    complex_pow(base, out, 3, 3);
    assert_complex_close(out[0], make(8.0, 0.0), TEST_EPS);
    assert_complex_close(out[1], make(0.0, -1.0), TEST_EPS);
    assert_complex_close(out[2], make(-2.0, 2.0), TEST_EPS);

    complex_pow(base, out, 3, 0);
    for (size_t i = 0; i < 3; ++i) {
        assert_complex_close(out[i], make(1.0, 0.0), TEST_EPS);
    }

    complex_copy(inplace, base, 3);
    complex_pow_inplace(inplace, 3, 4);
    assert_complex_close(inplace[0], make(16.0, 0.0), TEST_EPS);
    assert_complex_close(inplace[1], make(1.0, 0.0), TEST_EPS);
    assert_complex_close(inplace[2], make(-4.0, 0.0), TEST_EPS);

    complex_copy(inplace, base, 3);
    complex_pow_inplace(inplace, 3, 0);
    for (size_t i = 0; i < 3; ++i) {
        assert_complex_close(inplace[i], make(1.0, 0.0), TEST_EPS);
    }

    printf("test_nlo_complex_pow_variants: validates complex power helpers.\n");
}

static void test_nlo_complex_real_pow_variants(void)
{
    const nlo_complex base[4] = {
        make(1.0, 0.0),
        make(0.0, 1.0),
        make(-1.0, 0.0),
        make(0.0, 0.0)
    };
    nlo_complex out[4] = {0};
    nlo_complex inplace[4] = {0};

    complex_real_pow(base, out, 4, 0.5);
    assert_complex_close(out[0], make(1.0, 0.0), TEST_EPS);
    assert(fabs(RE(out[1]) - sqrt(0.5)) < 1e-12);
    assert(fabs(IM(out[1]) - sqrt(0.5)) < 1e-12);
    assert(fabs(RE(out[2])) < 1e-10);
    assert(fabs(IM(out[2]) - 1.0) < 1e-10);
    assert_complex_close(out[3], make(0.0, 0.0), TEST_EPS);

    complex_copy(inplace, base, 4);
    complex_real_pow_inplace(inplace, 4, 2.0);
    assert_complex_close(inplace[0], make(1.0, 0.0), TEST_EPS);
    assert_complex_close(inplace[1], make(-1.0, 0.0), TEST_EPS);
    assert_complex_close(inplace[2], make(1.0, 0.0), TEST_EPS);
    assert_complex_close(inplace[3], make(0.0, 0.0), TEST_EPS);

    complex_copy(inplace, base, 4);
    complex_real_pow_inplace(inplace, 4, 0.0);
    assert_complex_close(inplace[0], make(1.0, 0.0), TEST_EPS);
    assert_complex_close(inplace[1], make(1.0, 0.0), TEST_EPS);
    assert_complex_close(inplace[2], make(1.0, 0.0), TEST_EPS);
    assert_complex_close(inplace[3], make(1.0, 0.0), TEST_EPS);

    printf("test_nlo_complex_real_pow_variants: validates complex real-power helpers.\n");
}

static void test_calculate_magnitude_squared(void)
{
    const nlo_complex src[4] = {
        make(3.0, 4.0),
        make(-2.0, 5.0),
        make(0.0, -7.0),
        make(-1.5, -2.5)
    };
    nlo_complex dst[4];

    calculate_magnitude_squared(src, dst, 4);

    assert(fabs(RE(dst[0]) - 25.0) < TEST_EPS);
    assert(fabs(IM(dst[0])) < TEST_EPS);
    assert(fabs(RE(dst[1]) - 29.0) < TEST_EPS);
    assert(fabs(IM(dst[1])) < TEST_EPS);
    assert(fabs(RE(dst[2]) - 49.0) < TEST_EPS);
    assert(fabs(IM(dst[2])) < TEST_EPS);
    assert(fabs(RE(dst[3]) - (1.5 * 1.5 + 2.5 * 2.5)) < TEST_EPS);
    assert(fabs(IM(dst[3])) < TEST_EPS);

    printf("test_calculate_magnitude_squared: validates |z|^2 helper.\n");
}

static void test_nlo_complex_exp_inplace(void)
{
    const double pi = acos(-1.0);
    nlo_complex values[3] = {
        make(0.0, 0.0),
        make(log(2.0), 0.0),
        make(0.0, pi)
    };

    complex_exp_inplace(values, 3);

    assert(fabs(RE(values[0]) - 1.0) < TEST_EPS);
    assert(fabs(IM(values[0])) < TEST_EPS);
    assert(fabs(RE(values[1]) - 2.0) < TEST_EPS);
    assert(fabs(IM(values[1])) < TEST_EPS);
    assert(fabs(RE(values[2]) + 1.0) < 1e-10);
    assert(fabs(IM(values[2])) < 1e-10);

    printf("test_nlo_complex_exp_inplace: validates complex exp implementation.\n");
}

int main(void)
{
    test_nlo_real_factorial();
    test_nlo_real_fill_copy_mul_pow();
    test_nlo_complex_fill();
    test_nlo_complex_copy_add_axpy_scalar_mul();
    test_nlo_complex_mul_inplace();
    test_nlo_complex_pow_variants();
    test_nlo_complex_real_pow_variants();
    test_calculate_magnitude_squared();
    test_nlo_complex_exp_inplace();
    printf("test_nlo_numerics: all subtests completed.\n");
    return 0;
}
