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

static void test_nlo_real_factorial(void)
{
    assert(nlo_real_factorial(0) == 1);
    assert(nlo_real_factorial(1) == 1);
    assert(nlo_real_factorial(2) == 2);
    assert(nlo_real_factorial(3) == 6);
    assert(nlo_real_factorial(4) == 24);
    assert(nlo_real_factorial(6) == 720);
    assert(nlo_real_factorial(10) == 3628800);
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
}

int main(void)
{
    test_nlo_real_factorial();
    test_nlo_complex_fill();
    test_nlo_complex_mul_inplace();
    return 0;
}
