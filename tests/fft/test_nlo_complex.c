/**
 * @file test_nlo_complex.c
 * @dir tests/fft
 * @brief Unit tests for nlo_complex type and operations.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "backend/nlo_complex.h"
#include <assert.h>
#include <stdio.h>

/**
 * @brief Test nlo_complex creation
 */
void test_nlo_make()
{
    nlo_complex z = nlo_make(3.0, 4.0);
    assert(NLO_RE(z) == 3.0);
    assert(NLO_IM(z) == 4.0);
    printf("test_nlo_make: validates complex constructor values.\n");
}

/**
 * @brief Test nlo_complex addition
 */
void test_nlo_add()
{
    nlo_complex a = nlo_make(1.0, 2.0);
    nlo_complex b = nlo_make(3.0, 4.0);
    nlo_complex c = nlo_add(a, b);
    assert(NLO_RE(c) == 4.0);
    assert(NLO_IM(c) == 6.0);
    printf("test_nlo_add: validates complex addition.\n");
}

/**
 * @brief Test nlo_complex multiplication
 */
void test_nlo_mul()
{
    nlo_complex a = nlo_make(1.0, 2.0);
    nlo_complex b = nlo_make(3.0, 4.0);
    nlo_complex c = nlo_mul(a, b);
    assert(NLO_RE(c) == -5.0); // 1*3 - 2*4
    assert(NLO_IM(c) == 10.0); // 1*4 + 2*3
    printf("test_nlo_mul: validates complex multiplication.\n");
}

int main(void)
{
    test_nlo_make();
    test_nlo_add();
    test_nlo_mul();
    printf("test_nlo_complex: all subtests completed.\n");
    return 0;
}
