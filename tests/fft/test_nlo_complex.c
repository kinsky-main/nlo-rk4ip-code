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
    nlo_complex z = make(3.0, 4.0);
    assert(RE(z) == 3.0);
    assert(IM(z) == 4.0);
    printf("test_nlo_make: validates complex constructor values.\n");
}

/**
 * @brief Test nlo_complex addition
 */
void test_nlo_add()
{
    nlo_complex a = make(1.0, 2.0);
    nlo_complex b = make(3.0, 4.0);
    nlo_complex c = add(a, b);
    assert(RE(c) == 4.0);
    assert(IM(c) == 6.0);
    printf("test_nlo_add: validates complex addition.\n");
}

/**
 * @brief Test nlo_complex multiplication
 */
void test_nlo_mul()
{
    nlo_complex a = make(1.0, 2.0);
    nlo_complex b = make(3.0, 4.0);
    nlo_complex c = mul(a, b);
    assert(RE(c) == -5.0); // 1*3 - 2*4
    assert(IM(c) == 10.0); // 1*4 + 2*3
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
