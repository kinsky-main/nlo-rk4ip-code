/**
 * @file math_ops.c
 * @dir src/numerics
 * @brief Header file for mathematical operations in numerical kernels.
 * This file defines functions for basic mathematical operations
 * used in nonlinear optics simulations.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

// MARK: Includes

#include "math_ops.h"
#include <stddef.h>

// MARK: Public Definitions

size_t nlo_real_factorial(size_t n)
{
    if (n == 0 || n == 1) {
        return 1;
    }

    size_t result = 1;
    for (size_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}