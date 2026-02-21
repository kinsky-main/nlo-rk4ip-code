/**
 * @file math_ops.h
 * @dir src/numerics
 * @brief Header file for mathematical operations in numerical kernels.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#pragma once

// MARK: Includes

#include <stddef.h>

// MARK: Const & Macros

// MARK: Typedefs

// MARK: Function Declarations

/**
 * @brief Compute factorial using integer arithmetic.
 *
 * @param n Non-negative integer argument.
 * @return size_t Factorial value `n!`.
 */
size_t nlo_real_factorial(size_t n);
