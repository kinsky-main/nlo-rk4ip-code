/**
 * @file vector_ops.h
 * @dir src/numerics
 * @brief Vector operations for numerical kernels.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */
#pragma once

// MARK: Includes

#include "fft/nlo_complex.h"
#include <stddef.h>

// MARK: Function Declarations

/**
 * @brief Fill a real-valued vector with a constant.
 */
void nlo_real_fill(double *dst, size_t n, double value);

/**
 * @brief Copy a real-valued vector.
 */
void nlo_real_copy(double *dst, const double *src, size_t n);

/**
 * @brief Element-wise multiply: dst[i] *= src[i].
 */
void nlo_real_mul_inplace(double *dst, const double *src, size_t n);

/**
 * @brief Raise each element to an integer power.
 */
void nlo_real_pow_int(const double *base, double *out, size_t n, unsigned int power);

/**
 * @brief Fill a complex vector with a constant.
 */
void nlo_complex_fill(nlo_complex *dst, size_t n, nlo_complex value);

/**
 * @brief Complex axpy with real input: dst[i] += alpha * src[i].
 */
void nlo_complex_axpy_real(nlo_complex *dst, const double *src, nlo_complex alpha, size_t n);

/**
 * @brief Element-wise complex multiply: dst[i] *= src[i].
 */
void nlo_complex_mul_inplace(nlo_complex *dst, const nlo_complex *src, size_t n);

/**
 * @brief Element-wise complex power: out[i] = base[i] ^ exponent.
 */
void nlo_complex_pow(const nlo_complex *base, nlo_complex *out, size_t n, unsigned int exponent);
