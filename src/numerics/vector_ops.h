/**
 * @file vector_ops.h
 * @dir src/numerics
 * @brief Vector operations for numerical kernels.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */
#pragma once

// MARK: Includes

#include "backend/nlo_complex.h"
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
 * @brief Copy complex vectors.
 */
void nlo_complex_copy(nlo_complex *dst, const nlo_complex *src, size_t n);

/**
 * @brief Compute magnitude squared for each complex element: dst[i] = |src[i]|^2.
 */
void calculate_magnitude_squared(const nlo_complex *src, nlo_complex *dst, size_t n);

/**
 * @brief Complex axpy with real input: dst[i] += alpha * src[i].
 */
void nlo_complex_axpy_real(nlo_complex *dst, const double *src, nlo_complex alpha, size_t n);

/**
 * @brief Element-wise multiply by complex constant value: dst[i] *= alpha.
 */
void nlo_complex_scalar_mul_inplace(nlo_complex *dst, nlo_complex alpha, size_t n);

/**
 * @brief Element-wise complex scalar multiply helper: out[i] = dst[i] * alpha.
 */
void nlo_complex_scalar_mul(nlo_complex *dst, const nlo_complex *src, nlo_complex alpha, size_t n);

/**
 * @brief Element-wise complex multiply: dst[i] *= src[i].
 */
void nlo_complex_mul_inplace(nlo_complex *dst, const nlo_complex *src, size_t n);

/**
 * @brief Element-wise complex multiply helper: out[i] = a[i] * b[i].
 */
void nlo_complex_mul_vec(nlo_complex *dst, const nlo_complex *a, const nlo_complex *b, size_t n);

/**
 * @brief Element-wise complex power: out[i] = base[i] ^ exponent.
 */
void nlo_complex_pow(const nlo_complex *base, nlo_complex *out, size_t n, unsigned int exponent);

/**
 * @brief Element-wise complex power inplace: dst[i] ^= exponent.
 */
void nlo_complex_pow_inplace(nlo_complex *dst, size_t n, unsigned int exponent);

/**
 * @brief Element-wise sum of two complex vectors inplace: dst[i] += src[i].
 */
void nlo_complex_add_inplace(nlo_complex *dst, const nlo_complex *src, size_t n);

/**
 * @brief Element-wise complex add helper: out[i] = a[i] + b[i].
 */
void nlo_complex_add_vec(nlo_complex *dst, const nlo_complex *a, const nlo_complex *b, size_t n);

/**
 * @brief Exponent of complex vector: dst[i] = exp(dst[i]).
 */
void nlo_complex_exp_inplace(nlo_complex *dst, size_t n);
