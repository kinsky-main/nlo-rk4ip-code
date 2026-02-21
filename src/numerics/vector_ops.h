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
 *
 * @param dst Destination vector.
 * @param n Element count.
 * @param value Fill value.
 */
void nlo_real_fill(double *dst, size_t n, double value);

/**
 * @brief Copy a real-valued vector.
 *
 * @param dst Destination vector.
 * @param src Source vector.
 * @param n Element count.
 */
void nlo_real_copy(double *dst, const double *src, size_t n);

/**
 * @brief Element-wise multiply: dst[i] *= src[i].
 *
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @param n Element count.
 */
void nlo_real_mul_inplace(double *dst, const double *src, size_t n);

/**
 * @brief Raise each element to an integer power.
 *
 * @param base Input vector.
 * @param out Destination vector.
 * @param n Element count.
 * @param power Non-negative integer exponent.
 */
void nlo_real_pow_int(const double *base, double *out, size_t n, unsigned int power);

/**
 * @brief Fill a complex vector with a constant.
 *
 * @param dst Destination vector.
 * @param n Element count.
 * @param value Fill value.
 */
void nlo_complex_fill(nlo_complex *dst, size_t n, nlo_complex value);

/**
 * @brief Copy complex vectors.
 *
 * @param dst Destination vector.
 * @param src Source vector.
 * @param n Element count.
 */
void nlo_complex_copy(nlo_complex *dst, const nlo_complex *src, size_t n);

/**
 * @brief Compute magnitude squared for each complex element: dst[i] = |src[i]|^2.
 *
 * @param src Source complex vector.
 * @param dst Destination complex vector.
 * @param n Element count.
 */
void calculate_magnitude_squared(const nlo_complex *src, nlo_complex *dst, size_t n);

/**
 * @brief Complex axpy with real input: dst[i] += alpha * src[i].
 *
 * @param dst Destination complex vector.
 * @param src Source real vector.
 * @param alpha Complex scaling coefficient.
 * @param n Element count.
 */
void nlo_complex_axpy_real(nlo_complex *dst, const double *src, nlo_complex alpha, size_t n);

/**
 * @brief Element-wise multiply by complex constant value: dst[i] *= alpha.
 *
 * @param dst Destination complex vector.
 * @param alpha Complex scaling coefficient.
 * @param n Element count.
 */
void nlo_complex_scalar_mul_inplace(nlo_complex *dst, nlo_complex alpha, size_t n);

/**
 * @brief Element-wise complex scalar multiply helper: out[i] = dst[i] * alpha.
 *
 * @param dst Destination complex vector.
 * @param src Source complex vector.
 * @param alpha Complex scaling coefficient.
 * @param n Element count.
 */
void nlo_complex_scalar_mul(nlo_complex *dst, const nlo_complex *src, nlo_complex alpha, size_t n);

/**
 * @brief Element-wise complex multiply: dst[i] *= src[i].
 *
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @param n Element count.
 */
void nlo_complex_mul_inplace(nlo_complex *dst, const nlo_complex *src, size_t n);

/**
 * @brief Element-wise complex multiply helper: out[i] = a[i] * b[i].
 *
 * @param dst Destination vector.
 * @param a Left operand vector.
 * @param b Right operand vector.
 * @param n Element count.
 */
void nlo_complex_mul_vec(nlo_complex *dst, const nlo_complex *a, const nlo_complex *b, size_t n);

/**
 * @brief Element-wise complex power: out[i] = base[i] ^ exponent.
 *
 * @param base Input vector.
 * @param out Destination vector.
 * @param n Element count.
 * @param exponent Non-negative integer exponent.
 */
void nlo_complex_pow(const nlo_complex *base, nlo_complex *out, size_t n, unsigned int exponent);

/**
 * @brief Element-wise complex power inplace: dst[i] ^= exponent.
 *
 * @param dst Destination vector updated in place.
 * @param n Element count.
 * @param exponent Non-negative integer exponent.
 */
void nlo_complex_pow_inplace(nlo_complex *dst, size_t n, unsigned int exponent);

/**
 * @brief Element-wise complex power with complex exponent inplace:
 *        dst[i] = dst[i] ^ exponent[i].
 *
 * @param dst Destination/base vector updated in place.
 * @param exponent Exponent vector.
 * @param n Element count.
 */
void nlo_complex_pow_elementwise_inplace(nlo_complex *dst, const nlo_complex *exponent, size_t n);

/**
 * @brief Element-wise complex real power: out[i] = base[i] ^ exponent.
 *
 * @param base Input vector.
 * @param out Destination vector.
 * @param n Element count.
 * @param exponent Real exponent.
 */
void nlo_complex_real_pow(const nlo_complex *base, nlo_complex *out, size_t n, double exponent);

/**
 * @brief Element-wise complex real power inplace: dst[i] = dst[i] ^ exponent.
 *
 * @param dst Destination vector updated in place.
 * @param n Element count.
 * @param exponent Real exponent.
 */
void nlo_complex_real_pow_inplace(nlo_complex *dst, size_t n, double exponent);

/**
 * @brief Element-wise sum of two complex vectors inplace: dst[i] += src[i].
 *
 * @param dst Destination/left operand vector.
 * @param src Right operand vector.
 * @param n Element count.
 */
void nlo_complex_add_inplace(nlo_complex *dst, const nlo_complex *src, size_t n);

/**
 * @brief Element-wise complex add helper: out[i] = a[i] + b[i].
 *
 * @param dst Destination vector.
 * @param a Left operand vector.
 * @param b Right operand vector.
 * @param n Element count.
 */
void nlo_complex_add_vec(nlo_complex *dst, const nlo_complex *a, const nlo_complex *b, size_t n);

/**
 * @brief Exponent of complex vector: dst[i] = exp(dst[i]).
 *
 * @param dst Destination vector updated in place.
 * @param n Element count.
 */
void nlo_complex_exp_inplace(nlo_complex *dst, size_t n);

/**
 * @brief Natural logarithm of complex vector: dst[i] = log(dst[i]).
 *
 * @param dst Destination vector updated in place.
 * @param n Element count.
 */
void nlo_complex_log_inplace(nlo_complex *dst, size_t n);

/**
 * @brief Sine of complex vector: dst[i] = sin(dst[i]).
 *
 * @param dst Destination vector updated in place.
 * @param n Element count.
 */
void nlo_complex_sin_inplace(nlo_complex *dst, size_t n);

/**
 * @brief Cosine of complex vector: dst[i] = cos(dst[i]).
 *
 * @param dst Destination vector updated in place.
 * @param n Element count.
 */
void nlo_complex_cos_inplace(nlo_complex *dst, size_t n);
