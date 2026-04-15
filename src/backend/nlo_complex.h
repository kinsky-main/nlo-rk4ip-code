/**
 * @file nlo_complex.h
 * @dir src/backend
 * @brief Definition of the nlo_complex type for backend implementations.
 * @author Wenzel Kinsky
 * @date 2026-01-27
 */
#pragma once

// MARK: Includes

// MARK: Const & Macros

// MARK: Typedefs

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  #define STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
  #define STATIC_ASSERT_CONCAT_(a, b) a##b
  #define STATIC_ASSERT_CONCAT(a, b) STATIC_ASSERT_CONCAT_(a, b)
  #define STATIC_ASSERT(cond, msg) \
    typedef char STATIC_ASSERT_CONCAT(static_assert_, __LINE__)[(cond) ? 1 : -1]
#endif

typedef struct { double re, im; } nlo_complex;
#define RE(z) ((z).re)
#define IM(z) ((z).im)

// MARK: Function Declarations

/**
 * @brief Construct a complex scalar from real and imaginary parts.
 *
 * @param re Real component.
 * @param im Imaginary component.
 * @return nlo_complex Constructed complex value.
 */
static inline nlo_complex make(double re, double im)
{
    nlo_complex z;

    z.re = re; z.im = im;
    return z;
}

/**
 * @brief Complex addition helper.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return nlo_complex Sum `a + b`.
 */
static inline nlo_complex add(nlo_complex a, nlo_complex b)
{
    return make(RE(a) + RE(b),
                    IM(a) + IM(b));
}

/**
 * @brief Complex multiplication helper.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return nlo_complex Product `a * b`.
 */
static inline nlo_complex mul(nlo_complex a, nlo_complex b)
{
    return make(RE(a) * RE(b) - IM(a) * IM(b),
                    RE(a) * IM(b) + IM(a) * RE(b));
}
