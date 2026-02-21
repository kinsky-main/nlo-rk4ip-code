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
  #define NLO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
  #define NLO_STATIC_ASSERT_CONCAT_(a, b) a##b
  #define NLO_STATIC_ASSERT_CONCAT(a, b) NLO_STATIC_ASSERT_CONCAT_(a, b)
  #define NLO_STATIC_ASSERT(cond, msg) \
    typedef char NLO_STATIC_ASSERT_CONCAT(nlo_static_assert_, __LINE__)[(cond) ? 1 : -1]
#endif

typedef struct { double re, im; } nlo_complex;
#define NLO_RE(z) ((z).re)
#define NLO_IM(z) ((z).im)

// MARK: Function Declarations

/**
 * @brief Construct a complex scalar from real and imaginary parts.
 *
 * @param re Real component.
 * @param im Imaginary component.
 * @return nlo_complex Constructed complex value.
 */
static inline nlo_complex nlo_make(double re, double im)
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
static inline nlo_complex nlo_add(nlo_complex a, nlo_complex b)
{
    return nlo_make(NLO_RE(a) + NLO_RE(b),
                    NLO_IM(a) + NLO_IM(b));
}

/**
 * @brief Complex multiplication helper.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return nlo_complex Product `a * b`.
 */
static inline nlo_complex nlo_mul(nlo_complex a, nlo_complex b)
{
    return nlo_make(NLO_RE(a) * NLO_RE(b) - NLO_IM(a) * NLO_IM(b),
                    NLO_RE(a) * NLO_IM(b) + NLO_IM(a) * NLO_RE(b));
}
