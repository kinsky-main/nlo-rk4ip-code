/**
 * @brief Definition of nlo_complex type used in FFT operations
 * @file nlo_complex.h
 * @author Wenzel Kinsky
 * @date 2026-01-27
 */ 
#pragma once

// MARK: Includes

// MARK: Const & Macros

// MARK: Typedefs

#if defined(NLO_FFT_BACKEND_FFTW)

  #include <fftw3.h>
  typedef struct { double re, im; } nlo_complex;
  #define NLO_RE(z) ((z).re)
  #define NLO_IM(z) ((z).im)
  _Static_assert(sizeof(nlo_complex) == sizeof(fftw_complex),
                 "nlo_complex must match fftw_complex layout");

#elif defined(NLO_FFT_BACKEND_CUFFT)

  #include <cuComplex.h>
  typedef cuDoubleComplex nlo_complex;
  #define NLO_RE(z) ((z).x)
  #define NLO_IM(z) ((z).y)

#else
  typedef struct { double re, im; } nlo_complex;
  #define NLO_RE(z) ((z).re)
  #define NLO_IM(z) ((z).im)
#endif

// MARK: Function Declarations

static inline nlo_complex nlo_make(double re, double im)
{
    nlo_complex z;

#if defined(NLO_FFT_BACKEND_CUFFT)
    z.x = re; z.y = im;
#else
    z.re = re; z.im = im;
#endif
    return z;
}

static inline nlo_complex nlo_add(nlo_complex a, nlo_complex b)
{
    return nlo_make(NLO_RE(a) + NLO_RE(b),
                    NLO_IM(a) + NLO_IM(b));
}

static inline nlo_complex nlo_mul(nlo_complex a, nlo_complex b)
{
    return nlo_make(NLO_RE(a) * NLO_RE(b) - NLO_IM(a) * NLO_IM(b),
                    NLO_RE(a) * NLO_IM(b) + NLO_IM(a) * NLO_RE(b));
}
