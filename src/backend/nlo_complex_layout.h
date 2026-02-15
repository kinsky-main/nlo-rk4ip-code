/**
 * @file nlo_complex_layout.h
 * @dir src/backend
 * @brief Compile-time layout contract for nlo_complex across host/device backends.
 */
#pragma once

#include "backend/nlo_complex.h"
#include <stddef.h>

#define NLO_COMPLEX_HOST_SCALAR 1
#define NLO_COMPLEX_DEVICE_AOS64 1

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define NLO_COMPLEX_ALIGN ((size_t)_Alignof(nlo_complex))
#elif defined(_MSC_VER)
#define NLO_COMPLEX_ALIGN ((size_t)__alignof(nlo_complex))
#else
#define NLO_COMPLEX_ALIGN (sizeof(double))
#endif

NLO_STATIC_ASSERT(sizeof(nlo_complex) == (2u * sizeof(double)),
                  "nlo_complex must store exactly two doubles");

NLO_STATIC_ASSERT(offsetof(nlo_complex, re) == 0u,
                  "nlo_complex real component offset mismatch");
NLO_STATIC_ASSERT(offsetof(nlo_complex, im) == sizeof(double),
                  "nlo_complex imag component offset mismatch");

