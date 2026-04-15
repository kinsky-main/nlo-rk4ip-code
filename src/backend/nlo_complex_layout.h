/**
 * @file nlo_complex_layout.h
 * @dir src/backend
 * @brief Compile-time layout contract for nlo_complex across host/device backends.
 */
#pragma once

#include "backend/nlo_complex.h"
#include <stddef.h>

#define COMPLEX_HOST_SCALAR 1
#define COMPLEX_DEVICE_AOS64 1

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define COMPLEX_ALIGN ((size_t)_Alignof(nlo_complex))
#elif defined(_MSC_VER)
#define COMPLEX_ALIGN ((size_t)__alignof(nlo_complex))
#else
#define COMPLEX_ALIGN (sizeof(double))
#endif

STATIC_ASSERT(sizeof(nlo_complex) == (2u * sizeof(double)),
                  "nlo_complex must store exactly two doubles");

STATIC_ASSERT(offsetof(nlo_complex, re) == 0u,
                  "nlo_complex real component offset mismatch");
STATIC_ASSERT(offsetof(nlo_complex, im) == sizeof(double),
                  "nlo_complex imag component offset mismatch");

