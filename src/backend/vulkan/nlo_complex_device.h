/**
 * @file nlo_complex_device.h
 * @dir src/backend/vulkan
 * @brief Device-side memory layout mirror for complex numbers in Vulkan buffers.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    double re;
    double im;
} nlo_vk_complex64;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(nlo_vk_complex64) == (2u * sizeof(double)),
               "nlo_vk_complex64 must match AoS vec2<double> layout");
_Static_assert(offsetof(nlo_vk_complex64, re) == 0u,
               "nlo_vk_complex64 real offset mismatch");
_Static_assert(offsetof(nlo_vk_complex64, im) == sizeof(double),
               "nlo_vk_complex64 imag offset mismatch");
#endif

