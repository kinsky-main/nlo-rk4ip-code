/**
 * @file fft_backend.h
 * @brief Runtime selector for FFT implementation.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /** Choose backend automatically from runtime/backend capabilities. */
    NLO_FFT_BACKEND_AUTO = 0,
    /** Force CPU FFTW backend. */
    NLO_FFT_BACKEND_FFTW = 1,
    /** Force Vulkan VkFFT backend. */
    NLO_FFT_BACKEND_VKFFT = 2,
    /** Force CUDA cuFFT backend. */
    NLO_FFT_BACKEND_CUFFT = 3,
    /** Force CUDA cuFFT Xt backend for eligible multi-GPU tensor FFTs. */
    NLO_FFT_BACKEND_CUFFT_XT = 4
} nlo_fft_backend_type;

#ifdef __cplusplus
}
#endif
