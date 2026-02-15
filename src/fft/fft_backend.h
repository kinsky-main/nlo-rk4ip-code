/**
 * @file fft_backend.h
 * @brief Runtime selector for FFT implementation.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NLO_FFT_BACKEND_AUTO = 0,
    NLO_FFT_BACKEND_FFTW = 1,
    NLO_FFT_BACKEND_VKFFT = 2
} nlo_fft_backend_type;

#ifdef __cplusplus
}
#endif
