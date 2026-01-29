/**
 * @file fft.h
 * @dir src/fft
 * @brief Header file for FFT operations in nonlinear optics simulations.
 * This file declares functions for performing forward and inverse Fast Fourier Transforms and will link
 * into whichever FFT library is chosen during the build process.
 */

#pragma once

// MARK: Includes

#include "nlo_complex.h"
#include <cstddef>

// MARK: Const & Macros

// MARK: Typedefs

// MARK: Function Declarations

/**
 * @brief Performs a forward Fast Fourier Transform (FFT) on the input time-domain signal.
 * @param time_domain_signal Pointer to the input signal in the time domain.
 * @param frequency_domain_signal Pointer to the output signal in the frequency domain.
 * @param signal_size Size of the input signal.
 */
void forward_fft(
    const nlo_complex* time_domain_signal,
    nlo_complex* frequency_domain_signal,
    std::size_t signal_size);

/**
 * @brief Performs an inverse Fast Fourier Transform (IFFT) on the input frequency-domain signal
 * to obtain the time-domain signal.
 * @param frequency_domain_signal Pointer to the input signal in the frequency domain.
 * @param time_domain_signal Pointer to the output signal in the time domain.
 * @param signal_size Size of the input signal.
 */
void inverse_fft(
    const nlo_complex* frequency_domain_signal,
    nlo_complex* time_domain_signal,
    std::size_t signal_size);