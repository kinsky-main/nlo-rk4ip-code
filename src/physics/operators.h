/**
 * @file operators.h
 * @dir src/physics
 * @brief Header file for physics operators in nonlinear optics simulations.
 * This file declares functions for applying dispersion and nonlinear operators
 * to optical field envelopes represented in the frequency and time domains.
 */
#pragma once

// MARK: Includes

#include "fft/nlo_complex.h"
#include "core/state.h"

// MARK: Const & Macros

// MARK: Typedefs

// MARK: Function Declarations

/**
 * @brief Pre-calculates the factors of the dispersion operator in the frequency domain to operate on field envelopes
 * for a given step size in propagation.
 * @param num_dispersion_terms Pointer to number of dispersion terms
 * @param betas Pointer to array of dispersion coefficients
 * @param step_size Step size for propagation
 * @param dispersion_factor Pointer to output buffer for dispersion factors (length: num_time_samples)
 * @param frequency_grid Pointer to frequency grid (length: num_time_samples)
 * @param num_time_samples Number of time-domain samples
 */
void calculate_dispersion_factor(
    const size_t *num_dispersion_terms,
    const double *betas,
    const double step_size,
    nlo_complex *dispersion_factor,
    const nlo_complex *frequency_grid,
    const size_t num_time_samples);

/**
 * @brief Applies the dispersion operator factors to the frequency-domain envelope.
 * @param dispersion_factor Pointer to dispersion factors (length: num_time_samples)
 * @param freq_domain_envelope Pointer to frequency-domain envelope to be modified (length: num_time_samples)
 */
void dispersion_operator(
    const nlo_complex *dispersion_factor,
    nlo_complex *freq_domain_envelope,
    size_t num_time_samples);

void nonlinear_operator(
    const double *gamma,
    nlo_complex *time_domain_envelope,
    const nlo_complex *time_domain_magnitude_squared,
    size_t num_time_samples);
