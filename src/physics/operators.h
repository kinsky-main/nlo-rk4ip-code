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
 * @param disp_params Pointer to dispersion parameters
 * @param dispersion_factors Pointer to output buffer for dispersion factors (length: num_time_samples)
 * @param frequency_grid Pointer to frequency grid (length: num_time_samples)
 * @param num_time_samples Number of time-domain samples
 */
void calculate_dispersion_factors(
    const dispersion_params *disp_params,
    nlo_complex* dispersion_factors,
    const double *frequency_grid,
    size_t num_time_samples);

/**
 * @brief Applies the dispersion operator factors to the frequency-domain envelope.
 * @param dispersion_factors Pointer to dispersion factors (length: num_time_samples)
 * @param freq_domain_envelope Pointer to frequency-domain envelope to be modified (length: num_time_samples)
 */
void dispersion_operator(
    const nlo_complex* dispersion_factors,
    nlo_complex* freq_domain_envelope,
    size_t num_time_samples);

void nonlinear_operator(
    const nonlinear_params *nonlin_params,
    const nlo_complex *time_domain_envelope);
