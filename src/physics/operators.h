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

nlo_complex *dispersion_operator(
    const dispersion_params *disp_params,
    const double *frequency_grid,
    size_t num_time_samples,
    nlo_complex *freq_domain_envelope);

nlo_complex *nonlinear_operator(
    const nonlinear_params *nonlin_params,
    const nlo_complex *time_domain_envelope);
