/**
 * @file operators.c
 * @dir src/physics
 * @brief Implementation of physics operators for nonlinear optics simulations.
 * This file defines functions for applying dispersion and nonlinear operators
 * to optical field envelopes represented in the frequency and time domains.
 * 
 */

//  MARK: Includes

#include "operators.h"
#include "core/state.h"
#include "numerics/vector_ops.h"
#include "numerics/math_ops.h"
#include <stdlib.h>
#include <string.h>

// MARK: Static Declarations

static inline nlo_complex i_factor(size_t power);

// MARK: Public Definitions

void calculate_dispersion_factor(
    const size_t *num_dispersion_terms,
    const double *betas,
    const double step_size,
    nlo_complex *dispersion_factor,
    const nlo_complex *frequency_grid,
    const size_t num_time_samples)
{
    if (num_dispersion_terms == NULL || betas == NULL || frequency_grid == NULL || dispersion_factor == NULL) {
        return;
    }

    if (num_time_samples == 0 || *num_dispersion_terms == 0) {
        return;
    }

    nlo_complex_fill(dispersion_factor, num_time_samples, nlo_make(0.0, 0.0));

    nlo_complex *omega_power = (nlo_complex *)malloc(num_time_samples * sizeof(*omega_power));
    if (omega_power == NULL) {
        return;
    }

    memcpy(omega_power, frequency_grid, num_time_samples * sizeof(*frequency_grid));

    for (size_t i = 2; i < *num_dispersion_terms; ++i) {
        double beta = betas[i];
        
        size_t factorial = nlo_real_factorial(i);
        nlo_complex factorial_component = nlo_make(beta / (double)factorial, 0.0);

        size_t complex_power = i-1;
        nlo_complex complex_component = i_factor(complex_power);

        nlo_complex_mul_inplace(omega_power, frequency_grid, num_time_samples);

        for (size_t j = 0; j < num_time_samples; ++j) {
            nlo_complex term = nlo_mul(omega_power[j], complex_component);
            term = nlo_mul(term, factorial_component);
            dispersion_factor[j] = nlo_add(dispersion_factor[j], term);
        }

        nlo_complex_scalar_mul_inplace(
            dispersion_factor, nlo_make(step_size / 2.0, 0.0), num_time_samples);
        
        nlo_complex_exp_inplace(dispersion_factor, num_time_samples);
    }

    free(omega_power);
}

void dispersion_operator(
    const nlo_complex* dispersion_factor,
    nlo_complex* freq_domain_envelope,
    size_t num_time_samples)
{
    if (dispersion_factor == NULL || freq_domain_envelope == NULL) {
        return;
    }

    nlo_complex_mul_inplace(freq_domain_envelope, dispersion_factor, num_time_samples);
}

void nonlinear_operator(
    const double *gamma,
    nlo_complex *time_domain_envelope,
    const nlo_complex *time_domain_magnitude_squared,
    size_t num_time_samples)
{
    if (gamma == NULL || time_domain_envelope == NULL || time_domain_magnitude_squared == NULL) {
        return;
    }
}


// MARK: Static Definitions

static inline nlo_complex i_factor(size_t power)
{
    switch (power & 3u) {
        case 0u:
            return nlo_make(1.0, 0.0);
        case 1u:
            return nlo_make(0.0, 1.0);
        case 2u:
            return nlo_make(-1.0, 0.0);
        default:
            return nlo_make(0.0, -1.0);
    }
}
