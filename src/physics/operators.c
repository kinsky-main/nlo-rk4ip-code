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

// MARK: Public Definitions

void calculate_dispersion_factors(
    const dispersion_params* disp_params,
    nlo_complex* dispersion_factors,
    const nlo_complex* frequency_grid,
    const size_t num_time_samples)
{
    if (disp_params == NULL || frequency_grid == NULL || dispersion_factors == NULL) {
        return;
    }

    if (num_time_samples == 0 || disp_params->num_dispersion_terms == 0) {
        return;
    }

    nlo_complex *omega_power = (nlo_complex *)malloc(num_time_samples * sizeof(*omega_power));
    if (omega_power == NULL) {
        return;
    }

    for (size_t i = 2; i < disp_params->num_dispersion_terms; ++i) {
        double beta = disp_params->betas[i];
        
        size_t factorial = nlo_real_factorial(i);
        size_t complex_power = i-1;
        nlo_complex complex_component = i_factor(complex_power);
        nlo_complex_pow(frequency_grid, omega_power, num_time_samples, i);

    }

    free(omega_power);
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
