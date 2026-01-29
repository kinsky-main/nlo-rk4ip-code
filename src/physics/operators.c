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
#include <stdlib.h>

// MARK: Public Definitions

nlo_complex* dispersion_operator(
    const dispersion_params* disp_params,
    const double* frequency_grid,
    size_t num_time_samples,
    nlo_complex* freq_domain_envelope)
{
    if (disp_params == NULL || frequency_grid == NULL || freq_domain_envelope == NULL) {
        return NULL;
    }

    if (num_time_samples == 0 || disp_params->num_dispersion_terms == 0) {
        return freq_domain_envelope;
    }

    nlo_complex* dispersion = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    double* omega_power = (double*)malloc(num_time_samples * sizeof(double));
    if (dispersion == NULL || omega_power == NULL) {
        free(dispersion);
        free(omega_power);
        return NULL;
    }

    for (size_t i = 0; i < num_time_samples; ++i) {
        const double omega = frequency_grid[i];
        omega_power[i] = omega * omega;
    }

    double inv_factorial = 1.0;
    for (size_t term_idx = 0; term_idx < disp_params->num_dispersion_terms; ++term_idx) {
        const size_t order = term_idx + 2;
        inv_factorial /= (double)order;

        const double beta = disp_params->betas[term_idx];
        if (beta != 0.0) {
            const size_t i_power = order - 1;
            nlo_complex i_factor;
            switch (i_power & 3u) {
                case 0u:
                    i_factor = nlo_make(1.0, 0.0);
                    break;
                case 1u:
                    i_factor = nlo_make(0.0, 1.0);
                    break;
                case 2u:
                    i_factor = nlo_make(-1.0, 0.0);
                    break;
                default:
                    i_factor = nlo_make(0.0, -1.0);
                    break;
            }

            const double scale = beta * inv_factorial;
            nlo_complex coeff = nlo_make(NLO_RE(i_factor) * scale,
                                         NLO_IM(i_factor) * scale);
            nlo_complex_axpy_real(dispersion, omega_power, coeff, num_time_samples);
        }

        if (term_idx + 1 < disp_params->num_dispersion_terms) {
            nlo_real_mul_inplace(omega_power, frequency_grid, num_time_samples);
        }
    }

    nlo_complex_mul_inplace(freq_domain_envelope, dispersion, num_time_samples);

    free(dispersion);
    free(omega_power);
    return freq_domain_envelope;
}

// MARK: Static Declarations

// MARK: Static Definitions
