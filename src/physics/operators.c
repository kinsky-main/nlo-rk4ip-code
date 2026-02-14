/**
 * @file operators.c
 * @brief Backend vector operator implementation.
 */

#include "physics/operators.h"
#include "backend/nlo_complex.h"
#include "numerics/math_ops.h"

static inline nlo_complex nlo_i_factor(size_t power)
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

nlo_vec_status nlo_calculate_dispersion_factor_vec(nlo_vector_backend* backend,
                                                   size_t num_dispersion_terms,
                                                   const double* betas,
                                                   double step_size,
                                                   nlo_vec_buffer* dispersion_factor,
                                                   const nlo_vec_buffer* frequency_grid,
                                                   nlo_vec_buffer* omega_power,
                                                   nlo_vec_buffer* term_buffer)
{
    (void)step_size;

    if (backend == NULL || betas == NULL || dispersion_factor == NULL ||
        frequency_grid == NULL || omega_power == NULL || term_buffer == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vec_complex_fill(backend, dispersion_factor, nlo_make(0.0, 0.0));
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_copy(backend, omega_power, frequency_grid);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    for (size_t i = 2u; i < num_dispersion_terms; ++i) {
        const double beta = betas[i];
        const size_t factorial = nlo_real_factorial(i);
        if (factorial == 0u) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        status = nlo_vec_complex_mul_inplace(backend, omega_power, frequency_grid);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        status = nlo_vec_complex_copy(backend, term_buffer, omega_power);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        const nlo_complex coeff = nlo_mul(nlo_make(beta / (double)factorial, 0.0), nlo_i_factor(i - 1u));
        status = nlo_vec_complex_scalar_mul_inplace(backend, term_buffer, coeff);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        status = nlo_vec_complex_add_inplace(backend, dispersion_factor, term_buffer);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    return nlo_vec_complex_exp_inplace(backend, dispersion_factor);
}

nlo_vec_status nlo_apply_dispersion_operator_vec(nlo_vector_backend* backend,
                                                 const nlo_vec_buffer* dispersion_factor,
                                                 nlo_vec_buffer* freq_domain_envelope,
                                                 double exp_step_size)
{
    if (backend == NULL || dispersion_factor == NULL || freq_domain_envelope == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vec_complex_mul_inplace(backend, freq_domain_envelope, dispersion_factor);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vec_complex_scalar_mul_inplace(backend,
                                              freq_domain_envelope,
                                              nlo_make(exp_step_size, 0.0));
}

nlo_vec_status nlo_apply_nonlinear_operator_vec(nlo_vector_backend* backend,
                                                double gamma,
                                                const nlo_vec_buffer* field,
                                                nlo_vec_buffer* magnitude_squared,
                                                nlo_vec_buffer* out_field)
{
    if (backend == NULL || field == NULL || magnitude_squared == NULL || out_field == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vec_complex_copy(backend, out_field, field);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_magnitude_squared(backend, field, magnitude_squared);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_scalar_mul_inplace(backend,
                                                magnitude_squared,
                                                nlo_make(gamma, 0.0));
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vec_complex_mul_inplace(backend, out_field, magnitude_squared);
}
