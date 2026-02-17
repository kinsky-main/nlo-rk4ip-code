/**
 * @file operators.c
 * @brief Backend vector operator implementation.
 */

#include "physics/operators.h"
#include "backend/nlo_complex.h"
#include "numerics/math_ops.h"
#include "utility/rk4_debug.h"

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

nlo_vec_status nlo_calculate_dispersion_factor_vec(
    nlo_vector_backend* backend,
    size_t num_dispersion_terms,
    const double* betas,
    double step_size,
    nlo_vec_buffer* dispersion_factor,
    const nlo_vec_buffer* frequency_grid,
    nlo_vec_buffer* omega_power,
    nlo_vec_buffer* term_buffer
)
{
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

    status = nlo_vec_complex_exp_inplace(backend, dispersion_factor);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    nlo_rk4_debug_log_dispersion_factor(backend, dispersion_factor, num_dispersion_terms, step_size);

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_apply_dispersion_operator_vec(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* dispersion_factor,
    nlo_vec_buffer* freq_domain_envelope,
    nlo_vec_buffer* dispersion_working_vec,
    double half_step_size
)
{
    if (backend == NULL || dispersion_factor == NULL || freq_domain_envelope == NULL || dispersion_working_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vec_complex_copy(backend, dispersion_working_vec, dispersion_factor);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_real_pow_inplace(backend, dispersion_working_vec, half_step_size);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vec_complex_mul_inplace(backend, freq_domain_envelope, dispersion_working_vec);
}

nlo_vec_status nlo_apply_nonlinear_operator_vec(
    nlo_vector_backend* backend,
    double gamma,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* magnitude_squared,
    nlo_vec_buffer* out_field
)
{
    if (backend == NULL || field == NULL || magnitude_squared == NULL || out_field == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    // TODO: Replace this with a single custom kernel that computes a not in place multiplication of the field with the magnitude squared, to avoid redundant copying and temporary storage.
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
                                                nlo_make(0.0, gamma));
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vec_complex_mul_inplace(backend, out_field, magnitude_squared);
}

nlo_vec_status nlo_apply_nonlinear_operator_program_vec(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* multiplier_vec,
    nlo_vec_buffer* out_field,
    nlo_vec_buffer* const* runtime_stack_vec,
    size_t runtime_stack_count
)
{
    if (backend == NULL ||
        program == NULL ||
        field == NULL ||
        multiplier_vec == NULL ||
        out_field == NULL ||
        runtime_stack_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const nlo_operator_eval_context eval_ctx = {
        .frequency_grid = NULL,
        .field = field
    };

    nlo_vec_status status = nlo_operator_program_execute(backend,
                                                         program,
                                                         &eval_ctx,
                                                         runtime_stack_vec,
                                                         runtime_stack_count,
                                                         multiplier_vec);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_copy(backend, out_field, field);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vec_complex_mul_inplace(backend, out_field, multiplier_vec);
}

nlo_vec_status nlo_apply_grin_operator_vec(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* grin_phase_factor_base,
    nlo_vec_buffer* field,
    nlo_vec_buffer* grin_working_vec,
    double half_step_size
)
{
    if (backend == NULL ||
        grin_phase_factor_base == NULL ||
        field == NULL ||
        grin_working_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vec_complex_copy(backend,
                                                 grin_working_vec,
                                                 grin_phase_factor_base);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_real_pow_inplace(backend,
                                              grin_working_vec,
                                              half_step_size);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return nlo_vec_complex_mul_inplace(backend, field, grin_working_vec);
}

