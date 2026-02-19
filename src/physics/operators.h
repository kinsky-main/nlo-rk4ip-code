/**
 * @file operators.h
 * @brief Backend vector operators for dispersion and nonlinearity.
 */
#pragma once

#include "backend/vector_backend.h"
#include "physics/operator_program.h"
#include <stddef.h>

/**
 * @brief Build the linear dispersion generator L(omega) from beta-series coefficients.
 *
 * @param backend Active vector backend.
 * @param num_dispersion_terms Number of beta-series terms.
 * @param betas Dispersion beta-series coefficients.
 * @param step_size Current propagation step size (debug reporting only).
 * @param dispersion_factor Output dispersion generator vector L(omega).
 * @param frequency_grid Frequency grid vector.
 * @param omega_power Scratch vector used for frequency powers.
 * @param term_buffer Scratch vector used for accumulation.
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_calculate_dispersion_factor_vec(
    nlo_vector_backend* backend,
    size_t num_dispersion_terms,
    const double* betas,
    double step_size,
    nlo_vec_buffer* dispersion_factor,
    const nlo_vec_buffer* frequency_grid,
    nlo_vec_buffer* omega_power,
    nlo_vec_buffer* term_buffer
);

/**
 * @brief Apply the dispersion operator in frequency space.
 *
 * @param backend Active vector backend.
 * @param dispersion_factor Dispersion vector representation.
 * @param freq_domain_envelope Frequency-domain field updated in place.
 * @param dispersion_working_vec Scratch vector.
 * @param half_step_size Half propagation step.
 * @param factor_is_exponential Nonzero when dispersion_factor stores exp(L(omega)).
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_apply_dispersion_operator_vec(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* dispersion_factor,
    nlo_vec_buffer* freq_domain_envelope,
    nlo_vec_buffer* dispersion_working_vec,
    double half_step_size,
    int factor_is_exponential
);

/**
 * @brief Apply the legacy nonlinear operator: out = field * (i * gamma * |field|^2).
 *
 * @param backend Active vector backend.
 * @param gamma Nonlinear coefficient.
 * @param field Input field vector.
 * @param magnitude_squared Scratch/output vector for |field|^2 term.
 * @param out_field Output field vector.
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_apply_nonlinear_operator_vec(
    nlo_vector_backend* backend,
    double gamma,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* magnitude_squared,
    nlo_vec_buffer* out_field
);

/**
 * @brief Apply a runtime-compiled nonlinear multiplier program.
 *
 * @param backend Active vector backend.
 * @param program Compiled nonlinear program.
 * @param field Input field vector.
 * @param multiplier_vec Scratch/output multiplier vector.
 * @param out_field Output field vector.
 * @param runtime_stack_vec Runtime scratch stack vectors.
 * @param runtime_stack_count Number of runtime scratch stack vectors.
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_apply_nonlinear_operator_program_vec(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* multiplier_vec,
    nlo_vec_buffer* out_field,
    nlo_vec_buffer* const* runtime_stack_vec,
    size_t runtime_stack_count
);

/**
 * @brief Apply the graded-index phase operator in real space.
 *
 * @param backend Active vector backend.
 * @param grin_phase_factor_base Precomputed base phase factor per spatial sample.
 * @param field Field buffer updated in place.
 * @param grin_working_vec Temporary buffer (same shape as field).
 * @param half_step_size Propagation half step applied as real exponent.
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_apply_grin_operator_vec(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* grin_phase_factor_base,
    nlo_vec_buffer* field,
    nlo_vec_buffer* grin_working_vec,
    double half_step_size
);
