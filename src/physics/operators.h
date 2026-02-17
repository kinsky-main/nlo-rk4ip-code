/**
 * @file operators.h
 * @brief Backend vector operators for dispersion and nonlinearity.
 */
#pragma once

#include "backend/vector_backend.h"
#include <stddef.h>

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

nlo_vec_status nlo_apply_dispersion_operator_vec(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* dispersion_factor,
    nlo_vec_buffer* freq_domain_envelope,
    nlo_vec_buffer* dispersion_working_vec,
    double half_step_size
);

nlo_vec_status nlo_apply_nonlinear_operator_vec(
    nlo_vector_backend* backend,
    double gamma,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* magnitude_squared,
    nlo_vec_buffer* out_field
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
