/**
 * @file operators.h
 * @brief Backend vector operators for dispersion and nonlinearity.
 */
#pragma once

#include "core/state.h"

/**
 * @brief Apply configured dispersion operators to a frequency-domain field.
 *
 * @param state Mutable simulation state containing operator programs and buffers.
 * @param freq_domain_envelope Frequency-domain field updated in place.
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_apply_dispersion_operator_stage(
    simulation_state* state,
    nlo_vec_buffer* freq_domain_envelope
);

/**
 * @brief Apply configured nonlinear operator to an input field.
 *
 * @param state Mutable simulation state containing operator programs and buffers.
 * @param field Input field vector.
 * @param out_field Output field vector.
 * @return nlo_vec_status operation status.
 */
nlo_vec_status nlo_apply_nonlinear_operator_stage(
    simulation_state* state,
    const nlo_vec_buffer* field,
    nlo_vec_buffer* out_field
);
