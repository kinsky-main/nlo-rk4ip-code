/**
 * @file rk4_debug.h
 * @brief Debug-only diagnostics helpers for RK4 propagation isolation.
 */
#pragma once

#include "backend/vector_backend.h"
#include "core/state.h"
#include <stddef.h>

#if defined(NLO_ENABLE_RK4_DEBUG_DIAGNOSTICS) || !defined(NDEBUG)
#define NLO_RK4_DEBUG_ACTIVE 1
#else
#define NLO_RK4_DEBUG_ACTIVE 0
#endif

void nlo_rk4_debug_reset_run(void);

void nlo_rk4_debug_log_vec_stats(
    const simulation_state* state,
    const nlo_vec_buffer* vec,
    const char* stage,
    size_t step_index,
    double z,
    double step
);

void nlo_rk4_debug_log_error_control(
    size_t step_index,
    double z,
    double step,
    double error,
    double scale,
    double next_step
);

void nlo_rk4_debug_log_dispersion_factor(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* dispersion_factor,
    size_t num_dispersion_terms,
    double step_size
);
