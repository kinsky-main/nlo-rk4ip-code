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

/**
 * @brief Reset per-run RK4 debug accumulators/counters.
 */
void nlo_rk4_debug_reset_run(void);

/**
 * @brief Log vector summary statistics for a debug stage.
 *
 * @param state Active simulation state.
 * @param vec Vector to summarize.
 * @param stage Stable stage identifier.
 * @param step_index Zero-based RK4 step index.
 * @param z Current propagation coordinate.
 * @param step Current step size.
 */
void nlo_rk4_debug_log_vec_stats(
    const simulation_state* state,
    const nlo_vec_buffer* vec,
    const char* stage,
    size_t step_index,
    double z,
    double step
);

/**
 * @brief Log RK4 adaptive-step error-control state.
 *
 * @param step_index Zero-based RK4 step index.
 * @param z Current propagation coordinate.
 * @param step Current step size.
 * @param error Current estimated local error.
 * @param scale Applied error scale/normalization term.
 * @param next_step Candidate next step size.
 */
void nlo_rk4_debug_log_error_control(
    size_t step_index,
    double z,
    double step,
    double error,
    double scale,
    double next_step
);

/**
 * @brief Log dispersion-factor diagnostics for a propagation step.
 *
 * @param backend Active vector backend.
 * @param dispersion_factor Dispersion factor vector.
 * @param num_dispersion_terms Number of expression terms used.
 * @param step_size Current propagation step size.
 */
void nlo_rk4_debug_log_dispersion_factor(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* dispersion_factor,
    size_t num_dispersion_terms,
    double step_size
);
