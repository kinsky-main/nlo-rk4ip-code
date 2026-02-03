/**
 * @file solve_rk4.h
 * @dir src/numerics
 * @brief Header for RK4 solver implementation for nonlinear optics propagation.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#pragma once

// MARK: Includes

#include "core/state.h"
#include "backend/nlo_complex.h"
#include <stddef.h>

// MARK: Const & Macros

// MARK: Typedefs

// MARK: Function Declarations

/**
 * @brief Perform full RK4 propagation on the simulation state.
 */
void solve_rk4(simulation_state* state);

/**
 * @brief Perform a single RK4 step on the simulation state.
 */
void step_rk4(simulation_state* state);
