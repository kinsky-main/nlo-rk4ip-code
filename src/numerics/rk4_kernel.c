/**
 * @file rk4_kernel.c
 * @dir src/numerics
 * @brief Implementation of RK4 solver kernel for nonlinear optics propagation.
 * @author Wenzel Kinsky
 */

#include "numerics/rk4_kernel.h"
#include "core/state.h"
#include "fft/nlo_complex.h"
#include <stddef.h>

void solve_rk4(simulation_state* state)
{
    // apply dispersion
};