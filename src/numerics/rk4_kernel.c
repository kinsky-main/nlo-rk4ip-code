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
#include "rk4_kernel.h"

void solve_rk4(simulation_state *state) {
    
    double z_end = state->config->propagation.propagation_distance;
    double max_step = state->config->propagation.max_step_size;
    double min_step = state->config->propagation.min_step_size;
    state->current_step_size = state->config->propagation.starting_step_size;

    while (state->current_z < z_end) {
        if (state->current_z + state->current_step_size > z_end) {
            state->current_step_size = z_end - state->current_z;
        }

        step_rk4(state);

        state->current_z += state->current_step_size;

    }
};

void step_rk4(simulation_state *state)
{
    
};