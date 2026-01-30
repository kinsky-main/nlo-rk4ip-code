/**
 * @file state.h
 * @brief Core state definitions and buffer allocations for the nonlinear optics solver.
 *
 * @author Wenzel Kinsky
 * @date 2026-01-27
 */
#pragma once

// MARK: Includes

#include <stddef.h>
#include "fft/nlo_complex.h"

// MARK: Const & Macros

#ifndef NT_MAX
#define NT_MAX (1u<<20)
#endif

// MARK: Typedefs

/**
 * @brief Physical parameters related to nonlinear (Kerr) effect and Raman scattering
 * 
 * @param gamma Nonlinear coefficient (1/(Wm))
 */
typedef struct {
    double gamma;
} nonlinear_params;

/**
 * @brief Physical parameters related to dispersion
 * 
 * @param num_dispersion_terms Number of dispersion terms used
 * @param betas Array of dispersion coefficients (s^n/m)
 * @param alpha Attenuation coefficient (1/m)
 */
typedef struct {
    size_t num_dispersion_terms;
    double betas[NT_MAX];
    double alpha;
} dispersion_params;

/**
 * @brief Parameters related to propagation settings
 * 
 * @param starting_step_size Initial step size for propagation (m)
 * @param max_step_size Maximum allowable step size (m)
 * @param min_step_size Minimum allowable step size (m)
 * @param propagation_distance Total distance to propagate (m)
 */
typedef struct {
    double starting_step_size;
    double max_step_size;
    double min_step_size;
    double propagation_distance;
} propagation_params;

/**
 * @brief Parameters defining the time grid for simulation
 * 
 * @param pulse_period Time period of the pulse (s)
 * @param delta_time Time step between samples (s)
 */
typedef struct {
    double pulse_period;
    double delta_time;
} time_grid;

/**
 * @brief Overall simulation configuration parameters
 * 
 * @param nonlinear Nonlinear parameters
 * @param dispersion Dispersion parameters
 * @param propagation Propagation parameters
 * @param time Time grid parameters
 */
typedef struct {
    nonlinear_params nonlinear;
    dispersion_params dispersion;
    propagation_params propagation;
    time_grid time;
} sim_config;

/**
 * @brief Simulation state during propagation
 * 
 * @param config Pointer to simulation configuration
 * @param num_time_samples Number of time-domain samples
 * @param field_buffer Buffer for the electric field in time domain
 * @param ip_field_buffer Intermediate buffer for calculations
 * @param current_dispersion_factor Buffer for current dispersion factors
 * @param current_z Current propagation distance
 * @param current_step_size Current step size for propagation
 */
typedef struct {
    const sim_config* config;
    size_t num_time_samples;
    nlo_complex* field_buffer;
    nlo_complex* ip_field_buffer;
    nlo_complex* field_magnitude_buffer;
    nlo_complex* field_working_buffer;
    nlo_complex* current_dispersion_factor;
    double current_z;
    double current_step_size;
} simulation_state;

// MARK: Function Declarations

/**
 * @brief Create and initialize a new simulation state
 * @param config Pointer to simulation configuration
 * @param num_time_samples Number of time-domain samples
 */
simulation_state* create_simulation_state(const sim_config* config, size_t num_time_samples);

/**
 * @brief Free resources associated with a simulation state
 * @param state Pointer to simulation state to free
 */
void free_simulation_state(simulation_state* state);
