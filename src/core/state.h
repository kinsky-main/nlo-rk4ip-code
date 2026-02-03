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
 * @param error_tolerance Relative error tolerance for adaptive RK4 stepping
 * @param propagation_distance Total distance to propagate (m)
 */
typedef struct {
    double starting_step_size;
    double max_step_size;
    double min_step_size;
    double error_tolerance;
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
 * @brief Parameters and a buffer for the frequency grid
 */
typedef struct {
    nlo_complex* frequency_grid;
} frequency_grid;

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
    frequency_grid frequency;
} sim_config;

/**
 * @brief Simulation state during propagation
 * 
 * @param config Pointer to simulation configuration
 * @param num_time_samples Number of time-domain samples
 * @param num_recorded_samples Number of field iterations retained
 * @param field_buffer Contiguous buffer holding all recorded electric fields
 * @param current_field Pointer to the currently active field record
 * @param ip_field_buffer Intermediate buffer for calculations
 * @param current_dispersion_factor Buffer for current dispersion factors
 * @param current_z Current propagation distance
 * @param current_step_size Current step size for propagation
 */
typedef struct {
    const sim_config* config;
    size_t num_time_samples;
    size_t num_recorded_samples;
    size_t current_record_index;
    nlo_complex* field_buffer;
    nlo_complex* current_field;
    double current_z;
    double current_step_size;
} simulation_state;

/**
 * @brief Working buffers for intermediate calculations during simulation
 * @param ip_field_buffer Interaction picture field buffer
 * @param field_magnitude_buffer Buffer for field magnitude squared
 * @param field_working_buffer General working buffer
 * @param field_freq_buffer Frequency domain buffer
 * @param k_1_buffer RK4 k1 buffer
 * @param k_2_buffer RK4 k2 buffer
 * @param k_3_buffer RK4 k3 buffer
 * @param k_4_buffer RK4 k4 buffer
 */
typedef struct {
    nlo_complex* ip_field_buffer;
    nlo_complex* field_magnitude_buffer;
    nlo_complex* field_working_buffer;
    nlo_complex* field_freq_buffer;
    nlo_complex* k_1_buffer;
    nlo_complex* k_2_buffer;
    nlo_complex* k_3_buffer;
    nlo_complex* k_4_buffer;
    nlo_complex* current_dispersion_factor;
} simulation_working_buffers;

// MARK: Function Declarations

/**
 * @brief Create and initialize a new simulation state
 * @param config Pointer to simulation configuration
 * @param num_time_samples Number of time-domain samples
 * @param num_recorded_samples Number of field snapshots to retain during propagation.
 *        The requested value may be capped based on available system memory.
 */
simulation_state* create_simulation_state(const sim_config* config, size_t num_time_samples, size_t num_recorded_samples);

/**
 * @brief Free resources associated with a simulation state
 * @param state Pointer to simulation state to free
 */
void free_simulation_state(simulation_state* state);

/**
 * @brief Get a pointer to a recorded field slice.
 *        Returned pointer is within the contiguous field_buffer and sized to num_time_samples.
 */
static inline nlo_complex* simulation_state_get_field_record(simulation_state* state, size_t record_index)
{
    if (state == NULL || record_index >= state->num_recorded_samples) {
        return NULL;
    }

    return state->field_buffer + (record_index * state->num_time_samples);
}

/**
 * @brief Convenience accessor for the currently active field record.
 */
static inline nlo_complex* simulation_state_current_field(const simulation_state* state)
{
    if (state == NULL) {
        return NULL;
    }

    return state->current_field;
}

/**
 * @brief Create and initialize a new simulation configuration
 * @param num_dispersion_terms Number of dispersion terms to initialize
 * @param num_time_samples Number of time-domain samples for frequency grid allocation
 */
sim_config* create_sim_config(size_t num_dispersion_terms, size_t num_time_samples);

/**
 * @brief Free resources associated with a simulation configuration
 * @param config Pointer to simulation configuration to free
 */
void free_sim_config(sim_config* config);
