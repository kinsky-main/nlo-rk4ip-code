/**
 * @file init.h
 * @brief Simulation-state initialization entry points and allocation summary.
 */
#pragma once

#include "core/state.h"
#include <stddef.h>

#ifndef NLOLIB_API
  #if defined(_WIN32)
    #if defined(NLOLIB_EXPORTS)
      #define NLOLIB_API __declspec(dllexport)
    #else
      #define NLOLIB_API __declspec(dllimport)
    #endif
  #else
    #define NLOLIB_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocation summary produced during simulation state initialization.
 */
typedef struct {
    size_t requested_records;
    size_t allocated_records;
    size_t per_record_bytes;
    size_t host_snapshot_bytes;
    size_t working_vector_bytes;
    size_t device_ring_capacity;
    size_t device_budget_bytes;
    nlo_vector_backend_type backend_type;
} nlo_allocation_info;

/**
 * @brief Initialize solver state and compute allocation diagnostics.
 *
 * @param config Simulation configuration.
 * @param num_time_samples Flattened sample count per record.
 * @param num_recorded_samples Number of records requested by caller.
 * @param exec_options Optional backend/runtime options (NULL for defaults).
 * @param allocation_info Optional allocation summary output (can be NULL).
 * @param out_state Destination state handle.
 * @return int 0 on success, nonzero on failure.
 */
NLOLIB_API int nlo_init_simulation_state(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options,
    nlo_allocation_info* allocation_info,
    simulation_state** out_state
);

/**
 * @brief Initialize solver state with optional SQLite snapshot storage support.
 *
 * @param config Simulation configuration.
 * @param num_time_samples Flattened sample count per record.
 * @param num_recorded_samples Number of records requested by caller.
 * @param exec_options Optional backend/runtime options (NULL for defaults).
 * @param storage_options Optional storage options (NULL disables storage).
 * @param allocation_info Optional allocation summary output (can be NULL).
 * @param out_state Destination state handle.
 * @return int 0 on success, nonzero on failure.
 */
NLOLIB_API int nlo_init_simulation_state_with_storage(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options,
    const nlo_storage_options* storage_options,
    nlo_allocation_info* allocation_info,
    simulation_state** out_state
);

#ifdef __cplusplus
}
#endif
