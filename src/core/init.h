#pragma once

// MARK: Includes

#include <stddef.h>
#include "core/state.h"

// MARK: Const & Macros

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

// MARK: Typedefs

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocation sizing information for a simulation state block.
 */
typedef struct {
    size_t requested_records;
    size_t records_per_block;
    size_t num_blocks;
    size_t per_record_bytes;
    size_t working_bytes;
    size_t block_bytes;
    size_t total_bytes;
} nlo_allocation_info;

// MARK: Function Declarations

/**
 * @brief Initialize a simulation state and report allocation sizing.
 *
 * @param config Pointer to simulation configuration.
 * @param num_time_samples Number of time-domain samples.
 * @param num_recorded_samples Number of field snapshots to retain.
 * @param allocation_info Optional output allocation sizing info.
 * @param out_state Output pointer for the created simulation state.
 * @return 0 on success, -1 on failure.
 */
NLOLIB_API int nlo_init_simulation_state(const sim_config* config,
                                         size_t num_time_samples,
                                         size_t num_recorded_samples,
                                         nlo_allocation_info* allocation_info,
                                         simulation_state** out_state);

#ifdef __cplusplus
}
#endif
