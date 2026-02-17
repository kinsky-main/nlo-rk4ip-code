/**
 * @file nlolib.h
 * @dir src
 * @brief Main header file for the Nonlinear Optics Library (NLOLib).
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#pragma once

// MARK: Includes

#include <stddef.h>

#include "core/state.h"
#include "core/init.h"
#include "backend/nlo_complex.h"

#ifdef __cplusplus
extern "C" {
#endif

// MARK: Const & Macros

// DLL export/import for Windows builds.
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

typedef enum {
    NLOLIB_STATUS_OK = 0,
    NLOLIB_STATUS_INVALID_ARGUMENT = 1,
    NLOLIB_STATUS_ALLOCATION_FAILED = 2,
    NLOLIB_STATUS_NOT_IMPLEMENTED = 3
} nlolib_status;

// MARK: Function Declarations

/**
 * @brief Propagate an input field and return recorded envelopes across z.
 *
 * @param config Simulation configuration parameters.
 * @param num_time_samples Number of time-domain samples in the input/output arrays.
 * @param input_field Pointer to input field buffer (length: num_time_samples).
 * @param num_recorded_samples Number of envelope records to return.
 * @param output_records Pointer to output record buffer (length:
 *        num_recorded_samples * num_time_samples). The layout is record-major:
 *        output_records[record_index * num_time_samples + time_index].
 *        For num_recorded_samples == 1, record 0 is the final field at z_end.
 *        For num_recorded_samples > 1, records are evenly distributed over [0, z_end].
 * @param exec_options Runtime backend selection/options
 *        (NULL uses AUTO hardware-detected defaults).
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_propagate(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options
);

#ifdef __cplusplus
}
#endif
