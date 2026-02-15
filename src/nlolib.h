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
 * @brief Propagate an input field through the solver.
 *
 * @param config Simulation configuration parameters.
 * @param num_time_samples Number of time-domain samples in the input/output arrays.
 * @param input_field Pointer to input field buffer (length: num_time_samples).
 * @param output_field Pointer to output field buffer (length: num_time_samples).
 * @return nlolib_status status code.
 */
NLOLIB_API nlolib_status nlolib_propagate(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    nlo_complex* output_field
);

#ifdef __cplusplus
}
#endif
