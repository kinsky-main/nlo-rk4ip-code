/**
 * @file state_debug.h
 * @brief Optional debug logging helpers for simulation-state initialization.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Emit a state-initialization failure diagnostic when enabled.
 *
 * Logging is controlled by the `NLO_STATE_DEBUG` environment variable.
 * Any non-empty value except `0`, `false`, `off`, or `no` enables output.
 *
 * @param stage Stable stage identifier for the failure site.
 * @param status Backend/status code associated with the failure.
 */
void nlo_state_debug_log_failure(const char* stage, int status);

#ifdef __cplusplus
}
#endif

