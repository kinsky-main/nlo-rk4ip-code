/**
 * @file nlo_progress.h
 * @dir src
 * @brief Public propagation progress callback types.
 */

#pragma once

#include <stddef.h>

/**
 * @brief Progress event class reported during propagation.
 */
typedef enum {
    /** One adaptive/fixed step was accepted. */
    NLO_PROGRESS_EVENT_ACCEPTED = 0,
    /** One adaptive retry was rejected. */
    NLO_PROGRESS_EVENT_REJECTED = 1,
    /** Propagation is finishing. */
    NLO_PROGRESS_EVENT_FINISH = 2
} nlo_progress_event_type;

/**
 * @brief Per-event propagation progress payload for caller callbacks.
 */
typedef struct {
    nlo_progress_event_type event_type;
    size_t step_index;
    size_t reject_attempt;
    double z;
    double z_end;
    double percent;
    double step_size;
    double next_step_size;
    double error;
    double elapsed_seconds;
    double eta_seconds;
} nlo_progress_info;

/**
 * @brief Progress callback invoked during propagation.
 *
 * @param info Current progress payload.
 * @param user_data Caller-owned opaque context pointer.
 * @return int Nonzero continues propagation; zero requests abort.
 */
typedef int (*nlo_progress_callback)(const nlo_progress_info* info, void* user_data);
