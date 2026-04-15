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
    PROGRESS_EVENT_ACCEPTED = 0,
    /** One adaptive retry was rejected. */
    PROGRESS_EVENT_REJECTED = 1,
    /** Propagation is finishing. */
    PROGRESS_EVENT_FINISH = 2
} progress_event_type;

/**
 * @brief Per-event propagation progress payload for caller callbacks.
 */
typedef struct {
    progress_event_type event_type;
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
} progress_info;

/**
 * @brief Progress callback invoked during propagation.
 *
 * @param info Current progress payload.
 * @param user_data Caller-owned opaque context pointer.
 * @return int Nonzero continues propagation; zero requests abort.
 */
typedef int (*progress_callback)(const progress_info* info, void* user_data);
