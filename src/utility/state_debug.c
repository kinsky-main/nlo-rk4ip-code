/**
 * @file state_debug.c
 * @brief Optional debug logging helpers for simulation-state initialization.
 */

#include "utility/state_debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <strings.h>
#endif

static int state_debug_enabled_cached = -1;

static int state_debug_str_ieq(const char* lhs, const char* rhs)
{
    if (lhs == NULL || rhs == NULL) {
        return 0;
    }
#if defined(_WIN32)
    return _stricmp(lhs, rhs) == 0;
#else
    return strcasecmp(lhs, rhs) == 0;
#endif
}

static int state_debug_enabled(void)
{
    if (state_debug_enabled_cached >= 0) {
        return state_debug_enabled_cached;
    }

    const char* env = getenv("STATE_DEBUG");
    if (env == NULL || env[0] == '\0') {
        state_debug_enabled_cached = 0;
        return 0;
    }

    if (state_debug_str_ieq(env, "0") ||
        state_debug_str_ieq(env, "false") ||
        state_debug_str_ieq(env, "off") ||
        state_debug_str_ieq(env, "no")) {
        state_debug_enabled_cached = 0;
        return 0;
    }

    state_debug_enabled_cached = 1;
    return 1;
}

void state_debug_log_failure(const char* stage, int status)
{
    if (!state_debug_enabled()) {
        return;
    }

    fprintf(stderr,
            "[STATE_DEBUG] create_simulation_state failed stage=%s status=%d\n",
            (stage != NULL) ? stage : "unknown",
            status);
}

void state_debug_log_ring_capacity(
    size_t requested_records,
    size_t per_record_bytes,
    size_t active_bytes,
    size_t runtime_stack_slots,
    size_t budget_bytes,
    size_t ring_capacity
)
{
    if (!state_debug_enabled()) {
        return;
    }

    fprintf(stderr,
            "[STATE_DEBUG] ring_capacity requested=%zu per_record_bytes=%zu "
            "active_bytes=%zu runtime_stack_slots=%zu budget_bytes=%zu "
            "computed_capacity=%zu\n",
            requested_records,
            per_record_bytes,
            active_bytes,
            runtime_stack_slots,
            budget_bytes,
            ring_capacity);
}

void state_debug_log_memory_checkpoint(
    const char* stage,
    int query_status,
    size_t total_device_local_bytes,
    size_t available_device_local_bytes,
    size_t max_storage_buffer_range_bytes
)
{
    if (!state_debug_enabled()) {
        return;
    }

    fprintf(stderr,
            "[STATE_DEBUG] memory_checkpoint stage=%s query_status=%d "
            "device_local_total=%zu device_local_available=%zu "
            "max_storage_buffer_range=%zu\n",
            (stage != NULL) ? stage : "unknown",
            query_status,
            total_device_local_bytes,
            available_device_local_bytes,
            max_storage_buffer_range_bytes);
}
