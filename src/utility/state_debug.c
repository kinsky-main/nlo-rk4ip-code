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

static int nlo_state_debug_enabled_cached = -1;

static int nlo_state_debug_str_ieq(const char* lhs, const char* rhs)
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

static int nlo_state_debug_enabled(void)
{
    if (nlo_state_debug_enabled_cached >= 0) {
        return nlo_state_debug_enabled_cached;
    }

    const char* env = getenv("NLO_STATE_DEBUG");
    if (env == NULL || env[0] == '\0') {
        nlo_state_debug_enabled_cached = 0;
        return 0;
    }

    if (nlo_state_debug_str_ieq(env, "0") ||
        nlo_state_debug_str_ieq(env, "false") ||
        nlo_state_debug_str_ieq(env, "off") ||
        nlo_state_debug_str_ieq(env, "no")) {
        nlo_state_debug_enabled_cached = 0;
        return 0;
    }

    nlo_state_debug_enabled_cached = 1;
    return 1;
}

void nlo_state_debug_log_failure(const char* stage, int status)
{
    if (!nlo_state_debug_enabled()) {
        return;
    }

    fprintf(stderr,
            "[NLO_STATE_DEBUG] create_simulation_state failed stage=%s status=%d\n",
            (stage != NULL) ? stage : "unknown",
            status);
}

void nlo_state_debug_log_ring_capacity(
    size_t requested_records,
    size_t per_record_bytes,
    size_t active_bytes,
    size_t runtime_stack_slots,
    size_t budget_bytes,
    size_t ring_capacity
)
{
    if (!nlo_state_debug_enabled()) {
        return;
    }

    fprintf(stderr,
            "[NLO_STATE_DEBUG] ring_capacity requested=%zu per_record_bytes=%zu "
            "active_bytes=%zu runtime_stack_slots=%zu budget_bytes=%zu "
            "computed_capacity=%zu\n",
            requested_records,
            per_record_bytes,
            active_bytes,
            runtime_stack_slots,
            budget_bytes,
            ring_capacity);
}
