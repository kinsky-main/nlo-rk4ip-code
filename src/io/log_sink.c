/**
 * @file log_sink.c
 * @brief Central logging sink and progress emitters for nlolib runtime logs.
 */

#include "io/log_sink.h"
#include "io/log_format.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NLO_LOG_STATUS_OK 0
#define NLO_LOG_STATUS_INVALID_ARGUMENT 1
#define NLO_LOG_STATUS_ALLOCATION_FAILED 2

#define NLO_LOG_BUFFER_MIN_BYTES 4096u

typedef struct
{
    FILE* file;
    char* ring;
    size_t ring_capacity;
    size_t ring_start;
    size_t ring_size;
    int level;
    int progress_enabled;
    int progress_milestone_percent;
    int progress_emit_on_step_adjust;
    int progress_active;
    int progress_next_milestone_percent;
    double progress_z_start;
    double progress_z_end;
} nlo_log_state;

static nlo_log_state g_nlo_log_state = {
    NULL,
    NULL,
    0u,
    0u,
    0u,
    NLO_LOG_LEVEL_INFO,
    1,
    5,
    0,
    0,
    0,
    0.0,
    0.0
};

static int nlo_log_level_enabled(int level)
{
    const int normalized = (level < NLO_LOG_LEVEL_ERROR) ? NLO_LOG_LEVEL_ERROR : level;
    return (normalized <= g_nlo_log_state.level);
}

static void nlo_log_ring_append(const char* text, size_t text_len)
{
    if (g_nlo_log_state.ring == NULL || g_nlo_log_state.ring_capacity == 0u || text == NULL || text_len == 0u) {
        return;
    }

    if (text_len >= g_nlo_log_state.ring_capacity) {
        const size_t keep = g_nlo_log_state.ring_capacity;
        memcpy(g_nlo_log_state.ring, text + (text_len - keep), keep);
        g_nlo_log_state.ring_start = 0u;
        g_nlo_log_state.ring_size = keep;
        return;
    }

    if ((g_nlo_log_state.ring_size + text_len) > g_nlo_log_state.ring_capacity) {
        const size_t overflow = (g_nlo_log_state.ring_size + text_len) - g_nlo_log_state.ring_capacity;
        g_nlo_log_state.ring_start = (g_nlo_log_state.ring_start + overflow) % g_nlo_log_state.ring_capacity;
        g_nlo_log_state.ring_size -= overflow;
    }

    const size_t tail = (g_nlo_log_state.ring_start + g_nlo_log_state.ring_size) % g_nlo_log_state.ring_capacity;
    const size_t first_copy = g_nlo_log_state.ring_capacity - tail;
    if (text_len <= first_copy) {
        memcpy(g_nlo_log_state.ring + tail, text, text_len);
    } else {
        memcpy(g_nlo_log_state.ring + tail, text, first_copy);
        memcpy(g_nlo_log_state.ring, text + first_copy, text_len - first_copy);
    }
    g_nlo_log_state.ring_size += text_len;
}

int nlo_log_set_file(const char* path_utf8, int append)
{
    if (path_utf8 == NULL || path_utf8[0] == '\0') {
        if (g_nlo_log_state.file != NULL) {
            fclose(g_nlo_log_state.file);
            g_nlo_log_state.file = NULL;
        }
        return NLO_LOG_STATUS_OK;
    }

    FILE* new_file = fopen(path_utf8, (append != 0) ? "ab" : "wb");
    if (new_file == NULL) {
        return NLO_LOG_STATUS_INVALID_ARGUMENT;
    }

    if (g_nlo_log_state.file != NULL) {
        fclose(g_nlo_log_state.file);
    }
    g_nlo_log_state.file = new_file;
    return NLO_LOG_STATUS_OK;
}

int nlo_log_set_buffer(size_t capacity_bytes)
{
    if (capacity_bytes == 0u) {
        free(g_nlo_log_state.ring);
        g_nlo_log_state.ring = NULL;
        g_nlo_log_state.ring_capacity = 0u;
        g_nlo_log_state.ring_start = 0u;
        g_nlo_log_state.ring_size = 0u;
        return NLO_LOG_STATUS_OK;
    }

    size_t capacity = capacity_bytes;
    if (capacity < NLO_LOG_BUFFER_MIN_BYTES) {
        capacity = NLO_LOG_BUFFER_MIN_BYTES;
    }

    char* new_ring = (char*)malloc(capacity);
    if (new_ring == NULL) {
        return NLO_LOG_STATUS_ALLOCATION_FAILED;
    }

    free(g_nlo_log_state.ring);
    g_nlo_log_state.ring = new_ring;
    g_nlo_log_state.ring_capacity = capacity;
    g_nlo_log_state.ring_start = 0u;
    g_nlo_log_state.ring_size = 0u;
    return NLO_LOG_STATUS_OK;
}

int nlo_log_clear_buffer(void)
{
    g_nlo_log_state.ring_start = 0u;
    g_nlo_log_state.ring_size = 0u;
    return NLO_LOG_STATUS_OK;
}

int nlo_log_read_buffer(char* dst, size_t dst_bytes, size_t* out_written, int consume)
{
    if (out_written == NULL || dst == NULL || dst_bytes == 0u) {
        return NLO_LOG_STATUS_INVALID_ARGUMENT;
    }

    *out_written = 0u;
    dst[0] = '\0';
    if (g_nlo_log_state.ring == NULL || g_nlo_log_state.ring_capacity == 0u || g_nlo_log_state.ring_size == 0u) {
        return NLO_LOG_STATUS_OK;
    }

    size_t to_copy = g_nlo_log_state.ring_size;
    if (to_copy >= dst_bytes) {
        to_copy = dst_bytes - 1u;
    }

    const size_t first_copy = g_nlo_log_state.ring_capacity - g_nlo_log_state.ring_start;
    if (to_copy <= first_copy) {
        memcpy(dst, g_nlo_log_state.ring + g_nlo_log_state.ring_start, to_copy);
    } else {
        memcpy(dst, g_nlo_log_state.ring + g_nlo_log_state.ring_start, first_copy);
        memcpy(dst + first_copy, g_nlo_log_state.ring, to_copy - first_copy);
    }
    dst[to_copy] = '\0';
    *out_written = to_copy;

    if (consume != 0) {
        g_nlo_log_state.ring_start = (g_nlo_log_state.ring_start + to_copy) % g_nlo_log_state.ring_capacity;
        g_nlo_log_state.ring_size -= to_copy;
    }

    return NLO_LOG_STATUS_OK;
}

int nlo_log_set_level(int level)
{
    if (level < NLO_LOG_LEVEL_ERROR || level > NLO_LOG_LEVEL_DEBUG) {
        return NLO_LOG_STATUS_INVALID_ARGUMENT;
    }
    g_nlo_log_state.level = level;
    return NLO_LOG_STATUS_OK;
}

int nlo_log_set_progress_options(int enabled, int milestone_percent, int emit_on_step_adjust)
{
    int milestone = milestone_percent;
    if (milestone <= 0) {
        milestone = 5;
    } else if (milestone > 100) {
        milestone = 100;
    }

    g_nlo_log_state.progress_enabled = (enabled != 0) ? 1 : 0;
    g_nlo_log_state.progress_milestone_percent = milestone;
    g_nlo_log_state.progress_emit_on_step_adjust = (emit_on_step_adjust != 0) ? 1 : 0;
    g_nlo_log_state.progress_next_milestone_percent = milestone;
    return NLO_LOG_STATUS_OK;
}

void nlo_log_emit_raw(int level, const char* text, size_t text_len)
{
    if (!nlo_log_level_enabled(level) || text == NULL || text_len == 0u) {
        return;
    }

    int wrote_to_sink = 0;
    if (g_nlo_log_state.ring != NULL && g_nlo_log_state.ring_capacity > 0u) {
        nlo_log_ring_append(text, text_len);
        wrote_to_sink = 1;
    }

    if (g_nlo_log_state.file != NULL) {
        (void)fwrite(text, 1u, text_len, g_nlo_log_state.file);
        fflush(g_nlo_log_state.file);
        wrote_to_sink = 1;
    }

    if (!wrote_to_sink) {
        (void)fwrite(text, 1u, text_len, stderr);
        fflush(stderr);
    }
}

void nlo_log_emit(int level, const char* fmt, ...)
{
    if (!nlo_log_level_enabled(level) || fmt == NULL) {
        return;
    }

    char message[4096];
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    if (written < 0) {
        return;
    }

    size_t length = (size_t)written;
    if (length >= sizeof(message)) {
        length = sizeof(message) - 1u;
    }

    if (length == 0u) {
        return;
    }

    if (message[length - 1u] != '\n') {
        if (length + 1u < sizeof(message)) {
            message[length++] = '\n';
            message[length] = '\0';
        }
    }

    nlo_log_emit_raw(level, message, length);
}

static double nlo_progress_percent(double z, double z_start, double z_end)
{
    if (!(z_end > z_start)) {
        return 0.0;
    }

    double ratio = (z - z_start) / (z_end - z_start);
    if (ratio < 0.0) {
        ratio = 0.0;
    } else if (ratio > 1.0) {
        ratio = 1.0;
    }
    return 100.0 * ratio;
}

void nlo_log_progress_begin(double z_start, double z_end)
{
    if (g_nlo_log_state.progress_enabled == 0 || !(z_end > z_start)) {
        g_nlo_log_state.progress_active = 0;
        return;
    }

    g_nlo_log_state.progress_active = 1;
    g_nlo_log_state.progress_z_start = z_start;
    g_nlo_log_state.progress_z_end = z_end;
    g_nlo_log_state.progress_next_milestone_percent = g_nlo_log_state.progress_milestone_percent;

    nlo_log_emit(NLO_LOG_LEVEL_INFO,
                 "[nlolib] progress:\n"
                 "  - state: start\n"
                 "  - z_percent: 0.0%%\n"
                 "  - z_current: %.9e\n"
                 "  - z_target: %.9e",
                 z_start,
                 z_end);
}

void nlo_log_progress_step_accepted(
    size_t step_index,
    double z,
    double z_end,
    double step_size,
    double error,
    double next_step_size
)
{
    if (g_nlo_log_state.progress_active == 0 || g_nlo_log_state.progress_enabled == 0) {
        return;
    }

    const double percent = nlo_progress_percent(z, g_nlo_log_state.progress_z_start, z_end);
    char step_index_text[48];
    (void)nlo_log_format_u64_grouped(step_index_text, sizeof(step_index_text), (uint64_t)step_index);

    if (g_nlo_log_state.progress_emit_on_step_adjust != 0) {
        const double adjust_tol = fmax(fabs(step_size) * 1e-12, 1e-15);
        if (fabs(next_step_size - step_size) > adjust_tol) {
            nlo_log_emit(NLO_LOG_LEVEL_INFO,
                         "[nlolib] step_adjustment:\n"
                         "  - step_index: %s\n"
                         "  - z_percent: %.1f%%\n"
                         "  - z_current: %.9e\n"
                         "  - z_target: %.9e\n"
                         "  - step_size: %.9e\n"
                         "  - next_step_size: %.9e\n"
                         "  - error: %.9e",
                         step_index_text,
                         percent,
                         z,
                         z_end,
                         step_size,
                         next_step_size,
                         error);
        }
    }

    while (g_nlo_log_state.progress_next_milestone_percent <= 100 &&
           percent + 1e-12 >= (double)g_nlo_log_state.progress_next_milestone_percent) {
        nlo_log_emit(NLO_LOG_LEVEL_INFO,
                     "[nlolib] progress:\n"
                     "  - state: milestone\n"
                     "  - milestone_percent: %d%%\n"
                     "  - z_percent: %.1f%%\n"
                     "  - z_current: %.9e\n"
                     "  - z_target: %.9e\n"
                     "  - step_index: %s",
                     g_nlo_log_state.progress_next_milestone_percent,
                     percent,
                     z,
                     z_end,
                     step_index_text);

        g_nlo_log_state.progress_next_milestone_percent += g_nlo_log_state.progress_milestone_percent;
        if (g_nlo_log_state.progress_next_milestone_percent > 100 &&
            g_nlo_log_state.progress_next_milestone_percent <
                (100 + g_nlo_log_state.progress_milestone_percent)) {
            g_nlo_log_state.progress_next_milestone_percent = 101;
        }
    }
}

void nlo_log_progress_step_rejected(
    size_t step_index,
    double z,
    double z_end,
    double attempted_step,
    double error,
    double retry_step,
    size_t reject_attempt
)
{
    if (g_nlo_log_state.progress_active == 0 || g_nlo_log_state.progress_enabled == 0) {
        return;
    }

    const double percent = nlo_progress_percent(z, g_nlo_log_state.progress_z_start, z_end);
    char step_index_text[48];
    char reject_attempt_text[48];
    (void)nlo_log_format_u64_grouped(step_index_text, sizeof(step_index_text), (uint64_t)step_index);
    (void)nlo_log_format_u64_grouped(reject_attempt_text, sizeof(reject_attempt_text), (uint64_t)reject_attempt);

    nlo_log_emit(NLO_LOG_LEVEL_INFO,
                 "[nlolib] step_rejected:\n"
                 "  - step_index: %s\n"
                 "  - reject_attempt: %s\n"
                 "  - z_percent: %.1f%%\n"
                 "  - z_current: %.9e\n"
                 "  - z_target: %.9e\n"
                 "  - attempted_step: %.9e\n"
                 "  - retry_step: %.9e\n"
                 "  - error: %.9e",
                 step_index_text,
                 reject_attempt_text,
                 percent,
                 z,
                 z_end,
                 attempted_step,
                 retry_step,
                 error);
}

void nlo_log_progress_finish(double z, double z_end, int success)
{
    if (g_nlo_log_state.progress_active == 0 || g_nlo_log_state.progress_enabled == 0) {
        return;
    }

    const double percent = (success != 0)
                               ? 100.0
                               : nlo_progress_percent(z, g_nlo_log_state.progress_z_start, z_end);

    nlo_log_emit(NLO_LOG_LEVEL_INFO,
                 "[nlolib] progress:\n"
                 "  - state: %s\n"
                 "  - z_percent: %.1f%%\n"
                 "  - z_current: %.9e\n"
                 "  - z_target: %.9e",
                 (success != 0) ? "complete" : "aborted",
                 percent,
                 z,
                 z_end);

    g_nlo_log_state.progress_active = 0;
}
