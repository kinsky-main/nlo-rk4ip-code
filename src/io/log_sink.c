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
#if defined(_WIN32)
  #include <windows.h>
#else
  #include <sys/time.h>
#endif

#define NLO_LOG_STATUS_OK 0
#define NLO_LOG_STATUS_INVALID_ARGUMENT 1
#define NLO_LOG_STATUS_ALLOCATION_FAILED 2

#define NLO_LOG_BUFFER_MIN_BYTES 4096u
#define NLO_PROGRESS_BAR_WIDTH 30u
#define NLO_PROGRESS_RENDER_MIN_SECONDS 0.2

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
    double progress_start_seconds;
    double progress_last_render_seconds;
    double progress_last_percent;
    size_t progress_last_step_index;
    size_t progress_last_line_len;
    int progress_line_visible;
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
    0.0,
    0.0,
    0.0,
    -1.0,
    0u,
    0u,
    0
};

static double nlo_log_now_seconds(void)
{
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (QueryPerformanceFrequency(&frequency) == 0 || QueryPerformanceCounter(&counter) == 0 ||
        frequency.QuadPart <= 0) {
        return 0.0;
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) {
        return 0.0;
    }
    return (double)tv.tv_sec + ((double)tv.tv_usec * 1e-6);
#endif
}

static void nlo_progress_format_duration(double seconds, char* out, size_t out_len)
{
    if (out == NULL || out_len == 0u) {
        return;
    }

    if (!isfinite(seconds) || seconds < 0.0) {
        (void)snprintf(out, out_len, "--:--");
        return;
    }

    const unsigned long total_seconds = (unsigned long)(seconds + 0.5);
    const unsigned long hours = total_seconds / 3600u;
    const unsigned long minutes = (total_seconds % 3600u) / 60u;
    const unsigned long secs = total_seconds % 60u;

    if (hours > 99u) {
        (void)snprintf(out, out_len, ">99h");
    } else if (hours > 0u) {
        (void)snprintf(out, out_len, "%02lu:%02lu:%02lu", hours, minutes, secs);
    } else {
        (void)snprintf(out, out_len, "%02lu:%02lu", minutes, secs);
    }
}

static void nlo_progress_make_bar(double percent, char* out, size_t out_len)
{
    if (out == NULL || out_len <= NLO_PROGRESS_BAR_WIDTH) {
        return;
    }

    int filled = (int)floor((percent / 100.0) * (double)NLO_PROGRESS_BAR_WIDTH);
    if (filled < 0) {
        filled = 0;
    } else if (filled > (int)NLO_PROGRESS_BAR_WIDTH) {
        filled = (int)NLO_PROGRESS_BAR_WIDTH;
    }

    for (size_t idx = 0u; idx < NLO_PROGRESS_BAR_WIDTH; ++idx) {
        out[idx] = (idx < (size_t)filled) ? '#' : '-';
    }
    out[NLO_PROGRESS_BAR_WIDTH] = '\0';
}

static void nlo_progress_end_console_line(void)
{
    if (g_nlo_log_state.progress_line_visible == 0) {
        return;
    }

    (void)fwrite("\n", 1u, 1u, stderr);
    fflush(stderr);
    g_nlo_log_state.progress_line_visible = 0;
    g_nlo_log_state.progress_last_line_len = 0u;
}

static void nlo_progress_render(
    size_t step_index,
    double percent,
    double now_seconds,
    const char* state_text,
    size_t reject_attempt
)
{
    char bar[NLO_PROGRESS_BAR_WIDTH + 1u];
    char step_text[48];
    char eta_text[24];
    char elapsed_text[24];
    char line[320];

    if (percent < 0.0) {
        percent = 0.0;
    } else if (percent > 100.0) {
        percent = 100.0;
    }

    nlo_progress_make_bar(percent, bar, sizeof(bar));
    (void)nlo_log_format_u64_grouped(step_text, sizeof(step_text), (uint64_t)step_index);

    double elapsed_seconds = 0.0;
    if (g_nlo_log_state.progress_start_seconds > 0.0 && now_seconds > g_nlo_log_state.progress_start_seconds) {
        elapsed_seconds = now_seconds - g_nlo_log_state.progress_start_seconds;
    }

    double remaining_seconds = NAN;
    if (percent >= 100.0) {
        remaining_seconds = 0.0;
    } else if (percent > 0.0 && elapsed_seconds > 0.0) {
        remaining_seconds = elapsed_seconds * ((100.0 - percent) / percent);
    }

    nlo_progress_format_duration(remaining_seconds, eta_text, sizeof(eta_text));
    nlo_progress_format_duration(elapsed_seconds, elapsed_text, sizeof(elapsed_text));

    if (reject_attempt > 0u) {
        const int written = snprintf(line,
                                     sizeof(line),
                                     "[nlolib] [%s] %6.2f%% |%s| step %s eta %s elapsed %s retry#%llu",
                                     (state_text != NULL) ? state_text : "progress",
                                     percent,
                                     bar,
                                     step_text,
                                     eta_text,
                                     elapsed_text,
                                     (unsigned long long)reject_attempt);
        if (written < 0) {
            return;
        }
    } else {
        const int written = snprintf(line,
                                     sizeof(line),
                                     "[nlolib] [%s] %6.2f%% |%s| step %s eta %s elapsed %s",
                                     (state_text != NULL) ? state_text : "progress",
                                     percent,
                                     bar,
                                     step_text,
                                     eta_text,
                                     elapsed_text);
        if (written < 0) {
            return;
        }
    }

    size_t line_len = strlen(line);
    (void)fwrite("\r", 1u, 1u, stderr);
    (void)fwrite(line, 1u, line_len, stderr);

    if (g_nlo_log_state.progress_last_line_len > line_len) {
        const size_t extra = g_nlo_log_state.progress_last_line_len - line_len;
        for (size_t idx = 0u; idx < extra; ++idx) {
            (void)fwrite(" ", 1u, 1u, stderr);
        }
    }
    fflush(stderr);

    g_nlo_log_state.progress_last_line_len = line_len;
    g_nlo_log_state.progress_line_visible = 1;
    g_nlo_log_state.progress_last_render_seconds = now_seconds;
    g_nlo_log_state.progress_last_percent = percent;
}

static int nlo_progress_should_render(double percent, double now_seconds, int force)
{
    if (force != 0) {
        return 1;
    }

    if (g_nlo_log_state.progress_last_percent < 0.0) {
        return 1;
    }

    if (percent + 1e-12 >= (double)g_nlo_log_state.progress_next_milestone_percent) {
        return 1;
    }

    if (g_nlo_log_state.progress_last_render_seconds > 0.0 &&
        now_seconds > g_nlo_log_state.progress_last_render_seconds &&
        (now_seconds - g_nlo_log_state.progress_last_render_seconds) >= NLO_PROGRESS_RENDER_MIN_SECONDS &&
        percent > g_nlo_log_state.progress_last_percent + 1e-12) {
        return 1;
    }

    return 0;
}

static void nlo_progress_advance_milestones(double percent)
{
    while (g_nlo_log_state.progress_next_milestone_percent <= 100 &&
           percent + 1e-12 >= (double)g_nlo_log_state.progress_next_milestone_percent) {
        g_nlo_log_state.progress_next_milestone_percent += g_nlo_log_state.progress_milestone_percent;
        if (g_nlo_log_state.progress_next_milestone_percent > 100 &&
            g_nlo_log_state.progress_next_milestone_percent <
                (100 + g_nlo_log_state.progress_milestone_percent)) {
            g_nlo_log_state.progress_next_milestone_percent = 101;
        }
    }
}

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
        nlo_progress_end_console_line();
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
    g_nlo_log_state.progress_start_seconds = nlo_log_now_seconds();
    g_nlo_log_state.progress_last_render_seconds = 0.0;
    g_nlo_log_state.progress_last_percent = -1.0;
    g_nlo_log_state.progress_last_step_index = 0u;
    g_nlo_log_state.progress_last_line_len = 0u;
    g_nlo_log_state.progress_line_visible = 0;
    nlo_progress_render(0u, 0.0, g_nlo_log_state.progress_start_seconds, "start", 0u);
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
    const double now_seconds = nlo_log_now_seconds();
    g_nlo_log_state.progress_last_step_index = step_index;

    const double adjust_tol = fmax(fabs(step_size) * 1e-12, 1e-15);
    const int is_step_adjust = (fabs(next_step_size - step_size) > adjust_tol) ? 1 : 0;
    const int force_render = (g_nlo_log_state.progress_emit_on_step_adjust != 0 && is_step_adjust != 0) ? 1 : 0;

    if (nlo_progress_should_render(percent, now_seconds, force_render) != 0) {
        nlo_progress_render(step_index,
                            percent,
                            now_seconds,
                            (is_step_adjust != 0) ? "adjust" : "running",
                            0u);
        nlo_progress_advance_milestones(percent);
    } else if (percent + 1e-12 >= (double)g_nlo_log_state.progress_next_milestone_percent) {
        nlo_progress_advance_milestones(percent);
    }

    (void)error;
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
    const double now_seconds = nlo_log_now_seconds();
    g_nlo_log_state.progress_last_step_index = step_index;
    nlo_progress_render(step_index, percent, now_seconds, "reject", reject_attempt);
    nlo_progress_advance_milestones(percent);
    (void)attempted_step;
    (void)error;
    (void)retry_step;
}

void nlo_log_progress_finish(double z, double z_end, int success)
{
    if (g_nlo_log_state.progress_active == 0 || g_nlo_log_state.progress_enabled == 0) {
        return;
    }

    const double now_seconds = nlo_log_now_seconds();
    const double percent = (success != 0)
                               ? 100.0
                               : nlo_progress_percent(z, g_nlo_log_state.progress_z_start, z_end);
    double elapsed_seconds = 0.0;
    if (g_nlo_log_state.progress_start_seconds > 0.0 && now_seconds > g_nlo_log_state.progress_start_seconds) {
        elapsed_seconds = now_seconds - g_nlo_log_state.progress_start_seconds;
    }

    nlo_progress_render(g_nlo_log_state.progress_last_step_index,
                        percent,
                        now_seconds,
                        (success != 0) ? "complete" : "aborted",
                        0u);
    nlo_progress_end_console_line();
    nlo_log_emit(NLO_LOG_LEVEL_INFO,
                 "[nlolib] progress_summary:\n"
                 "  - state: %s\n"
                 "  - z_percent: %.1f%%\n"
                 "  - elapsed_seconds: %.3f\n"
                 "  - z_current: %.9e\n"
                 "  - z_target: %.9e",
                 (success != 0) ? "complete" : "aborted",
                 percent,
                 elapsed_seconds,
                 z,
                 z_end);

    g_nlo_log_state.progress_active = 0;
    g_nlo_log_state.progress_start_seconds = 0.0;
    g_nlo_log_state.progress_last_render_seconds = 0.0;
    g_nlo_log_state.progress_last_percent = -1.0;
    g_nlo_log_state.progress_last_step_index = 0u;
    g_nlo_log_state.progress_line_visible = 0;
    g_nlo_log_state.progress_last_line_len = 0u;
}
