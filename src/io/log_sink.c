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

#define LOG_STATUS_OK 0
#define LOG_STATUS_INVALID_ARGUMENT 1
#define LOG_STATUS_ALLOCATION_FAILED 2

#define LOG_BUFFER_MIN_BYTES 4096u
#define PROGRESS_BAR_WIDTH 30u
#define PROGRESS_RENDER_MIN_SECONDS 0.2

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
    int progress_stream_mode;
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
    progress_callback progress_callback;
    void* progress_callback_user_data;
    int progress_abort_requested;
} log_state;

static log_state g_nlo_log_state = {
    NULL,
    NULL,
    0u,
    0u,
    0u,
    LOG_LEVEL_INFO,
    1,
    5,
    0,
    LOG_PROGRESS_STREAM_STDERR,
    0,
    0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    0u,
    0u,
    0,
    NULL,
    NULL,
    0
};

static void progress_write_bytes(const char* text, size_t text_len)
{
    if (text == NULL || text_len == 0u) {
        return;
    }

    if (g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_STDOUT ||
        g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_BOTH) {
        (void)fwrite(text, 1u, text_len, stdout);
    }
    if (g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_STDERR ||
        g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_BOTH) {
        (void)fwrite(text, 1u, text_len, stderr);
    }
}

static void progress_flush(void)
{
    if (g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_STDOUT ||
        g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_BOTH) {
        fflush(stdout);
    }
    if (g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_STDERR ||
        g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_BOTH) {
        fflush(stderr);
    }
}

static double log_now_seconds(void)
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

static void progress_format_duration(double seconds, char* out, size_t out_len)
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

static void progress_make_bar(double percent, char* out, size_t out_len)
{
    if (out == NULL || out_len <= PROGRESS_BAR_WIDTH) {
        return;
    }

    int filled = (int)floor((percent / 100.0) * (double)PROGRESS_BAR_WIDTH);
    if (filled < 0) {
        filled = 0;
    } else if (filled > (int)PROGRESS_BAR_WIDTH) {
        filled = (int)PROGRESS_BAR_WIDTH;
    }

    for (size_t idx = 0u; idx < PROGRESS_BAR_WIDTH; ++idx) {
        out[idx] = (idx < (size_t)filled) ? '#' : '-';
    }
    out[PROGRESS_BAR_WIDTH] = '\0';
}

static void progress_end_console_line(void)
{
    if (g_nlo_log_state.progress_line_visible == 0) {
        return;
    }

    progress_write_bytes("\n", 1u);
    progress_flush();
    g_nlo_log_state.progress_line_visible = 0;
    g_nlo_log_state.progress_last_line_len = 0u;
}

static void progress_render(
    size_t step_index,
    double percent,
    double now_seconds,
    const char* state_text,
    size_t reject_attempt
)
{
    char bar[PROGRESS_BAR_WIDTH + 1u];
    char step_text[48];
    char eta_text[24];
    char elapsed_text[24];
    char line[320];

    if (percent < 0.0) {
        percent = 0.0;
    } else if (percent > 100.0) {
        percent = 100.0;
    }

    progress_make_bar(percent, bar, sizeof(bar));
    (void)log_format_u64_grouped(step_text, sizeof(step_text), (uint64_t)step_index);

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

    progress_format_duration(remaining_seconds, eta_text, sizeof(eta_text));
    progress_format_duration(elapsed_seconds, elapsed_text, sizeof(elapsed_text));

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
    progress_write_bytes("\r", 1u);
    progress_write_bytes(line, line_len);

    if (g_nlo_log_state.progress_last_line_len > line_len) {
        const size_t extra = g_nlo_log_state.progress_last_line_len - line_len;
        for (size_t idx = 0u; idx < extra; ++idx) {
            progress_write_bytes(" ", 1u);
        }
    }
    progress_flush();

    g_nlo_log_state.progress_last_line_len = line_len;
    g_nlo_log_state.progress_line_visible = 1;
    g_nlo_log_state.progress_last_render_seconds = now_seconds;
    g_nlo_log_state.progress_last_percent = percent;
}

static void progress_compute_timing(
    double percent,
    double now_seconds,
    double* out_elapsed_seconds,
    double* out_remaining_seconds
)
{
    double elapsed_seconds = 0.0;
    double remaining_seconds = NAN;

    if (g_nlo_log_state.progress_start_seconds > 0.0 && now_seconds > g_nlo_log_state.progress_start_seconds) {
        elapsed_seconds = now_seconds - g_nlo_log_state.progress_start_seconds;
    }

    if (percent >= 100.0) {
        remaining_seconds = 0.0;
    } else if (percent > 0.0 && elapsed_seconds > 0.0) {
        remaining_seconds = elapsed_seconds * ((100.0 - percent) / percent);
    }

    if (out_elapsed_seconds != NULL) {
        *out_elapsed_seconds = elapsed_seconds;
    }
    if (out_remaining_seconds != NULL) {
        *out_remaining_seconds = remaining_seconds;
    }
}

static void progress_notify_callback(
    progress_event_type event_type,
    size_t step_index,
    size_t reject_attempt,
    double z,
    double z_end,
    double percent,
    double step_size,
    double next_step_size,
    double error,
    double now_seconds
)
{
    if (g_nlo_log_state.progress_callback == NULL ||
        (g_nlo_log_state.progress_abort_requested != 0 && event_type != PROGRESS_EVENT_FINISH)) {
        return;
    }

    double elapsed_seconds = 0.0;
    double eta_seconds = NAN;
    progress_compute_timing(percent, now_seconds, &elapsed_seconds, &eta_seconds);

    const progress_info info = {
        event_type,
        step_index,
        reject_attempt,
        z,
        z_end,
        percent,
        step_size,
        next_step_size,
        error,
        elapsed_seconds,
        isfinite(eta_seconds) ? eta_seconds : -1.0
    };

    if (g_nlo_log_state.progress_callback(&info, g_nlo_log_state.progress_callback_user_data) == 0) {
        g_nlo_log_state.progress_abort_requested = 1;
    }
}

static int progress_should_render(double percent, double now_seconds, int force)
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
        (now_seconds - g_nlo_log_state.progress_last_render_seconds) >= PROGRESS_RENDER_MIN_SECONDS &&
        percent > g_nlo_log_state.progress_last_percent + 1e-12) {
        return 1;
    }

    return 0;
}

static void progress_advance_milestones(double percent)
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

static int log_level_enabled(int level)
{
    const int normalized = (level < LOG_LEVEL_ERROR) ? LOG_LEVEL_ERROR : level;
    return (normalized <= g_nlo_log_state.level);
}

static void log_ring_append(const char* text, size_t text_len)
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

int log_set_file(const char* path_utf8, int append)
{
    if (path_utf8 == NULL || path_utf8[0] == '\0') {
        if (g_nlo_log_state.file != NULL) {
            fclose(g_nlo_log_state.file);
            g_nlo_log_state.file = NULL;
        }
        return LOG_STATUS_OK;
    }

    FILE* new_file = fopen(path_utf8, (append != 0) ? "ab" : "wb");
    if (new_file == NULL) {
        return LOG_STATUS_INVALID_ARGUMENT;
    }

    if (g_nlo_log_state.file != NULL) {
        fclose(g_nlo_log_state.file);
    }
    g_nlo_log_state.file = new_file;
    return LOG_STATUS_OK;
}

int log_set_buffer(size_t capacity_bytes)
{
    if (capacity_bytes == 0u) {
        free(g_nlo_log_state.ring);
        g_nlo_log_state.ring = NULL;
        g_nlo_log_state.ring_capacity = 0u;
        g_nlo_log_state.ring_start = 0u;
        g_nlo_log_state.ring_size = 0u;
        return LOG_STATUS_OK;
    }

    size_t capacity = capacity_bytes;
    if (capacity < LOG_BUFFER_MIN_BYTES) {
        capacity = LOG_BUFFER_MIN_BYTES;
    }

    char* new_ring = (char*)malloc(capacity);
    if (new_ring == NULL) {
        return LOG_STATUS_ALLOCATION_FAILED;
    }

    free(g_nlo_log_state.ring);
    g_nlo_log_state.ring = new_ring;
    g_nlo_log_state.ring_capacity = capacity;
    g_nlo_log_state.ring_start = 0u;
    g_nlo_log_state.ring_size = 0u;
    return LOG_STATUS_OK;
}

int log_clear_buffer(void)
{
    g_nlo_log_state.ring_start = 0u;
    g_nlo_log_state.ring_size = 0u;
    return LOG_STATUS_OK;
}

int log_read_buffer(char* dst, size_t dst_bytes, size_t* out_written, int consume)
{
    if (out_written == NULL || dst == NULL || dst_bytes == 0u) {
        return LOG_STATUS_INVALID_ARGUMENT;
    }

    *out_written = 0u;
    dst[0] = '\0';
    if (g_nlo_log_state.ring == NULL || g_nlo_log_state.ring_capacity == 0u || g_nlo_log_state.ring_size == 0u) {
        return LOG_STATUS_OK;
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

    return LOG_STATUS_OK;
}

int log_set_level(int level)
{
    if (level < LOG_LEVEL_ERROR || level > LOG_LEVEL_DEBUG) {
        return LOG_STATUS_INVALID_ARGUMENT;
    }
    g_nlo_log_state.level = level;
    return LOG_STATUS_OK;
}

int log_set_progress_options(int enabled, int milestone_percent, int emit_on_step_adjust)
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
    return LOG_STATUS_OK;
}

int log_set_progress_stream(int stream_mode)
{
    if (stream_mode < LOG_PROGRESS_STREAM_STDERR || stream_mode > LOG_PROGRESS_STREAM_BOTH) {
        return LOG_STATUS_INVALID_ARGUMENT;
    }

    g_nlo_log_state.progress_stream_mode = stream_mode;
    return LOG_STATUS_OK;
}

int log_set_progress_callback(progress_callback callback, void* user_data)
{
    g_nlo_log_state.progress_callback = callback;
    g_nlo_log_state.progress_callback_user_data = user_data;
    g_nlo_log_state.progress_abort_requested = 0;
    return LOG_STATUS_OK;
}

int log_progress_abort_requested(void)
{
    return g_nlo_log_state.progress_abort_requested;
}

void log_emit_raw(int level, const char* text, size_t text_len)
{
    if (!log_level_enabled(level) || text == NULL || text_len == 0u) {
        return;
    }

    int wrote_to_sink = 0;
    if (g_nlo_log_state.ring != NULL && g_nlo_log_state.ring_capacity > 0u) {
        log_ring_append(text, text_len);
        wrote_to_sink = 1;
    }

    if (g_nlo_log_state.file != NULL) {
        (void)fwrite(text, 1u, text_len, g_nlo_log_state.file);
        fflush(g_nlo_log_state.file);
        wrote_to_sink = 1;
    }

    if (!wrote_to_sink) {
        progress_end_console_line();
        (void)fwrite(text, 1u, text_len, stderr);
        fflush(stderr);
    }
}

void log_emit(int level, const char* fmt, ...)
{
    if (!log_level_enabled(level) || fmt == NULL) {
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

    log_emit_raw(level, message, length);
}

static double progress_percent(double z, double z_start, double z_end)
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

void log_progress_begin(double z_start, double z_end)
{
    if ((g_nlo_log_state.progress_enabled == 0 && g_nlo_log_state.progress_callback == NULL) ||
        !(z_end > z_start)) {
        g_nlo_log_state.progress_active = 0;
        return;
    }

    g_nlo_log_state.progress_active = 1;
    g_nlo_log_state.progress_z_start = z_start;
    g_nlo_log_state.progress_z_end = z_end;
    g_nlo_log_state.progress_next_milestone_percent = g_nlo_log_state.progress_milestone_percent;
    g_nlo_log_state.progress_start_seconds = log_now_seconds();
    g_nlo_log_state.progress_last_render_seconds = 0.0;
    g_nlo_log_state.progress_last_percent = -1.0;
    g_nlo_log_state.progress_last_step_index = 0u;
    g_nlo_log_state.progress_last_line_len = 0u;
    g_nlo_log_state.progress_line_visible = 0;
    g_nlo_log_state.progress_abort_requested = 0;
}

void log_progress_step_accepted(
    size_t step_index,
    double z,
    double z_end,
    double step_size,
    double error,
    double next_step_size
)
{
    if (g_nlo_log_state.progress_active == 0) {
        return;
    }

    const double percent = progress_percent(z, g_nlo_log_state.progress_z_start, z_end);
    const double now_seconds = log_now_seconds();
    g_nlo_log_state.progress_last_step_index = step_index;

    const double adjust_tol = fmax(fabs(step_size) * 1e-12, 1e-15);
    const int is_step_adjust = (fabs(next_step_size - step_size) > adjust_tol) ? 1 : 0;
    const int force_render = (g_nlo_log_state.progress_emit_on_step_adjust != 0 && is_step_adjust != 0) ? 1 : 0;

    if (g_nlo_log_state.progress_enabled != 0 &&
        progress_should_render(percent, now_seconds, force_render) != 0) {
        progress_render(step_index,
                            percent,
                            now_seconds,
                            (is_step_adjust != 0) ? "adjust" : "running",
                            0u);
        progress_advance_milestones(percent);
    } else if (g_nlo_log_state.progress_enabled != 0 &&
               percent + 1e-12 >= (double)g_nlo_log_state.progress_next_milestone_percent) {
        progress_advance_milestones(percent);
    }

    progress_notify_callback(PROGRESS_EVENT_ACCEPTED,
                                 step_index,
                                 0u,
                                 z,
                                 z_end,
                                 percent,
                                 step_size,
                                 next_step_size,
                                 error,
                                 now_seconds);
}

void log_progress_step_rejected(
    size_t step_index,
    double z,
    double z_end,
    double attempted_step,
    double error,
    double retry_step,
    size_t reject_attempt
)
{
    if (g_nlo_log_state.progress_active == 0) {
        return;
    }

    const double percent = progress_percent(z, g_nlo_log_state.progress_z_start, z_end);
    const double now_seconds = log_now_seconds();
    g_nlo_log_state.progress_last_step_index = step_index;
    if (g_nlo_log_state.progress_enabled != 0) {
        progress_render(step_index, percent, now_seconds, "reject", reject_attempt);
        progress_advance_milestones(percent);
    }
    progress_notify_callback(PROGRESS_EVENT_REJECTED,
                                 step_index,
                                 reject_attempt,
                                 z,
                                 z_end,
                                 percent,
                                 attempted_step,
                                 retry_step,
                                 error,
                                 now_seconds);
}

void log_progress_finish(double z, double z_end, int success)
{
    if (g_nlo_log_state.progress_active == 0) {
        return;
    }

    const double now_seconds = log_now_seconds();
    const double percent = (success != 0)
                               ? 100.0
                               : progress_percent(z, g_nlo_log_state.progress_z_start, z_end);
    double elapsed_seconds = 0.0;
    if (g_nlo_log_state.progress_start_seconds > 0.0 && now_seconds > g_nlo_log_state.progress_start_seconds) {
        elapsed_seconds = now_seconds - g_nlo_log_state.progress_start_seconds;
    }

    if (g_nlo_log_state.progress_enabled != 0) {
        progress_render(g_nlo_log_state.progress_last_step_index,
                            percent,
                            now_seconds,
                            (success != 0) ? "complete" : "aborted",
                            0u);
        progress_end_console_line();
        log_emit(LOG_LEVEL_INFO,
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
    }
    progress_notify_callback(PROGRESS_EVENT_FINISH,
                                 g_nlo_log_state.progress_last_step_index,
                                 0u,
                                 z,
                                 z_end,
                                 percent,
                                 0.0,
                                 0.0,
                                 0.0,
                                 now_seconds);

    g_nlo_log_state.progress_active = 0;
    g_nlo_log_state.progress_start_seconds = 0.0;
    g_nlo_log_state.progress_last_render_seconds = 0.0;
    g_nlo_log_state.progress_last_percent = -1.0;
    g_nlo_log_state.progress_last_step_index = 0u;
    g_nlo_log_state.progress_line_visible = 0;
    g_nlo_log_state.progress_last_line_len = 0u;
}
