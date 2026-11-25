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
  #include <io.h>
/* The ring is appended to by the propagation thread of an asynchronous job
 * while the caller's thread reads it, so every ring access takes this lock. */
static SRWLOCK g_ring_lock = SRWLOCK_INIT;
static void ring_lock(void) { AcquireSRWLockExclusive(&g_ring_lock); }
static void ring_unlock(void) { ReleaseSRWLockExclusive(&g_ring_lock); }
#else
  #include <pthread.h>
  #include <sys/ioctl.h>
  #include <sys/time.h>
  #include <unistd.h>
static pthread_mutex_t g_ring_lock = PTHREAD_MUTEX_INITIALIZER;
static void ring_lock(void) { (void)pthread_mutex_lock(&g_ring_lock); }
static void ring_unlock(void) { (void)pthread_mutex_unlock(&g_ring_lock); }
#endif

#define LOG_STATUS_OK 0
#define LOG_STATUS_INVALID_ARGUMENT 1
#define LOG_STATUS_ALLOCATION_FAILED 2

#define LOG_BUFFER_MIN_BYTES 4096u
#define PROGRESS_BAR_WIDTH 30u
#define PROGRESS_MIN_BAR_WIDTH 10u
#define PROGRESS_SUB_BAR_WIDTH 10u
/* ENABLE_VIRTUAL_TERMINAL_PROCESSING, absent from older Windows SDKs. */
#define PROGRESS_ENABLE_VT 0x0004u
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
    size_t progress_accepted_steps;
    size_t progress_rejected_steps;
    double progress_min_step_size;
    double progress_max_step_size;
    double progress_last_step_size;
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

/* What one output stream can show, queried at every render so that a resized
 * terminal is followed. */
typedef struct
{
    int cursor;    /* ANSI cursor movement is interpreted */
    int color;     /* colour is wanted as well (NO_COLOR unset) */
    int utf8;      /* box-drawing glyphs can be written */
    int columns;   /* terminal width, 0 when unknown or not a terminal */
    void* console; /* Windows console handle, NULL elsewhere */
} progress_stream_caps;

static void progress_query_caps(FILE* stream, progress_stream_caps* caps)
{
    memset(caps, 0, sizeof(*caps));
#if defined(_WIN32)
    const int fd = _fileno(stream);
    if (fd < 0 || _isatty(fd) == 0) {
        return;
    }
    const HANDLE handle = (HANDLE)_get_osfhandle(fd);
    DWORD mode = 0u;
    if (handle == INVALID_HANDLE_VALUE || GetConsoleMode(handle, &mode) == 0) {
        return;
    }
    /* Written with WriteConsoleW, so glyphs do not depend on the code page. */
    caps->console = handle;
    caps->utf8 = 1;
    if ((mode & PROGRESS_ENABLE_VT) != 0u || SetConsoleMode(handle, mode | PROGRESS_ENABLE_VT) != 0) {
        caps->cursor = 1;
    }
    CONSOLE_SCREEN_BUFFER_INFO screen;
    if (GetConsoleScreenBufferInfo(handle, &screen) != 0) {
        caps->columns = (int)(screen.srWindow.Right - screen.srWindow.Left) + 1;
    }
#else
    const int fd = fileno(stream);
    if (fd < 0 || isatty(fd) == 0) {
        return;
    }
    const char* term = getenv("TERM");
    caps->cursor = (term != NULL && strcmp(term, "dumb") != 0) ? 1 : 0;
    const char* locale_names[3] = {"LC_ALL", "LC_CTYPE", "LANG"};
    for (size_t idx = 0u; idx < 3u; ++idx) {
        const char* value = getenv(locale_names[idx]);
        if (value != NULL && value[0] != '\0') {
            caps->utf8 = (strstr(value, "UTF-8") != NULL || strstr(value, "utf8") != NULL ||
                          strstr(value, "UTF8") != NULL || strstr(value, "utf-8") != NULL)
                             ? 1
                             : 0;
            break;
        }
    }
    struct winsize size;
    if (ioctl(fd, TIOCGWINSZ, &size) == 0 && size.ws_col > 0) {
        caps->columns = (int)size.ws_col;
    }
#endif
    const char* no_color = getenv("NO_COLOR");
    caps->color = (caps->cursor != 0 && (no_color == NULL || no_color[0] == '\0')) ? 1 : 0;
}

static void progress_stream_write(FILE* stream, const progress_stream_caps* caps, const char* text, size_t text_len)
{
#if defined(_WIN32)
    if (caps->console != NULL) {
        wchar_t wide[2048];
        const int wide_len = MultiByteToWideChar(CP_UTF8, 0, text, (int)text_len, wide, 2048);
        if (wide_len > 0) {
            DWORD written = 0u;
            fflush(stream);
            (void)WriteConsoleW((HANDLE)caps->console, wide, (DWORD)wide_len, &written, NULL);
            return;
        }
    }
#else
    (void)caps;
#endif
    (void)fwrite(text, 1u, text_len, stream);
    fflush(stream);
}

static size_t progress_append(char* out, size_t out_len, size_t pos, const char* fmt, ...)
{
    if (pos >= out_len) {
        return pos;
    }
    va_list args;
    va_start(args, fmt);
    const int written = vsnprintf(out + pos, out_len - pos, fmt, args);
    va_end(args);
    if (written < 0) {
        return pos;
    }
    pos += (size_t)written;
    return (pos < out_len) ? pos : (out_len - 1u);
}

/* One bar of `width` cells, `fraction` of them filled. With colour the filled
 * part takes `color` (a 256-colour index) and the rest is dim; without it the
 * two parts differ by glyph. */
static size_t progress_append_bar(
    char* out,
    size_t out_len,
    size_t pos,
    const progress_stream_caps* caps,
    size_t width,
    double fraction,
    int color
)
{
    if (!(fraction > 0.0)) {
        fraction = 0.0;
    } else if (fraction > 1.0) {
        fraction = 1.0;
    }
    const size_t filled = (size_t)(fraction * (double)width + 0.5);
    const char* heavy = "#";
    const char* light = "-";
    if (caps->color != 0) {
        heavy = (caps->utf8 != 0) ? "\xE2\x94\x81" : "=";
        light = heavy;
        pos = progress_append(out, out_len, pos, "\x1b[38;5;%dm", color);
    }
    for (size_t idx = 0u; idx < filled; ++idx) {
        pos = progress_append(out, out_len, pos, "%s", heavy);
    }
    if (caps->color != 0) {
        pos = progress_append(out, out_len, pos, "\x1b[38;5;240m");
    }
    for (size_t idx = filled; idx < width; ++idx) {
        pos = progress_append(out, out_len, pos, "%s", light);
    }
    if (caps->color != 0) {
        pos = progress_append(out, out_len, pos, "\x1b[0m");
    }
    return pos;
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

/* Position of the current step between the step bounds, on a log scale since
 * the bounds are usually decades apart: 0 at min_step_size, 1 at max. */
static double progress_step_fraction(double step_size)
{
    const double min_step = g_nlo_log_state.progress_min_step_size;
    const double max_step = g_nlo_log_state.progress_max_step_size;
    if (!(step_size > 0.0)) {
        return 0.0;
    }
    if (!(min_step > 0.0) || !(max_step > min_step)) {
        return 1.0;
    }
    return log(step_size / min_step) / log(max_step / min_step);
}

/* One row of the stacked display: a label column, a bar, then text. The bar
 * narrows and then the text is cut to keep the row inside `limit` columns,
 * because a wrapped row cannot be redrawn in place. */
static size_t progress_append_row(
    char* out,
    size_t out_len,
    size_t pos,
    const progress_stream_caps* caps,
    size_t limit,
    const char* label,
    int bold_label,
    double fraction,
    int color,
    const char* text
)
{
    const size_t label_len = strlen(label);
    size_t text_len = strlen(text);
    size_t bar_width = PROGRESS_BAR_WIDTH;
    while (label_len + bar_width + text_len > limit && bar_width > PROGRESS_MIN_BAR_WIDTH) {
        bar_width -= 1u;
    }
    if (label_len + bar_width + text_len > limit) {
        text_len = (limit > label_len + bar_width) ? (limit - label_len - bar_width) : 0u;
    }

    pos = progress_append(out, out_len, pos,
                          (bold_label != 0 && caps->color != 0) ? "\x1b[1m%s\x1b[0m" : "%s", label);
    pos = progress_append_bar(out, out_len, pos, caps, bar_width, fraction, color);
    pos = progress_append(out, out_len, pos, "%.*s\x1b[K", (int)text_len, text);
    return pos;
}

/* Draw the progress display on one stream. A terminal gets three rows redrawn
 * in place:
 *
 *   nlolib     ==========----------  34.5%  step 5,181  elapsed 00:07  eta 00:12
 *   rejected   ==------------------  12.5%  740 of 5,921 attempts
 *   step size  =============-------  0.0331 um  (min 1e-05, max 0.5)
 *
 * Anything else (a pipe, a file, a console without cursor control) gets the
 * same values on one line, which needs only a carriage return. `redraw` is
 * nonzero when the previous display is still the last thing on the stream.
 * Returns the visible length of the one-line form, 0 for the stacked form. */
static size_t progress_draw(
    FILE* stream,
    size_t step_index,
    double percent,
    double elapsed_seconds,
    double remaining_seconds,
    const char* final_state,
    int redraw
)
{
    progress_stream_caps caps;
    progress_query_caps(stream, &caps);

    char step_text[48];
    char eta_text[24];
    char elapsed_text[24];
    char main_text[160];
    (void)log_format_u64_grouped(step_text, sizeof(step_text), (uint64_t)step_index);
    progress_format_duration(remaining_seconds, eta_text, sizeof(eta_text));
    progress_format_duration(elapsed_seconds, elapsed_text, sizeof(elapsed_text));
    if (final_state != NULL) {
        (void)snprintf(main_text, sizeof(main_text), " %5.1f%%  step %s  %s in %s",
                       percent, step_text, final_state, elapsed_text);
    } else {
        (void)snprintf(main_text, sizeof(main_text), " %5.1f%%  step %s  elapsed %s  eta %s",
                       percent, step_text, elapsed_text, eta_text);
    }

    const size_t rejected = g_nlo_log_state.progress_rejected_steps;
    const size_t attempts = g_nlo_log_state.progress_accepted_steps + rejected;
    const double reject_fraction = (attempts > 0u) ? ((double)rejected / (double)attempts) : 0.0;
    const double step_size = g_nlo_log_state.progress_last_step_size;
    const double step_fraction = progress_step_fraction(step_size);
    int main_color = 208;
    if (final_state != NULL) {
        main_color = (percent >= 100.0) ? 40 : 196;
    }

    char line[2048];
    size_t pos = 0u;
    if (caps.cursor != 0) {
        char rejected_text[48];
        char attempts_text[48];
        char reject_text[160];
        char size_text[160];
        (void)log_format_u64_grouped(rejected_text, sizeof(rejected_text), (uint64_t)rejected);
        (void)log_format_u64_grouped(attempts_text, sizeof(attempts_text), (uint64_t)attempts);
        (void)snprintf(reject_text, sizeof(reject_text), " %5.1f%%  %s of %s attempts",
                       100.0 * reject_fraction, rejected_text, attempts_text);
        if (step_size > 0.0) {
            (void)snprintf(size_text, sizeof(size_text), "  %.3g um  (min %.3g, max %.3g)",
                           step_size,
                           g_nlo_log_state.progress_min_step_size,
                           g_nlo_log_state.progress_max_step_size);
        } else {
            (void)snprintf(size_text, sizeof(size_text), "  -- um  (min %.3g, max %.3g)",
                           g_nlo_log_state.progress_min_step_size,
                           g_nlo_log_state.progress_max_step_size);
        }

        const size_t limit = (caps.columns > 1) ? (size_t)(caps.columns - 1) : (size_t)120u;
        pos = progress_append(line, sizeof(line), pos, (redraw != 0) ? "\r\x1b[2A" : "\r");
        pos = progress_append_row(line, sizeof(line), pos, &caps, limit,
                                  "nlolib     ", 1, percent / 100.0, main_color, main_text);
        pos = progress_append(line, sizeof(line), pos, "\n");
        pos = progress_append_row(line, sizeof(line), pos, &caps, limit,
                                  "rejected   ", 0, reject_fraction, 197, reject_text);
        pos = progress_append(line, sizeof(line), pos, "\n");
        pos = progress_append_row(line, sizeof(line), pos, &caps, limit,
                                  "step size  ", 0, step_fraction, 39, size_text);
        progress_stream_write(stream, &caps, line, pos);
        return 0u;
    }

    char tail_text[160];
    if (step_size > 0.0) {
        (void)snprintf(tail_text, sizeof(tail_text), "  rejected %.1f%%  step size %.3g um",
                       100.0 * reject_fraction, step_size);
    } else {
        (void)snprintf(tail_text, sizeof(tail_text), "  rejected %.1f%%  step size -- um",
                       100.0 * reject_fraction);
    }
    static const char k_name[] = "nlolib ";
    const size_t visible = (sizeof(k_name) - 1u) + PROGRESS_BAR_WIDTH + strlen(main_text) + strlen(tail_text);
    pos = progress_append(line, sizeof(line), pos, "\r%s", k_name);
    pos = progress_append_bar(line, sizeof(line), pos, &caps, PROGRESS_BAR_WIDTH, percent / 100.0, main_color);
    pos = progress_append(line, sizeof(line), pos, "%s%s", main_text, tail_text);
    /* Blank what is left of a longer previous line. */
    for (size_t idx = visible; idx < g_nlo_log_state.progress_last_line_len; ++idx) {
        pos = progress_append(line, sizeof(line), pos, " ");
    }
    progress_stream_write(stream, &caps, line, pos);
    return visible;
}

static void progress_render(
    size_t step_index,
    double percent,
    double now_seconds,
    const char* final_state
)
{
    if (percent < 0.0) {
        percent = 0.0;
    } else if (percent > 100.0) {
        percent = 100.0;
    }

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

    const int redraw = g_nlo_log_state.progress_line_visible;
    size_t visible = 0u;
    if (g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_STDOUT ||
        g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_BOTH) {
        visible = progress_draw(stdout, step_index, percent, elapsed_seconds, remaining_seconds, final_state, redraw);
    }
    if (g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_STDERR ||
        g_nlo_log_state.progress_stream_mode == LOG_PROGRESS_STREAM_BOTH) {
        visible = progress_draw(stderr, step_index, percent, elapsed_seconds, remaining_seconds, final_state, redraw);
    }

    g_nlo_log_state.progress_last_line_len = visible;
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
        isfinite(eta_seconds) ? eta_seconds : -1.0,
        g_nlo_log_state.progress_accepted_steps,
        g_nlo_log_state.progress_rejected_steps,
        g_nlo_log_state.progress_min_step_size,
        g_nlo_log_state.progress_max_step_size
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
        ring_lock();
        free(g_nlo_log_state.ring);
        g_nlo_log_state.ring = NULL;
        g_nlo_log_state.ring_capacity = 0u;
        g_nlo_log_state.ring_start = 0u;
        g_nlo_log_state.ring_size = 0u;
        ring_unlock();
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

    ring_lock();
    free(g_nlo_log_state.ring);
    g_nlo_log_state.ring = new_ring;
    g_nlo_log_state.ring_capacity = capacity;
    g_nlo_log_state.ring_start = 0u;
    g_nlo_log_state.ring_size = 0u;
    ring_unlock();
    return LOG_STATUS_OK;
}

int log_clear_buffer(void)
{
    ring_lock();
    g_nlo_log_state.ring_start = 0u;
    g_nlo_log_state.ring_size = 0u;
    ring_unlock();
    return LOG_STATUS_OK;
}

int log_read_buffer(char* dst, size_t dst_bytes, size_t* out_written, int consume)
{
    if (out_written == NULL || dst == NULL || dst_bytes == 0u) {
        return LOG_STATUS_INVALID_ARGUMENT;
    }

    *out_written = 0u;
    dst[0] = '\0';
    ring_lock();
    if (g_nlo_log_state.ring == NULL || g_nlo_log_state.ring_capacity == 0u || g_nlo_log_state.ring_size == 0u) {
        ring_unlock();
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
    ring_unlock();

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
    ring_lock();
    if (g_nlo_log_state.ring != NULL && g_nlo_log_state.ring_capacity > 0u) {
        log_ring_append(text, text_len);
        wrote_to_sink = 1;
    }
    ring_unlock();

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

void log_progress_begin(double z_start, double z_end, double min_step_size, double max_step_size)
{
    if ((g_nlo_log_state.progress_enabled == 0 && g_nlo_log_state.progress_callback == NULL) ||
        !(z_end > z_start)) {
        g_nlo_log_state.progress_active = 0;
        return;
    }

    g_nlo_log_state.progress_active = 1;
    g_nlo_log_state.progress_z_start = z_start;
    g_nlo_log_state.progress_z_end = z_end;
    g_nlo_log_state.progress_accepted_steps = 0u;
    g_nlo_log_state.progress_rejected_steps = 0u;
    g_nlo_log_state.progress_min_step_size = min_step_size;
    g_nlo_log_state.progress_max_step_size = max_step_size;
    g_nlo_log_state.progress_last_step_size = 0.0;
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
    g_nlo_log_state.progress_accepted_steps += 1u;
    g_nlo_log_state.progress_last_step_size = step_size;

    const double adjust_tol = fmax(fabs(step_size) * 1e-12, 1e-15);
    const int is_step_adjust = (fabs(next_step_size - step_size) > adjust_tol) ? 1 : 0;
    const int force_render = (g_nlo_log_state.progress_emit_on_step_adjust != 0 && is_step_adjust != 0) ? 1 : 0;

    if (g_nlo_log_state.progress_enabled != 0 &&
        progress_should_render(percent, now_seconds, force_render) != 0) {
        progress_render(step_index, percent, now_seconds, NULL);
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
    g_nlo_log_state.progress_rejected_steps += 1u;
    g_nlo_log_state.progress_last_step_size = attempted_step;
    /* Rejections do not advance z, so they are drawn on the clock alone: a
     * run of them must not redraw the line once per attempt. */
    if (g_nlo_log_state.progress_enabled != 0 &&
        (g_nlo_log_state.progress_last_percent < 0.0 ||
         (now_seconds - g_nlo_log_state.progress_last_render_seconds) >= PROGRESS_RENDER_MIN_SECONDS)) {
        progress_render(step_index, percent, now_seconds, NULL);
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
                            (success != 0) ? "complete" : "aborted");
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
