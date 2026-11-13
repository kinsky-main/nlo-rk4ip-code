/**
 * @file progress_mex.c
 * @brief Live progress and log output for the nlolib MATLAB wrapper.
 *
 * loadlibrary() cannot pass MATLAB functions as C callbacks, and nlolib's own
 * stdout/stderr output never reaches the MATLAB Command Window.  This MEX file
 * provides a C progress_callback that nlolib calls on the MATLAB thread during
 * nlolib_propagate().  Each call drains nlolib's in-memory log buffer and
 * prints progress with mexPrintf, flushing the Command Window at most once
 * per RENDER_MIN_SECONDS.  Ctrl+C makes the callback return 0, which aborts
 * the propagation.
 *
 * Usage from MATLAB (see NLolib.propagate):
 *   address = progress_mex('callback', libraryPath, style, milestonePercent)
 *     Returns the callback address as uint64 for
 *     propagate_options.progress_callback.  libraryPath is the nlolib shared
 *     library MATLAB already loaded; it is used to look up
 *     nlolib_read_log_buffer in that same instance.  style is 'inline'
 *     (one line redrawn with backspaces) or 'lines' (a new line each
 *     milestonePercent, for Command Windows that do not honour backspace).
 */

#include "mex.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

/* Undocumented but long-standing libut entry points for Ctrl+C detection. */
#ifdef __cplusplus
extern "C" bool utIsInterruptPending(void);
#else
extern bool utIsInterruptPending(void);
#endif

/* Mirrors progress_event_type / progress_info in nlolib.h. */
typedef enum
{
    PROGRESS_EVENT_ACCEPTED = 0,
    PROGRESS_EVENT_REJECTED = 1,
    PROGRESS_EVENT_FINISH = 2
} progress_event_type;

typedef struct
{
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

typedef int (*read_log_buffer_fn)(char* dst, size_t dst_bytes, size_t* out_written, int consume);

#define BAR_WIDTH 30
#define RENDER_MIN_SECONDS 1.0
#define LOG_CHUNK_BYTES 65536

static read_log_buffer_fn g_read_log_buffer = NULL;
static char g_log_chunk[LOG_CHUNK_BYTES];
static size_t g_line_len = 0u;      /* characters of the progress line on screen */
static double g_last_render = -1.0; /* elapsed_seconds of the last redraw */
static int g_lines_style = 0;       /* 1: print milestone lines, no backspaces */
static double g_milestone = 5.0;    /* percent between lines in lines style */
static double g_next_milestone = 0.0;

static void format_duration(double seconds, char* out, size_t out_len)
{
    if (!(seconds >= 0.0) || !isfinite(seconds)) {
        snprintf(out, out_len, "--:--");
        return;
    }
    const unsigned long total = (unsigned long)(seconds + 0.5);
    const unsigned long hours = total / 3600u;
    if (hours > 0u) {
        snprintf(out, out_len, "%lu:%02lu:%02lu", hours, (total / 60u) % 60u, total % 60u);
    } else {
        snprintf(out, out_len, "%02lu:%02lu", total / 60u, total % 60u);
    }
}

/* Erase the progress line so that log text starts on a clean line. */
static void erase_progress_line(void)
{
    if (g_line_len == 0u) {
        return;
    }
    char erase[512];
    size_t n = (g_line_len < sizeof(erase) - 1u) ? g_line_len : sizeof(erase) - 1u;
    memset(erase, '\b', n);
    erase[n] = '\0';
    mexPrintf("%s", erase);
    g_line_len = 0u;
}

/* Print everything nlolib logged since the last drain. Returns 1 if any. */
static int drain_logs(void)
{
    if (g_read_log_buffer == NULL) {
        return 0;
    }
    int printed = 0;
    for (;;) {
        size_t written = 0u;
        if (g_read_log_buffer(g_log_chunk, sizeof(g_log_chunk), &written, 1) != 0 || written == 0u) {
            break;
        }
        if (!printed) {
            erase_progress_line();
            printed = 1;
        }
        g_log_chunk[(written < sizeof(g_log_chunk)) ? written : sizeof(g_log_chunk) - 1u] = '\0';
        mexPrintf("%s", g_log_chunk);
        if (written < sizeof(g_log_chunk) - 1u) {
            break;
        }
    }
    return printed;
}

static void render(const progress_info* info, const char* state)
{
    char bar[BAR_WIDTH + 1];
    double percent = info->percent < 0.0 ? 0.0 : (info->percent > 100.0 ? 100.0 : info->percent);
    int filled = (int)(percent / 100.0 * BAR_WIDTH + 0.5);
    for (int idx = 0; idx < BAR_WIDTH; ++idx) {
        bar[idx] = (idx < filled) ? '#' : '-';
    }
    bar[BAR_WIDTH] = '\0';

    char eta[24];
    char elapsed[24];
    format_duration(info->eta_seconds, eta, sizeof(eta));
    format_duration(info->elapsed_seconds, elapsed, sizeof(elapsed));

    char line[256];
    int len = snprintf(line, sizeof(line),
                       "[nlolib] [%s] %6.2f%% |%s| step %llu eta %s elapsed %s",
                       state, percent, bar, (unsigned long long)info->step_index, eta, elapsed);
    if (len < 0) {
        return;
    }
    if (g_lines_style) {
        mexPrintf("%s\n", line);
        return;
    }
    size_t new_len = strlen(line);

    /* Backspace over the old line, then pad so leftovers are overwritten. */
    char out[768];
    size_t pos = 0u;
    for (size_t idx = 0u; idx < g_line_len && pos < sizeof(out) - 1u; ++idx) {
        out[pos++] = '\b';
    }
    for (size_t idx = 0u; idx < new_len && pos < sizeof(out) - 1u; ++idx) {
        out[pos++] = line[idx];
    }
    size_t padded = new_len;
    while (padded < g_line_len && pos + 2u < sizeof(out)) {
        out[pos++] = ' ';
        ++padded;
    }
    while (padded > new_len && pos < sizeof(out) - 1u) {
        out[pos++] = '\b';
        --padded;
    }
    out[pos] = '\0';
    mexPrintf("%s", out);
    g_line_len = new_len;
}

static void flush_command_window(void)
{
    /* drawnow repaints the Command Window while MATLAB is busy in calllib. */
    (void)mexEvalString("drawnow;");
}

static int progress_callback(const progress_info* info, void* user_data)
{
    (void)user_data;
    if (info == NULL) {
        return 1;
    }

    const int finished = (info->event_type == PROGRESS_EVENT_FINISH);
    const int logged = drain_logs();
    const int due = (g_last_render < 0.0) ||
                    (info->elapsed_seconds - g_last_render >= RENDER_MIN_SECONDS);

    if (finished) {
        const int complete = (info->percent >= 100.0 - 1e-9);
        render(info, complete ? "complete" : "aborted");
        if (!g_lines_style) {
            mexPrintf("\n");
        }
        g_line_len = 0u;
        g_last_render = -1.0;
        g_next_milestone = 0.0;
        flush_command_window();
        return 1;
    }

    /* Redraw and flush on a fixed cadence only, so the Command Window never
     * receives more updates than it can paint.  Lines style additionally
     * waits for the next milestone. */
    if (due) {
        int draw = 1;
        if (g_lines_style) {
            /* The FINISH event prints the 100% line. */
            draw = (info->percent >= g_next_milestone) && (info->percent < 100.0 - 1e-9);
            while (g_next_milestone <= info->percent) {
                g_next_milestone += g_milestone;
            }
        }
        if (draw) {
            render(info, (info->event_type == PROGRESS_EVENT_REJECTED) ? "reject" : "running");
        }
        if (draw || logged) {
            g_last_render = info->elapsed_seconds;
            flush_command_window();
        }
    }

    return utIsInterruptPending() ? 0 : 1;
}

static read_log_buffer_fn resolve_read_log_buffer(const char* library_path)
{
#if defined(_WIN32)
    /* LoadLibrary on an already-loaded module returns that same instance. */
    HMODULE module = LoadLibraryA(library_path);
    if (module == NULL) {
        return NULL;
    }
    return (read_log_buffer_fn)(void*)GetProcAddress(module, "nlolib_read_log_buffer");
#else
    /* RTLD_NOLOAD: only attach to the copy loadlibrary already mapped. */
    void* handle = dlopen(library_path, RTLD_NOW | RTLD_NOLOAD);
    if (handle == NULL) {
        return NULL;
    }
    return (read_log_buffer_fn)dlsym(handle, "nlolib_read_log_buffer");
#endif
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 4 || !mxIsChar(prhs[0]) || !mxIsChar(prhs[1]) || !mxIsChar(prhs[2]) ||
        !mxIsNumeric(prhs[3]) || mxGetNumberOfElements(prhs[3]) != 1) {
        mexErrMsgIdAndTxt("nlolib:progressMexUsage",
                          "Usage: address = progress_mex('callback', libraryPath, style, milestonePercent)");
    }
    char command[32];
    if (mxGetString(prhs[0], command, sizeof(command)) != 0 || strcmp(command, "callback") != 0) {
        mexErrMsgIdAndTxt("nlolib:progressMexUsage", "Unknown command; expected 'callback'.");
    }
    char* library_path = mxArrayToUTF8String(prhs[1]);
    if (library_path == NULL) {
        mexErrMsgIdAndTxt("nlolib:progressMexUsage", "libraryPath must be a char vector.");
    }
    g_read_log_buffer = resolve_read_log_buffer(library_path);
    mxFree(library_path);

    char style[16];
    if (mxGetString(prhs[2], style, sizeof(style)) != 0 ||
        (strcmp(style, "inline") != 0 && strcmp(style, "lines") != 0)) {
        mexErrMsgIdAndTxt("nlolib:progressMexUsage", "style must be 'inline' or 'lines'.");
    }
    g_lines_style = (strcmp(style, "lines") == 0);
    const double milestone = mxGetScalar(prhs[3]);
    g_milestone = (milestone >= 1.0 && milestone <= 100.0) ? milestone : 5.0;
    g_next_milestone = 0.0;

    /* nlolib holds this address during propagate; keep the MEX loaded. */
    if (!mexIsLocked()) {
        mexLock();
    }
    g_line_len = 0u;
    g_last_render = -1.0;

    (void)nlhs;
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    ((uint64_t*)mxGetData(plhs[0]))[0] = (uint64_t)(uintptr_t)&progress_callback;
}
