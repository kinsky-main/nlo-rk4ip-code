/**
 * @file bench_solver_backend.c
 * @brief End-to-end CPU vs GPU solver benchmark harness.
 */

#include "backend/nlo_complex.h"
#include "core/init.h"
#include "core/state.h"
#include "numerics/rk4_kernel.h"
#include "vulkan_bench_context.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifndef NLO_BENCH_MAX_SIZES
#define NLO_BENCH_MAX_SIZES 32u
#endif

#ifndef NLO_BENCH_MAX_PATH
#define NLO_BENCH_MAX_PATH 1024u
#endif

#ifndef NLO_BENCH_NOTE_CAP
#define NLO_BENCH_NOTE_CAP 256u
#endif

#ifndef NLO_BENCH_PI
#define NLO_BENCH_PI 3.14159265358979323846
#endif

typedef enum {
    NLO_BENCH_BACKEND_CPU = 0,
    NLO_BENCH_BACKEND_GPU = 1,
    NLO_BENCH_BACKEND_BOTH = 2
} nlo_bench_backend_request;

typedef enum {
    NLO_BENCH_RUNTIME_CPU = 0,
    NLO_BENCH_RUNTIME_GPU = 1
} nlo_bench_runtime_backend;

typedef struct {
    nlo_bench_backend_request backend;
    size_t sizes[NLO_BENCH_MAX_SIZES];
    size_t size_count;
    size_t warmup_runs;
    size_t measured_runs;
    char csv_path[NLO_BENCH_MAX_PATH];
} nlo_bench_options;

typedef struct {
    size_t sample_count;
    sim_config* config;
    nlo_complex* input_field;
    nlo_complex* output_field;
} nlo_bench_case_data;

typedef struct {
    double init_ms;
    double upload_ms;
    double solve_ms;
    double download_ms;
    double teardown_ms;
    double total_ms;
    double samples_per_sec;
} nlo_bench_run_metrics;

typedef struct {
    double mean_ms;
    double median_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
} nlo_bench_summary;

typedef struct {
#if defined(_WIN32)
    LARGE_INTEGER value;
#else
    struct timespec value;
#endif
} nlo_bench_timestamp;

static void nlo_bench_copy_note(
    char* note,
    size_t note_capacity,
    const char* text
)
{
    if (note == NULL || note_capacity == 0u) {
        return;
    }

    if (text == NULL) {
        note[0] = '\0';
        return;
    }

#if defined(_MSC_VER)
    strncpy_s(note, note_capacity, text, _TRUNCATE);
#else
    snprintf(note, note_capacity, "%s", text);
#endif
}

static int nlo_bench_string_equals_ci(const char* lhs, const char* rhs)
{
    if (lhs == NULL || rhs == NULL) {
        return 0;
    }

    while (*lhs != '\0' && *rhs != '\0') {
        char a = *lhs;
        char b = *rhs;

        if (a >= 'A' && a <= 'Z') {
            a = (char)(a - 'A' + 'a');
        }
        if (b >= 'A' && b <= 'Z') {
            b = (char)(b - 'A' + 'a');
        }
        if (a != b) {
            return 0;
        }

        ++lhs;
        ++rhs;
    }

    return (*lhs == '\0' && *rhs == '\0') ? 1 : 0;
}

static void nlo_bench_print_usage(const char* executable_name)
{
    printf("Usage: %s [options]\n", executable_name);
    printf("Options:\n");
    printf("  --backend=cpu|gpu|both\n");
    printf("  --sizes=1024,4096,16384,65536\n");
    printf("  --warmup=N\n");
    printf("  --runs=N\n");
    printf("  --csv=path/to/results.csv\n");
    printf("  --help\n");
}

static void nlo_bench_set_default_options(nlo_bench_options* options)
{
    static const size_t default_sizes[] = {1024u, 4096u, 16384u, 65536u};
    if (options == NULL) {
        return;
    }

    memset(options, 0, sizeof(*options));
    options->backend = NLO_BENCH_BACKEND_BOTH;
    options->warmup_runs = 2u;
    options->measured_runs = 8u;

    options->size_count = sizeof(default_sizes) / sizeof(default_sizes[0]);
    for (size_t i = 0u; i < options->size_count; ++i) {
        options->sizes[i] = default_sizes[i];
    }

    snprintf(options->csv_path,
             sizeof(options->csv_path),
             "benchmarks/results/solver_backend.csv");
}

static int nlo_bench_parse_unsigned_size(const char* text, size_t* out_value)
{
    if (text == NULL || out_value == NULL || *text == '\0') {
        return -1;
    }

    errno = 0;
    char* end_ptr = NULL;
    unsigned long long value = strtoull(text, &end_ptr, 10);
    if (errno != 0 || end_ptr == text || *end_ptr != '\0' || value == 0u) {
        return -1;
    }

    *out_value = (size_t)value;
    return 0;
}

static int nlo_bench_parse_size_list(const char* text, nlo_bench_options* options)
{
    if (text == NULL || options == NULL || *text == '\0') {
        return -1;
    }

    size_t count = 0u;
    const char* cursor = text;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }

        errno = 0;
        char* end_ptr = NULL;
        unsigned long long value = strtoull(cursor, &end_ptr, 10);
        if (errno != 0 || end_ptr == cursor || value == 0u) {
            return -1;
        }
        if (count >= NLO_BENCH_MAX_SIZES) {
            return -1;
        }

        options->sizes[count] = (size_t)value;
        count += 1u;

        cursor = end_ptr;
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == ',') {
            ++cursor;
            continue;
        }
        if (*cursor != '\0') {
            return -1;
        }
    }

    if (count == 0u) {
        return -1;
    }

    options->size_count = count;
    return 0;
}

static int nlo_bench_parse_options(
    int argc,
    char** argv,
    nlo_bench_options* options
)
{
    if (argv == NULL || options == NULL) {
        return -1;
    }

    nlo_bench_set_default_options(options);
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            nlo_bench_print_usage(argv[0]);
            return 1;
        }

        if (strncmp(arg, "--backend=", 10) == 0) {
            const char* value = arg + 10;
            if (nlo_bench_string_equals_ci(value, "cpu")) {
                options->backend = NLO_BENCH_BACKEND_CPU;
                continue;
            }
            if (nlo_bench_string_equals_ci(value, "gpu")) {
                options->backend = NLO_BENCH_BACKEND_GPU;
                continue;
            }
            if (nlo_bench_string_equals_ci(value, "both")) {
                options->backend = NLO_BENCH_BACKEND_BOTH;
                continue;
            }

            fprintf(stderr, "Invalid backend option: %s\n", value);
            nlo_bench_print_usage(argv[0]);
            return -1;
        }

        if (strncmp(arg, "--sizes=", 8) == 0) {
            if (nlo_bench_parse_size_list(arg + 8, options) != 0) {
                fprintf(stderr, "Invalid sizes list: %s\n", arg + 8);
                return -1;
            }
            continue;
        }

        if (strncmp(arg, "--warmup=", 9) == 0) {
            size_t warmup_runs = 0u;
            if (nlo_bench_parse_unsigned_size(arg + 9, &warmup_runs) != 0) {
                fprintf(stderr, "Invalid warmup count: %s\n", arg + 9);
                return -1;
            }
            options->warmup_runs = warmup_runs;
            continue;
        }

        if (strncmp(arg, "--runs=", 7) == 0) {
            size_t measured_runs = 0u;
            if (nlo_bench_parse_unsigned_size(arg + 7, &measured_runs) != 0) {
                fprintf(stderr, "Invalid measured run count: %s\n", arg + 7);
                return -1;
            }
            options->measured_runs = measured_runs;
            continue;
        }

        if (strncmp(arg, "--csv=", 6) == 0) {
            const char* value = arg + 6;
            if (*value == '\0') {
                fprintf(stderr, "Invalid CSV path.\n");
                return -1;
            }
            snprintf(options->csv_path, sizeof(options->csv_path), "%s", value);
            continue;
        }

        fprintf(stderr, "Unknown option: %s\n", arg);
        nlo_bench_print_usage(argv[0]);
        return -1;
    }

    return 0;
}

static void nlo_bench_now(nlo_bench_timestamp* out_time)
{
    if (out_time == NULL) {
        return;
    }

#if defined(_WIN32)
    (void)QueryPerformanceCounter(&out_time->value);
#else
    (void)clock_gettime(CLOCK_MONOTONIC, &out_time->value);
#endif
}

static double nlo_bench_elapsed_ms(
    const nlo_bench_timestamp* start_time,
    const nlo_bench_timestamp* end_time
)
{
    if (start_time == NULL || end_time == NULL) {
        return 0.0;
    }

#if defined(_WIN32)
    static LARGE_INTEGER frequency = {0};
    if (frequency.QuadPart == 0) {
        (void)QueryPerformanceFrequency(&frequency);
    }

    const LONGLONG ticks = end_time->value.QuadPart - start_time->value.QuadPart;
    if (frequency.QuadPart == 0) {
        return 0.0;
    }
    return ((double)ticks * 1000.0) / (double)frequency.QuadPart;
#else
    const time_t sec = end_time->value.tv_sec - start_time->value.tv_sec;
    const long nsec = end_time->value.tv_nsec - start_time->value.tv_nsec;
    return ((double)sec * 1000.0) + ((double)nsec / 1000000.0);
#endif
}

static const char* nlo_bench_backend_label(nlo_bench_runtime_backend backend)
{
    return (backend == NLO_BENCH_RUNTIME_CPU) ? "cpu" : "gpu";
}

static int nlo_bench_file_exists(const char* path)
{
    if (path == NULL) {
        return 0;
    }

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    fclose(file);
    return 1;
}

static int nlo_bench_mkdir_single(const char* path)
{
    if (path == NULL || *path == '\0') {
        return 0;
    }

#if defined(_WIN32)
    int status = _mkdir(path);
#else
    int status = mkdir(path, 0777);
#endif

    if (status == 0 || errno == EEXIST) {
        return 0;
    }
    return -1;
}

static int nlo_bench_make_parent_dirs(const char* file_path)
{
    if (file_path == NULL || *file_path == '\0') {
        return -1;
    }

    char buffer[NLO_BENCH_MAX_PATH];
    snprintf(buffer, sizeof(buffer), "%s", file_path);

    char* last_slash = strrchr(buffer, '/');
    char* last_backslash = strrchr(buffer, '\\');
    char* separator = last_slash;
    if (last_backslash != NULL && (separator == NULL || last_backslash > separator)) {
        separator = last_backslash;
    }

    if (separator == NULL) {
        return 0;
    }

    *separator = '\0';
    const size_t len = strlen(buffer);
    if (len == 0u) {
        return 0;
    }

    for (size_t i = 0u; i < len; ++i) {
        if (buffer[i] != '/' && buffer[i] != '\\') {
            continue;
        }

        if (i == 0u) {
            continue;
        }
        if (i == 2u && buffer[1] == ':') {
            continue;
        }

        const char saved = buffer[i];
        buffer[i] = '\0';
        if (nlo_bench_mkdir_single(buffer) != 0) {
            buffer[i] = saved;
            return -1;
        }
        buffer[i] = saved;
    }

    return nlo_bench_mkdir_single(buffer);
}

static void nlo_bench_iso8601_utc(char* out_text, size_t out_capacity)
{
    if (out_text == NULL || out_capacity == 0u) {
        return;
    }

    const time_t now = time(NULL);
    struct tm utc_tm;

#if defined(_WIN32)
    (void)gmtime_s(&utc_tm, &now);
#else
    (void)gmtime_r(&now, &utc_tm);
#endif

    if (strftime(out_text, out_capacity, "%Y-%m-%dT%H:%M:%SZ", &utc_tm) == 0u) {
        out_text[0] = '\0';
    }
}

static void nlo_bench_sanitize_note(char* text)
{
    if (text == NULL) {
        return;
    }

    for (char* p = text; *p != '\0'; ++p) {
        if (*p == ',' || *p == '\n' || *p == '\r') {
            *p = ';';
        }
    }
}

static int nlo_bench_write_csv_header(FILE* csv_file)
{
    if (csv_file == NULL) {
        return -1;
    }

    const int written = fprintf(csv_file,
                                "timestamp_utc,backend,size,warmup_runs,measured_runs,run_index,"
                                "init_ms,upload_ms,solve_ms,download_ms,teardown_ms,total_ms,"
                                "samples_per_sec,status,notes\n");
    if (written < 0) {
        return -1;
    }

    return fflush(csv_file);
}

static int nlo_bench_write_csv_row(
    FILE* csv_file,
    const char* backend_label,
    size_t size,
    size_t warmup_runs,
    size_t measured_runs,
    size_t run_index,
    const nlo_bench_run_metrics* metrics,
    const char* status,
    const char* notes
)
{
    if (csv_file == NULL || backend_label == NULL || metrics == NULL || status == NULL) {
        return -1;
    }

    char timestamp_utc[32];
    nlo_bench_iso8601_utc(timestamp_utc, sizeof(timestamp_utc));

    char note_buffer[NLO_BENCH_NOTE_CAP];
    nlo_bench_copy_note(note_buffer, sizeof(note_buffer), (notes == NULL) ? "" : notes);
    nlo_bench_sanitize_note(note_buffer);

    const int written = fprintf(csv_file,
                                "%s,%s,%zu,%zu,%zu,%zu,"
                                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s\n",
                                timestamp_utc,
                                backend_label,
                                size,
                                warmup_runs,
                                measured_runs,
                                run_index,
                                metrics->init_ms,
                                metrics->upload_ms,
                                metrics->solve_ms,
                                metrics->download_ms,
                                metrics->teardown_ms,
                                metrics->total_ms,
                                metrics->samples_per_sec,
                                status,
                                note_buffer);
    if (written < 0) {
        return -1;
    }

    return fflush(csv_file);
}

static int nlo_bench_prepare_case_data(
    size_t sample_count,
    nlo_bench_case_data* out_case_data,
    char* note,
    size_t note_capacity
)
{
    if (sample_count == 0u || out_case_data == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Invalid benchmark case configuration.");
        return -1;
    }

    memset(out_case_data, 0, sizeof(*out_case_data));
    out_case_data->sample_count = sample_count;
    out_case_data->config = create_sim_config(sample_count);
    if (out_case_data->config == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Failed to allocate sim_config.");
        return -1;
    }

    out_case_data->input_field = (nlo_complex*)calloc(sample_count, sizeof(nlo_complex));
    out_case_data->output_field = (nlo_complex*)calloc(sample_count, sizeof(nlo_complex));
    if (out_case_data->input_field == NULL || out_case_data->output_field == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Failed to allocate benchmark input/output buffers.");
        free(out_case_data->input_field);
        free(out_case_data->output_field);
        out_case_data->input_field = NULL;
        out_case_data->output_field = NULL;
        free_sim_config(out_case_data->config);
        out_case_data->config = NULL;
        return -1;
    }

    sim_config* config = out_case_data->config;
    config->runtime.dispersion_factor_expr = NULL;
    config->runtime.dispersion_expr = NULL;
    config->runtime.nonlinear_expr = NULL;
    config->runtime.num_constants = 0u;

    config->propagation.starting_step_size = 0.002;
    config->propagation.max_step_size = 0.020;
    config->propagation.min_step_size = 0.0002;
    config->propagation.error_tolerance = 1e-6;
    config->propagation.propagation_distance = 0.4;

    config->time.delta_time = 1.0 / (double)sample_count;
    config->time.pulse_period = 1.0;

    for (size_t i = 0u; i < sample_count; ++i) {
        const long centered = (i <= sample_count / 2u)
                                  ? (long)i
                                  : (long)i - (long)sample_count;
        const double omega = (2.0 * NLO_BENCH_PI * (double)centered) / (double)sample_count;
        config->frequency.frequency_grid[i] = nlo_make(omega, 0.0);
    }

    const double half_span = 0.5 * (double)(sample_count - 1u);
    for (size_t i = 0u; i < sample_count; ++i) {
        const double t = ((double)i - half_span) * config->time.delta_time;
        const double envelope = exp(-(t * t) / 0.02);
        const double phase = 4.0 * t * t;
        out_case_data->input_field[i] = nlo_make(envelope * cos(phase), envelope * sin(phase));
    }

    nlo_bench_copy_note(note, note_capacity, "");
    return 0;
}

static void nlo_bench_destroy_case_data(nlo_bench_case_data* case_data)
{
    if (case_data == NULL) {
        return;
    }

    free(case_data->input_field);
    free(case_data->output_field);
    free_sim_config(case_data->config);

    case_data->sample_count = 0u;
    case_data->config = NULL;
    case_data->input_field = NULL;
    case_data->output_field = NULL;
}

static int nlo_bench_execute_single_run(
    const nlo_execution_options* exec_options,
    const nlo_bench_case_data* case_data,
    nlo_bench_run_metrics* out_metrics,
    char* note,
    size_t note_capacity
)
{
    if (exec_options == NULL || case_data == NULL || out_metrics == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Invalid run arguments.");
        return -1;
    }

    if (case_data->config == NULL ||
        case_data->input_field == NULL ||
        case_data->output_field == NULL ||
        case_data->sample_count == 0u) {
        nlo_bench_copy_note(note, note_capacity, "Benchmark run data is uninitialized.");
        return -1;
    }

    memset(out_metrics, 0, sizeof(*out_metrics));

    simulation_state* state = NULL;
    nlo_bench_timestamp total_start;
    nlo_bench_timestamp total_end;
    nlo_bench_timestamp phase_start;
    nlo_bench_timestamp phase_end;

    nlo_bench_now(&total_start);

    nlo_bench_now(&phase_start);
    if (nlo_init_simulation_state(case_data->config,
                                  case_data->sample_count,
                                  1u,
                                  exec_options,
                                  NULL,
                                  &state) != 0 || state == NULL) {
        nlo_bench_copy_note(note, note_capacity, "State initialization failed.");
        return -1;
    }
    nlo_bench_now(&phase_end);
    out_metrics->init_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

    nlo_bench_now(&phase_start);
    nlo_vec_status upload_status = simulation_state_upload_initial_field(state, case_data->input_field);
    nlo_bench_now(&phase_end);
    out_metrics->upload_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
    if (upload_status != NLO_VEC_STATUS_OK) {
        nlo_bench_now(&phase_start);
        free_simulation_state(state);
        nlo_bench_now(&phase_end);
        out_metrics->teardown_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
        nlo_bench_now(&total_end);
        out_metrics->total_ms = nlo_bench_elapsed_ms(&total_start, &total_end);
        snprintf(note,
                 note_capacity,
                 "Initial field upload failed (status=%d).",
                 (int)upload_status);
        return -1;
    }

    nlo_bench_now(&phase_start);
    solve_rk4(state);
    nlo_bench_now(&phase_end);
    out_metrics->solve_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

    nlo_bench_now(&phase_start);
    nlo_vec_status download_status = simulation_state_download_current_field(state, case_data->output_field);
    nlo_bench_now(&phase_end);
    out_metrics->download_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
    if (download_status != NLO_VEC_STATUS_OK) {
        nlo_bench_now(&phase_start);
        free_simulation_state(state);
        nlo_bench_now(&phase_end);
        out_metrics->teardown_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
        nlo_bench_now(&total_end);
        out_metrics->total_ms = nlo_bench_elapsed_ms(&total_start, &total_end);
        snprintf(note,
                 note_capacity,
                 "Current field download failed (status=%d).",
                 (int)download_status);
        return -1;
    }

    nlo_bench_now(&phase_start);
    free_simulation_state(state);
    nlo_bench_now(&phase_end);
    out_metrics->teardown_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

    nlo_bench_now(&total_end);
    out_metrics->total_ms = nlo_bench_elapsed_ms(&total_start, &total_end);

    if (out_metrics->total_ms > 0.0) {
        out_metrics->samples_per_sec = ((double)case_data->sample_count * 1000.0) / out_metrics->total_ms;
    } else {
        out_metrics->samples_per_sec = 0.0;
    }

    nlo_bench_copy_note(note, note_capacity, "");
    return 0;
}

static int nlo_bench_double_compare(const void* lhs, const void* rhs)
{
    const double left = *(const double*)lhs;
    const double right = *(const double*)rhs;
    if (left < right) {
        return -1;
    }
    if (left > right) {
        return 1;
    }
    return 0;
}

static int nlo_bench_compute_summary(
    const double* values,
    size_t count,
    nlo_bench_summary* out_summary
)
{
    if (values == NULL || count == 0u || out_summary == NULL) {
        return -1;
    }

    memset(out_summary, 0, sizeof(*out_summary));
    out_summary->min_ms = values[0];
    out_summary->max_ms = values[0];

    double sum = 0.0;
    for (size_t i = 0u; i < count; ++i) {
        const double value = values[i];
        sum += value;
        if (value < out_summary->min_ms) {
            out_summary->min_ms = value;
        }
        if (value > out_summary->max_ms) {
            out_summary->max_ms = value;
        }
    }
    out_summary->mean_ms = sum / (double)count;

    double variance = 0.0;
    for (size_t i = 0u; i < count; ++i) {
        const double delta = values[i] - out_summary->mean_ms;
        variance += delta * delta;
    }
    out_summary->stddev_ms = sqrt(variance / (double)count);

    double* sorted = (double*)calloc(count, sizeof(double));
    if (sorted == NULL) {
        return -1;
    }
    memcpy(sorted, values, count * sizeof(double));
    qsort(sorted, count, sizeof(double), nlo_bench_double_compare);

    if ((count % 2u) == 0u) {
        out_summary->median_ms = 0.5 * (sorted[count / 2u - 1u] + sorted[count / 2u]);
    } else {
        out_summary->median_ms = sorted[count / 2u];
    }

    free(sorted);
    return 0;
}

static int nlo_bench_run_backend_case(
    nlo_bench_runtime_backend backend,
    const nlo_bench_options* options,
    size_t sample_count,
    FILE* csv_file,
    const char* skip_reason,
    const nlo_bench_vk_context* vk_context,
    int* out_error_count
)
{
    if (options == NULL || csv_file == NULL || out_error_count == NULL) {
        return -1;
    }

    const char* backend_label = nlo_bench_backend_label(backend);
    nlo_bench_run_metrics empty_metrics = {0};

    if (skip_reason != NULL && *skip_reason != '\0') {
        printf("  %-3s skipped: %s\n", backend_label, skip_reason);
        if (nlo_bench_write_csv_row(csv_file,
                                    backend_label,
                                    sample_count,
                                    options->warmup_runs,
                                    options->measured_runs,
                                    0u,
                                    &empty_metrics,
                                    "SKIPPED",
                                    skip_reason) != 0) {
            *out_error_count += 1;
            return -1;
        }
        return 0;
    }

    nlo_bench_case_data case_data;
    char note[NLO_BENCH_NOTE_CAP];
    if (nlo_bench_prepare_case_data(sample_count,
                                    &case_data,
                                    note,
                                    sizeof(note)) != 0) {
        printf("  %-3s error: %s\n", backend_label, note);
        if (nlo_bench_write_csv_row(csv_file,
                                    backend_label,
                                    sample_count,
                                    options->warmup_runs,
                                    options->measured_runs,
                                    0u,
                                    &empty_metrics,
                                    "ERROR",
                                    note) != 0) {
            *out_error_count += 1;
        }
        *out_error_count += 1;
        return -1;
    }

    nlo_execution_options exec_options;
    if (backend == NLO_BENCH_RUNTIME_CPU) {
        exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    } else {
        if (vk_context == NULL) {
            nlo_bench_destroy_case_data(&case_data);
            nlo_bench_copy_note(note, sizeof(note), "Missing Vulkan context.");
            if (nlo_bench_write_csv_row(csv_file,
                                        backend_label,
                                        sample_count,
                                        options->warmup_runs,
                                        options->measured_runs,
                                        0u,
                                        &empty_metrics,
                                        "ERROR",
                                        note) != 0) {
                *out_error_count += 1;
            }
            *out_error_count += 1;
            return -1;
        }

        exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_VULKAN);
        exec_options.vulkan.physical_device = vk_context->physical_device;
        exec_options.vulkan.device = vk_context->device;
        exec_options.vulkan.queue = vk_context->queue;
        exec_options.vulkan.queue_family_index = vk_context->queue_family_index;
        exec_options.vulkan.command_pool = VK_NULL_HANDLE;
    }

    const size_t total_runs = options->warmup_runs + options->measured_runs;
    double* measured_totals = (double*)calloc(options->measured_runs, sizeof(double));
    if (measured_totals == NULL) {
        nlo_bench_destroy_case_data(&case_data);
        nlo_bench_copy_note(note, sizeof(note), "Failed to allocate measurement buffer.");
        *out_error_count += 1;
        return -1;
    }

    size_t recorded = 0u;
    bool had_error = false;
    bool skipped_backend = false;
    for (size_t run = 0u; run < total_runs; ++run) {
        nlo_bench_run_metrics metrics;
        if (nlo_bench_execute_single_run(&exec_options,
                                         &case_data,
                                         &metrics,
                                         note,
                                         sizeof(note)) != 0) {
            if (backend == NLO_BENCH_RUNTIME_GPU && recorded == 0u) {
                skipped_backend = true;
                printf("  %-3s skipped: %s\n", backend_label, note);
                if (nlo_bench_write_csv_row(csv_file,
                                            backend_label,
                                            sample_count,
                                            options->warmup_runs,
                                            options->measured_runs,
                                            0u,
                                            &metrics,
                                            "SKIPPED",
                                            note) != 0) {
                    *out_error_count += 1;
                    had_error = true;
                }
            } else {
                had_error = true;
                *out_error_count += 1;
                printf("  %-3s error: %s\n", backend_label, note);
                if (nlo_bench_write_csv_row(csv_file,
                                            backend_label,
                                            sample_count,
                                            options->warmup_runs,
                                            options->measured_runs,
                                            run + 1u,
                                            &metrics,
                                            "ERROR",
                                            note) != 0) {
                    *out_error_count += 1;
                }
            }
            break;
        }

        if (run < options->warmup_runs) {
            continue;
        }

        const size_t measured_index = run - options->warmup_runs;
        measured_totals[measured_index] = metrics.total_ms;
        recorded += 1u;

        if (nlo_bench_write_csv_row(csv_file,
                                    backend_label,
                                    sample_count,
                                    options->warmup_runs,
                                    options->measured_runs,
                                    measured_index + 1u,
                                    &metrics,
                                    "OK",
                                    "") != 0) {
            had_error = true;
            *out_error_count += 1;
            break;
        }
    }

    if (!had_error && recorded > 0u) {
        nlo_bench_summary summary;
        if (nlo_bench_compute_summary(measured_totals, recorded, &summary) == 0) {
            printf("  %-3s total(ms): mean=%.3f median=%.3f min=%.3f max=%.3f std=%.3f\n",
                   backend_label,
                   summary.mean_ms,
                   summary.median_ms,
                   summary.min_ms,
                   summary.max_ms,
                   summary.stddev_ms);
        } else {
            printf("  %-3s warning: failed to compute summary statistics.\n", backend_label);
        }
    } else if (!had_error && recorded == 0u && !skipped_backend) {
        printf("  %-3s warning: no measured runs recorded.\n", backend_label);
    }

    free(measured_totals);
    nlo_bench_destroy_case_data(&case_data);
    return had_error ? -1 : 0;
}

int main(int argc, char** argv)
{
    nlo_bench_options options;
    const int parse_status = nlo_bench_parse_options(argc, argv, &options);
    if (parse_status > 0) {
        return 0;
    }
    if (parse_status != 0) {
        return 1;
    }

    if (nlo_bench_make_parent_dirs(options.csv_path) != 0) {
        fprintf(stderr, "Failed to create parent directories for CSV output: %s\n", options.csv_path);
        return 1;
    }

    const int csv_exists = nlo_bench_file_exists(options.csv_path);
    FILE* csv_file = fopen(options.csv_path, "a");
    if (csv_file == NULL) {
        fprintf(stderr, "Failed to open CSV output path: %s\n", options.csv_path);
        return 1;
    }

    if (!csv_exists && nlo_bench_write_csv_header(csv_file) != 0) {
        fclose(csv_file);
        fprintf(stderr, "Failed to write CSV header.\n");
        return 1;
    }

    printf("Benchmark configuration:\n");
    printf("  backend: %s\n",
           (options.backend == NLO_BENCH_BACKEND_CPU) ? "cpu" :
           (options.backend == NLO_BENCH_BACKEND_GPU) ? "gpu" : "both");
    printf("  warmup runs: %zu\n", options.warmup_runs);
    printf("  measured runs: %zu\n", options.measured_runs);
    printf("  csv: %s\n", options.csv_path);

    int error_count = 0;
    const bool run_cpu = (options.backend == NLO_BENCH_BACKEND_CPU ||
                          options.backend == NLO_BENCH_BACKEND_BOTH);
    const bool run_gpu = (options.backend == NLO_BENCH_BACKEND_GPU ||
                          options.backend == NLO_BENCH_BACKEND_BOTH);

    bool gpu_available = false;
    char gpu_skip_reason[NLO_BENCH_NOTE_CAP];
    nlo_bench_copy_note(gpu_skip_reason, sizeof(gpu_skip_reason), "");

    nlo_bench_vk_context vk_context;
    memset(&vk_context, 0, sizeof(vk_context));

    if (run_gpu) {
        if (nlo_bench_vk_context_init(&vk_context,
                                      gpu_skip_reason,
                                      sizeof(gpu_skip_reason)) == 0) {
            gpu_available = true;
        } else {
            gpu_available = false;
        }
    }

    for (size_t i = 0u; i < options.size_count; ++i) {
        const size_t sample_count = options.sizes[i];
        printf("\nSize %zu samples\n", sample_count);

        if (run_cpu) {
            (void)nlo_bench_run_backend_case(NLO_BENCH_RUNTIME_CPU,
                                             &options,
                                             sample_count,
                                             csv_file,
                                             NULL,
                                             NULL,
                                             &error_count);
        }

        if (run_gpu) {
            if (!gpu_available) {
                (void)nlo_bench_run_backend_case(NLO_BENCH_RUNTIME_GPU,
                                                 &options,
                                                 sample_count,
                                                 csv_file,
                                                 gpu_skip_reason,
                                                 NULL,
                                                 &error_count);
            } else {
                (void)nlo_bench_run_backend_case(NLO_BENCH_RUNTIME_GPU,
                                                 &options,
                                                 sample_count,
                                                 csv_file,
                                                 NULL,
                                                 &vk_context,
                                                 &error_count);
            }
        }
    }

    if (run_gpu && gpu_available) {
        nlo_bench_vk_context_destroy(&vk_context);
    }

    fclose(csv_file);
    printf("\nBenchmark completed with %d error(s).\n", error_count);
    return (error_count == 0) ? 0 : 1;
}
