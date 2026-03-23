/**
 * @file bench_runtime_breakdown.c
 * @brief Detailed runtime walltime benchmark for init, solver, and switch costs.
 */

#include "backend/nlo_complex.h"
#include "core/init.h"
#include "core/state.h"
#include "numerics/rk4_kernel.h"
#include "physics/operator_program_jit.h"
#include "tensor_scaling_plan.h"
#include "utility/perf_profile.h"
#include "vulkan_bench_context.h"

#include <errno.h>
#include <math.h>
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

#ifndef NLO_BENCH_MAX_PATH
#define NLO_BENCH_MAX_PATH 1024u
#endif

#ifndef NLO_BENCH_MAX_VALUES
#define NLO_BENCH_MAX_VALUES 32u
#endif

#ifndef NLO_BENCH_NOTE_CAP
#define NLO_BENCH_NOTE_CAP 256u
#endif

#ifndef NLO_BENCH_PI
#define NLO_BENCH_PI 3.14159265358979323846
#endif

typedef enum {
    NLO_BENCH_SUITE_INIT = 0,
    NLO_BENCH_SUITE_SOLVER = 1,
    NLO_BENCH_SUITE_SWITCH = 2,
    NLO_BENCH_SUITE_ALL = 3
} nlo_breakdown_suite;

typedef enum {
    NLO_BENCH_SCENARIO_TEMPORAL_FIXED = 0,
    NLO_BENCH_SCENARIO_TEMPORAL_ADAPTIVE = 1,
    NLO_BENCH_SCENARIO_TENSOR_FIXED = 2,
    NLO_BENCH_SCENARIO_TENSOR_ADAPTIVE = 3
} nlo_breakdown_scenario;

typedef enum {
    NLO_BENCH_BACKEND_CPU = 0,
    NLO_BENCH_BACKEND_GPU = 1,
    NLO_BENCH_BACKEND_BOTH = 2
} nlo_breakdown_backend_request;

typedef enum {
    NLO_BENCH_JIT_ON = 0,
    NLO_BENCH_JIT_OFF = 1,
    NLO_BENCH_JIT_BOTH = 2
} nlo_breakdown_jit_request;

typedef enum {
    NLO_BENCH_TIMESTAMP_AUTO = 0,
    NLO_BENCH_TIMESTAMP_ON = 1,
    NLO_BENCH_TIMESTAMP_OFF = 2
} nlo_breakdown_timestamp_mode;

typedef struct {
    nlo_breakdown_suite suite;
    nlo_breakdown_scenario scenario;
    nlo_breakdown_backend_request backend;
    nlo_breakdown_jit_request jit;
    nlo_breakdown_timestamp_mode timestamp_queries;
    size_t sizes[NLO_BENCH_MAX_VALUES];
    size_t size_count;
    size_t tensor_scales[NLO_BENCH_MAX_VALUES];
    size_t tensor_scale_count;
    size_t warmup_runs;
    size_t measured_runs;
    size_t trace_steps;
    int dry_run;
    char summary_csv[NLO_BENCH_MAX_PATH];
    char trace_csv[NLO_BENCH_MAX_PATH];
} nlo_breakdown_options;

typedef struct {
    size_t sample_count;
    sim_config* config;
    nlo_complex* input_field;
    nlo_complex* output_field;
    nlo_complex* switch_records;
    size_t switch_record_capacity;
} nlo_breakdown_case_data;

typedef struct {
    double init_ms;
    double upload_ms;
    double solve_ms;
    double download_ms;
    double teardown_ms;
    double total_ms;
    nlo_perf_profile_snapshot init_profile;
    nlo_perf_profile_snapshot solve_profile;
    nlo_perf_profile_snapshot profile;
} nlo_breakdown_run_metrics;

typedef struct {
    nlo_breakdown_suite suite;
    nlo_breakdown_scenario scenario;
    int backend_kind;
    nlo_operator_jit_mode jit_mode;
    size_t size;
    size_t scale;
    size_t run_index;
    nlo_breakdown_run_metrics metrics;
    char status[16];
    char note[NLO_BENCH_NOTE_CAP];
} nlo_breakdown_result_row;

typedef struct {
#if defined(_WIN32)
    LARGE_INTEGER value;
#else
    struct timespec value;
#endif
} nlo_breakdown_timestamp;

static void nlo_bench_now(nlo_breakdown_timestamp* out_time)
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

static double nlo_bench_elapsed_ms(const nlo_breakdown_timestamp* start_time,
                                   const nlo_breakdown_timestamp* end_time)
{
    if (start_time == NULL || end_time == NULL) {
        return 0.0;
    }
#if defined(_WIN32)
    static LARGE_INTEGER frequency = {0};
    if (frequency.QuadPart == 0) {
        (void)QueryPerformanceFrequency(&frequency);
    }
    if (frequency.QuadPart == 0) {
        return 0.0;
    }
    return ((double)(end_time->value.QuadPart - start_time->value.QuadPart) * 1000.0) /
           (double)frequency.QuadPart;
#else
    return ((double)(end_time->value.tv_sec - start_time->value.tv_sec) * 1000.0) +
           ((double)(end_time->value.tv_nsec - start_time->value.tv_nsec) / 1000000.0);
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
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
        ++lhs;
        ++rhs;
    }
    return (*lhs == '\0' && *rhs == '\0') ? 1 : 0;
}

static void nlo_bench_copy_note(char* note, size_t note_capacity, const char* text)
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

static int nlo_bench_parse_unsigned_size(const char* text, size_t* out_value)
{
    char* end_ptr = NULL;
    unsigned long long value = 0u;
    if (text == NULL || out_value == NULL || *text == '\0') {
        return -1;
    }
    errno = 0;
    value = strtoull(text, &end_ptr, 10);
    if (errno != 0 || end_ptr == text || *end_ptr != '\0' || value == 0u) {
        return -1;
    }
    *out_value = (size_t)value;
    return 0;
}

static int nlo_bench_parse_nonnegative_size(const char* text, size_t* out_value)
{
    char* end_ptr = NULL;
    unsigned long long value = 0u;
    if (text == NULL || out_value == NULL || *text == '\0') {
        return -1;
    }
    errno = 0;
    value = strtoull(text, &end_ptr, 10);
    if (errno != 0 || end_ptr == text || *end_ptr != '\0') {
        return -1;
    }
    *out_value = (size_t)value;
    return 0;
}

static int nlo_bench_parse_size_values(const char* text,
                                       size_t* out_values,
                                       size_t max_values,
                                       size_t* out_count)
{
    size_t count = 0u;
    const char* cursor = text;
    if (text == NULL || out_values == NULL || out_count == NULL || *text == '\0') {
        return -1;
    }
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        if (count >= max_values || nlo_bench_parse_unsigned_size(cursor, &out_values[count]) != 0) {
            return -1;
        }
        count += 1u;
        while (*cursor != '\0' && *cursor != ',') {
            ++cursor;
        }
        if (*cursor == ',') {
            ++cursor;
        }
    }
    if (count == 0u) {
        return -1;
    }
    *out_count = count;
    return 0;
}

static void nlo_breakdown_profile_snapshot_accumulate(
    nlo_perf_profile_snapshot* dst,
    const nlo_perf_profile_snapshot* src
)
{
    if (dst == NULL || src == NULL) {
        return;
    }

    dst->dispersion_ms += src->dispersion_ms;
    dst->nonlinear_ms += src->nonlinear_ms;
    dst->dispersion_calls += src->dispersion_calls;
    dst->nonlinear_calls += src->nonlinear_calls;
    dst->gpu_dispatch_count += src->gpu_dispatch_count;
    dst->gpu_copy_count += src->gpu_copy_count;
    dst->gpu_device_copy_count += src->gpu_device_copy_count;
    dst->gpu_device_copy_bytes += src->gpu_device_copy_bytes;
    dst->gpu_host_transfer_copy_count += src->gpu_host_transfer_copy_count;
    dst->gpu_host_transfer_copy_bytes += src->gpu_host_transfer_copy_bytes;
    dst->gpu_memory_pass_count += src->gpu_memory_pass_count;
    dst->gpu_memory_pass_bytes += src->gpu_memory_pass_bytes;
    dst->gpu_upload_count += src->gpu_upload_count;
    dst->gpu_download_count += src->gpu_download_count;
    dst->gpu_upload_bytes += src->gpu_upload_bytes;
    dst->gpu_download_bytes += src->gpu_download_bytes;
    dst->trace_row_count += src->trace_row_count;
    dst->trace_dropped_count += src->trace_dropped_count;
    dst->gpu_timestamps_available =
        (dst->gpu_timestamps_available != 0 || src->gpu_timestamps_available != 0) ? 1 : 0;

    for (size_t i = 0u; i < NLO_PERF_EVENT_COUNT; ++i) {
        dst->event_total_ms[i] += src->event_total_ms[i];
        dst->event_gpu_exec_ms[i] += src->event_gpu_exec_ms[i];
        dst->event_call_count[i] += src->event_call_count[i];
        dst->event_bytes[i] += src->event_bytes[i];
    }
}

static void nlo_breakdown_print_usage(const char* executable_name)
{
    printf("Usage: %s [options]\n", executable_name);
    printf("  --suite=init|solver|switch|all\n");
    printf("  --scenario=temporal_fixed|temporal_adaptive|tensor_fixed|tensor_adaptive\n");
    printf("  --backend=cpu|gpu|both\n");
    printf("  --jit=on|off|both\n");
    printf("  --sizes=1024,4096,16384\n");
    printf("  --tensor-scales=8,16,32\n");
    printf("  --warmup=N\n");
    printf("  --runs=N\n");
    printf("  --trace-steps=N\n");
    printf("  --summary-csv=path.csv\n");
    printf("  --trace-csv=path.csv\n");
    printf("  --timestamp-queries=auto|on|off\n");
    printf("  --dry-run\n");
}

static void nlo_breakdown_set_default_options(nlo_breakdown_options* options)
{
    static const size_t default_sizes[] = {1024u, 4096u, 16384u};
    static const size_t default_scales[] = {8u, 16u};
    memset(options, 0, sizeof(*options));
    options->suite = NLO_BENCH_SUITE_SOLVER;
    options->scenario = NLO_BENCH_SCENARIO_TEMPORAL_FIXED;
    options->backend = NLO_BENCH_BACKEND_BOTH;
    options->jit = NLO_BENCH_JIT_BOTH;
    options->timestamp_queries = NLO_BENCH_TIMESTAMP_AUTO;
    options->warmup_runs = 1u;
    options->measured_runs = 3u;
    options->trace_steps = 0u;
    options->size_count = sizeof(default_sizes) / sizeof(default_sizes[0]);
    options->tensor_scale_count = sizeof(default_scales) / sizeof(default_scales[0]);
    memcpy(options->sizes, default_sizes, sizeof(default_sizes));
    memcpy(options->tensor_scales, default_scales, sizeof(default_scales));
    snprintf(options->summary_csv, sizeof(options->summary_csv),
             "benchmarks/results/runtime_breakdown_summary.csv");
    snprintf(options->trace_csv, sizeof(options->trace_csv),
             "benchmarks/results/runtime_breakdown_trace.csv");
}

static int nlo_breakdown_parse_options(int argc, char** argv, nlo_breakdown_options* options)
{
    nlo_breakdown_set_default_options(options);
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            nlo_breakdown_print_usage(argv[0]);
            return 1;
        }
        if (strcmp(arg, "--dry-run") == 0) {
            options->dry_run = 1;
            continue;
        }
        if (strncmp(arg, "--suite=", 8) == 0) {
            const char* value = arg + 8;
            if (nlo_bench_string_equals_ci(value, "init")) options->suite = NLO_BENCH_SUITE_INIT;
            else if (nlo_bench_string_equals_ci(value, "solver")) options->suite = NLO_BENCH_SUITE_SOLVER;
            else if (nlo_bench_string_equals_ci(value, "switch")) options->suite = NLO_BENCH_SUITE_SWITCH;
            else if (nlo_bench_string_equals_ci(value, "all")) options->suite = NLO_BENCH_SUITE_ALL;
            else return -1;
            continue;
        }
        if (strncmp(arg, "--scenario=", 11) == 0) {
            const char* value = arg + 11;
            if (nlo_bench_string_equals_ci(value, "temporal_fixed")) options->scenario = NLO_BENCH_SCENARIO_TEMPORAL_FIXED;
            else if (nlo_bench_string_equals_ci(value, "temporal_adaptive")) options->scenario = NLO_BENCH_SCENARIO_TEMPORAL_ADAPTIVE;
            else if (nlo_bench_string_equals_ci(value, "tensor_fixed")) options->scenario = NLO_BENCH_SCENARIO_TENSOR_FIXED;
            else if (nlo_bench_string_equals_ci(value, "tensor_adaptive")) options->scenario = NLO_BENCH_SCENARIO_TENSOR_ADAPTIVE;
            else return -1;
            continue;
        }
        if (strncmp(arg, "--backend=", 10) == 0) {
            const char* value = arg + 10;
            if (nlo_bench_string_equals_ci(value, "cpu")) options->backend = NLO_BENCH_BACKEND_CPU;
            else if (nlo_bench_string_equals_ci(value, "gpu")) options->backend = NLO_BENCH_BACKEND_GPU;
            else if (nlo_bench_string_equals_ci(value, "both")) options->backend = NLO_BENCH_BACKEND_BOTH;
            else return -1;
            continue;
        }
        if (strncmp(arg, "--jit=", 6) == 0) {
            const char* value = arg + 6;
            if (nlo_bench_string_equals_ci(value, "on")) options->jit = NLO_BENCH_JIT_ON;
            else if (nlo_bench_string_equals_ci(value, "off")) options->jit = NLO_BENCH_JIT_OFF;
            else if (nlo_bench_string_equals_ci(value, "both")) options->jit = NLO_BENCH_JIT_BOTH;
            else return -1;
            continue;
        }
        if (strncmp(arg, "--sizes=", 8) == 0) {
            if (nlo_bench_parse_size_values(arg + 8, options->sizes, NLO_BENCH_MAX_VALUES, &options->size_count) != 0) return -1;
            continue;
        }
        if (strncmp(arg, "--tensor-scales=", 16) == 0) {
            if (nlo_bench_parse_size_values(arg + 16, options->tensor_scales, NLO_BENCH_MAX_VALUES, &options->tensor_scale_count) != 0) return -1;
            continue;
        }
        if (strncmp(arg, "--warmup=", 9) == 0) {
            if (nlo_bench_parse_nonnegative_size(arg + 9, &options->warmup_runs) != 0) return -1;
            continue;
        }
        if (strncmp(arg, "--runs=", 7) == 0) {
            if (nlo_bench_parse_unsigned_size(arg + 7, &options->measured_runs) != 0) return -1;
            continue;
        }
        if (strncmp(arg, "--trace-steps=", 14) == 0) {
            if (nlo_bench_parse_nonnegative_size(arg + 14, &options->trace_steps) != 0) return -1;
            continue;
        }
        if (strncmp(arg, "--summary-csv=", 14) == 0) {
            snprintf(options->summary_csv, sizeof(options->summary_csv), "%s", arg + 14);
            continue;
        }
        if (strncmp(arg, "--trace-csv=", 12) == 0) {
            snprintf(options->trace_csv, sizeof(options->trace_csv), "%s", arg + 12);
            continue;
        }
        if (strncmp(arg, "--timestamp-queries=", 20) == 0) {
            const char* value = arg + 20;
            if (nlo_bench_string_equals_ci(value, "auto")) options->timestamp_queries = NLO_BENCH_TIMESTAMP_AUTO;
            else if (nlo_bench_string_equals_ci(value, "on")) options->timestamp_queries = NLO_BENCH_TIMESTAMP_ON;
            else if (nlo_bench_string_equals_ci(value, "off")) options->timestamp_queries = NLO_BENCH_TIMESTAMP_OFF;
            else return -1;
            continue;
        }
        return -1;
    }
    return 0;
}

static int nlo_bench_mkdir_single(const char* path)
{
    if (path == NULL || *path == '\0') {
        return 0;
    }
#if defined(_WIN32)
    return (_mkdir(path) == 0 || errno == EEXIST) ? 0 : -1;
#else
    return (mkdir(path, 0777) == 0 || errno == EEXIST) ? 0 : -1;
#endif
}

static int nlo_bench_make_parent_dirs(const char* file_path)
{
    char buffer[NLO_BENCH_MAX_PATH];
    char* sep = NULL;
    snprintf(buffer, sizeof(buffer), "%s", file_path);
    sep = strrchr(buffer, '/');
    {
        char* backslash = strrchr(buffer, '\\');
        if (backslash != NULL && (sep == NULL || backslash > sep)) {
            sep = backslash;
        }
    }
    if (sep == NULL) {
        return 0;
    }
    *sep = '\0';
    for (size_t i = 0u; buffer[i] != '\0'; ++i) {
        if (buffer[i] != '/' && buffer[i] != '\\') {
            continue;
        }
        if (i == 0u || (i == 2u && buffer[1] == ':')) {
            continue;
        }
        {
            const char saved = buffer[i];
            buffer[i] = '\0';
            if (nlo_bench_mkdir_single(buffer) != 0) {
                return -1;
            }
            buffer[i] = saved;
        }
    }
    return nlo_bench_mkdir_single(buffer);
}

static void nlo_bench_iso8601_utc(char* out_text, size_t out_capacity)
{
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

static const char* nlo_breakdown_suite_label(nlo_breakdown_suite suite)
{
    switch (suite) {
        case NLO_BENCH_SUITE_INIT: return "init";
        case NLO_BENCH_SUITE_SOLVER: return "solver";
        case NLO_BENCH_SUITE_SWITCH: return "switch";
        case NLO_BENCH_SUITE_ALL: return "all";
        default: return "unknown";
    }
}

static const char* nlo_breakdown_scenario_label(nlo_breakdown_scenario scenario)
{
    switch (scenario) {
        case NLO_BENCH_SCENARIO_TEMPORAL_FIXED: return "temporal_fixed";
        case NLO_BENCH_SCENARIO_TEMPORAL_ADAPTIVE: return "temporal_adaptive";
        case NLO_BENCH_SCENARIO_TENSOR_FIXED: return "tensor_fixed";
        case NLO_BENCH_SCENARIO_TENSOR_ADAPTIVE: return "tensor_adaptive";
        default: return "unknown";
    }
}

static const char* nlo_breakdown_backend_label(int backend_kind)
{
    return (backend_kind == NLO_BENCH_BACKEND_CPU) ? "cpu" : "gpu";
}

static const char* nlo_breakdown_jit_label(nlo_operator_jit_mode mode)
{
    return (mode == NLO_OPERATOR_JIT_MODE_OFF) ? "off" : "on";
}

static nlo_perf_gpu_timestamp_mode nlo_breakdown_gpu_timestamp_mode(
    nlo_breakdown_timestamp_mode mode
)
{
    switch (mode) {
        case NLO_BENCH_TIMESTAMP_ON: return NLO_PERF_GPU_TIMESTAMPS_ON;
        case NLO_BENCH_TIMESTAMP_OFF: return NLO_PERF_GPU_TIMESTAMPS_OFF;
        case NLO_BENCH_TIMESTAMP_AUTO:
        default: return NLO_PERF_GPU_TIMESTAMPS_AUTO;
    }
}

static int nlo_breakdown_write_summary_header(FILE* file)
{
    if (fprintf(file,
                "timestamp_utc,suite,scenario,backend,jit,size,scale,run_index,"
                "init_ms,upload_ms,solve_ms,download_ms,teardown_ms,total_ms,"
                "gpu_timestamps_available,trace_rows,trace_dropped,status,notes") < 0) {
        return -1;
    }
    for (size_t i = 0u; i < nlo_perf_profile_event_count(); ++i) {
        const char* name = nlo_perf_profile_event_name((nlo_perf_event_id)i);
        if (fprintf(file, ",%s_ms,%s_gpu_ms,%s_calls", name, name, name) < 0) {
            return -1;
        }
    }
    if (fprintf(file, "\n") < 0) {
        return -1;
    }
    return fflush(file);
}

static int nlo_breakdown_write_trace_header(FILE* file)
{
    return (fprintf(file,
                    "run_id,backend_kind,jit_mode,scenario_kind,step_index,reject_attempt,"
                    "event_id,event_name,call_count,bytes,host_wall_ms,gpu_exec_ms\n") >= 0 &&
            fflush(file) == 0)
               ? 0
               : -1;
}

static int nlo_breakdown_prepare_temporal_case(size_t sample_count,
                                               int fixed_step,
                                               nlo_breakdown_case_data* case_data,
                                               char* note,
                                               size_t note_capacity)
{
    memset(case_data, 0, sizeof(*case_data));
    case_data->sample_count = sample_count;
    case_data->config = create_sim_config(sample_count);
    if (case_data->config == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Failed to allocate sim_config.");
        return -1;
    }
    case_data->input_field = (nlo_complex*)calloc(sample_count, sizeof(nlo_complex));
    case_data->output_field = (nlo_complex*)calloc(sample_count, sizeof(nlo_complex));
    case_data->switch_record_capacity = 4u;
    case_data->switch_records =
        (nlo_complex*)calloc(sample_count * case_data->switch_record_capacity, sizeof(nlo_complex));
    if (case_data->input_field == NULL ||
        case_data->output_field == NULL ||
        case_data->switch_records == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Failed to allocate temporal buffers.");
        return -1;
    }

    {
        sim_config* config = case_data->config;
        config->runtime.dispersion_factor_expr = "i*c0*w*w-c1";
        config->runtime.dispersion_expr = "exp(h*D)";
        config->runtime.nonlinear_expr = "i*A*(c2*I)";
        config->runtime.num_constants = 3u;
        config->runtime.constants[0] = -0.5;
        config->runtime.constants[1] = 0.0;
        config->runtime.constants[2] = 1.0;
        config->propagation.propagation_distance = 0.4;
        config->propagation.starting_step_size = fixed_step ? 0.01 : 0.004;
        config->propagation.max_step_size = fixed_step ? 0.01 : 0.02;
        config->propagation.min_step_size = fixed_step ? 0.01 : 0.0002;
        config->propagation.error_tolerance = 1e-6;
        config->time.delta_time = 1.0 / (double)sample_count;
        config->time.pulse_period = 1.0;
        for (size_t i = 0u; i < sample_count; ++i) {
            const long centered = (i <= sample_count / 2u) ? (long)i : (long)i - (long)sample_count;
            const double omega = (2.0 * NLO_BENCH_PI * (double)centered) / (double)sample_count;
            const double t = ((double)i - 0.5 * (double)(sample_count - 1u)) * config->time.delta_time;
            const double envelope = exp(-(t * t) / 0.02);
            const double phase = 4.0 * t * t;
            config->frequency.frequency_grid[i] = nlo_make(omega, 0.0);
            case_data->input_field[i] = nlo_make(envelope * cos(phase), envelope * sin(phase));
        }
    }
    nlo_bench_copy_note(note, note_capacity, "");
    return 0;
}

static int nlo_breakdown_prepare_tensor_case(size_t scale,
                                             int fixed_step,
                                             nlo_breakdown_case_data* case_data,
                                             char* note,
                                             size_t note_capacity)
{
    nlo_bench_tensor_shape shape = {0};
    if (nlo_bench_tensor_shape_from_scale(scale, &shape) != 0) {
        nlo_bench_copy_note(note, note_capacity, "Invalid tensor scale.");
        return -1;
    }
    memset(case_data, 0, sizeof(*case_data));
    case_data->sample_count = shape.total_samples;
    case_data->config = create_sim_config(shape.total_samples);
    if (case_data->config == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Failed to allocate tensor sim_config.");
        return -1;
    }
    case_data->input_field = (nlo_complex*)calloc(shape.total_samples, sizeof(nlo_complex));
    case_data->output_field = (nlo_complex*)calloc(shape.total_samples, sizeof(nlo_complex));
    case_data->switch_record_capacity = 4u;
    case_data->switch_records =
        (nlo_complex*)calloc(shape.total_samples * case_data->switch_record_capacity,
                             sizeof(nlo_complex));
    if (case_data->input_field == NULL ||
        case_data->output_field == NULL ||
        case_data->switch_records == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Failed to allocate tensor buffers.");
        return -1;
    }
    {
        sim_config* config = case_data->config;
        config->tensor.nt = shape.nt;
        config->tensor.nx = shape.nx;
        config->tensor.ny = shape.ny;
        config->tensor.layout = NLO_TENSOR_LAYOUT_XYT_T_FAST;
        config->time.nt = shape.nt;
        config->time.delta_time = 0.04;
        config->time.pulse_period = (double)shape.nt * config->time.delta_time;
        config->spatial.nx = shape.nx;
        config->spatial.ny = shape.ny;
        config->spatial.delta_x = 0.15;
        config->spatial.delta_y = 0.15;
        config->propagation.propagation_distance = 0.2;
        config->propagation.starting_step_size = fixed_step ? 0.02 : 0.01;
        config->propagation.max_step_size = fixed_step ? 0.02 : 0.02;
        config->propagation.min_step_size = fixed_step ? 0.02 : 0.0005;
        config->propagation.error_tolerance = 1e-6;
        config->runtime.linear_factor_expr = "i*(c0*wt*wt + c1*(kx*kx + ky*ky))";
        config->runtime.dispersion_factor_expr = "i*(c0*wt*wt + c1*(kx*kx + ky*ky))";
        config->runtime.potential_expr = "c2*(x*x + y*y + 0.25*t*t)";
        config->runtime.nonlinear_expr = "i*A*(c3*I + V)";
        config->runtime.num_constants = 4u;
        config->runtime.constants[0] = 0.04;
        config->runtime.constants[1] = -0.20;
        config->runtime.constants[2] = 0.08;
        config->runtime.constants[3] = 0.90;
        for (size_t t = 0u; t < shape.nt; ++t) {
            const long centered = (t <= shape.nt / 2u) ? (long)t : (long)t - (long)shape.nt;
            const double omega = (2.0 * NLO_BENCH_PI * (double)centered) /
                                 ((double)shape.nt * config->time.delta_time);
            config->frequency.frequency_grid[t] = nlo_make(omega, 0.0);
        }
        for (size_t x = 0u; x < shape.nx; ++x) {
            const double x_value = ((double)x - 0.5 * (double)(shape.nx - 1u)) * config->spatial.delta_x;
            for (size_t y = 0u; y < shape.ny; ++y) {
                const double y_value = ((double)y - 0.5 * (double)(shape.ny - 1u)) * config->spatial.delta_y;
                const double transverse = exp(-((x_value / 0.60) * (x_value / 0.60)) -
                                               ((y_value / 0.70) * (y_value / 0.70)));
                for (size_t t = 0u; t < shape.nt; ++t) {
                    const double centered_t = ((double)t - 0.5 * (double)(shape.nt - 1u)) * config->time.delta_time;
                    const double temporal = exp(-((centered_t / 0.24) * (centered_t / 0.24)));
                    const double phase = 0.15 * centered_t * centered_t;
                    const size_t index = ((x * shape.ny) + y) * shape.nt + t;
                    case_data->input_field[index] =
                        nlo_make(temporal * transverse * cos(phase), temporal * transverse * sin(phase));
                }
            }
        }
    }
    nlo_bench_copy_note(note, note_capacity, "");
    return 0;
}

static void nlo_breakdown_destroy_case_data(nlo_breakdown_case_data* case_data)
{
    if (case_data == NULL) {
        return;
    }
    free(case_data->input_field);
    free(case_data->output_field);
    free(case_data->switch_records);
    free_sim_config(case_data->config);
    memset(case_data, 0, sizeof(*case_data));
}

static nlo_execution_options nlo_breakdown_exec_options_for_backend(int backend_kind,
                                                                    const nlo_bench_vk_context* vk_context)
{
    nlo_execution_options options = nlo_execution_options_default(
        (backend_kind == NLO_BENCH_BACKEND_CPU) ? NLO_VECTOR_BACKEND_CPU : NLO_VECTOR_BACKEND_VULKAN);
    if (backend_kind == NLO_BENCH_BACKEND_GPU && vk_context != NULL) {
        options.vulkan.physical_device = vk_context->physical_device;
        options.vulkan.device = vk_context->device;
        options.vulkan.queue = vk_context->queue;
        options.vulkan.queue_family_index = vk_context->queue_family_index;
        options.vulkan.command_pool = VK_NULL_HANDLE;
    }
    return options;
}

static int nlo_breakdown_execute_run(nlo_breakdown_suite suite,
                                     const nlo_execution_options* exec_options,
                                     const nlo_breakdown_case_data* case_data,
                                     nlo_operator_jit_mode jit_mode,
                                     nlo_breakdown_scenario scenario,
                                     uint64_t run_id,
                                     size_t trace_steps,
                                     FILE* trace_file,
                                     nlo_breakdown_run_metrics* out_metrics,
                                     char* note,
                                     size_t note_capacity)
{
    simulation_state* state = NULL;
    nlo_breakdown_timestamp total_start;
    nlo_breakdown_timestamp total_end;
    nlo_breakdown_timestamp phase_start;
    nlo_breakdown_timestamp phase_end;
    const size_t num_records = (suite == NLO_BENCH_SUITE_SWITCH) ? 4u : 1u;
    size_t trace_capacity = (suite == NLO_BENCH_SUITE_SOLVER && trace_steps > 0u) ? (trace_steps * 128u) : 0u;
    nlo_perf_profile_trace_row* trace_rows =
        (trace_capacity > 0u) ? (nlo_perf_profile_trace_row*)calloc(trace_capacity, sizeof(*trace_rows)) : NULL;

    memset(out_metrics, 0, sizeof(*out_metrics));
    nlo_operator_program_set_jit_mode(jit_mode);
    nlo_perf_profile_set_enabled(1);
    nlo_perf_profile_reset();
    nlo_perf_profile_trace_configure(trace_rows, trace_capacity);
    nlo_perf_profile_trace_set_run_metadata(run_id,
                                            (uint64_t)exec_options->backend_type,
                                            (uint64_t)jit_mode,
                                            (uint64_t)scenario);
    nlo_perf_profile_trace_set_step(UINT64_MAX, UINT64_MAX);
    nlo_perf_profile_mark_gpu_timestamps_available(0);

    nlo_bench_now(&phase_start);
    if (nlo_init_simulation_state(case_data->config, case_data->sample_count, num_records, exec_options, NULL, &state) != 0 ||
        state == NULL) {
        nlo_bench_copy_note(note, note_capacity, "State initialization failed.");
        goto fail;
    }
    nlo_bench_now(&phase_end);
    out_metrics->init_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
    nlo_perf_profile_snapshot_read(&out_metrics->init_profile);

    if (suite == NLO_BENCH_SUITE_SWITCH) {
        state->output_records = case_data->switch_records;
        state->output_record_capacity = case_data->switch_record_capacity;
    }

    if (suite == NLO_BENCH_SUITE_INIT) {
        nlo_bench_now(&phase_start);
        free_simulation_state(state);
        state = NULL;
        nlo_bench_now(&phase_end);
        out_metrics->teardown_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
        out_metrics->total_ms = out_metrics->init_ms + out_metrics->teardown_ms;
        out_metrics->profile = out_metrics->init_profile;
    } else {
        nlo_perf_profile_reset();
        nlo_perf_profile_trace_configure(trace_rows, trace_capacity);
        nlo_perf_profile_trace_set_run_metadata(run_id,
                                                (uint64_t)exec_options->backend_type,
                                                (uint64_t)jit_mode,
                                                (uint64_t)scenario);
        nlo_perf_profile_trace_set_step(UINT64_MAX, UINT64_MAX);
        nlo_perf_profile_mark_gpu_timestamps_available(0);

        nlo_bench_now(&total_start);
        nlo_bench_now(&phase_start);
        if (simulation_state_upload_initial_field(state, case_data->input_field) != NLO_VEC_STATUS_OK) {
            nlo_bench_copy_note(note, note_capacity, "Initial field upload failed.");
            goto fail;
        }
        nlo_bench_now(&phase_end);
        out_metrics->upload_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

        if (suite == NLO_BENCH_SUITE_SWITCH) {
            for (size_t i = 0u; i < 8u; ++i) {
                if (nlo_vec_begin_simulation(state->backend) != NLO_VEC_STATUS_OK ||
                    nlo_vec_end_simulation(state->backend) != NLO_VEC_STATUS_OK) {
                    nlo_bench_copy_note(note, note_capacity, "Empty begin/end cycle failed.");
                    goto fail;
                }
            }
            for (size_t i = 0u; i < 4u; ++i) {
                if (nlo_vec_begin_simulation(state->backend) != NLO_VEC_STATUS_OK ||
                    simulation_state_capture_snapshot(state) != NLO_VEC_STATUS_OK ||
                    nlo_vec_end_simulation(state->backend) != NLO_VEC_STATUS_OK) {
                    nlo_bench_copy_note(note, note_capacity, "Snapshot switch cycle failed.");
                    goto fail;
                }
            }
        } else {
            nlo_bench_now(&phase_start);
            solve_rk4(state);
            nlo_bench_now(&phase_end);
            out_metrics->solve_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
        }

        nlo_bench_now(&phase_start);
        if (simulation_state_download_current_field(state, case_data->output_field) != NLO_VEC_STATUS_OK) {
            nlo_bench_copy_note(note, note_capacity, "Current field download failed.");
            goto fail;
        }
        nlo_bench_now(&phase_end);
        out_metrics->download_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

        nlo_bench_now(&phase_start);
        free_simulation_state(state);
        state = NULL;
        nlo_bench_now(&phase_end);
        out_metrics->teardown_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
        nlo_bench_now(&total_end);
        out_metrics->total_ms = nlo_bench_elapsed_ms(&total_start, &total_end) + out_metrics->init_ms;
        nlo_perf_profile_snapshot_read(&out_metrics->solve_profile);
        out_metrics->profile = out_metrics->init_profile;
        nlo_breakdown_profile_snapshot_accumulate(&out_metrics->profile, &out_metrics->solve_profile);
    }
    if (trace_file != NULL && trace_rows != NULL) {
        for (uint64_t i = 0u; i < out_metrics->solve_profile.trace_row_count; ++i) {
            const nlo_perf_profile_trace_row* row = &trace_rows[i];
            fprintf(trace_file,
                    "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%s,%llu,%llu,%.9f,%.9f\n",
                    (unsigned long long)row->run_id,
                    (unsigned long long)row->backend_kind,
                    (unsigned long long)row->jit_mode,
                    (unsigned long long)row->scenario_kind,
                    (unsigned long long)row->step_index,
                    (unsigned long long)row->reject_attempt,
                    (unsigned long long)row->event_id,
                    nlo_perf_profile_event_name((nlo_perf_event_id)row->event_id),
                    (unsigned long long)row->call_count,
                    (unsigned long long)row->bytes,
                    row->host_wall_ms,
                    row->gpu_exec_ms);
        }
        fflush(trace_file);
    }
    free(trace_rows);
    nlo_perf_profile_trace_configure(NULL, 0u);
    nlo_perf_profile_set_enabled(0);
    nlo_bench_copy_note(note, note_capacity, "");
    return 0;

fail:
    if (state != NULL) {
        free_simulation_state(state);
    }
    free(trace_rows);
    nlo_perf_profile_trace_configure(NULL, 0u);
    nlo_perf_profile_set_enabled(0);
    return -1;
}

static int nlo_breakdown_write_summary_row(FILE* file,
                                           nlo_breakdown_suite suite,
                                           nlo_breakdown_scenario scenario,
                                           int backend_kind,
                                           nlo_operator_jit_mode jit_mode,
                                           size_t size,
                                           size_t scale,
                                           size_t run_index,
                                           const nlo_breakdown_run_metrics* metrics,
                                           const char* status,
                                           const char* note)
{
    char timestamp_utc[32];
    char note_buffer[NLO_BENCH_NOTE_CAP];
    nlo_bench_iso8601_utc(timestamp_utc, sizeof(timestamp_utc));
    nlo_bench_copy_note(note_buffer, sizeof(note_buffer), note);
    nlo_bench_sanitize_note(note_buffer);
    if (fprintf(file,
                "%s,%s,%s,%s,%s,%zu,%zu,%zu,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%llu,%llu,%s,%s",
                timestamp_utc,
                nlo_breakdown_suite_label(suite),
                nlo_breakdown_scenario_label(scenario),
                nlo_breakdown_backend_label(backend_kind),
                nlo_breakdown_jit_label(jit_mode),
                size,
                scale,
                run_index,
                metrics->init_ms,
                metrics->upload_ms,
                metrics->solve_ms,
                metrics->download_ms,
                metrics->teardown_ms,
                metrics->total_ms,
                metrics->profile.gpu_timestamps_available,
                (unsigned long long)metrics->profile.trace_row_count,
                (unsigned long long)metrics->profile.trace_dropped_count,
                status,
                note_buffer) < 0) {
        fprintf(stderr, "summary row write failed at fixed columns\n");
        return -1;
    }
    for (size_t i = 0u; i < nlo_perf_profile_event_count(); ++i) {
        if (fprintf(file,
                    ",%.9f,%.9f,%llu",
                    metrics->profile.event_total_ms[i],
                    metrics->profile.event_gpu_exec_ms[i],
                    (unsigned long long)metrics->profile.event_call_count[i]) < 0) {
            fprintf(stderr, "summary row write failed at event %zu\n", i);
            return -1;
        }
    }
    return (fprintf(file, "\n") >= 0 && fflush(file) == 0) ? 0 : -1;
}

static int nlo_breakdown_compare_doubles(const void* lhs, const void* rhs)
{
    const double a = *(const double*)lhs;
    const double b = *(const double*)rhs;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static double nlo_breakdown_median(double* values, size_t count)
{
    if (values == NULL || count == 0u) {
        return 0.0;
    }
    qsort(values, count, sizeof(*values), nlo_breakdown_compare_doubles);
    if ((count & 1u) != 0u) {
        return values[count / 2u];
    }
    return 0.5 * (values[(count / 2u) - 1u] + values[count / 2u]);
}

static void nlo_breakdown_print_hotspots(const nlo_breakdown_result_row* rows, size_t row_count)
{
    typedef struct {
        nlo_perf_event_id event;
        double median_ms;
        double median_share;
    } nlo_hotspot_summary;

    nlo_hotspot_summary summaries[NLO_PERF_EVENT_COUNT];
    size_t summary_count = 0u;

    for (size_t event_index = 0u; event_index < nlo_perf_profile_event_count(); ++event_index) {
        double values[128];
        double shares[128];
        size_t count = 0u;
        for (size_t i = 0u; i < row_count && count < 128u; ++i) {
            if (rows[i].suite != NLO_BENCH_SUITE_SOLVER ||
                strcmp(rows[i].status, "OK") != 0 ||
                rows[i].metrics.solve_ms <= 0.0) {
                continue;
            }
            values[count] = rows[i].metrics.solve_profile.event_total_ms[event_index];
            shares[count] = values[count] / rows[i].metrics.solve_ms;
            count += 1u;
        }
        if (count == 0u) {
            continue;
        }

        const double median_ms = nlo_breakdown_median(values, count);
        const double median_share = nlo_breakdown_median(shares, count);
        if (median_ms < 0.25 || median_share < 0.05) {
            continue;
        }

        summaries[summary_count].event = (nlo_perf_event_id)event_index;
        summaries[summary_count].median_ms = median_ms;
        summaries[summary_count].median_share = median_share;
        summary_count += 1u;
    }

    for (size_t i = 0u; i < summary_count; ++i) {
        for (size_t j = i + 1u; j < summary_count; ++j) {
            if (summaries[j].median_share > summaries[i].median_share) {
                const nlo_hotspot_summary tmp = summaries[i];
                summaries[i] = summaries[j];
                summaries[j] = tmp;
            }
        }
    }

    if (summary_count == 0u) {
        return;
    }

    printf("\nSolver hotspots (median share of solve walltime):\n");
    for (size_t i = 0u; i < summary_count && i < 10u; ++i) {
        printf("  %s: %.3f ms (%.1f%%)%s\n",
               nlo_perf_profile_event_name(summaries[i].event),
               summaries[i].median_ms,
               summaries[i].median_share * 100.0,
               (summaries[i].median_ms >= 0.25 && summaries[i].median_share >= 0.05) ? " hotspot" : "");
    }
}

static void nlo_breakdown_print_jit_deltas(const nlo_breakdown_result_row* rows, size_t row_count)
{
    typedef struct {
        nlo_perf_event_id event;
        double median_abs_ms;
        double median_rel_delta;
    } nlo_jit_delta_summary;

    nlo_jit_delta_summary summaries[NLO_PERF_EVENT_COUNT + 1u];
    size_t summary_count = 0u;

    for (size_t event_index = 0u; event_index <= nlo_perf_profile_event_count(); ++event_index) {
        double deltas[128];
        double rel_deltas[128];
        size_t count = 0u;

        for (size_t i = 0u; i < row_count && count < 128u; ++i) {
            if (rows[i].backend_kind != NLO_BENCH_BACKEND_GPU ||
                rows[i].jit_mode != NLO_OPERATOR_JIT_MODE_ON ||
                strcmp(rows[i].status, "OK") != 0) {
                continue;
            }

            for (size_t j = 0u; j < row_count; ++j) {
                if (rows[j].backend_kind != NLO_BENCH_BACKEND_GPU ||
                    rows[j].jit_mode != NLO_OPERATOR_JIT_MODE_OFF ||
                    strcmp(rows[j].status, "OK") != 0 ||
                    rows[j].suite != rows[i].suite ||
                    rows[j].scenario != rows[i].scenario ||
                    rows[j].size != rows[i].size ||
                    rows[j].scale != rows[i].scale ||
                    rows[j].run_index != rows[i].run_index) {
                    continue;
                }

                const double jit_value =
                    (event_index == nlo_perf_profile_event_count())
                        ? rows[i].metrics.solve_ms
                        : rows[i].metrics.solve_profile.event_total_ms[event_index];
                const double base_value =
                    (event_index == nlo_perf_profile_event_count())
                        ? rows[j].metrics.solve_ms
                        : rows[j].metrics.solve_profile.event_total_ms[event_index];
                const double abs_delta = base_value - jit_value;
                double rel_delta = 0.0;
                if (base_value > 0.0) {
                    rel_delta = abs_delta / base_value;
                }
                deltas[count] = abs_delta;
                rel_deltas[count] = rel_delta;
                count += 1u;
                break;
            }
        }

        if (count == 0u) {
            continue;
        }

        const double median_abs_ms = nlo_breakdown_median(deltas, count);
        const double median_rel = nlo_breakdown_median(rel_deltas, count);
        if (fabs(median_abs_ms) < 0.25 || fabs(median_rel) < 0.15) {
            continue;
        }

        summaries[summary_count].event =
            (event_index == nlo_perf_profile_event_count()) ? NLO_PERF_EVENT_COUNT : (nlo_perf_event_id)event_index;
        summaries[summary_count].median_abs_ms = median_abs_ms;
        summaries[summary_count].median_rel_delta = median_rel;
        summary_count += 1u;
    }

    for (size_t i = 0u; i < summary_count; ++i) {
        for (size_t j = i + 1u; j < summary_count; ++j) {
            if (fabs(summaries[j].median_abs_ms) > fabs(summaries[i].median_abs_ms)) {
                const nlo_jit_delta_summary tmp = summaries[i];
                summaries[i] = summaries[j];
                summaries[j] = tmp;
            }
        }
    }

    if (summary_count == 0u) {
        return;
    }

    printf("\nJIT vs interpreter deltas (GPU median, positive means JIT faster):\n");
    for (size_t i = 0u; i < summary_count && i < 10u; ++i) {
        const char* label = (summaries[i].event == NLO_PERF_EVENT_COUNT)
                                ? "solve_total"
                                : nlo_perf_profile_event_name(summaries[i].event);
        printf("  %s: %.3f ms (%.1f%%)%s\n",
               label,
               summaries[i].median_abs_ms,
               summaries[i].median_rel_delta * 100.0,
               (fabs(summaries[i].median_abs_ms) >= 0.25 &&
                fabs(summaries[i].median_rel_delta) >= 0.15)
                   ? " significant_jit_delta"
                   : "");
    }
}

static void nlo_breakdown_print_switch_summary(const nlo_breakdown_result_row* rows, size_t row_count)
{
    double total_ms = 0.0;
    double begin_end_ms = 0.0;
    double snapshot_switch_ms = 0.0;
    size_t count = 0u;
    for (size_t i = 0u; i < row_count; ++i) {
        if (rows[i].suite != NLO_BENCH_SUITE_SWITCH || strcmp(rows[i].status, "OK") != 0) {
            continue;
        }
        total_ms += rows[i].metrics.total_ms;
        begin_end_ms += rows[i].metrics.solve_profile.event_total_ms[NLO_PERF_EVENT_BEGIN_SIMULATION] +
                        rows[i].metrics.solve_profile.event_total_ms[NLO_PERF_EVENT_END_SIMULATION];
        snapshot_switch_ms += rows[i].metrics.solve_profile.event_total_ms[NLO_PERF_EVENT_SNAPSHOT_PAUSE_END_SIM] +
                              rows[i].metrics.solve_profile.event_total_ms[NLO_PERF_EVENT_SNAPSHOT_DOWNLOAD] +
                              rows[i].metrics.solve_profile.event_total_ms[NLO_PERF_EVENT_SNAPSHOT_RESUME_BEGIN_SIM];
        count += 1u;
    }
    if (count == 0u || total_ms <= 0.0) {
        return;
    }

    const double begin_end_share = begin_end_ms / total_ms;
    const double snapshot_share = snapshot_switch_ms / total_ms;
    printf("\nSwitch-cost summary:\n");
    printf("  begin/end: %.3f ms average (%.1f%%)%s\n",
           begin_end_ms / (double)count,
           begin_end_share * 100.0,
           (begin_end_share >= 0.05) ? " significant_switch_cost" : "");
    printf("  snapshot switch: %.3f ms average (%.1f%%)%s\n",
           snapshot_switch_ms / (double)count,
           snapshot_share * 100.0,
           (snapshot_share >= 0.05) ? " significant_switch_cost" : "");
}

static void nlo_breakdown_print_init_summary(const nlo_breakdown_result_row* rows, size_t row_count)
{
    double event_totals[NLO_PERF_EVENT_COUNT] = {0.0};
    size_t count = 0u;
    for (size_t i = 0u; i < row_count; ++i) {
        if ((rows[i].suite != NLO_BENCH_SUITE_INIT && rows[i].suite != NLO_BENCH_SUITE_SOLVER) ||
            strcmp(rows[i].status, "OK") != 0) {
            continue;
        }
        for (size_t event_index = 0u; event_index < nlo_perf_profile_event_count(); ++event_index) {
            event_totals[event_index] += rows[i].metrics.init_profile.event_total_ms[event_index];
        }
        count += 1u;
    }
    if (count == 0u) {
        return;
    }

    printf("\nInitialization breakdown (average):\n");
    for (size_t event_index = 0u; event_index <= (size_t)NLO_PERF_EVENT_FFT_PLAN_CREATE; ++event_index) {
        const double average_ms = event_totals[event_index] / (double)count;
        if (average_ms <= 0.0) {
            continue;
        }
        printf("  %s: %.3f ms\n",
               nlo_perf_profile_event_name((nlo_perf_event_id)event_index),
               average_ms);
    }
}

int main(int argc, char** argv)
{
    nlo_breakdown_options options;
    nlo_bench_vk_context vk_context;
    char gpu_reason[NLO_BENCH_NOTE_CAP];
    int gpu_available = 0;
    FILE* summary_file = NULL;
    FILE* trace_file = NULL;
    size_t run_id = 1u;

    const int parse_status = nlo_breakdown_parse_options(argc, argv, &options);
    if (parse_status > 0) return 0;
    if (parse_status != 0) {
        nlo_breakdown_print_usage(argv[0]);
        return 1;
    }

    if (options.dry_run) {
        printf("bench_runtime_breakdown dry-run ok\n");
        return 0;
    }

    memset(&vk_context, 0, sizeof(vk_context));
    gpu_reason[0] = '\0';
    if (options.backend != NLO_BENCH_BACKEND_CPU &&
        nlo_bench_vk_context_init(&vk_context, gpu_reason, sizeof(gpu_reason)) == 0) {
        gpu_available = 1;
    }
    if (nlo_bench_make_parent_dirs(options.summary_csv) != 0 ||
        nlo_bench_make_parent_dirs(options.trace_csv) != 0) {
        return 1;
    }
    summary_file = fopen(options.summary_csv, "a");
    trace_file = fopen(options.trace_csv, "a");
    if (summary_file == NULL || trace_file == NULL) {
        if (summary_file != NULL) fclose(summary_file);
        if (trace_file != NULL) fclose(trace_file);
        return 1;
    }
    if (ftell(summary_file) == 0) (void)nlo_breakdown_write_summary_header(summary_file);
    if (ftell(trace_file) == 0) (void)nlo_breakdown_write_trace_header(trace_file);

    const int tensor = (options.scenario == NLO_BENCH_SCENARIO_TENSOR_FIXED ||
                        options.scenario == NLO_BENCH_SCENARIO_TENSOR_ADAPTIVE) ? 1 : 0;
    const size_t case_count = tensor ? options.tensor_scale_count : options.size_count;
    const size_t suite_count = (options.suite == NLO_BENCH_SUITE_ALL) ? 3u : 1u;
    const size_t results_capacity =
        case_count * suite_count * 2u * 2u * ((options.measured_runs > 0u) ? options.measured_runs : 1u);
    nlo_breakdown_result_row* results =
        (results_capacity > 0u)
            ? (nlo_breakdown_result_row*)calloc(results_capacity, sizeof(*results))
            : NULL;
    size_t result_count = 0u;

    for (size_t case_index = 0u; case_index < case_count; ++case_index) {
        const size_t size = tensor ? 0u : options.sizes[case_index];
        const size_t scale = tensor ? options.tensor_scales[case_index] : 0u;
        const int fixed_step = (options.scenario == NLO_BENCH_SCENARIO_TEMPORAL_FIXED ||
                                options.scenario == NLO_BENCH_SCENARIO_TENSOR_FIXED) ? 1 : 0;
        nlo_breakdown_case_data case_data;
        char note[NLO_BENCH_NOTE_CAP];
        int prep_ok = tensor
                          ? nlo_breakdown_prepare_tensor_case(scale, fixed_step, &case_data, note, sizeof(note))
                          : nlo_breakdown_prepare_temporal_case(size, fixed_step, &case_data, note, sizeof(note));
        if (prep_ok != 0) {
            continue;
        }

        for (int backend_kind = NLO_BENCH_BACKEND_CPU; backend_kind <= NLO_BENCH_BACKEND_GPU; ++backend_kind) {
            if (options.backend != NLO_BENCH_BACKEND_BOTH && backend_kind != (int)options.backend) {
                continue;
            }
            if (backend_kind == NLO_BENCH_BACKEND_GPU && !gpu_available) {
                nlo_breakdown_run_metrics empty = {0};
                (void)nlo_breakdown_write_summary_row(summary_file,
                                                      options.suite,
                                                      options.scenario,
                                                      backend_kind,
                                                      NLO_OPERATOR_JIT_MODE_ON,
                                                      size,
                                                      scale,
                                                      0u,
                                                      &empty,
                                                      "SKIPPED",
                                                      gpu_reason);
                continue;
            }

            const nlo_execution_options exec_options =
                nlo_breakdown_exec_options_for_backend(backend_kind, gpu_available ? &vk_context : NULL);
            for (int jit_index = 0; jit_index < 2; ++jit_index) {
                const int jit_mode = (jit_index == 0) ? NLO_OPERATOR_JIT_MODE_ON : NLO_OPERATOR_JIT_MODE_OFF;
                if (backend_kind == NLO_BENCH_BACKEND_CPU && jit_mode == NLO_OPERATOR_JIT_MODE_OFF) continue;
                if (options.jit == NLO_BENCH_JIT_ON && jit_mode != NLO_OPERATOR_JIT_MODE_ON) continue;
                if (options.jit == NLO_BENCH_JIT_OFF && jit_mode != NLO_OPERATOR_JIT_MODE_OFF) continue;

                const nlo_breakdown_suite suite_first =
                    (options.suite == NLO_BENCH_SUITE_ALL) ? NLO_BENCH_SUITE_INIT : options.suite;
                const nlo_breakdown_suite suite_last =
                    (options.suite == NLO_BENCH_SUITE_ALL) ? NLO_BENCH_SUITE_SWITCH : options.suite;

                for (int suite_value = (int)suite_first; suite_value <= (int)suite_last; ++suite_value) {
                    const nlo_breakdown_suite active_suite = (nlo_breakdown_suite)suite_value;
                    const nlo_perf_gpu_timestamp_mode timestamp_mode =
                        (backend_kind == NLO_BENCH_BACKEND_GPU)
                            ? nlo_breakdown_gpu_timestamp_mode(options.timestamp_queries)
                            : NLO_PERF_GPU_TIMESTAMPS_OFF;

                    for (size_t run_index = 0u; run_index < options.warmup_runs + options.measured_runs; ++run_index) {
                        nlo_breakdown_run_metrics metrics;
                        nlo_perf_profile_set_gpu_timestamp_mode(timestamp_mode);
                        const int status = nlo_breakdown_execute_run(active_suite,
                                                                     &exec_options,
                                                                     &case_data,
                                                                     (nlo_operator_jit_mode)jit_mode,
                                                                     options.scenario,
                                                                     (uint64_t)run_id,
                                                                     options.trace_steps,
                                                                     trace_file,
                                                                     &metrics,
                                                                     note,
                                                                     sizeof(note));
                        if (run_index >= options.warmup_runs) {
                            const char* status_text = (status == 0) ? "OK" : "ERROR";
                            (void)nlo_breakdown_write_summary_row(summary_file,
                                                                  active_suite,
                                                                  options.scenario,
                                                                  backend_kind,
                                                                  (nlo_operator_jit_mode)jit_mode,
                                                                  size,
                                                                  scale,
                                                                  run_index - options.warmup_runs,
                                                                  &metrics,
                                                                  status_text,
                                                                  note);
                            if (results != NULL && result_count < results_capacity) {
                                nlo_breakdown_result_row* row = &results[result_count];
                                row->suite = active_suite;
                                row->scenario = options.scenario;
                                row->backend_kind = backend_kind;
                                row->jit_mode = (nlo_operator_jit_mode)jit_mode;
                                row->size = size;
                                row->scale = scale;
                                row->run_index = run_index - options.warmup_runs;
                                row->metrics = metrics;
                                snprintf(row->status, sizeof(row->status), "%s", status_text);
                                snprintf(row->note, sizeof(row->note), "%s", note);
                                result_count += 1u;
                            }
                        }
                        run_id += 1u;
                    }
                }
            }
        }
        nlo_breakdown_destroy_case_data(&case_data);
    }

    fclose(summary_file);
    fclose(trace_file);
    nlo_perf_profile_set_gpu_timestamp_mode(NLO_PERF_GPU_TIMESTAMPS_AUTO);
    if (gpu_available) {
        nlo_bench_vk_context_destroy(&vk_context);
    }
    nlo_breakdown_print_hotspots(results, result_count);
    nlo_breakdown_print_jit_deltas(results, result_count);
    nlo_breakdown_print_switch_summary(results, result_count);
    nlo_breakdown_print_init_summary(results, result_count);
    free(results);
    return 0;
}
