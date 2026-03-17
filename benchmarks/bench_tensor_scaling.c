/**
 * @file bench_tensor_scaling.c
 * @brief Tensor 3D CPU/GPU scaling benchmark runner.
 */

#include "bench_tensor_scaling.h"

#include "backend/nlo_complex.h"
#include "core/init.h"
#include "core/init_internal.h"
#include "core/state.h"
#include "numerics/rk4_kernel.h"
#include "tensor_scaling_plan.h"
#include "utility/perf_profile.h"

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

#ifndef NLO_BENCH_MAX_PATH
#define NLO_BENCH_MAX_PATH 1024u
#endif

#ifndef NLO_BENCH_NOTE_CAP
#define NLO_BENCH_NOTE_CAP 256u
#endif

#ifndef NLO_BENCH_PI
#define NLO_BENCH_PI 3.14159265358979323846
#endif

#ifndef NLO_BENCH_MAX_TENSOR_CANDIDATES
#define NLO_BENCH_MAX_TENSOR_CANDIDATES 16u
#endif

#ifndef NLO_BENCH_SPILL_TARGET_COUNT
#define NLO_BENCH_SPILL_TARGET_COUNT 3u
#endif

typedef enum {
    NLO_BENCH_RUNTIME_CPU = 0,
    NLO_BENCH_RUNTIME_GPU = 1
} nlo_bench_runtime_backend;

typedef struct {
    size_t total_samples;
    size_t nt;
    size_t nx;
    size_t ny;
    size_t num_records;
    int storage_enabled;
    sim_config* config;
    nlo_complex* input_field;
    nlo_complex* output_field;
    char storage_path[NLO_BENCH_MAX_PATH];
} nlo_bench_tensor_case_data;

typedef struct {
    size_t total_samples;
    size_t nt;
    size_t nx;
    size_t ny;
    size_t num_records;
    size_t per_record_bytes;
    size_t working_set_bytes;
    size_t output_bytes;
    size_t system_memory_bytes;
    size_t gpu_device_budget_bytes;
    size_t allocated_records;
    size_t device_ring_capacity;
    size_t host_snapshot_bytes;
    size_t working_vector_bytes;
    size_t records_spilled;
    size_t chunks_written;
    size_t db_size_bytes;

    double init_ms;
    double upload_ms;
    double solve_ms;
    double dispersion_ms;
    double nonlinear_ms;
    double download_ms;
    double teardown_ms;
    double total_ms;
    double samples_per_sec;

    uint64_t gpu_dispatch_count;
    uint64_t gpu_copy_count;
    uint64_t gpu_memory_pass_count;
    uint64_t gpu_memory_pass_bytes;
    uint64_t gpu_upload_count;
    uint64_t gpu_download_count;
    uint64_t gpu_upload_bytes;
    uint64_t gpu_download_bytes;
} nlo_bench_tensor_metrics;

typedef struct {
    nlo_bench_tensor_shape shape;
    nlo_bench_tensor_region region;
    size_t working_set_bytes;
    size_t per_record_bytes;
    size_t system_budget_bytes;
    size_t gpu_budget_bytes;
    int cpu_probe_ok;
    int gpu_probe_ok;
} nlo_bench_tensor_candidate;

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

static const char* nlo_bench_backend_label(nlo_bench_runtime_backend backend)
{
    return (backend == NLO_BENCH_RUNTIME_CPU) ? "cpu" : "gpu";
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

static int nlo_bench_checked_mul_size_t(size_t lhs, size_t rhs, size_t* out_value)
{
    if (out_value == NULL) {
        return -1;
    }
    if (lhs == 0u || rhs == 0u) {
        *out_value = 0u;
        return 0;
    }
    if (lhs > (SIZE_MAX / rhs)) {
        return -1;
    }
    *out_value = lhs * rhs;
    return 0;
}

static size_t nlo_bench_effective_host_budget_bytes(const nlo_bench_tensor_options* options)
{
    if (options != NULL && options->planner_host_bytes > 0u) {
        return options->planner_host_bytes;
    }
    return nlo_apply_memory_headroom(nlo_query_available_system_memory_bytes());
}

static size_t nlo_bench_scale_size_t_double(size_t value, double factor)
{
    const double scaled = floor((double)value * factor);
    if (!(scaled > 0.0)) {
        return 0u;
    }
    if (scaled >= (double)SIZE_MAX) {
        return SIZE_MAX;
    }
    return (size_t)scaled;
}

static int nlo_bench_file_exists(const char* path)
{
    FILE* file = NULL;
    if (path == NULL) {
        return 0;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    fclose(file);
    return 1;
}

static int nlo_bench_mkdir_single(const char* path)
{
    int status = 0;

    if (path == NULL || *path == '\0') {
        return 0;
    }

#if defined(_WIN32)
    status = _mkdir(path);
#else
    status = mkdir(path, 0777);
#endif

    if (status == 0 || errno == EEXIST) {
        return 0;
    }
    return -1;
}

static int nlo_bench_make_parent_dirs(const char* file_path)
{
    char buffer[NLO_BENCH_MAX_PATH];
    char* separator = NULL;
    size_t len = 0u;

    if (file_path == NULL || *file_path == '\0') {
        return -1;
    }

    snprintf(buffer, sizeof(buffer), "%s", file_path);
    separator = strrchr(buffer, '/');
    {
        char* backslash = strrchr(buffer, '\\');
        if (backslash != NULL && (separator == NULL || backslash > separator)) {
            separator = backslash;
        }
    }
    if (separator == NULL) {
        return nlo_bench_mkdir_single(buffer);
    }

    *separator = '\0';
    len = strlen(buffer);
    if (len == 0u) {
        return 0;
    }

    for (size_t i = 0u; i < len; ++i) {
        const char ch = buffer[i];
        if (ch != '/' && ch != '\\') {
            continue;
        }
        if (i == 0u) {
            continue;
        }
        if (i == 2u && buffer[1] == ':') {
            continue;
        }

        buffer[i] = '\0';
        if (nlo_bench_mkdir_single(buffer) != 0) {
            buffer[i] = ch;
            return -1;
        }
        buffer[i] = ch;
    }

    return nlo_bench_mkdir_single(buffer);
}

static int nlo_bench_make_directory_tree(const char* dir_path)
{
    char probe_path[NLO_BENCH_MAX_PATH];

    if (dir_path == NULL || *dir_path == '\0') {
        return -1;
    }

    snprintf(probe_path, sizeof(probe_path), "%s/.nlo_bench_dir_probe", dir_path);
    if (nlo_bench_make_parent_dirs(probe_path) != 0) {
        return -1;
    }

    return nlo_bench_mkdir_single(dir_path);
}

static void nlo_bench_iso8601_utc(char* out_text, size_t out_capacity)
{
    const time_t now = time(NULL);
    struct tm utc_tm;

    if (out_text == NULL || out_capacity == 0u) {
        return;
    }

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

    for (char* cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor == ',' || *cursor == '\n' || *cursor == '\r') {
            *cursor = ';';
        }
    }
}

static size_t nlo_bench_compute_working_set_bytes(const nlo_allocation_info* info)
{
    size_t ring_bytes = 0u;
    size_t total = 0u;

    if (info == NULL) {
        return 0u;
    }

    if (nlo_bench_checked_mul_size_t(info->per_record_bytes,
                                     info->device_ring_capacity,
                                     &ring_bytes) != 0) {
        ring_bytes = 0u;
    }

    total = info->host_snapshot_bytes + info->working_vector_bytes;
    if (SIZE_MAX - total < ring_bytes) {
        return SIZE_MAX;
    }
    return total + ring_bytes;
}

static int nlo_bench_write_csv_header(FILE* csv_file)
{
    const int written = fprintf(csv_file,
                                "timestamp_utc,scenario,region,backend,status,notes,"
                                "total_samples,nt,nx,ny,num_records,per_record_bytes,working_set_bytes,output_bytes,"
                                "system_memory_bytes,gpu_device_budget_bytes,storage_enabled,allocated_records,"
                                "device_ring_capacity,host_snapshot_bytes,working_vector_bytes,"
                                "warmup_runs,measured_runs,run_index,"
                                "init_ms,upload_ms,solve_ms,dispersion_ms,nonlinear_ms,download_ms,teardown_ms,total_ms,"
                                "samples_per_sec,gpu_dispatch_count,gpu_copy_count,gpu_memory_pass_count,gpu_memory_pass_bytes,"
                                "gpu_upload_count,gpu_download_count,gpu_upload_bytes,gpu_download_bytes,"
                                "records_spilled,chunks_written,db_size_bytes\n");
    if (written < 0) {
        return -1;
    }
    return fflush(csv_file);
}

static int nlo_bench_write_csv_row(
    FILE* csv_file,
    const char* region_label,
    const char* backend_label,
    size_t warmup_runs,
    size_t measured_runs,
    size_t run_index,
    const nlo_bench_tensor_metrics* metrics,
    const char* status,
    const char* notes
)
{
    char timestamp_utc[32];
    char note_buffer[NLO_BENCH_NOTE_CAP];
    int written = 0;

    if (csv_file == NULL || region_label == NULL || backend_label == NULL || metrics == NULL || status == NULL) {
        return -1;
    }

    nlo_bench_iso8601_utc(timestamp_utc, sizeof(timestamp_utc));
    nlo_bench_copy_note(note_buffer, sizeof(note_buffer), (notes == NULL) ? "" : notes);
    nlo_bench_sanitize_note(note_buffer);

    written = fprintf(csv_file,
                      "%s,tensor3d_scaling,%s,%s,%s,%s,"
                      "%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
                      "%zu,%zu,%d,%zu,"
                      "%zu,%zu,%zu,"
                      "%zu,%zu,%zu,"
                      "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                      "%.6f,%llu,%llu,%llu,%llu,"
                      "%llu,%llu,%llu,%llu,"
                      "%zu,%zu,%zu\n",
                      timestamp_utc,
                      region_label,
                      backend_label,
                      status,
                      note_buffer,
                      metrics->total_samples,
                      metrics->nt,
                      metrics->nx,
                      metrics->ny,
                      metrics->num_records,
                      metrics->per_record_bytes,
                      metrics->working_set_bytes,
                      metrics->output_bytes,
                      metrics->system_memory_bytes,
                      metrics->gpu_device_budget_bytes,
                      (metrics->output_bytes > 0u) ? 1 : 0,
                      metrics->allocated_records,
                      metrics->device_ring_capacity,
                      metrics->host_snapshot_bytes,
                      metrics->working_vector_bytes,
                      warmup_runs,
                      measured_runs,
                      run_index,
                      metrics->init_ms,
                      metrics->upload_ms,
                      metrics->solve_ms,
                      metrics->dispersion_ms,
                      metrics->nonlinear_ms,
                      metrics->download_ms,
                      metrics->teardown_ms,
                      metrics->total_ms,
                      metrics->samples_per_sec,
                      (unsigned long long)metrics->gpu_dispatch_count,
                      (unsigned long long)metrics->gpu_copy_count,
                      (unsigned long long)metrics->gpu_memory_pass_count,
                      (unsigned long long)metrics->gpu_memory_pass_bytes,
                      (unsigned long long)metrics->gpu_upload_count,
                      (unsigned long long)metrics->gpu_download_count,
                      (unsigned long long)metrics->gpu_upload_bytes,
                      (unsigned long long)metrics->gpu_download_bytes,
                      metrics->records_spilled,
                      metrics->chunks_written,
                      metrics->db_size_bytes);
    if (written < 0) {
        return -1;
    }

    return fflush(csv_file);
}

static int nlo_bench_prepare_tensor_case_data(
    const nlo_bench_tensor_shape* shape,
    size_t num_records,
    int storage_enabled,
    const char* storage_path,
    nlo_bench_tensor_case_data* out_case_data,
    char* note,
    size_t note_capacity
)
{
    static const char* tensor_linear_expr = "i*(c0*wt*wt + c1*(kx*kx + ky*ky))";
    const double delta_time = 0.04;
    const double delta_x = 0.15;
    const double delta_y = 0.15;
    const double temporal_width = 0.24;
    const double x_width = 0.60;
    const double y_width = 0.70;

    if (shape == NULL || out_case_data == NULL || shape->total_samples == 0u || num_records == 0u) {
        nlo_bench_copy_note(note, note_capacity, "Invalid tensor benchmark case configuration.");
        return -1;
    }

    memset(out_case_data, 0, sizeof(*out_case_data));
    out_case_data->total_samples = shape->total_samples;
    out_case_data->nt = shape->nt;
    out_case_data->nx = shape->nx;
    out_case_data->ny = shape->ny;
    out_case_data->num_records = num_records;
    out_case_data->storage_enabled = storage_enabled;
    if (storage_path != NULL) {
        snprintf(out_case_data->storage_path, sizeof(out_case_data->storage_path), "%s", storage_path);
    }

    out_case_data->config = create_sim_config(shape->total_samples);
    if (out_case_data->config == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Failed to allocate tensor sim_config.");
        return -1;
    }

    out_case_data->input_field = (nlo_complex*)calloc(shape->total_samples, sizeof(nlo_complex));
    out_case_data->output_field = (nlo_complex*)calloc(shape->total_samples, sizeof(nlo_complex));
    if (out_case_data->input_field == NULL || out_case_data->output_field == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Failed to allocate tensor benchmark buffers.");
        free(out_case_data->input_field);
        free(out_case_data->output_field);
        free_sim_config(out_case_data->config);
        memset(out_case_data, 0, sizeof(*out_case_data));
        return -1;
    }

    out_case_data->config->tensor.nt = shape->nt;
    out_case_data->config->tensor.nx = shape->nx;
    out_case_data->config->tensor.ny = shape->ny;
    out_case_data->config->tensor.layout = NLO_TENSOR_LAYOUT_XYT_T_FAST;
    out_case_data->config->time.nt = shape->nt;
    out_case_data->config->time.delta_time = delta_time;
    out_case_data->config->time.pulse_period = (double)shape->nt * delta_time;
    out_case_data->config->spatial.nx = shape->nx;
    out_case_data->config->spatial.ny = shape->ny;
    out_case_data->config->spatial.delta_x = delta_x;
    out_case_data->config->spatial.delta_y = delta_y;
    out_case_data->config->propagation.propagation_distance = 0.2;
    out_case_data->config->propagation.starting_step_size = 0.02;
    out_case_data->config->propagation.max_step_size = 0.02;
    out_case_data->config->propagation.min_step_size = 0.02;
    out_case_data->config->propagation.error_tolerance = 1e-9;
    out_case_data->config->runtime.linear_factor_expr = tensor_linear_expr;
    out_case_data->config->runtime.dispersion_factor_expr = tensor_linear_expr;
    out_case_data->config->runtime.nonlinear_expr = "0";
    out_case_data->config->runtime.num_constants = 2u;
    out_case_data->config->runtime.constants[0] = 0.04;
    out_case_data->config->runtime.constants[1] = -0.20;

    for (size_t t = 0u; t < shape->nt; ++t) {
        const long centered = (t <= shape->nt / 2u)
                                  ? (long)t
                                  : (long)t - (long)shape->nt;
        const double omega = (2.0 * NLO_BENCH_PI * (double)centered) /
                             ((double)shape->nt * delta_time);
        out_case_data->config->frequency.frequency_grid[t] = nlo_make(omega, 0.0);
    }

    for (size_t x = 0u; x < shape->nx; ++x) {
        const double x_value = ((double)x - 0.5 * (double)(shape->nx - 1u)) * delta_x;
        for (size_t y = 0u; y < shape->ny; ++y) {
            const double y_value = ((double)y - 0.5 * (double)(shape->ny - 1u)) * delta_y;
            const double transverse =
                exp(-((x_value / x_width) * (x_value / x_width)) -
                    ((y_value / y_width) * (y_value / y_width)));
            for (size_t t = 0u; t < shape->nt; ++t) {
                size_t index = 0u;
                const double centered_t =
                    ((double)t - 0.5 * (double)(shape->nt - 1u)) * delta_time;
                const double temporal = exp(-((centered_t / temporal_width) * (centered_t / temporal_width)));
                const double phase = 0.15 * centered_t * centered_t;

                if (nlo_bench_checked_mul_size_t(x, shape->ny, &index) != 0 ||
                    index > SIZE_MAX - y) {
                    nlo_bench_copy_note(note, note_capacity, "Tensor index overflow while generating input field.");
                    return -1;
                }
                index += y;
                if (nlo_bench_checked_mul_size_t(index, shape->nt, &index) != 0 ||
                    index > SIZE_MAX - t) {
                    nlo_bench_copy_note(note, note_capacity, "Tensor index overflow while generating input field.");
                    return -1;
                }
                index += t;
                out_case_data->input_field[index] =
                    nlo_make(temporal * transverse * cos(phase),
                             temporal * transverse * sin(phase));
            }
        }
    }

    nlo_bench_copy_note(note, note_capacity, "");
    return 0;
}

static void nlo_bench_destroy_case_data(nlo_bench_tensor_case_data* case_data)
{
    if (case_data == NULL) {
        return;
    }

    free(case_data->input_field);
    free(case_data->output_field);
    free_sim_config(case_data->config);
    memset(case_data, 0, sizeof(*case_data));
}

static int nlo_bench_probe_case(
    const nlo_execution_options* exec_options,
    const sim_config* config,
    size_t total_samples,
    size_t num_records,
    const nlo_storage_options* storage_options,
    nlo_allocation_info* out_info,
    char* note,
    size_t note_capacity
)
{
    simulation_state* state = NULL;
    int status = 0;

    if (exec_options == NULL || config == NULL || total_samples == 0u || out_info == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Invalid benchmark probe arguments.");
        return -1;
    }

    memset(out_info, 0, sizeof(*out_info));
    if (storage_options != NULL) {
        status = nlo_init_simulation_state_with_storage(config,
                                                        total_samples,
                                                        num_records,
                                                        exec_options,
                                                        storage_options,
                                                        out_info,
                                                        &state);
    } else {
        status = nlo_init_simulation_state(config,
                                           total_samples,
                                           num_records,
                                           exec_options,
                                           out_info,
                                           &state);
    }
    if (status != 0 || state == NULL) {
        nlo_bench_copy_note(note, note_capacity, "State initialization failed.");
        return -1;
    }

    free_simulation_state(state);
    nlo_bench_copy_note(note, note_capacity, "");
    return 0;
}

static int nlo_bench_execute_single_run(
    const nlo_execution_options* exec_options,
    const nlo_bench_tensor_case_data* case_data,
    size_t system_memory_bytes,
    size_t gpu_device_budget_bytes,
    nlo_bench_tensor_metrics* out_metrics,
    char* note,
    size_t note_capacity
)
{
    simulation_state* state = NULL;
    nlo_storage_options storage_options;
    nlo_allocation_info allocation_info;
    nlo_bench_timestamp total_start;
    nlo_bench_timestamp total_end;
    nlo_bench_timestamp phase_start;
    nlo_bench_timestamp phase_end;
    int init_status = 0;

    if (exec_options == NULL || case_data == NULL || out_metrics == NULL) {
        nlo_bench_copy_note(note, note_capacity, "Invalid benchmark run arguments.");
        return -1;
    }

    memset(out_metrics, 0, sizeof(*out_metrics));
    memset(&storage_options, 0, sizeof(storage_options));
    memset(&allocation_info, 0, sizeof(allocation_info));

    out_metrics->total_samples = case_data->total_samples;
    out_metrics->nt = case_data->nt;
    out_metrics->nx = case_data->nx;
    out_metrics->ny = case_data->ny;
    out_metrics->num_records = case_data->num_records;
    out_metrics->system_memory_bytes = system_memory_bytes;
    out_metrics->gpu_device_budget_bytes = gpu_device_budget_bytes;
    if (nlo_bench_checked_mul_size_t(case_data->num_records,
                                     case_data->total_samples * sizeof(nlo_complex),
                                     &out_metrics->output_bytes) != 0) {
        out_metrics->output_bytes = SIZE_MAX;
    }

    if (case_data->storage_enabled) {
        storage_options = nlo_storage_options_default();
        storage_options.sqlite_path = case_data->storage_path;
        storage_options.chunk_records = 1u;
        storage_options.cap_policy = NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES;
    }

    nlo_perf_profile_set_enabled(1);
    nlo_perf_profile_reset();
    nlo_bench_now(&total_start);

    nlo_bench_now(&phase_start);
    if (case_data->storage_enabled) {
        init_status = nlo_init_simulation_state_with_storage(case_data->config,
                                                             case_data->total_samples,
                                                             case_data->num_records,
                                                             exec_options,
                                                             &storage_options,
                                                             &allocation_info,
                                                             &state);
    } else {
        init_status = nlo_init_simulation_state(case_data->config,
                                                case_data->total_samples,
                                                case_data->num_records,
                                                exec_options,
                                                &allocation_info,
                                                &state);
    }
    nlo_bench_now(&phase_end);
    out_metrics->init_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

    out_metrics->per_record_bytes = allocation_info.per_record_bytes;
    out_metrics->allocated_records = allocation_info.allocated_records;
    out_metrics->host_snapshot_bytes = allocation_info.host_snapshot_bytes;
    out_metrics->working_vector_bytes = allocation_info.working_vector_bytes;
    out_metrics->device_ring_capacity = allocation_info.device_ring_capacity;
    out_metrics->working_set_bytes = nlo_bench_compute_working_set_bytes(&allocation_info);

    if (init_status != 0 || state == NULL) {
        nlo_bench_now(&total_end);
        out_metrics->total_ms = nlo_bench_elapsed_ms(&total_start, &total_end);
        nlo_bench_copy_note(note, note_capacity, "State initialization failed.");
        nlo_perf_profile_set_enabled(0);
        return -1;
    }

    nlo_bench_now(&phase_start);
    if (simulation_state_upload_initial_field(state, case_data->input_field) != NLO_VEC_STATUS_OK) {
        nlo_bench_now(&phase_end);
        out_metrics->upload_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
        nlo_bench_copy_note(note, note_capacity, "Initial field upload failed.");
        free_simulation_state(state);
        nlo_perf_profile_set_enabled(0);
        return -1;
    }
    nlo_bench_now(&phase_end);
    out_metrics->upload_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

    nlo_bench_now(&phase_start);
    solve_rk4(state);
    nlo_bench_now(&phase_end);
    out_metrics->solve_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

    if (state->snapshot_status != NLO_VEC_STATUS_OK) {
        out_metrics->records_spilled = state->snapshot_result.records_spilled;
        out_metrics->chunks_written = state->snapshot_result.chunks_written;
        out_metrics->db_size_bytes = state->snapshot_result.db_size_bytes;
        nlo_bench_copy_note(note, note_capacity, "Snapshot spill path reported an error.");
        free_simulation_state(state);
        nlo_perf_profile_set_enabled(0);
        return -1;
    }

    nlo_bench_now(&phase_start);
    if (simulation_state_download_current_field(state, case_data->output_field) != NLO_VEC_STATUS_OK) {
        nlo_bench_now(&phase_end);
        out_metrics->download_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);
        out_metrics->records_spilled = state->snapshot_result.records_spilled;
        out_metrics->chunks_written = state->snapshot_result.chunks_written;
        out_metrics->db_size_bytes = state->snapshot_result.db_size_bytes;
        nlo_bench_copy_note(note, note_capacity, "Current field download failed.");
        free_simulation_state(state);
        nlo_perf_profile_set_enabled(0);
        return -1;
    }
    nlo_bench_now(&phase_end);
    out_metrics->download_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

    out_metrics->records_spilled = state->snapshot_result.records_spilled;
    out_metrics->chunks_written = state->snapshot_result.chunks_written;
    out_metrics->db_size_bytes = state->snapshot_result.db_size_bytes;

    nlo_bench_now(&phase_start);
    free_simulation_state(state);
    nlo_bench_now(&phase_end);
    out_metrics->teardown_ms = nlo_bench_elapsed_ms(&phase_start, &phase_end);

    nlo_bench_now(&total_end);
    out_metrics->total_ms = nlo_bench_elapsed_ms(&total_start, &total_end);
    if (out_metrics->total_ms > 0.0) {
        out_metrics->samples_per_sec =
            ((double)case_data->total_samples * 1000.0) / out_metrics->total_ms;
    }

    {
        nlo_perf_profile_snapshot snapshot;
        memset(&snapshot, 0, sizeof(snapshot));
        nlo_perf_profile_snapshot_read(&snapshot);
        out_metrics->dispersion_ms = snapshot.dispersion_ms;
        out_metrics->nonlinear_ms = snapshot.nonlinear_ms;
        out_metrics->gpu_dispatch_count = snapshot.gpu_dispatch_count;
        out_metrics->gpu_copy_count = snapshot.gpu_copy_count;
        out_metrics->gpu_memory_pass_count = snapshot.gpu_memory_pass_count;
        out_metrics->gpu_memory_pass_bytes = snapshot.gpu_memory_pass_bytes;
        out_metrics->gpu_upload_count = snapshot.gpu_upload_count;
        out_metrics->gpu_download_count = snapshot.gpu_download_count;
        out_metrics->gpu_upload_bytes = snapshot.gpu_upload_bytes;
        out_metrics->gpu_download_bytes = snapshot.gpu_download_bytes;
    }

    nlo_perf_profile_set_enabled(0);
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
    double sum = 0.0;
    double variance = 0.0;
    double* sorted = NULL;

    if (values == NULL || count == 0u || out_summary == NULL) {
        return -1;
    }

    memset(out_summary, 0, sizeof(*out_summary));
    out_summary->min_ms = values[0];
    out_summary->max_ms = values[0];

    for (size_t i = 0u; i < count; ++i) {
        sum += values[i];
        if (values[i] < out_summary->min_ms) {
            out_summary->min_ms = values[i];
        }
        if (values[i] > out_summary->max_ms) {
            out_summary->max_ms = values[i];
        }
    }
    out_summary->mean_ms = sum / (double)count;

    for (size_t i = 0u; i < count; ++i) {
        const double delta = values[i] - out_summary->mean_ms;
        variance += delta * delta;
    }
    out_summary->stddev_ms = sqrt(variance / (double)count);

    sorted = (double*)calloc(count, sizeof(double));
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

static int nlo_bench_default_tensor_scales(size_t* out_scales, size_t max_scales, size_t* out_count)
{
    size_t scale = 8u;
    size_t count = 0u;

    if (out_scales == NULL || out_count == NULL || max_scales == 0u) {
        return -1;
    }

    while (count < max_scales) {
        out_scales[count] = scale;
        count += 1u;
        if (scale > (SIZE_MAX / 2u)) {
            break;
        }
        scale *= 2u;
        if (scale > 1024u) {
            break;
        }
    }

    *out_count = count;
    return 0;
}

static int nlo_bench_plan_tensor_candidates(
    const nlo_bench_tensor_options* options,
    const nlo_bench_vk_context* vk_context,
    int gpu_available,
    nlo_bench_tensor_candidate* out_candidates,
    size_t* out_count
)
{
    size_t scales[NLO_BENCH_MAX_TENSOR_CANDIDATES];
    size_t scale_count = 0u;
    size_t system_budget_bytes = 0u;

    if (options == NULL || out_candidates == NULL || out_count == NULL) {
        return -1;
    }

    if (options->tensor_scales != NULL && options->tensor_scale_count > 0u) {
        scale_count = options->tensor_scale_count;
        for (size_t i = 0u; i < scale_count && i < NLO_BENCH_MAX_TENSOR_CANDIDATES; ++i) {
            scales[i] = options->tensor_scales[i];
        }
    } else if (nlo_bench_default_tensor_scales(scales,
                                               NLO_BENCH_MAX_TENSOR_CANDIDATES,
                                               &scale_count) != 0) {
        return -1;
    }

    system_budget_bytes = nlo_bench_effective_host_budget_bytes(options);
    *out_count = 0u;

    for (size_t i = 0u; i < scale_count && *out_count < NLO_BENCH_MAX_TENSOR_CANDIDATES; ++i) {
        nlo_bench_tensor_candidate* candidate = &out_candidates[*out_count];
        nlo_bench_tensor_case_data probe_case;
        nlo_allocation_info cpu_info;
        nlo_allocation_info gpu_info;
        nlo_execution_options cpu_options;
        nlo_execution_options gpu_options;
        nlo_bench_tensor_region_inputs region_inputs;
        char note[NLO_BENCH_NOTE_CAP];
        size_t working_set_bytes = 0u;
        size_t gpu_budget_bytes = 0u;
        int cpu_probe_ok = 0;
        int gpu_probe_ok = 0;

        memset(candidate, 0, sizeof(*candidate));
        memset(&probe_case, 0, sizeof(probe_case));
        memset(&cpu_info, 0, sizeof(cpu_info));
        memset(&gpu_info, 0, sizeof(gpu_info));

        if (nlo_bench_tensor_shape_from_scale(scales[i], &candidate->shape) != 0) {
            continue;
        }
        if (nlo_bench_prepare_tensor_case_data(&candidate->shape,
                                               1u,
                                               0,
                                               NULL,
                                               &probe_case,
                                               note,
                                               sizeof(note)) != 0) {
            continue;
        }

        cpu_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
        cpu_probe_ok = (nlo_bench_probe_case(&cpu_options,
                                             probe_case.config,
                                             probe_case.total_samples,
                                             1u,
                                             NULL,
                                             &cpu_info,
                                             note,
                                             sizeof(note)) == 0);
        if (cpu_probe_ok) {
            working_set_bytes = nlo_bench_compute_working_set_bytes(&cpu_info);
            candidate->per_record_bytes = cpu_info.per_record_bytes;
        }

        if (gpu_available) {
            gpu_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_VULKAN);
            gpu_options.vulkan.physical_device = vk_context->physical_device;
            gpu_options.vulkan.device = vk_context->device;
            gpu_options.vulkan.queue = vk_context->queue;
            gpu_options.vulkan.queue_family_index = vk_context->queue_family_index;
            gpu_options.vulkan.command_pool = VK_NULL_HANDLE;
            gpu_probe_ok = (nlo_bench_probe_case(&gpu_options,
                                                 probe_case.config,
                                                 probe_case.total_samples,
                                                 1u,
                                                 NULL,
                                                 &gpu_info,
                                                 note,
                                                 sizeof(note)) == 0);
        }

        if (gpu_probe_ok) {
            const size_t gpu_working_set = nlo_bench_compute_working_set_bytes(&gpu_info);
            if (gpu_working_set > working_set_bytes) {
                working_set_bytes = gpu_working_set;
            }
            if (candidate->per_record_bytes == 0u) {
                candidate->per_record_bytes = gpu_info.per_record_bytes;
            }
            gpu_budget_bytes = gpu_info.device_budget_bytes;
        }

        if (options->planner_gpu_bytes > 0u) {
            gpu_budget_bytes = options->planner_gpu_bytes;
        }

        region_inputs.working_set_bytes = working_set_bytes;
        region_inputs.host_budget_bytes = system_budget_bytes;
        region_inputs.gpu_budget_bytes = gpu_budget_bytes;
        region_inputs.cpu_init_ok = cpu_probe_ok;
        region_inputs.gpu_init_ok = (options->planner_gpu_bytes > 0u) ? 1 : gpu_probe_ok;

        candidate->region = nlo_bench_tensor_classify_fit_region(&region_inputs);
        candidate->working_set_bytes = working_set_bytes;
        candidate->system_budget_bytes = system_budget_bytes;
        candidate->gpu_budget_bytes = gpu_budget_bytes;
        candidate->cpu_probe_ok = cpu_probe_ok;
        candidate->gpu_probe_ok = gpu_probe_ok;

        nlo_bench_destroy_case_data(&probe_case);
        *out_count += 1u;

        if (!cpu_probe_ok && options->tensor_scale_count == 0u) {
            break;
        }
    }

    return 0;
}

static void nlo_bench_print_tensor_plan(
    const nlo_bench_tensor_candidate* candidates,
    size_t candidate_count,
    const nlo_bench_tensor_options* options
)
{
    static const double spill_factors[NLO_BENCH_SPILL_TARGET_COUNT] = {1.25, 1.50, 2.00};
    const nlo_bench_tensor_candidate* spill_candidate = NULL;

    if (candidates == NULL || options == NULL) {
        return;
    }

    printf("Tensor scaling plan:\n");
    printf("  effective_host_budget: %zu bytes\n", nlo_bench_effective_host_budget_bytes(options));
    printf("  effective_gpu_budget: %zu bytes\n", options->planner_gpu_bytes);

    for (size_t i = 0u; i < candidate_count; ++i) {
        const nlo_bench_tensor_candidate* candidate = &candidates[i];
        printf("  scale=%zu shape=(nt=%zu nx=%zu ny=%zu total=%zu) region=%s working_set=%zu cpu_probe=%s gpu_probe=%s\n",
               candidate->shape.scale,
               candidate->shape.nt,
               candidate->shape.nx,
               candidate->shape.ny,
               candidate->shape.total_samples,
               nlo_bench_tensor_region_label(candidate->region),
               candidate->working_set_bytes,
               candidate->cpu_probe_ok ? "ok" : "fail",
               candidate->gpu_probe_ok ? "ok" : "fail");
        if (candidate->region == NLO_BENCH_TENSOR_REGION_GPU_FIT) {
            spill_candidate = candidate;
        }
    }

    if (spill_candidate != NULL) {
        printf("  output_spill baseline: scale=%zu total=%zu per_record=%zu bytes\n",
               spill_candidate->shape.scale,
               spill_candidate->shape.total_samples,
               spill_candidate->per_record_bytes);
        for (size_t i = 0u; i < NLO_BENCH_SPILL_TARGET_COUNT; ++i) {
            const size_t target_output_bytes =
                nlo_bench_scale_size_t_double(spill_candidate->system_budget_bytes, spill_factors[i]);
            const size_t spill_records =
                nlo_bench_tensor_records_for_output_bytes(spill_candidate->per_record_bytes, target_output_bytes);
            printf("    output_spill factor=%.2f records=%zu output_bytes=%zu\n",
                   spill_factors[i],
                   spill_records,
                   spill_records * spill_candidate->per_record_bytes);
        }
    } else {
        printf("  output_spill skipped: no gpu_fit tensor candidate available.\n");
    }
}

static void nlo_bench_print_backend_summary(
    const char* backend_label,
    const nlo_bench_tensor_metrics* last_metrics,
    const double* measured_totals,
    size_t measured_count
)
{
    nlo_bench_summary total_summary;
    if (backend_label == NULL || last_metrics == NULL || measured_totals == NULL || measured_count == 0u) {
        return;
    }

    if (nlo_bench_compute_summary(measured_totals, measured_count, &total_summary) != 0) {
        printf("  %-3s warning: failed to compute timing summary.\n", backend_label);
        return;
    }

    printf("  %-3s total(ms): mean=%.3f median=%.3f min=%.3f max=%.3f std=%.3f\n",
           backend_label,
           total_summary.mean_ms,
           total_summary.median_ms,
           total_summary.min_ms,
           total_summary.max_ms,
           total_summary.stddev_ms);
    if (last_metrics->gpu_dispatch_count > 0u || last_metrics->gpu_copy_count > 0u) {
        printf("      gpu ops/run: dispatches=%llu copies=%llu memory_passes=%llu\n",
               (unsigned long long)last_metrics->gpu_dispatch_count,
               (unsigned long long)last_metrics->gpu_copy_count,
               (unsigned long long)last_metrics->gpu_memory_pass_count);
    }
    if (last_metrics->records_spilled > 0u || last_metrics->chunks_written > 0u) {
        printf("      storage spill: records=%zu chunks=%zu db_size=%.2f MiB\n",
               last_metrics->records_spilled,
               last_metrics->chunks_written,
               (double)last_metrics->db_size_bytes / (1024.0 * 1024.0));
    }
}

static int nlo_bench_run_case(
    nlo_bench_runtime_backend backend,
    const nlo_bench_tensor_options* options,
    const char* region_label,
    const nlo_bench_tensor_case_data* case_data,
    size_t system_memory_bytes,
    size_t gpu_device_budget_bytes,
    const nlo_bench_vk_context* vk_context,
    FILE* csv_file,
    const char* skip_reason,
    const char* skip_status,
    int* out_error_count
)
{
    const char* backend_label = nlo_bench_backend_label(backend);
    nlo_execution_options exec_options;
    nlo_bench_tensor_metrics metrics;
    double* measured_totals = NULL;
    size_t recorded = 0u;
    char note[NLO_BENCH_NOTE_CAP];

    if (options == NULL || region_label == NULL || case_data == NULL || out_error_count == NULL) {
        return -1;
    }

    memset(&metrics, 0, sizeof(metrics));
    if (skip_reason != NULL && *skip_reason != '\0') {
        printf("  %-3s %s: %s\n", backend_label, skip_status, skip_reason);
        if (csv_file != NULL &&
            nlo_bench_write_csv_row(csv_file,
                                    region_label,
                                    backend_label,
                                    options->warmup_runs,
                                    options->measured_runs,
                                    0u,
                                    &metrics,
                                    (skip_status != NULL) ? skip_status : "SKIPPED",
                                    skip_reason) != 0) {
            *out_error_count += 1;
            return -1;
        }
        return 0;
    }

    exec_options = (backend == NLO_BENCH_RUNTIME_CPU)
                       ? nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU)
                       : nlo_execution_options_default(NLO_VECTOR_BACKEND_VULKAN);
    if (case_data->storage_enabled) {
        exec_options.record_ring_target = 1u;
    }
    if (backend == NLO_BENCH_RUNTIME_GPU) {
        if (vk_context == NULL) {
            *out_error_count += 1;
            return -1;
        }
        exec_options.vulkan.physical_device = vk_context->physical_device;
        exec_options.vulkan.device = vk_context->device;
        exec_options.vulkan.queue = vk_context->queue;
        exec_options.vulkan.queue_family_index = vk_context->queue_family_index;
        exec_options.vulkan.command_pool = VK_NULL_HANDLE;
    }

    measured_totals = (double*)calloc(options->measured_runs, sizeof(double));
    if (measured_totals == NULL) {
        *out_error_count += 1;
        return -1;
    }

    for (size_t run = 0u; run < options->warmup_runs + options->measured_runs; ++run) {
        if (nlo_bench_execute_single_run(&exec_options,
                                         case_data,
                                         system_memory_bytes,
                                         gpu_device_budget_bytes,
                                         &metrics,
                                         note,
                                         sizeof(note)) != 0) {
            const char* status = (skip_status != NULL) ? skip_status : "ERROR";
            printf("  %-3s %s: %s\n", backend_label, status, note);
            if (csv_file != NULL) {
                (void)nlo_bench_write_csv_row(csv_file,
                                              region_label,
                                              backend_label,
                                              options->warmup_runs,
                                              options->measured_runs,
                                              run + 1u,
                                              &metrics,
                                              status,
                                              note);
            }
            if (skip_status == NULL || strcmp(skip_status, "EXPECTED_LIMIT") != 0) {
                *out_error_count += 1;
                free(measured_totals);
                return -1;
            }
            free(measured_totals);
            return 0;
        }

        if (run < options->warmup_runs) {
            continue;
        }

        measured_totals[recorded] = metrics.total_ms;
        recorded += 1u;
        if (csv_file != NULL &&
            nlo_bench_write_csv_row(csv_file,
                                    region_label,
                                    backend_label,
                                    options->warmup_runs,
                                    options->measured_runs,
                                    recorded,
                                    &metrics,
                                    "OK",
                                    "") != 0) {
            *out_error_count += 1;
            free(measured_totals);
            return -1;
        }
    }

    if (recorded > 0u) {
        nlo_bench_print_backend_summary(backend_label, &metrics, measured_totals, recorded);
    }

    free(measured_totals);
    return 0;
}

int nlo_bench_run_tensor_scaling(
    const nlo_bench_tensor_options* options,
    const nlo_bench_vk_context* vk_context,
    int gpu_available,
    const char* gpu_skip_reason,
    int* out_error_count
)
{
    static const double spill_factors[NLO_BENCH_SPILL_TARGET_COUNT] = {1.25, 1.50, 2.00};
    nlo_bench_tensor_candidate candidates[NLO_BENCH_MAX_TENSOR_CANDIDATES];
    const nlo_bench_tensor_candidate* spill_candidate = NULL;
    size_t candidate_count = 0u;
    FILE* csv_file = NULL;

    if (options == NULL || out_error_count == NULL) {
        return -1;
    }

    memset(candidates, 0, sizeof(candidates));
    if (nlo_bench_plan_tensor_candidates(options,
                                         vk_context,
                                         gpu_available,
                                         candidates,
                                         &candidate_count) != 0) {
        *out_error_count += 1;
        return -1;
    }

    nlo_bench_print_tensor_plan(candidates, candidate_count, options);
    if (options->dry_run) {
        return 0;
    }

    if (options->csv_path == NULL || options->storage_dir == NULL) {
        *out_error_count += 1;
        return -1;
    }
    if (nlo_bench_make_parent_dirs(options->csv_path) != 0 ||
        nlo_bench_make_directory_tree(options->storage_dir) != 0) {
        *out_error_count += 1;
        return -1;
    }

    {
        const int csv_exists = nlo_bench_file_exists(options->csv_path);
        csv_file = fopen(options->csv_path, "a");
        if (csv_file == NULL) {
            *out_error_count += 1;
            return -1;
        }
        if (!csv_exists && nlo_bench_write_csv_header(csv_file) != 0) {
            fclose(csv_file);
            *out_error_count += 1;
            return -1;
        }
    }

    for (size_t i = 0u; i < candidate_count; ++i) {
        const nlo_bench_tensor_candidate* candidate = &candidates[i];
        nlo_bench_tensor_case_data case_data;
        char note[NLO_BENCH_NOTE_CAP];

        if (candidate->region == NLO_BENCH_TENSOR_REGION_TOO_LARGE) {
            continue;
        }
        if (candidate->region == NLO_BENCH_TENSOR_REGION_GPU_FIT) {
            spill_candidate = candidate;
        }

        printf("\nTensor scale %zu (%s)\n",
               candidate->shape.scale,
               nlo_bench_tensor_region_label(candidate->region));

        if (nlo_bench_prepare_tensor_case_data(&candidate->shape,
                                               1u,
                                               0,
                                               NULL,
                                               &case_data,
                                               note,
                                               sizeof(note)) != 0) {
            fprintf(stderr, "Failed to prepare tensor case: %s\n", note);
            *out_error_count += 1;
            continue;
        }

        if (options->backend_request == 0 || options->backend_request == 2) {
            (void)nlo_bench_run_case(NLO_BENCH_RUNTIME_CPU,
                                     options,
                                     nlo_bench_tensor_region_label(candidate->region),
                                     &case_data,
                                     candidate->system_budget_bytes,
                                     candidate->gpu_budget_bytes,
                                     NULL,
                                     csv_file,
                                     NULL,
                                     NULL,
                                     out_error_count);
        }
        if (options->backend_request == 1 || options->backend_request == 2) {
            const char* skip_reason = NULL;
            const char* skip_status = NULL;
            if (!gpu_available) {
                skip_reason = gpu_skip_reason;
                skip_status = "SKIPPED";
            } else if (candidate->region == NLO_BENCH_TENSOR_REGION_HOST_FIT_ONLY) {
                skip_reason = "Tensor working set exceeds effective GPU budget for this machine.";
                skip_status = "EXPECTED_LIMIT";
            }
            (void)nlo_bench_run_case(NLO_BENCH_RUNTIME_GPU,
                                     options,
                                     nlo_bench_tensor_region_label(candidate->region),
                                     &case_data,
                                     candidate->system_budget_bytes,
                                     candidate->gpu_budget_bytes,
                                     vk_context,
                                     csv_file,
                                     skip_reason,
                                     skip_status,
                                     out_error_count);
        }

        nlo_bench_destroy_case_data(&case_data);
    }

    if (spill_candidate != NULL) {
        for (size_t i = 0u; i < NLO_BENCH_SPILL_TARGET_COUNT; ++i) {
            nlo_bench_tensor_case_data case_data;
            char note[NLO_BENCH_NOTE_CAP];
            char storage_path[NLO_BENCH_MAX_PATH];
            const size_t target_output_bytes =
                nlo_bench_scale_size_t_double(spill_candidate->system_budget_bytes, spill_factors[i]);
            const size_t spill_records =
                nlo_bench_tensor_records_for_output_bytes(spill_candidate->per_record_bytes, target_output_bytes);

            snprintf(storage_path,
                     sizeof(storage_path),
                     "%s/tensor_output_spill_scale%zu_factor%zu.sqlite",
                     options->storage_dir,
                     spill_candidate->shape.scale,
                     (size_t)lround(spill_factors[i] * 100.0));

            printf("\nTensor output spill factor %.2f (records=%zu)\n", spill_factors[i], spill_records);
            if (nlo_bench_prepare_tensor_case_data(&spill_candidate->shape,
                                                   spill_records,
                                                   1,
                                                   storage_path,
                                                   &case_data,
                                                   note,
                                                   sizeof(note)) != 0) {
                fprintf(stderr, "Failed to prepare tensor spill case: %s\n", note);
                *out_error_count += 1;
                continue;
            }

            if (options->backend_request == 0 || options->backend_request == 2) {
                (void)nlo_bench_run_case(NLO_BENCH_RUNTIME_CPU,
                                         options,
                                         nlo_bench_tensor_region_label(NLO_BENCH_TENSOR_REGION_OUTPUT_SPILL),
                                         &case_data,
                                         spill_candidate->system_budget_bytes,
                                         spill_candidate->gpu_budget_bytes,
                                         NULL,
                                         csv_file,
                                         NULL,
                                         NULL,
                                         out_error_count);
            }
            if (options->backend_request == 1 || options->backend_request == 2) {
                (void)nlo_bench_run_case(NLO_BENCH_RUNTIME_GPU,
                                         options,
                                         nlo_bench_tensor_region_label(NLO_BENCH_TENSOR_REGION_OUTPUT_SPILL),
                                         &case_data,
                                         spill_candidate->system_budget_bytes,
                                         spill_candidate->gpu_budget_bytes,
                                         vk_context,
                                         csv_file,
                                         gpu_available ? NULL : gpu_skip_reason,
                                         "SKIPPED",
                                         out_error_count);
            }

            nlo_bench_destroy_case_data(&case_data);
        }
    }

    if (csv_file != NULL) {
        fclose(csv_file);
    }
    return 0;
}
