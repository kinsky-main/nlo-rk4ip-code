/**
 * @file perf_profile.c
 * @brief Lightweight runtime performance counters for benchmarking diagnostics.
 */

#include "utility/perf_profile.h"

#include <string.h>

#if NLO_ENABLE_RUNTIME_PROFILING

#ifndef NLO_PERF_TRACE_UNSET_U64
#define NLO_PERF_TRACE_UNSET_U64 UINT64_MAX
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct {
    nlo_perf_profile_trace_row* rows;
    size_t capacity;
    size_t count;
    uint64_t dropped_count;
    uint64_t run_id;
    uint64_t backend_kind;
    uint64_t jit_mode;
    uint64_t scenario_kind;
    uint64_t step_index;
    uint64_t reject_attempt;
} nlo_perf_profile_trace_state;

typedef struct {
    int enabled;
    nlo_perf_gpu_timestamp_mode gpu_timestamp_mode;
    nlo_perf_profile_snapshot snapshot;
    nlo_perf_profile_trace_state trace;
} nlo_perf_profile_state;

static const char* const g_nlo_perf_profile_event_names[NLO_PERF_EVENT_COUNT] = {
    "backend_create",
    "state_allocate_vectors",
    "operator_compile_lower",
    "operator_jit_warmup",
    "frequency_grid_init",
    "tensor_axis_init",
    "potential_init",
    "dispersion_factor_init",
    "fft_plan_create",
    "initial_field_upload",
    "begin_simulation",
    "end_simulation",
    "host_step_admin",
    "host_error_control",
    "host_progress_logging",
    "host_snapshot_bookkeeping",
    "snapshot_pause_end_sim",
    "snapshot_download",
    "snapshot_resume_begin_sim",
    "fft_forward",
    "fft_inverse",
    "dispersion_apply",
    "nonlinear_apply",
    "operator_program_jit_execute",
    "operator_program_interpreter_execute",
    "weighted_rms_error",
    "vec_copy",
    "vec_add",
    "vec_scalar_mul",
    "vec_mul",
    "vec_magnitude_squared",
    "vec_axpy",
    "vec_affine_comb2",
    "vec_affine_comb3",
    "vec_affine_comb4",
    "vec_embedded_error_pair",
    "vec_lerp",
    "vk_command_begin",
    "vk_command_submit_wait",
    "vk_simulation_flush",
    "vk_host_to_device_transfer",
    "vk_device_to_host_transfer"
};

static nlo_perf_profile_state g_nlo_perf_profile_state = {
    0,
    NLO_PERF_GPU_TIMESTAMPS_AUTO,
    {0},
    {
        NULL,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        NLO_PERF_TRACE_UNSET_U64,
        NLO_PERF_TRACE_UNSET_U64
    }
};

static void nlo_perf_profile_trace_reset_metadata(void)
{
    g_nlo_perf_profile_state.trace.run_id = 0u;
    g_nlo_perf_profile_state.trace.backend_kind = 0u;
    g_nlo_perf_profile_state.trace.jit_mode = 0u;
    g_nlo_perf_profile_state.trace.scenario_kind = 0u;
    g_nlo_perf_profile_state.trace.step_index = NLO_PERF_TRACE_UNSET_U64;
    g_nlo_perf_profile_state.trace.reject_attempt = NLO_PERF_TRACE_UNSET_U64;
}

static void nlo_perf_profile_trace_append(
    nlo_perf_event_id event,
    double host_wall_ms,
    double gpu_exec_ms,
    uint64_t bytes
)
{
    if (!g_nlo_perf_profile_state.enabled) {
        return;
    }
    if (event < 0 || event >= NLO_PERF_EVENT_COUNT) {
        return;
    }

    nlo_perf_profile_trace_state* trace = &g_nlo_perf_profile_state.trace;
    if (trace->rows == NULL || trace->capacity == 0u) {
        return;
    }

    if (trace->count >= trace->capacity) {
        trace->dropped_count += 1u;
        g_nlo_perf_profile_state.snapshot.trace_dropped_count = trace->dropped_count;
        return;
    }

    nlo_perf_profile_trace_row* row = &trace->rows[trace->count];
    row->run_id = trace->run_id;
    row->backend_kind = trace->backend_kind;
    row->jit_mode = trace->jit_mode;
    row->scenario_kind = trace->scenario_kind;
    row->step_index = trace->step_index;
    row->reject_attempt = trace->reject_attempt;
    row->event_id = (uint64_t)event;
    row->call_count = 1u;
    row->bytes = bytes;
    row->host_wall_ms = host_wall_ms;
    row->gpu_exec_ms = gpu_exec_ms;

    trace->count += 1u;
    g_nlo_perf_profile_state.snapshot.trace_row_count = (uint64_t)trace->count;
}

void nlo_perf_profile_set_enabled(int enabled)
{
    g_nlo_perf_profile_state.enabled = (enabled != 0) ? 1 : 0;
}

int nlo_perf_profile_is_enabled(void)
{
    return g_nlo_perf_profile_state.enabled;
}

void nlo_perf_profile_reset(void)
{
    memset(&g_nlo_perf_profile_state.snapshot, 0, sizeof(g_nlo_perf_profile_state.snapshot));
    g_nlo_perf_profile_state.trace.count = 0u;
    g_nlo_perf_profile_state.trace.dropped_count = 0u;
    nlo_perf_profile_trace_reset_metadata();
}

void nlo_perf_profile_snapshot_read(nlo_perf_profile_snapshot* out_snapshot)
{
    if (out_snapshot == NULL) {
        return;
    }

    *out_snapshot = g_nlo_perf_profile_state.snapshot;
}

void nlo_perf_profile_add_dispersion_time(double elapsed_ms)
{
    nlo_perf_profile_add_event(NLO_PERF_EVENT_DISPERSION_APPLY, elapsed_ms, 0u);
}

void nlo_perf_profile_add_nonlinear_time(double elapsed_ms)
{
    nlo_perf_profile_add_event(NLO_PERF_EVENT_NONLINEAR_APPLY, elapsed_ms, 0u);
}

void nlo_perf_profile_add_event(nlo_perf_event_id event, double elapsed_ms, uint64_t bytes)
{
    if (!g_nlo_perf_profile_state.enabled || elapsed_ms < 0.0) {
        return;
    }
    if (event < 0 || event >= NLO_PERF_EVENT_COUNT) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.event_total_ms[event] += elapsed_ms;
    g_nlo_perf_profile_state.snapshot.event_call_count[event] += 1u;
    g_nlo_perf_profile_state.snapshot.event_bytes[event] += bytes;

    if (event == NLO_PERF_EVENT_DISPERSION_APPLY) {
        g_nlo_perf_profile_state.snapshot.dispersion_ms += elapsed_ms;
        g_nlo_perf_profile_state.snapshot.dispersion_calls += 1u;
    } else if (event == NLO_PERF_EVENT_NONLINEAR_APPLY) {
        g_nlo_perf_profile_state.snapshot.nonlinear_ms += elapsed_ms;
        g_nlo_perf_profile_state.snapshot.nonlinear_calls += 1u;
    }

    nlo_perf_profile_trace_append(event, elapsed_ms, 0.0, bytes);
}

void nlo_perf_profile_add_event_gpu_time(nlo_perf_event_id event, double elapsed_ms)
{
    if (!g_nlo_perf_profile_state.enabled || elapsed_ms < 0.0) {
        return;
    }
    if (event < 0 || event >= NLO_PERF_EVENT_COUNT) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.event_gpu_exec_ms[event] += elapsed_ms;
    nlo_perf_profile_trace_append(event, 0.0, elapsed_ms, 0u);
}

void nlo_perf_profile_add_gpu_dispatch(
    uint64_t dispatch_count,
    uint64_t pass_count,
    uint64_t pass_bytes
)
{
    if (!g_nlo_perf_profile_state.enabled || dispatch_count == 0u) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.gpu_dispatch_count += dispatch_count;
    g_nlo_perf_profile_state.snapshot.gpu_memory_pass_count += pass_count;
    g_nlo_perf_profile_state.snapshot.gpu_memory_pass_bytes += pass_bytes;
}

void nlo_perf_profile_add_gpu_copy(uint64_t copy_count, uint64_t bytes)
{
    if (!g_nlo_perf_profile_state.enabled || copy_count == 0u) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.gpu_copy_count += copy_count;
    g_nlo_perf_profile_state.snapshot.gpu_device_copy_count += copy_count;
    g_nlo_perf_profile_state.snapshot.gpu_device_copy_bytes += bytes;
    g_nlo_perf_profile_state.snapshot.gpu_memory_pass_count += 2u * copy_count;
    g_nlo_perf_profile_state.snapshot.gpu_memory_pass_bytes += 2u * bytes;
}

void nlo_perf_profile_add_gpu_device_copy(uint64_t copy_count, uint64_t bytes)
{
    nlo_perf_profile_add_gpu_copy(copy_count, bytes);
}

void nlo_perf_profile_add_gpu_host_transfer_copy(uint64_t copy_count, uint64_t bytes)
{
    if (!g_nlo_perf_profile_state.enabled || copy_count == 0u) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.gpu_copy_count += copy_count;
    g_nlo_perf_profile_state.snapshot.gpu_host_transfer_copy_count += copy_count;
    g_nlo_perf_profile_state.snapshot.gpu_host_transfer_copy_bytes += bytes;
    g_nlo_perf_profile_state.snapshot.gpu_memory_pass_count += 2u * copy_count;
    g_nlo_perf_profile_state.snapshot.gpu_memory_pass_bytes += 2u * bytes;
}

void nlo_perf_profile_add_gpu_upload(uint64_t count, uint64_t bytes)
{
    if (!g_nlo_perf_profile_state.enabled || count == 0u) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.gpu_upload_count += count;
    g_nlo_perf_profile_state.snapshot.gpu_upload_bytes += bytes;
}

void nlo_perf_profile_add_gpu_download(uint64_t count, uint64_t bytes)
{
    if (!g_nlo_perf_profile_state.enabled || count == 0u) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.gpu_download_count += count;
    g_nlo_perf_profile_state.snapshot.gpu_download_bytes += bytes;
}

const char* nlo_perf_profile_event_name(nlo_perf_event_id event)
{
    if (event < 0 || event >= NLO_PERF_EVENT_COUNT) {
        return "unknown";
    }
    return g_nlo_perf_profile_event_names[event];
}

size_t nlo_perf_profile_event_count(void)
{
    return (size_t)NLO_PERF_EVENT_COUNT;
}

void nlo_perf_profile_trace_configure(nlo_perf_profile_trace_row* rows, size_t capacity)
{
    g_nlo_perf_profile_state.trace.rows = rows;
    g_nlo_perf_profile_state.trace.capacity = capacity;
    g_nlo_perf_profile_state.trace.count = 0u;
    g_nlo_perf_profile_state.trace.dropped_count = 0u;
    g_nlo_perf_profile_state.snapshot.trace_row_count = 0u;
    g_nlo_perf_profile_state.snapshot.trace_dropped_count = 0u;
}

void nlo_perf_profile_trace_set_run_metadata(
    uint64_t run_id,
    uint64_t backend_kind,
    uint64_t jit_mode,
    uint64_t scenario_kind
)
{
    g_nlo_perf_profile_state.trace.run_id = run_id;
    g_nlo_perf_profile_state.trace.backend_kind = backend_kind;
    g_nlo_perf_profile_state.trace.jit_mode = jit_mode;
    g_nlo_perf_profile_state.trace.scenario_kind = scenario_kind;
}

void nlo_perf_profile_trace_set_step(uint64_t step_index, uint64_t reject_attempt)
{
    g_nlo_perf_profile_state.trace.step_index = step_index;
    g_nlo_perf_profile_state.trace.reject_attempt = reject_attempt;
}

void nlo_perf_profile_mark_gpu_timestamps_available(int available)
{
    g_nlo_perf_profile_state.snapshot.gpu_timestamps_available = (available != 0) ? 1 : 0;
}

void nlo_perf_profile_set_gpu_timestamp_mode(nlo_perf_gpu_timestamp_mode mode)
{
    if (mode < NLO_PERF_GPU_TIMESTAMPS_AUTO || mode > NLO_PERF_GPU_TIMESTAMPS_OFF) {
        g_nlo_perf_profile_state.gpu_timestamp_mode = NLO_PERF_GPU_TIMESTAMPS_AUTO;
        return;
    }
    g_nlo_perf_profile_state.gpu_timestamp_mode = mode;
}

nlo_perf_gpu_timestamp_mode nlo_perf_profile_get_gpu_timestamp_mode(void)
{
    return g_nlo_perf_profile_state.gpu_timestamp_mode;
}

double nlo_perf_profile_now_ms(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER frequency = {0};
    LARGE_INTEGER now;
    if (frequency.QuadPart == 0) {
        (void)QueryPerformanceFrequency(&frequency);
    }
    (void)QueryPerformanceCounter(&now);
    if (frequency.QuadPart == 0) {
        return 0.0;
    }
    return ((double)now.QuadPart * 1000.0) / (double)frequency.QuadPart;
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0.0;
    }
    return ((double)now.tv_sec * 1000.0) + ((double)now.tv_nsec / 1000000.0);
#endif
}

#else

void nlo_perf_profile_set_enabled(int enabled)
{
    (void)enabled;
}

int nlo_perf_profile_is_enabled(void)
{
    return 0;
}

void nlo_perf_profile_reset(void)
{
}

void nlo_perf_profile_snapshot_read(nlo_perf_profile_snapshot* out_snapshot)
{
    if (out_snapshot != NULL) {
        memset(out_snapshot, 0, sizeof(*out_snapshot));
    }
}

void nlo_perf_profile_add_dispersion_time(double elapsed_ms)
{
    (void)elapsed_ms;
}

void nlo_perf_profile_add_nonlinear_time(double elapsed_ms)
{
    (void)elapsed_ms;
}

void nlo_perf_profile_add_event(nlo_perf_event_id event, double elapsed_ms, uint64_t bytes)
{
    (void)event;
    (void)elapsed_ms;
    (void)bytes;
}

void nlo_perf_profile_add_event_gpu_time(nlo_perf_event_id event, double elapsed_ms)
{
    (void)event;
    (void)elapsed_ms;
}

void nlo_perf_profile_add_gpu_dispatch(uint64_t dispatch_count, uint64_t pass_count, uint64_t pass_bytes)
{
    (void)dispatch_count;
    (void)pass_count;
    (void)pass_bytes;
}

void nlo_perf_profile_add_gpu_device_copy(uint64_t copy_count, uint64_t bytes)
{
    (void)copy_count;
    (void)bytes;
}

void nlo_perf_profile_add_gpu_host_transfer_copy(uint64_t copy_count, uint64_t bytes)
{
    (void)copy_count;
    (void)bytes;
}

void nlo_perf_profile_add_gpu_copy(uint64_t copy_count, uint64_t bytes)
{
    (void)copy_count;
    (void)bytes;
}

void nlo_perf_profile_add_gpu_upload(uint64_t count, uint64_t bytes)
{
    (void)count;
    (void)bytes;
}

void nlo_perf_profile_add_gpu_download(uint64_t count, uint64_t bytes)
{
    (void)count;
    (void)bytes;
}

const char* nlo_perf_profile_event_name(nlo_perf_event_id event)
{
    (void)event;
    return "profiling_disabled";
}

size_t nlo_perf_profile_event_count(void)
{
    return (size_t)NLO_PERF_EVENT_COUNT;
}

void nlo_perf_profile_trace_configure(nlo_perf_profile_trace_row* rows, size_t capacity)
{
    (void)rows;
    (void)capacity;
}

void nlo_perf_profile_trace_set_run_metadata(
    uint64_t run_id,
    uint64_t backend_kind,
    uint64_t jit_mode,
    uint64_t scenario_kind
)
{
    (void)run_id;
    (void)backend_kind;
    (void)jit_mode;
    (void)scenario_kind;
}

void nlo_perf_profile_trace_set_step(uint64_t step_index, uint64_t reject_attempt)
{
    (void)step_index;
    (void)reject_attempt;
}

void nlo_perf_profile_mark_gpu_timestamps_available(int available)
{
    (void)available;
}

void nlo_perf_profile_set_gpu_timestamp_mode(nlo_perf_gpu_timestamp_mode mode)
{
    (void)mode;
}

nlo_perf_gpu_timestamp_mode nlo_perf_profile_get_gpu_timestamp_mode(void)
{
    return NLO_PERF_GPU_TIMESTAMPS_OFF;
}

double nlo_perf_profile_now_ms(void)
{
    return 0.0;
}

#endif
