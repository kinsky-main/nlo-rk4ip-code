/**
 * @file perf_profile.h
 * @brief Lightweight runtime performance counters for benchmarking diagnostics.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef NLO_ENABLE_RUNTIME_PROFILING
#define NLO_ENABLE_RUNTIME_PROFILING 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NLO_PERF_EVENT_BACKEND_CREATE = 0,
    NLO_PERF_EVENT_STATE_ALLOCATE_VECTORS = 1,
    NLO_PERF_EVENT_OPERATOR_COMPILE_LOWER = 2,
    NLO_PERF_EVENT_OPERATOR_JIT_WARMUP = 3,
    NLO_PERF_EVENT_FREQUENCY_GRID_INIT = 4,
    NLO_PERF_EVENT_TENSOR_AXIS_INIT = 5,
    NLO_PERF_EVENT_POTENTIAL_INIT = 6,
    NLO_PERF_EVENT_DISPERSION_FACTOR_INIT = 7,
    NLO_PERF_EVENT_FFT_PLAN_CREATE = 8,
    NLO_PERF_EVENT_INITIAL_FIELD_UPLOAD = 9,
    NLO_PERF_EVENT_BEGIN_SIMULATION = 10,
    NLO_PERF_EVENT_END_SIMULATION = 11,
    NLO_PERF_EVENT_HOST_STEP_ADMIN = 12,
    NLO_PERF_EVENT_HOST_ERROR_CONTROL = 13,
    NLO_PERF_EVENT_HOST_PROGRESS_LOGGING = 14,
    NLO_PERF_EVENT_HOST_SNAPSHOT_BOOKKEEPING = 15,
    NLO_PERF_EVENT_SNAPSHOT_PAUSE_END_SIM = 16,
    NLO_PERF_EVENT_SNAPSHOT_DOWNLOAD = 17,
    NLO_PERF_EVENT_SNAPSHOT_RESUME_BEGIN_SIM = 18,
    NLO_PERF_EVENT_FFT_FORWARD = 19,
    NLO_PERF_EVENT_FFT_INVERSE = 20,
    NLO_PERF_EVENT_DISPERSION_APPLY = 21,
    NLO_PERF_EVENT_NONLINEAR_APPLY = 22,
    NLO_PERF_EVENT_OPERATOR_PROGRAM_JIT_EXECUTE = 23,
    NLO_PERF_EVENT_OPERATOR_PROGRAM_INTERPRETER_EXECUTE = 24,
    NLO_PERF_EVENT_WEIGHTED_RMS_ERROR = 25,
    NLO_PERF_EVENT_VEC_COPY = 26,
    NLO_PERF_EVENT_VEC_ADD = 27,
    NLO_PERF_EVENT_VEC_SCALAR_MUL = 28,
    NLO_PERF_EVENT_VEC_MUL = 29,
    NLO_PERF_EVENT_VEC_MAGNITUDE_SQUARED = 30,
    NLO_PERF_EVENT_VEC_AXPY = 31,
    NLO_PERF_EVENT_VEC_AFFINE_COMB2 = 32,
    NLO_PERF_EVENT_VEC_AFFINE_COMB3 = 33,
    NLO_PERF_EVENT_VEC_AFFINE_COMB4 = 34,
    NLO_PERF_EVENT_VEC_EMBEDDED_ERROR_PAIR = 35,
    NLO_PERF_EVENT_VEC_LERP = 36,
    NLO_PERF_EVENT_VK_COMMAND_BEGIN = 37,
    NLO_PERF_EVENT_VK_COMMAND_SUBMIT_WAIT = 38,
    NLO_PERF_EVENT_VK_SIMULATION_FLUSH = 39,
    NLO_PERF_EVENT_VK_HOST_TO_DEVICE_TRANSFER = 40,
    NLO_PERF_EVENT_VK_DEVICE_TO_HOST_TRANSFER = 41,
    NLO_PERF_EVENT_COUNT = 42
} nlo_perf_event_id;

typedef enum {
    NLO_PERF_GPU_TIMESTAMPS_AUTO = 0,
    NLO_PERF_GPU_TIMESTAMPS_ON = 1,
    NLO_PERF_GPU_TIMESTAMPS_OFF = 2
} nlo_perf_gpu_timestamp_mode;

typedef struct {
    uint64_t run_id;
    uint64_t backend_kind;
    uint64_t jit_mode;
    uint64_t scenario_kind;
    uint64_t step_index;
    uint64_t reject_attempt;
    uint64_t event_id;
    uint64_t call_count;
    uint64_t bytes;
    double host_wall_ms;
    double gpu_exec_ms;
} nlo_perf_profile_trace_row;

/**
 * @brief Snapshot of accumulated performance counters.
 */
typedef struct {
    double dispersion_ms;
    double nonlinear_ms;

    uint64_t dispersion_calls;
    uint64_t nonlinear_calls;

    uint64_t gpu_dispatch_count;
    uint64_t gpu_copy_count;
    uint64_t gpu_device_copy_count;
    uint64_t gpu_device_copy_bytes;
    uint64_t gpu_host_transfer_copy_count;
    uint64_t gpu_host_transfer_copy_bytes;
    uint64_t gpu_memory_pass_count;
    uint64_t gpu_memory_pass_bytes;

    uint64_t gpu_upload_count;
    uint64_t gpu_download_count;
    uint64_t gpu_upload_bytes;
    uint64_t gpu_download_bytes;

    double event_total_ms[NLO_PERF_EVENT_COUNT];
    double event_gpu_exec_ms[NLO_PERF_EVENT_COUNT];
    uint64_t event_call_count[NLO_PERF_EVENT_COUNT];
    uint64_t event_bytes[NLO_PERF_EVENT_COUNT];

    uint64_t trace_row_count;
    uint64_t trace_dropped_count;
    int gpu_timestamps_available;
} nlo_perf_profile_snapshot;

typedef struct {
    double start_ms;
    int active;
} nlo_perf_scope;

/**
 * @brief Enable or disable profiling counter accumulation globally.
 *
 * @param enabled Non-zero enables profiling; zero disables profiling.
 */
void nlo_perf_profile_set_enabled(int enabled);

/**
 * @brief Query whether profiling counters are currently enabled.
 *
 * @return int Non-zero when enabled, zero when disabled.
 */
int nlo_perf_profile_is_enabled(void);

/**
 * @brief Reset all counters and accumulated timings.
 */
void nlo_perf_profile_reset(void);

/**
 * @brief Copy current counters into @p out_snapshot.
 *
 * @param out_snapshot Destination snapshot.
 */
void nlo_perf_profile_snapshot_read(nlo_perf_profile_snapshot* out_snapshot);

/**
 * @brief Record elapsed dispersion-operator time.
 *
 * @param elapsed_ms Wall time in milliseconds.
 */
void nlo_perf_profile_add_dispersion_time(double elapsed_ms);

/**
 * @brief Record elapsed nonlinear-operator time.
 *
 * @param elapsed_ms Wall time in milliseconds.
 */
void nlo_perf_profile_add_nonlinear_time(double elapsed_ms);

/**
 * @brief Record elapsed walltime for one logical profiling event.
 *
 * @param event Event identifier.
 * @param elapsed_ms Wall time in milliseconds.
 * @param bytes Optional byte count associated with the event.
 */
void nlo_perf_profile_add_event(nlo_perf_event_id event, double elapsed_ms, uint64_t bytes);

/**
 * @brief Record device execution time for one logical profiling event.
 *
 * @param event Event identifier.
 * @param elapsed_ms Device execution time in milliseconds.
 */
void nlo_perf_profile_add_event_gpu_time(nlo_perf_event_id event, double elapsed_ms);

/**
 * @brief Record one or more GPU compute dispatches and approximate memory passes.
 *
 * @param dispatch_count Number of compute dispatches.
 * @param pass_count Approximate device-memory pass count.
 * @param pass_bytes Approximate total bytes traversed by those passes.
 */
void nlo_perf_profile_add_gpu_dispatch(
    uint64_t dispatch_count,
    uint64_t pass_count,
    uint64_t pass_bytes
);

/**
 * @brief Record one or more GPU device-to-device buffer-copy operations.
 *
 * This counter excludes host transfer staging copies.
 *
 * @param copy_count Number of GPU copy operations.
 * @param bytes Bytes copied per operation aggregate.
 */
void nlo_perf_profile_add_gpu_device_copy(uint64_t copy_count, uint64_t bytes);

/**
 * @brief Record one or more GPU copies used for host-transfer staging.
 *
 * @param copy_count Number of GPU copy operations.
 * @param bytes Bytes copied per operation aggregate.
 */
void nlo_perf_profile_add_gpu_host_transfer_copy(uint64_t copy_count, uint64_t bytes);

/**
 * @brief Backward-compatible alias for GPU device-to-device copy accounting.
 *
 * @param copy_count Number of GPU copy operations.
 * @param bytes Bytes copied per operation aggregate.
 */
void nlo_perf_profile_add_gpu_copy(uint64_t copy_count, uint64_t bytes);

/**
 * @brief Record host-to-device transfer operation(s).
 *
 * @param count Number of transfer operations.
 * @param bytes Total transfer bytes.
 */
void nlo_perf_profile_add_gpu_upload(uint64_t count, uint64_t bytes);

/**
 * @brief Record device-to-host transfer operation(s).
 *
 * @param count Number of transfer operations.
 * @param bytes Total transfer bytes.
 */
void nlo_perf_profile_add_gpu_download(uint64_t count, uint64_t bytes);

/**
 * @brief Return a stable string name for a profiling event.
 *
 * @param event Event identifier.
 * @return const char* Human-readable event name.
 */
const char* nlo_perf_profile_event_name(nlo_perf_event_id event);

/**
 * @brief Return the number of defined profiling events.
 *
 * @return size_t Event count.
 */
size_t nlo_perf_profile_event_count(void);

/**
 * @brief Configure optional preallocated trace storage.
 *
 * Passing NULL or zero capacity disables trace capture.
 *
 * @param rows Caller-owned trace storage.
 * @param capacity Maximum number of rows.
 */
void nlo_perf_profile_trace_configure(nlo_perf_profile_trace_row* rows, size_t capacity);

/**
 * @brief Set run-level metadata used for trace rows.
 *
 * @param run_id Benchmark run identifier.
 * @param backend_kind Backend identifier chosen by the benchmark.
 * @param jit_mode Internal JIT mode identifier.
 * @param scenario_kind Scenario identifier chosen by the benchmark.
 */
void nlo_perf_profile_trace_set_run_metadata(
    uint64_t run_id,
    uint64_t backend_kind,
    uint64_t jit_mode,
    uint64_t scenario_kind
);

/**
 * @brief Set step-level metadata used for subsequent trace rows.
 *
 * @param step_index Current solver step index, or UINT64_MAX when unset.
 * @param reject_attempt Current reject-attempt index, or UINT64_MAX when unset.
 */
void nlo_perf_profile_trace_set_step(uint64_t step_index, uint64_t reject_attempt);

/**
 * @brief Mark whether device timestamp data is available for the current run.
 *
 * @param available Non-zero when timestamps are available.
 */
void nlo_perf_profile_mark_gpu_timestamps_available(int available);

/**
 * @brief Override internal GPU timestamp-query mode for benchmark runs.
 *
 * @param mode Requested timestamp-query mode.
 */
void nlo_perf_profile_set_gpu_timestamp_mode(nlo_perf_gpu_timestamp_mode mode);

/**
 * @brief Query the current internal GPU timestamp-query mode.
 *
 * @return nlo_perf_gpu_timestamp_mode Current timestamp-query mode.
 */
nlo_perf_gpu_timestamp_mode nlo_perf_profile_get_gpu_timestamp_mode(void);

/**
 * @brief Read a monotonic wall-clock timestamp in milliseconds.
 *
 * @return double Monotonic timestamp in milliseconds.
 */
double nlo_perf_profile_now_ms(void);

#if NLO_ENABLE_RUNTIME_PROFILING
#define NLO_PERF_SCOPE_BEGIN(scope_name)                                                \
    do {                                                                                \
        (scope_name).active = nlo_perf_profile_is_enabled();                            \
        if ((scope_name).active) {                                                      \
            (scope_name).start_ms = nlo_perf_profile_now_ms();                          \
        }                                                                               \
    } while (0)

#define NLO_PERF_SCOPE_END(scope_name, event_id, byte_count)                            \
    do {                                                                                \
        if ((scope_name).active) {                                                      \
            nlo_perf_profile_add_event((event_id),                                      \
                                       nlo_perf_profile_now_ms() - (scope_name).start_ms, \
                                       (byte_count));                                   \
        }                                                                               \
    } while (0)

#define NLO_PERF_ADD_GPU_TIME(event_id, elapsed_ms)                                     \
    do {                                                                                \
        if (nlo_perf_profile_is_enabled()) {                                            \
            nlo_perf_profile_add_event_gpu_time((event_id), (elapsed_ms));              \
        }                                                                               \
    } while (0)

#define NLO_PERF_MARK_GPU_TIMESTAMPS_AVAILABLE(available_value)                         \
    do {                                                                                \
        if (nlo_perf_profile_is_enabled()) {                                            \
            nlo_perf_profile_mark_gpu_timestamps_available((available_value));           \
        }                                                                               \
    } while (0)
#else
#define NLO_PERF_SCOPE_BEGIN(scope_name) do { (void)&(scope_name); } while (0)
#define NLO_PERF_SCOPE_END(scope_name, event_id, byte_count)                            \
    do {                                                                                \
        (void)&(scope_name);                                                            \
        (void)(event_id);                                                               \
        (void)(byte_count);                                                             \
    } while (0)
#define NLO_PERF_ADD_GPU_TIME(event_id, elapsed_ms)                                     \
    do {                                                                                \
        (void)(event_id);                                                               \
        (void)(elapsed_ms);                                                             \
    } while (0)
#define NLO_PERF_MARK_GPU_TIMESTAMPS_AVAILABLE(available_value)                         \
    do {                                                                                \
        (void)(available_value);                                                        \
    } while (0)
#endif

#ifdef __cplusplus
}
#endif
