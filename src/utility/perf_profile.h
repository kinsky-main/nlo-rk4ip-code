/**
 * @file perf_profile.h
 * @brief Lightweight runtime performance counters for benchmarking diagnostics.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Snapshot of accumulated performance counters.
 */
#ifndef NLO_PERF_PROFILE_SNAPSHOT_DEFINED
#define NLO_PERF_PROFILE_SNAPSHOT_DEFINED 1
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
} nlo_perf_profile_snapshot;
#endif

/**
 * @brief Enable or disable profiling counter accumulation globally.
 *
 * @param enabled Non-zero enables profiling; zero disables profiling.
 */
void perf_profile_set_enabled(int enabled);

/**
 * @brief Query whether profiling counters are currently enabled.
 *
 * @return int Non-zero when enabled, zero when disabled.
 */
int perf_profile_is_enabled(void);

/**
 * @brief Reset all counters and accumulated timings.
 */
void perf_profile_reset(void);

/**
 * @brief Copy current counters into @p out_snapshot.
 *
 * @param out_snapshot Destination snapshot.
 */
void perf_profile_snapshot_read(nlo_perf_profile_snapshot* out_snapshot);

/**
 * @brief Record elapsed dispersion-operator time.
 *
 * @param elapsed_ms Wall time in milliseconds.
 */
void perf_profile_add_dispersion_time(double elapsed_ms);

/**
 * @brief Record elapsed nonlinear-operator time.
 *
 * @param elapsed_ms Wall time in milliseconds.
 */
void perf_profile_add_nonlinear_time(double elapsed_ms);

/**
 * @brief Record one or more GPU compute dispatches and approximate memory passes.
 *
 * @param dispatch_count Number of compute dispatches.
 * @param pass_count Approximate device-memory pass count.
 * @param pass_bytes Approximate total bytes traversed by those passes.
 */
void perf_profile_add_gpu_dispatch(
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
void perf_profile_add_gpu_device_copy(uint64_t copy_count, uint64_t bytes);

/**
 * @brief Record one or more GPU copies used for host-transfer staging.
 *
 * @param copy_count Number of GPU copy operations.
 * @param bytes Bytes copied per operation aggregate.
 */
void perf_profile_add_gpu_host_transfer_copy(uint64_t copy_count, uint64_t bytes);

/**
 * @brief Backward-compatible alias for GPU device-to-device copy accounting.
 *
 * @param copy_count Number of GPU copy operations.
 * @param bytes Bytes copied per operation aggregate.
 */
void perf_profile_add_gpu_copy(uint64_t copy_count, uint64_t bytes);

/**
 * @brief Record host-to-device transfer operation(s).
 *
 * @param count Number of transfer operations.
 * @param bytes Total transfer bytes.
 */
void perf_profile_add_gpu_upload(uint64_t count, uint64_t bytes);

/**
 * @brief Record device-to-host transfer operation(s).
 *
 * @param count Number of transfer operations.
 * @param bytes Total transfer bytes.
 */
void perf_profile_add_gpu_download(uint64_t count, uint64_t bytes);

/**
 * @brief GPU work categories used by assessment timestamp instrumentation.
 */
typedef enum {
    PERF_GPU_PHASE_OTHER = 0,
    PERF_GPU_PHASE_FFT = 1,
    PERF_GPU_PHASE_LINEAR = 2,
    PERF_GPU_PHASE_NONLINEAR = 3,
    PERF_GPU_PHASE_RK_ALGEBRA = 4,
    PERF_GPU_PHASE_ERROR_NORM = 5,
    PERF_GPU_PHASE_COUNT = 6
} perf_gpu_phase;

/**
 * @brief Host-side slots recorded by assessment instrumentation.
 */
typedef enum {
    PERF_HOST_SLOT_INIT_MS = 0,
    PERF_HOST_SLOT_SOLVE_MS = 1,
    PERF_HOST_SLOT_SUBMITS = 2,
    PERF_HOST_SLOT_TIMESTAMP_DROPS = 3,
    PERF_HOST_SLOT_COUNT = 4
} perf_host_slot;

/**
 * @brief Set the GPU phase that subsequently recorded GPU work is attributed to.
 *
 * @param phase One of ::perf_gpu_phase.
 * @return int Previous phase, for restoring.
 */
int perf_profile_set_gpu_phase(int phase);

/**
 * @brief Return the current GPU attribution phase.
 *
 * @return int One of ::perf_gpu_phase.
 */
int perf_profile_get_gpu_phase(void);

/**
 * @brief Accumulate measured GPU execution time for one phase.
 *
 * @param phase One of ::perf_gpu_phase.
 * @param elapsed_ms GPU time in milliseconds.
 */
void perf_profile_add_gpu_phase_time(int phase, double elapsed_ms);

/**
 * @brief Accumulate a host-side slot.
 *
 * @param slot One of ::perf_host_slot.
 * @param value Value to add (milliseconds or a count).
 */
void perf_profile_add_host_slot(int slot, double value);

/**
 * @brief Copy phase GPU times, interval counts and host slots.
 *
 * @param out_phase_ms Array of ::PERF_GPU_PHASE_COUNT doubles, or NULL.
 * @param out_phase_intervals Array of ::PERF_GPU_PHASE_COUNT counts, or NULL.
 * @param out_host_slots Array of ::PERF_HOST_SLOT_COUNT doubles, or NULL.
 */
void perf_profile_gpu_phases_read(
    double* out_phase_ms,
    uint64_t* out_phase_intervals,
    double* out_host_slots
);

/**
 * @brief Read a monotonic wall-clock timestamp in milliseconds.
 *
 * @return double Monotonic timestamp in milliseconds.
 */
double perf_profile_now_ms(void);

#ifdef __cplusplus
}
#endif
