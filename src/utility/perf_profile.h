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
 * @brief Read a monotonic wall-clock timestamp in milliseconds.
 *
 * @return double Monotonic timestamp in milliseconds.
 */
double nlo_perf_profile_now_ms(void);

#ifdef __cplusplus
}
#endif
