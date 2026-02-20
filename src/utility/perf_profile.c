/**
 * @file perf_profile.c
 * @brief Lightweight runtime performance counters for benchmarking diagnostics.
 */

#include "utility/perf_profile.h"

#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct {
    int enabled;
    nlo_perf_profile_snapshot snapshot;
} nlo_perf_profile_state;

static nlo_perf_profile_state g_nlo_perf_profile_state = {
    0,
    {0}
};

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
    if (!g_nlo_perf_profile_state.enabled || elapsed_ms < 0.0) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.dispersion_ms += elapsed_ms;
    g_nlo_perf_profile_state.snapshot.dispersion_calls += 1u;
}

void nlo_perf_profile_add_nonlinear_time(double elapsed_ms)
{
    if (!g_nlo_perf_profile_state.enabled || elapsed_ms < 0.0) {
        return;
    }

    g_nlo_perf_profile_state.snapshot.nonlinear_ms += elapsed_ms;
    g_nlo_perf_profile_state.snapshot.nonlinear_calls += 1u;
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
