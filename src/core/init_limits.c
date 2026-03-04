/**
 * @file init_limits.c
 * @brief Runtime limit estimation and shared initialization math helpers.
 */

#include "core/state.h"
#include "core/init_internal.h"
#include <limits.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
#include <unistd.h>
#endif

#ifndef NLO_MEMORY_HEADROOM_NUM
#define NLO_MEMORY_HEADROOM_NUM 8u
#endif

#ifndef NLO_MEMORY_HEADROOM_DEN
#define NLO_MEMORY_HEADROOM_DEN 10u
#endif

#ifndef NLO_DEVICE_RING_BUDGET_HEADROOM_NUM
#define NLO_DEVICE_RING_BUDGET_HEADROOM_NUM 9u
#endif

#ifndef NLO_DEVICE_RING_BUDGET_HEADROOM_DEN
#define NLO_DEVICE_RING_BUDGET_HEADROOM_DEN 10u
#endif
int nlo_checked_mul_size_t(size_t a, size_t b, size_t* out)
{
    if (out == NULL) {
        return -1;
    }

    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }

    if (a > SIZE_MAX / b) {
        return -1;
    }

    *out = a * b;
    return 0;
}

size_t nlo_query_available_system_memory_bytes(void)
{
#if defined(_WIN32)
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status) == 0) {
        return 0;
    }

    if (mem_status.ullAvailPhys > (unsigned long long)SIZE_MAX) {
        return SIZE_MAX;
    }

    return (size_t)mem_status.ullAvailPhys;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        return 0;
    }

    unsigned long long bytes = (unsigned long long)info.freeram * (unsigned long long)info.mem_unit;
    if (bytes > (unsigned long long)SIZE_MAX) {
        return SIZE_MAX;
    }

    return (size_t)bytes;
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }

    unsigned long long bytes = (unsigned long long)pages * (unsigned long long)page_size;
    if (bytes > (unsigned long long)SIZE_MAX) {
        return SIZE_MAX;
    }

    return (size_t)bytes;
#else
    return 0;
#endif
}

size_t nlo_apply_memory_headroom(size_t available_bytes)
{
    if (available_bytes == 0u) {
        return 0u;
    }

    return (available_bytes / NLO_MEMORY_HEADROOM_DEN) * NLO_MEMORY_HEADROOM_NUM;
}

size_t nlo_compute_host_record_capacity(size_t num_time_samples, size_t requested_records)
{
    size_t available_bytes = nlo_query_available_system_memory_bytes();
    if (available_bytes == 0u) {
        return requested_records;
    }

    available_bytes = nlo_apply_memory_headroom(available_bytes);
    if (available_bytes == 0u) {
        return 0u;
    }

    size_t per_record_bytes = 0u;
    if (nlo_checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &per_record_bytes) != 0 || per_record_bytes == 0u) {
        return 0u;
    }

    size_t working_bytes = 0u;
    if (nlo_checked_mul_size_t(per_record_bytes, NLO_WORK_VECTOR_COUNT, &working_bytes) != 0) {
        return 0u;
    }

    if (working_bytes >= available_bytes) {
        return 0u;
    }

    size_t max_records = (available_bytes - working_bytes) / per_record_bytes;
    if (max_records == 0u) {
        return 0u;
    }

    return (max_records < requested_records) ? max_records : requested_records;
}

static size_t nlo_estimate_active_vector_count(const sim_config* config)
{
    size_t active_vec_count = 2u + NLO_WORK_VECTOR_COUNT + NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS;
    if (config != NULL) {
        const size_t nt = (config->tensor.nt > 0u) ? config->tensor.nt : config->time.nt;
        const size_t nx = (config->tensor.nx > 0u) ? config->tensor.nx : config->spatial.nx;
        const size_t ny = (config->tensor.ny > 0u) ? config->tensor.ny : config->spatial.ny;
        const int tensor_mode_active = (config->tensor.nt > 0u) ? 1 : 0;
        const int explicit_nd = (nt > 0u) ? 1 : 0;
        const int enable_transverse = (explicit_nd != 0) && (nx > 1u || ny > 1u);
        if (enable_transverse && !tensor_mode_active) {
            if (active_vec_count <= SIZE_MAX - 3u) {
                active_vec_count += 3u;
            } else {
                return SIZE_MAX;
            }
        }
        if (config->tensor.nt > 0u) {
            if (active_vec_count <= SIZE_MAX - 12u) {
                active_vec_count += 12u;
            } else {
                return SIZE_MAX;
            }
        }
    }
    return active_vec_count;
}

static size_t nlo_resolve_runtime_num_time_samples_from_config(const sim_config* config)
{
    if (config == NULL) {
        return 0u;
    }

    if (config->tensor.nt > 0u) {
        size_t ntx = 0u;
        size_t total = 0u;
        if (config->tensor.nx == 0u || config->tensor.ny == 0u) {
            return 0u;
        }
        if (nlo_checked_mul_size_t(config->tensor.nt, config->tensor.nx, &ntx) != 0 ||
            nlo_checked_mul_size_t(ntx, config->tensor.ny, &total) != 0) {
            return 0u;
        }
        return total;
    }

    const size_t nt = config->time.nt;
    const size_t nx = config->spatial.nx;
    const size_t ny = config->spatial.ny;

    if (nt > 0u) {
        const size_t resolved_nx = (nx > 0u) ? nx : 1u;
        const size_t resolved_ny = (ny > 0u) ? ny : 1u;
        size_t ntx = 0u;
        size_t total = 0u;
        if (nlo_checked_mul_size_t(nt, resolved_nx, &ntx) != 0 ||
            nlo_checked_mul_size_t(ntx, resolved_ny, &total) != 0) {
            return 0u;
        }
        return total;
    }

    if (nx == 0u && ny == 0u) {
        return 0u;
    }
    if (nx == 0u || ny == 0u) {
        return 0u;
    }

    size_t total = 0u;
    if (nlo_checked_mul_size_t(nx, ny, &total) != 0) {
        return 0u;
    }
    return total;
}

int nlo_query_runtime_limits_internal(
    const sim_config* config,
    const nlo_execution_options* exec_options,
    nlo_runtime_limits* out_limits
)
{
    if (out_limits == NULL) {
        return -1;
    }

    *out_limits = nlo_runtime_limits_default();
    const nlo_execution_options options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_AUTO);

    const size_t active_vec_count = nlo_estimate_active_vector_count(config);
    const size_t requested_num_time_samples = nlo_resolve_runtime_num_time_samples_from_config(config);
    const size_t max_samples_by_element_size = SIZE_MAX / sizeof(nlo_complex);
    size_t bytes_per_sample = 0u;
    if (active_vec_count == SIZE_MAX ||
        nlo_checked_mul_size_t(active_vec_count, sizeof(nlo_complex), &bytes_per_sample) != 0 ||
        bytes_per_sample == 0u) {
        return -1;
    }

    if (requested_num_time_samples > 0u) {
        (void)nlo_checked_mul_size_t(requested_num_time_samples,
                                 bytes_per_sample,
                                 &out_limits->estimated_required_working_set_bytes);
        out_limits->max_num_recorded_samples_in_memory =
            nlo_compute_host_record_capacity(requested_num_time_samples, SIZE_MAX);
    }

    size_t runtime_limit = max_samples_by_element_size;
    const size_t host_available = nlo_apply_memory_headroom(nlo_query_available_system_memory_bytes());
    if (host_available > 0u) {
        const size_t host_limit = host_available / bytes_per_sample;
        if (host_limit > 0u && host_limit < runtime_limit) {
            runtime_limit = host_limit;
        }
    }

    nlo_vector_backend* backend = NULL;
    if (options.backend_type == NLO_VECTOR_BACKEND_CPU) {
        backend = nlo_vector_backend_create_cpu();
    } else if (options.backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        backend = nlo_vector_backend_create_vulkan(&options.vulkan);
    } else {
        backend = nlo_vector_backend_create_vulkan(NULL);
        if (backend == NULL) {
            backend = nlo_vector_backend_create_cpu();
        }
    }

    if (backend != NULL) {
        nlo_vec_backend_memory_info mem_info = {0};
        if (nlo_vec_query_memory_info(backend, &mem_info) == NLO_VEC_STATUS_OK &&
            nlo_vector_backend_get_type(backend) == NLO_VECTOR_BACKEND_VULKAN) {
            const double frac = (options.device_heap_fraction > 0.0 &&
                                 options.device_heap_fraction <= 1.0)
                                    ? options.device_heap_fraction
                                    : NLO_DEFAULT_DEVICE_HEAP_FRACTION;
            size_t budget_bytes = options.forced_device_budget_bytes;
            if (budget_bytes == 0u) {
                budget_bytes = (size_t)((double)mem_info.device_local_available_bytes * frac);
            }
            budget_bytes =
                (budget_bytes / NLO_DEVICE_RING_BUDGET_HEADROOM_DEN) *
                NLO_DEVICE_RING_BUDGET_HEADROOM_NUM;
            out_limits->estimated_device_budget_bytes = budget_bytes;

            if (budget_bytes > 0u) {
                const size_t device_budget_limit = budget_bytes / bytes_per_sample;
                if (device_budget_limit > 0u && device_budget_limit < runtime_limit) {
                    runtime_limit = device_budget_limit;
                }
            }
            if (mem_info.max_storage_buffer_range > 0u) {
                const size_t max_buffer_samples = mem_info.max_storage_buffer_range / sizeof(nlo_complex);
                if (max_buffer_samples > 0u && max_buffer_samples < runtime_limit) {
                    runtime_limit = max_buffer_samples;
                }
            }
        } else if (nlo_vector_backend_get_type(backend) == NLO_VECTOR_BACKEND_CPU &&
                   options.fft_backend == NLO_FFT_BACKEND_FFTW &&
                   runtime_limit > (size_t)INT_MAX) {
            runtime_limit = (size_t)INT_MAX;
        }
        nlo_vector_backend_destroy(backend);
    }

    if (runtime_limit > max_samples_by_element_size) {
        runtime_limit = max_samples_by_element_size;
    }

    out_limits->max_num_time_samples_runtime = runtime_limit;
    return 0;
}

