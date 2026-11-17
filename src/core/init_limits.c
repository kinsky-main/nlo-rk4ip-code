/**
 * @file init_limits.c
 * @brief Runtime limit estimation and shared initialization math helpers.
 */

#include "core/state.h"
#include "core/init_internal.h"
#include <limits.h>

#if defined(_WIN32)
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
#include <unistd.h>
#endif

/* Safety margin on host memory: only 8/10 of what the OS reports as available
 * is budgeted. The remaining fifth absorbs what the estimate cannot see --
 * allocator and driver overhead, the FFT plan, the caller's own copies of the
 * records, and other processes allocating between the query and the run. */
#ifndef MEMORY_HEADROOM_NUM
#define MEMORY_HEADROOM_NUM 8u
#endif

#ifndef MEMORY_HEADROOM_DEN
#define MEMORY_HEADROOM_DEN 10u
#endif

#ifndef DEVICE_RING_BUDGET_HEADROOM_NUM
#define DEVICE_RING_BUDGET_HEADROOM_NUM 9u
#endif

#ifndef DEVICE_RING_BUDGET_HEADROOM_DEN
#define DEVICE_RING_BUDGET_HEADROOM_DEN 10u
#endif

static size_t estimate_active_vector_count(const sim_config* config);

static size_t available_system_memory_override_bytes = 0u;

void set_available_system_memory_override_bytes(size_t bytes)
{
    available_system_memory_override_bytes = bytes;
}

int checked_mul_size_t(size_t a, size_t b, size_t* out)
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

size_t query_available_system_memory_bytes(void)
{
    if (available_system_memory_override_bytes > 0u) {
        return available_system_memory_override_bytes;
    }
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
    // MemAvailable counts reclaimable page cache; sysinfo().freeram does not,
    // and on a machine with a warm cache it under-reports by an order of
    // magnitude.  Fall back to freeram on kernels without MemAvailable.
    unsigned long long bytes = 0u;
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (meminfo != NULL) {
        char line[128];
        unsigned long long available_kib = 0u;
        while (fgets(line, sizeof(line), meminfo) != NULL) {
            if (sscanf(line, "MemAvailable: %llu kB", &available_kib) == 1) {
                bytes = available_kib * 1024ull;
                break;
            }
        }
        fclose(meminfo);
    }
    if (bytes == 0u) {
        struct sysinfo info;
        if (sysinfo(&info) != 0) {
            return 0;
        }
        bytes = (unsigned long long)info.freeram * (unsigned long long)info.mem_unit;
    }
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

size_t apply_memory_headroom(size_t available_bytes)
{
    if (available_bytes == 0u) {
        return 0u;
    }

    return (available_bytes / MEMORY_HEADROOM_DEN) * MEMORY_HEADROOM_NUM;
}

size_t query_resident_bytes(const void* address, size_t length)
{
    if (address == NULL || length == 0u || available_system_memory_override_bytes > 0u) {
        return 0u;
    }
#if defined(__linux__)
    const long page_size_l = sysconf(_SC_PAGESIZE);
    if (page_size_l <= 0) {
        return 0u;
    }
    const size_t page_size = (size_t)page_size_l;
    const uintptr_t begin = (uintptr_t)address & ~((uintptr_t)page_size - 1u);
    if (length > UINTPTR_MAX - (uintptr_t)address) {
        return 0u;
    }
    const uintptr_t end = (uintptr_t)address + length;
    unsigned char page_state[4096];
    size_t resident = 0u;
    uintptr_t cursor = begin;
    while (cursor < end) {
        size_t pages = (size_t)((end - cursor + page_size - 1u) / page_size);
        if (pages > sizeof(page_state)) {
            pages = sizeof(page_state);
        }
        if (mincore((void*)cursor, pages * page_size, page_state) != 0) {
            break;
        }
        for (size_t i = 0u; i < pages; ++i) {
            if ((page_state[i] & 1u) != 0u) {
                resident += page_size;
            }
        }
        cursor += (uintptr_t)(pages * page_size);
    }
    return (resident < length) ? resident : length;
#elif defined(_WIN32)
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    const size_t page_size = (size_t)system_info.dwPageSize;
    if (page_size == 0u || length > UINTPTR_MAX - (uintptr_t)address) {
        return 0u;
    }
    const uintptr_t begin = (uintptr_t)address & ~((uintptr_t)page_size - 1u);
    const uintptr_t end = (uintptr_t)address + length;
    PSAPI_WORKING_SET_EX_INFORMATION page_state[256];
    const size_t page_state_count = sizeof(page_state) / sizeof(page_state[0]);
    const HANDLE process = GetCurrentProcess();
    size_t resident = 0u;
    uintptr_t cursor = begin;
    while (cursor < end) {
        size_t pages = (size_t)((end - cursor + page_size - 1u) / page_size);
        if (pages > page_state_count) {
            pages = page_state_count;
        }
        for (size_t i = 0u; i < pages; ++i) {
            page_state[i].VirtualAddress = (PVOID)(cursor + (uintptr_t)(i * page_size));
        }
        if (QueryWorkingSetEx(process,
                              page_state,
                              (DWORD)(pages * sizeof(page_state[0]))) == 0) {
            break;
        }
        for (size_t i = 0u; i < pages; ++i) {
            if (page_state[i].VirtualAttributes.Valid != 0u) {
                resident += page_size;
            }
        }
        cursor += (uintptr_t)(pages * page_size);
    }
    return (resident < length) ? resident : length;
#else
    return 0u;
#endif
}

static size_t estimate_active_vector_count(const sim_config* config)
{
    size_t active_vec_count = 2u + WORK_VECTOR_COUNT + OPERATOR_PROGRAM_MAX_STACK_SLOTS;
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
            if (active_vec_count <= SIZE_MAX - 6u) {
                active_vec_count += 6u;
            } else {
                return SIZE_MAX;
            }
        }
    }
    return active_vec_count;
}

static size_t resolve_runtime_num_time_samples_from_config(const sim_config* config)
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
        if (checked_mul_size_t(config->tensor.nt, config->tensor.nx, &ntx) != 0 ||
            checked_mul_size_t(ntx, config->tensor.ny, &total) != 0) {
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
        if (checked_mul_size_t(nt, resolved_nx, &ntx) != 0 ||
            checked_mul_size_t(ntx, resolved_ny, &total) != 0) {
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
    if (checked_mul_size_t(nx, ny, &total) != 0) {
        return 0u;
    }
    return total;
}

int resolve_dense_record_limit(
    const sim_config* config,
    size_t num_time_samples,
    size_t resident_credit_bytes,
    dense_record_limit* out_limit
)
{
    if (out_limit == NULL) {
        return -1;
    }
    *out_limit = (dense_record_limit){0};

    const size_t active_vec_count = estimate_active_vector_count(config);
    size_t bytes_per_sample = 0u;
    if (active_vec_count == SIZE_MAX ||
        checked_mul_size_t(active_vec_count, sizeof(nlo_complex), &bytes_per_sample) != 0 ||
        bytes_per_sample == 0u) {
        return -1;
    }
    if (num_time_samples == 0u) {
        return 0;
    }
    if (checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &out_limit->bytes_per_record) != 0 ||
        checked_mul_size_t(num_time_samples, bytes_per_sample, &out_limit->working_set_bytes) != 0) {
        *out_limit = (dense_record_limit){0};
        return -1;
    }

    /* Dense recording keeps every record resident in the caller's buffer, and
     * the solver downloads each record straight into it: the core holds no
     * second host copy, so a record costs exactly bytes_per_record. Beside the
     * records the host must hold the working vectors. On the CPU backend that
     * is literally the working set. On Vulkan the vectors live on the device,
     * but the reserve is kept: it covers the host-visible staging and driver
     * mappings, and it keeps the answer independent of which backend AUTO
     * resolves to, so validating a request never has to create a device. */
    const size_t host_available = query_available_system_memory_bytes();
    if (host_available == 0u) {
        return 0;
    }
    out_limit->derivable = 1;
    out_limit->host_budget_bytes = apply_memory_headroom(host_available);

    size_t record_budget =
        (out_limit->host_budget_bytes > out_limit->working_set_bytes)
            ? (out_limit->host_budget_bytes - out_limit->working_set_bytes)
            : 0u;
    /* Buffers the caller has already touched are no longer counted as
     * available, so without this credit a count the query reported before
     * they were allocated would be refused once they exist. */
    record_budget = (record_budget > SIZE_MAX - resident_credit_bytes)
                        ? SIZE_MAX
                        : (record_budget + resident_credit_bytes);
    out_limit->record_budget_bytes = record_budget;

    size_t max_records = record_budget / out_limit->bytes_per_record;
    const size_t precision_ceiling = runtime_limits_default().max_num_recorded_samples_with_storage;
    if (max_records > precision_ceiling) {
        max_records = precision_ceiling;
    }
    out_limit->max_records = max_records;
    return 0;
}

int query_runtime_limits_internal(
    const sim_config* config,
    const execution_options* exec_options,
    runtime_limits* out_limits
)
{
    if (out_limits == NULL) {
        return -1;
    }

    *out_limits = runtime_limits_default();
    const execution_options options =
        (exec_options != NULL)
            ? *exec_options
            : execution_options_default(VECTOR_BACKEND_AUTO);

    const size_t active_vec_count = estimate_active_vector_count(config);
    const size_t requested_num_time_samples = resolve_runtime_num_time_samples_from_config(config);
    const size_t max_samples_by_element_size = SIZE_MAX / sizeof(nlo_complex);
    size_t bytes_per_sample = 0u;
    if (active_vec_count == SIZE_MAX ||
        checked_mul_size_t(active_vec_count, sizeof(nlo_complex), &bytes_per_sample) != 0 ||
        bytes_per_sample == 0u) {
        return -1;
    }

    if (requested_num_time_samples > 0u) {
        (void)checked_mul_size_t(requested_num_time_samples,
                                 bytes_per_sample,
                                 &out_limits->estimated_required_working_set_bytes);
    }

    size_t runtime_limit = max_samples_by_element_size;
    const size_t host_available = apply_memory_headroom(query_available_system_memory_bytes());

    /* The in-memory record capacity comes from the same derivation propagate
     * validates a dense request against. Zero means no record fits, or that it
     * is not derivable -- the config does not state its sample count, or the
     * platform does not report available memory -- in which case propagate
     * enforces no dense limit either. */
    dense_record_limit dense_limit;
    (void)resolve_dense_record_limit(config, requested_num_time_samples, 0u, &dense_limit);
    out_limits->max_num_recorded_samples_in_memory = dense_limit.max_records;
    if (host_available > 0u) {
        const size_t host_limit = host_available / bytes_per_sample;
        if (host_limit > 0u && host_limit < runtime_limit) {
            runtime_limit = host_limit;
        }
    }

    vector_backend* backend = NULL;
    if (options.backend_type == VECTOR_BACKEND_CPU) {
        backend = vector_backend_create_cpu();
    } else if (options.backend_type == VECTOR_BACKEND_VULKAN) {
        backend = vector_backend_create_vulkan(&options.vulkan);
    } else {
        backend = vector_backend_create_vulkan(NULL);
        if (backend == NULL) {
            backend = vector_backend_create_cpu();
        }
    }

    if (backend != NULL) {
        vec_backend_memory_info mem_info = {0};
        if (vec_query_memory_info(backend, &mem_info) == VEC_STATUS_OK &&
            vector_backend_get_type(backend) == VECTOR_BACKEND_VULKAN) {
            const double frac = (options.device_heap_fraction > 0.0 &&
                                 options.device_heap_fraction <= 1.0)
                                    ? options.device_heap_fraction
                                    : DEFAULT_DEVICE_HEAP_FRACTION;
            size_t budget_bytes = options.forced_device_budget_bytes;
            if (budget_bytes == 0u) {
                budget_bytes = (size_t)((double)mem_info.device_local_available_bytes * frac);
            }
            budget_bytes =
                (budget_bytes / DEVICE_RING_BUDGET_HEADROOM_DEN) *
                DEVICE_RING_BUDGET_HEADROOM_NUM;
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
        } else if (vector_backend_get_type(backend) == VECTOR_BACKEND_CPU &&
                   options.fft_backend == FFT_BACKEND_FFTW &&
                   runtime_limit > (size_t)INT_MAX) {
            runtime_limit = (size_t)INT_MAX;
        }
        vector_backend_destroy(backend);
    }

    if (runtime_limit > max_samples_by_element_size) {
        runtime_limit = max_samples_by_element_size;
    }

    out_limits->max_num_time_samples_runtime = runtime_limit;
    return 0;
}

