/**
 * @brief State management for nonlinear optics solver
 * @file state.c
 * @author Wenzel Kinsky
 * @date 2026-01-27
 */
#include "core/state.h"
#include "fft/fft.h"
#include <stdint.h>
#include <stdlib.h>

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

#define NLO_WORK_BUFFER_COUNT 9u // Keep in sync with work_buffers list in create_simulation_state.

static int allocate_nlo_complex_buffer(nlo_complex **buffer, size_t num_elements);
static int checked_mul_size_t(size_t a, size_t b, size_t *out);
static size_t query_available_system_memory_bytes(void);
static size_t apply_memory_headroom(size_t available_bytes);
static size_t compute_record_block_size(size_t num_time_samples, size_t requested_records, size_t work_buffer_count);

// MARK: Simulation Config Management

sim_config *create_sim_config(size_t num_dispersion_terms, size_t num_time_samples)
{
    if (num_dispersion_terms > NT_MAX || num_time_samples == 0 || num_time_samples > NT_MAX)
    {
        return NULL;
    }

    sim_config *config = (sim_config *)calloc(1, sizeof(sim_config));
    if (config == NULL)
    {
        return NULL;
    }

    config->dispersion.num_dispersion_terms = num_dispersion_terms;
    config->frequency.frequency_grid = (nlo_complex *)calloc(num_time_samples, sizeof(nlo_complex));
    if (config->frequency.frequency_grid == NULL)
    {
        free(config);
        return NULL;
    }

    return config;
}

void free_sim_config(sim_config *config)
{
    if (config == NULL)
    {
        return;
    }

    free(config->frequency.frequency_grid);
    free(config);
}

// MARK: Simulation State Management

simulation_state *create_simulation_state(const sim_config *config, size_t num_time_samples, size_t num_recorded_samples)
{
    size_t max_recorded_samples = 0;
    size_t field_buffer_elements = 0;

    if (num_time_samples == 0 ||
        num_time_samples > NT_MAX ||
        num_recorded_samples == 0 ||
        num_recorded_samples > NT_MAX ||
        config == NULL)
    {
        return NULL;
    }

    max_recorded_samples = compute_record_block_size(num_time_samples, num_recorded_samples, NLO_WORK_BUFFER_COUNT);
    if (max_recorded_samples == 0)
    {
        return NULL;
    }

    simulation_state *state = (simulation_state *)calloc(1, sizeof(simulation_state));
    if (state == NULL)
    {
        return NULL;
    }

    state->config = config;
    state->num_time_samples = num_time_samples;
    state->num_recorded_samples = max_recorded_samples;
    state->current_record_index = 0u;
    state->current_z = 0.0;
    state->current_step_size = config->propagation.starting_step_size;

    if (checked_mul_size_t(num_time_samples, state->num_recorded_samples, &field_buffer_elements) != 0)
    {
        free_simulation_state(state);
        return NULL;
    }

    if (allocate_nlo_complex_buffer(&state->field_buffer, field_buffer_elements) != 0)
    {
        free_simulation_state(state);
        return NULL;
    }

    state->current_field = state->field_buffer;

    nlo_complex **work_buffers[] = {
        &state->ip_field_buffer,
        &state->field_magnitude_buffer,
        &state->field_working_buffer,
        &state->field_freq_buffer,
        &state->k_1_buffer,
        &state->k_2_buffer,
        &state->k_3_buffer,
        &state->k_4_buffer,
        &state->current_dispersion_factor};

    const size_t num_work_buffers = sizeof(work_buffers) / sizeof(work_buffers[0]);
    for (size_t i = 0; i < num_work_buffers; ++i)
    {
        if (allocate_nlo_complex_buffer(work_buffers[i], num_time_samples) != 0)
        {
            free_simulation_state(state);
            return NULL;
        }
    }

    return state;
}

void free_simulation_state(simulation_state *state)
{
    if (state != NULL)
    {
        free(state->field_buffer);
        free(state->ip_field_buffer);
        free(state->field_magnitude_buffer);
        free(state->field_working_buffer);
        free(state->field_freq_buffer);
        free(state->k_1_buffer);
        free(state->k_2_buffer);
        free(state->k_3_buffer);
        free(state->k_4_buffer);
        free(state->current_dispersion_factor);
        free(state);
    }
}

static int allocate_nlo_complex_buffer(nlo_complex **buffer, size_t num_elements)
{
    *buffer = (nlo_complex *)calloc(num_elements, sizeof(nlo_complex));
    return (*buffer == NULL) ? -1 : 0;
}

static int checked_mul_size_t(size_t a, size_t b, size_t *out)
{
    if (out == NULL)
    {
        return -1;
    }

    if (a == 0 || b == 0)
    {
        *out = 0;
        return 0;
    }

    if (a > SIZE_MAX / b)
    {
        return -1;
    }

    *out = a * b;
    return 0;
}

static size_t query_available_system_memory_bytes(void)
{
#if defined(_WIN32)
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status) == 0)
    {
        return 0;
    }

    if (mem_status.ullAvailPhys > (unsigned long long)SIZE_MAX)
    {
        return SIZE_MAX;
    }

    return (size_t)mem_status.ullAvailPhys;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) != 0)
    {
        return 0;
    }

    unsigned long long bytes = (unsigned long long)info.freeram * (unsigned long long)info.mem_unit;
    if (bytes > (unsigned long long)SIZE_MAX)
    {
        return SIZE_MAX;
    }

    return (size_t)bytes;
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
    {
        return 0;
    }

    unsigned long long bytes = (unsigned long long)pages * (unsigned long long)page_size;
    if (bytes > (unsigned long long)SIZE_MAX)
    {
        return SIZE_MAX;
    }

    return (size_t)bytes;
#else
    return 0;
#endif
}

static size_t apply_memory_headroom(size_t available_bytes)
{
    if (available_bytes == 0)
    {
        return 0;
    }

    return (available_bytes / NLO_MEMORY_HEADROOM_DEN) * NLO_MEMORY_HEADROOM_NUM;
}

static size_t compute_record_block_size(size_t num_time_samples, size_t requested_records, size_t work_buffer_count)
{
    size_t available_bytes = 0;
    size_t per_record_bytes = 0;
    size_t work_bytes = 0;
    size_t max_records = 0;

    if (num_time_samples == 0 || requested_records == 0 || work_buffer_count == 0)
    {
        return 0;
    }

    available_bytes = query_available_system_memory_bytes();
    if (available_bytes == 0)
    {
        return requested_records;
    }

    available_bytes = apply_memory_headroom(available_bytes);
    if (available_bytes == 0)
    {
        return 0;
    }

    if (checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &per_record_bytes) != 0 ||
        per_record_bytes == 0)
    {
        return 0;
    }

    if (checked_mul_size_t(per_record_bytes, work_buffer_count, &work_bytes) != 0)
    {
        return 0;
    }

    if (work_bytes >= available_bytes)
    {
        return 0;
    }

    max_records = (available_bytes - work_bytes) / per_record_bytes;
    if (max_records == 0)
    {
        return 0;
    }

    if (max_records > NT_MAX)
    {
        max_records = NT_MAX;
    }

    return (max_records < requested_records) ? max_records : requested_records;
}
