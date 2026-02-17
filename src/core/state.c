/**
 * @file state.c
 * @brief State management for backend-resident simulation buffers.
 */

#include "core/state.h"
#include "fft/fft.h"
#include "physics/operators.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

#ifndef NLO_MIN_DEVICE_RING_CAPACITY
#define NLO_MIN_DEVICE_RING_CAPACITY 1u
#endif

static int checked_mul_size_t(size_t a, size_t b, size_t* out);
static size_t query_available_system_memory_bytes(void);
static size_t apply_memory_headroom(size_t available_bytes);
static size_t compute_host_record_capacity(size_t num_time_samples, size_t requested_records);

static void nlo_destroy_vec_if_set(nlo_vector_backend* backend, nlo_vec_buffer** vec)
{
    if (backend == NULL || vec == NULL || *vec == NULL) {
        return;
    }

    nlo_vec_destroy(backend, *vec);
    *vec = NULL;
}

static nlo_vec_status nlo_create_complex_vec(nlo_vector_backend* backend, size_t length, nlo_vec_buffer** out_vec)
{
    return nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, length, out_vec);
}

static size_t nlo_compute_device_ring_capacity(const simulation_state* state, size_t requested_records)
{
    if (state == NULL || state->backend == NULL || requested_records == 0u) {
        return 0u;
    }

    if (nlo_vector_backend_get_type(state->backend) != NLO_VECTOR_BACKEND_VULKAN) {
        return 0u;
    }

    const double frac = (state->exec_options.device_heap_fraction > 0.0 &&
                         state->exec_options.device_heap_fraction <= 1.0)
                            ? state->exec_options.device_heap_fraction
                            : NLO_DEFAULT_DEVICE_HEAP_FRACTION;

    nlo_vec_backend_memory_info mem_info = {0};
    if (nlo_vec_query_memory_info(state->backend, &mem_info) != NLO_VEC_STATUS_OK) {
        return NLO_MIN_DEVICE_RING_CAPACITY;
    }

    const size_t per_record_bytes = state->num_time_samples * sizeof(nlo_complex);
    if (per_record_bytes == 0u) {
        return 0u;
    }

    size_t budget_bytes = state->exec_options.forced_device_budget_bytes;
    if (budget_bytes == 0u) {
        budget_bytes = (size_t)((double)mem_info.device_local_available_bytes * frac);
    }

    const size_t active_vec_count = 2u + NLO_WORK_VECTOR_COUNT;
    size_t active_bytes = 0u;
    if (checked_mul_size_t(active_vec_count, per_record_bytes, &active_bytes) != 0) {
        return NLO_MIN_DEVICE_RING_CAPACITY;
    }

    if (budget_bytes <= active_bytes + per_record_bytes) {
        return NLO_MIN_DEVICE_RING_CAPACITY;
    }

    size_t ring_capacity = (budget_bytes - active_bytes) / per_record_bytes;
    if (ring_capacity < NLO_MIN_DEVICE_RING_CAPACITY) {
        ring_capacity = NLO_MIN_DEVICE_RING_CAPACITY;
    }

    if (state->exec_options.record_ring_target > 0u &&
        ring_capacity > state->exec_options.record_ring_target) {
        ring_capacity = state->exec_options.record_ring_target;
    }

    if (ring_capacity > requested_records) {
        ring_capacity = requested_records;
    }

    return ring_capacity;
}

nlo_execution_options nlo_execution_options_default(nlo_vector_backend_type backend_type)
{
    nlo_execution_options options;
    memset(&options, 0, sizeof(options));
    options.backend_type = backend_type;
    options.fft_backend = NLO_FFT_BACKEND_AUTO;
    options.device_heap_fraction = NLO_DEFAULT_DEVICE_HEAP_FRACTION;
    options.record_ring_target = 0u;
    options.forced_device_budget_bytes = 0u;
    return options;
}

sim_config* create_sim_config(size_t num_dispersion_terms, size_t num_time_samples)
{
    if (num_dispersion_terms > NT_MAX || num_time_samples == 0 || num_time_samples > NT_MAX) {
        return NULL;
    }

    sim_config* config = (sim_config*)calloc(1, sizeof(sim_config));
    if (config == NULL) {
        return NULL;
    }

    config->propagation.error_tolerance = 1e-6;
    config->dispersion.num_dispersion_terms = num_dispersion_terms;
    config->frequency.frequency_grid = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    if (config->frequency.frequency_grid == NULL) {
        free(config);
        return NULL;
    }

    return config;
}

void free_sim_config(sim_config* config)
{
    if (config == NULL) {
        return;
    }

    free(config->frequency.frequency_grid);
    free(config);
}

simulation_state* create_simulation_state(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options
)
{
    if (config == NULL || exec_options == NULL ||
        num_time_samples == 0 || num_time_samples > NT_MAX ||
        num_recorded_samples == 0 || num_recorded_samples > NT_MAX) {
        return NULL;
    }

    simulation_state* state = (simulation_state*)calloc(1, sizeof(simulation_state));
    if (state == NULL) {
        return NULL;
    }

    state->config = config;
    state->exec_options = *exec_options;
    state->num_time_samples = num_time_samples;
    state->num_recorded_samples = compute_host_record_capacity(num_time_samples, num_recorded_samples);
    if (state->num_recorded_samples == 0u) {
        free_simulation_state(state);
        return NULL;
    }

    state->current_z = 0.0;
    state->current_step_size = config->propagation.starting_step_size;
    state->current_half_step_exp = 0.5 * state->current_step_size;
    state->last_dispersion_step_size = 0.0;
    state->dispersion_valid = 0;

    size_t host_elements = 0u;
    if (checked_mul_size_t(state->num_time_samples, state->num_recorded_samples, &host_elements) != 0) {
        free_simulation_state(state);
        return NULL;
    }

    state->field_buffer = (nlo_complex*)calloc(host_elements, sizeof(nlo_complex));
    if (state->field_buffer == NULL) {
        free_simulation_state(state);
        return NULL;
    }

    if (exec_options->backend_type == NLO_VECTOR_BACKEND_CPU) {
        state->backend = nlo_vector_backend_create_cpu();
    }
    else if (exec_options->backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        state->backend = nlo_vector_backend_create_vulkan(&exec_options->vulkan);
    }
    else if (exec_options->backend_type == NLO_VECTOR_BACKEND_AUTO) {
        state->backend = nlo_vector_backend_create_vulkan(NULL);
    }
    else {
        state->backend = NULL;
    }

    if (state->backend == NULL) {
        free_simulation_state(state);
        return NULL;
    }

    if (nlo_create_complex_vec(state->backend, num_time_samples, &state->current_field_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->frequency_grid_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.ip_field_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.field_magnitude_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.field_working_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.field_freq_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.omega_power_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.k_1_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.k_2_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.k_3_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.k_4_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.dispersion_factor_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.previous_field_vec) != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    nlo_vec_status status = nlo_vec_complex_fill(state->backend, state->current_field_vec, nlo_make(0.0, 0.0));
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    if (config->frequency.frequency_grid != NULL) {
        status = nlo_vec_upload(state->backend,
                                state->frequency_grid_vec,
                                config->frequency.frequency_grid,
                                num_time_samples * sizeof(nlo_complex));
    } else {
        status = nlo_vec_complex_fill(state->backend, state->frequency_grid_vec, nlo_make(0.0, 0.0));
    }
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    status = nlo_calculate_dispersion_factor_vec(state->backend,
                                                 config->dispersion.num_dispersion_terms,
                                                 config->dispersion.betas,
                                                 state->current_step_size,
                                                 state->working_vectors.dispersion_factor_vec,
                                                 state->frequency_grid_vec,
                                                 state->working_vectors.omega_power_vec,
                                                 state->working_vectors.field_working_vec);
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    state->last_dispersion_step_size = state->current_step_size;
    state->dispersion_valid = 1;

    if (nlo_fft_plan_create_with_backend(state->backend,
                                         num_time_samples,
                                         state->exec_options.fft_backend,
                                         &state->fft_plan) != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    state->record_ring_capacity = nlo_compute_device_ring_capacity(state, state->num_recorded_samples);
    if (state->record_ring_capacity > 0u) {
        state->record_ring_vec = (nlo_vec_buffer**)calloc(state->record_ring_capacity, sizeof(nlo_vec_buffer*));
        if (state->record_ring_vec == NULL) {
            free_simulation_state(state);
            return NULL;
        }

        for (size_t i = 0; i < state->record_ring_capacity; ++i) {
            if (nlo_create_complex_vec(state->backend, num_time_samples, &state->record_ring_vec[i]) != NLO_VEC_STATUS_OK) {
                free_simulation_state(state);
                return NULL;
            }
        }
    }

    return state;
}

void free_simulation_state(simulation_state* state)
{
    if (state == NULL) {
        return;
    }

    if (state->fft_plan != NULL) {
        nlo_fft_plan_destroy(state->fft_plan);
        state->fft_plan = NULL;
    }

    if (state->backend != NULL) {
        nlo_destroy_vec_if_set(state->backend, &state->current_field_vec);
        nlo_destroy_vec_if_set(state->backend, &state->frequency_grid_vec);

        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.ip_field_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.field_magnitude_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.field_working_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.field_freq_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.omega_power_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.k_1_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.k_2_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.k_3_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.k_4_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.dispersion_factor_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.previous_field_vec);

        if (state->record_ring_vec != NULL) {
            for (size_t i = 0; i < state->record_ring_capacity; ++i) {
                nlo_destroy_vec_if_set(state->backend, &state->record_ring_vec[i]);
            }
        }

        nlo_vector_backend_destroy(state->backend);
        state->backend = NULL;
    }

    free(state->record_ring_vec);
    free(state->field_buffer);
    free(state);
}

nlo_vec_status simulation_state_upload_initial_field(simulation_state* state, const nlo_complex* field)
{
    if (state == NULL || field == NULL || state->backend == NULL || state->current_field_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t bytes = state->num_time_samples * sizeof(nlo_complex);
    nlo_vec_status status = nlo_vec_upload(state->backend, state->current_field_vec, field, bytes);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    nlo_complex* record0 = simulation_state_get_field_record(state, 0u);
    if (record0 != NULL) {
        memcpy(record0, field, bytes);
    }

    state->current_record_index = 1u;
    if (state->current_record_index > state->num_recorded_samples) {
        state->current_record_index = state->num_recorded_samples;
    }
    state->record_ring_flushed_count = (state->num_recorded_samples > 0u) ? 1u : 0u;

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status simulation_state_download_current_field(const simulation_state* state, nlo_complex* out_field)
{
    if (state == NULL || out_field == NULL || state->backend == NULL || state->current_field_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    return nlo_vec_download(state->backend,
                            state->current_field_vec,
                            out_field,
                            state->num_time_samples * sizeof(nlo_complex));
}

static nlo_vec_status simulation_state_flush_oldest_ring_entry(simulation_state* state)
{
    if (state == NULL || state->record_ring_size == 0u || state->record_ring_capacity == 0u) {
        return NLO_VEC_STATUS_OK;
    }

    if (state->record_ring_flushed_count >= state->num_recorded_samples) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_complex* host_record = simulation_state_get_field_record(state, state->record_ring_flushed_count);
    if (host_record == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const size_t slot = state->record_ring_head;
    nlo_vec_buffer* src = state->record_ring_vec[slot];
    if (src == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const bool resume_simulation = nlo_vec_is_in_simulation(state->backend);
    if (resume_simulation) {
        nlo_vec_status status = nlo_vec_end_simulation(state->backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    nlo_vec_status status = nlo_vec_download(state->backend,
                                             src,
                                             host_record,
                                             state->num_time_samples * sizeof(nlo_complex));

    if (resume_simulation) {
        nlo_vec_status begin_status = nlo_vec_begin_simulation(state->backend);
        if (status == NLO_VEC_STATUS_OK) {
            status = begin_status;
        }
    }

    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    state->record_ring_head = (state->record_ring_head + 1u) % state->record_ring_capacity;
    state->record_ring_size -= 1u;
    state->record_ring_flushed_count += 1u;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status simulation_state_capture_snapshot(simulation_state* state)
{
    if (state == NULL || state->backend == NULL || state->current_field_vec == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (state->current_record_index >= state->num_recorded_samples) {
        return NLO_VEC_STATUS_OK;
    }

    if (nlo_vector_backend_get_type(state->backend) == NLO_VECTOR_BACKEND_CPU) {
        const void* host_src = NULL;
        nlo_vec_status status = nlo_vec_get_const_host_ptr(state->backend,
                                                           state->current_field_vec,
                                                           &host_src);
        if (status != NLO_VEC_STATUS_OK || host_src == NULL) {
            return (status == NLO_VEC_STATUS_OK) ? NLO_VEC_STATUS_BACKEND_UNAVAILABLE : status;
        }

        nlo_complex* host_record = simulation_state_get_field_record(state, state->current_record_index);
        if (host_record == NULL) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        memcpy(host_record, host_src, state->num_time_samples * sizeof(nlo_complex));
        state->current_record_index += 1u;
        return NLO_VEC_STATUS_OK;
    }

    if (state->record_ring_capacity == 0u) {
        nlo_complex* host_record = simulation_state_get_field_record(state, state->current_record_index);
        if (host_record == NULL) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        const bool resume_simulation = nlo_vec_is_in_simulation(state->backend);
        if (resume_simulation) {
            nlo_vec_status status = nlo_vec_end_simulation(state->backend);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
        }

        nlo_vec_status status = nlo_vec_download(state->backend,
                                                 state->current_field_vec,
                                                 host_record,
                                                 state->num_time_samples * sizeof(nlo_complex));

        if (resume_simulation) {
            nlo_vec_status begin_status = nlo_vec_begin_simulation(state->backend);
            if (status == NLO_VEC_STATUS_OK) {
                status = begin_status;
            }
        }

        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        state->current_record_index += 1u;
        state->record_ring_flushed_count += 1u;
        return NLO_VEC_STATUS_OK;
    }

    if (state->record_ring_size == state->record_ring_capacity) {
        nlo_vec_status flush_status = simulation_state_flush_oldest_ring_entry(state);
        if (flush_status != NLO_VEC_STATUS_OK) {
            return flush_status;
        }
    }

    const size_t slot = (state->record_ring_head + state->record_ring_size) % state->record_ring_capacity;
    nlo_vec_buffer* ring_dst = state->record_ring_vec[slot];
    if (ring_dst == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status copy_status = nlo_vec_complex_copy(state->backend, ring_dst, state->current_field_vec);
    if (copy_status != NLO_VEC_STATUS_OK) {
        return copy_status;
    }

    state->record_ring_size += 1u;
    state->current_record_index += 1u;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status simulation_state_flush_snapshots(simulation_state* state)
{
    if (state == NULL || state->backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    while (state->record_ring_size > 0u) {
        nlo_vec_status status = simulation_state_flush_oldest_ring_entry(state);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    return NLO_VEC_STATUS_OK;
}

static int checked_mul_size_t(size_t a, size_t b, size_t* out)
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

static size_t query_available_system_memory_bytes(void)
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

static size_t apply_memory_headroom(size_t available_bytes)
{
    if (available_bytes == 0u) {
        return 0u;
    }

    return (available_bytes / NLO_MEMORY_HEADROOM_DEN) * NLO_MEMORY_HEADROOM_NUM;
}

static size_t compute_host_record_capacity(size_t num_time_samples, size_t requested_records)
{
    size_t available_bytes = query_available_system_memory_bytes();
    if (available_bytes == 0u) {
        return requested_records;
    }

    available_bytes = apply_memory_headroom(available_bytes);
    if (available_bytes == 0u) {
        return 0u;
    }

    size_t per_record_bytes = 0u;
    if (checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &per_record_bytes) != 0 || per_record_bytes == 0u) {
        return 0u;
    }

    size_t working_bytes = 0u;
    if (checked_mul_size_t(per_record_bytes, NLO_WORK_VECTOR_COUNT, &working_bytes) != 0) {
        return 0u;
    }

    if (working_bytes >= available_bytes) {
        return 0u;
    }

    size_t max_records = (available_bytes - working_bytes) / per_record_bytes;
    if (max_records == 0u) {
        return 0u;
    }

    if (max_records > NT_MAX) {
        max_records = NT_MAX;
    }

    return (max_records < requested_records) ? max_records : requested_records;
}
