/**
 * @file state.c
 * @brief State management for backend-resident simulation buffers.
 */

#include "core/state.h"
#include "fft/fft.h"
#include "physics/operators.h"
#include <float.h>
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

#ifndef NLO_TWO_PI
#define NLO_TWO_PI 6.283185307179586476925286766559
#endif

#ifndef NLO_FREQ_GRID_REL_TOL
#define NLO_FREQ_GRID_REL_TOL 1e-9
#endif

static int checked_mul_size_t(size_t a, size_t b, size_t* out);
static int nlo_resolve_spatial_dimensions(
    const sim_config* config,
    size_t num_time_samples,
    size_t* out_nx,
    size_t* out_ny
);
static size_t query_available_system_memory_bytes(void);
static size_t apply_memory_headroom(size_t available_bytes);
static size_t compute_host_record_capacity(size_t num_time_samples, size_t requested_records);
static double nlo_resolve_delta_time(const sim_config* config, size_t num_time_samples);
static void nlo_fill_default_omega_grid(nlo_complex* out_grid, size_t num_time_samples, double delta_time);
static int nlo_frequency_grid_matches_expected_unshifted(
    const nlo_complex* grid,
    size_t num_time_samples,
    double delta_time
);

#ifndef NLO_DEFAULT_DISPERSION_FACTOR_EXPR
#define NLO_DEFAULT_DISPERSION_FACTOR_EXPR "i*c0*(w^2)-c1"
#endif

#ifndef NLO_DEFAULT_DISPERSION_EXPR
#define NLO_DEFAULT_DISPERSION_EXPR "exp(h*D)"
#endif

#ifndef NLO_DEFAULT_NONLINEAR_EXPR
#define NLO_DEFAULT_NONLINEAR_EXPR "i*c2*I + i*V"
#endif

#ifndef NLO_DEFAULT_C0
#define NLO_DEFAULT_C0 -0.5
#endif

#ifndef NLO_DEFAULT_C1
#define NLO_DEFAULT_C1 0.0
#endif

#ifndef NLO_DEFAULT_C2
#define NLO_DEFAULT_C2 1.0
#endif

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

static int nlo_resolve_spatial_dimensions(
    const sim_config* config,
    size_t num_time_samples,
    size_t* out_nx,
    size_t* out_ny
)
{
    if (config == NULL || out_nx == NULL || out_ny == NULL || num_time_samples == 0u) {
        return -1;
    }

    size_t nx = config->spatial.nx;
    size_t ny = config->spatial.ny;
    if (nx == 0u && ny == 0u) {
        nx = num_time_samples;
        ny = 1u;
    } else if (nx == 0u || ny == 0u) {
        return -1;
    }

    size_t total_points = 0u;
    if (checked_mul_size_t(nx, ny, &total_points) != 0 || total_points != num_time_samples) {
        return -1;
    }

    *out_nx = nx;
    *out_ny = ny;
    return 0;
}

static double nlo_expected_omega_unshifted(size_t index, size_t num_time_samples, double omega_step)
{
    if (num_time_samples == 0u) {
        return 0.0;
    }

    const size_t positive_limit = (num_time_samples - 1u) / 2u;
    if (index <= positive_limit) {
        return (double)index * omega_step;
    }

    return -((double)num_time_samples - (double)index) * omega_step;
}

static double nlo_resolve_delta_time(const sim_config* config, size_t num_time_samples)
{
    if (config == NULL || num_time_samples == 0u) {
        return 1.0;
    }

    if (config->time.delta_time > 0.0) {
        return config->time.delta_time;
    }

    if (config->time.pulse_period > 0.0) {
        return config->time.pulse_period / (double)num_time_samples;
    }

    return 1.0;
}

static void nlo_fill_default_omega_grid(nlo_complex* out_grid, size_t num_time_samples, double delta_time)
{
    if (out_grid == NULL || num_time_samples == 0u) {
        return;
    }

    const double safe_delta_time = (delta_time > 0.0) ? delta_time : 1.0;
    const double omega_step = NLO_TWO_PI / ((double)num_time_samples * safe_delta_time);

    for (size_t i = 0u; i < num_time_samples; ++i) {
        out_grid[i] = nlo_make(nlo_expected_omega_unshifted(i, num_time_samples, omega_step), 0.0);
    }
}

static int nlo_frequency_grid_matches_expected_unshifted(
    const nlo_complex* grid,
    size_t num_time_samples,
    double delta_time
)
{
    if (grid == NULL || num_time_samples == 0u) {
        return 0;
    }

    const double safe_delta_time = (delta_time > 0.0) ? delta_time : 1.0;
    const double omega_step = NLO_TWO_PI / ((double)num_time_samples * safe_delta_time);

    for (size_t i = 0u; i < num_time_samples; ++i) {
        const double expected_real = nlo_expected_omega_unshifted(i, num_time_samples, omega_step);
        const double real_tol = NLO_FREQ_GRID_REL_TOL * fmax(1.0, fabs(expected_real));
        if (fabs(grid[i].re - expected_real) > real_tol) {
            return 0;
        }
        if (fabs(grid[i].im) > real_tol) {
            return 0;
        }
    }

    return 1;
}

static const char* nlo_resolve_operator_expr(const char* expr, const char* fallback)
{
    if (expr != NULL && expr[0] != '\0') {
        return expr;
    }

    return fallback;
}

static size_t nlo_resolve_runtime_constants(const runtime_operator_params* runtime, double out_constants[16])
{
    if (out_constants == NULL) {
        return 0u;
    }

    for (size_t i = 0u; i < NLO_RUNTIME_OPERATOR_CONSTANTS_MAX; ++i) {
        out_constants[i] = 0.0;
    }
    out_constants[0] = NLO_DEFAULT_C0;
    out_constants[1] = NLO_DEFAULT_C1;
    out_constants[2] = NLO_DEFAULT_C2;

    size_t count = 3u;
    if (runtime == NULL) {
        return count;
    }

    const size_t runtime_count = runtime->num_constants;
    if (runtime_count > NLO_RUNTIME_OPERATOR_CONSTANTS_MAX) {
        return 0u;
    }

    for (size_t i = 0u; i < runtime_count; ++i) {
        out_constants[i] = runtime->constants[i];
    }

    if (runtime_count > count) {
        count = runtime_count;
    }

    return count;
}

static double nlo_default_heap_fraction_for_device(VkPhysicalDeviceType device_type)
{
    switch (device_type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return NLO_DEFAULT_DEVICE_HEAP_FRACTION_DISCRETE;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return NLO_DEFAULT_DEVICE_HEAP_FRACTION_INTEGRATED;
        default:
            return NLO_DEFAULT_DEVICE_HEAP_FRACTION;
    }
}

static size_t nlo_compute_device_ring_capacity(const simulation_state* state, size_t requested_records)
{
    if (state == NULL || state->backend == NULL || requested_records == 0u) {
        return 0u;
    }

    if (nlo_vector_backend_get_type(state->backend) != NLO_VECTOR_BACKEND_VULKAN) {
        return 0u;
    }

    nlo_vec_backend_memory_info mem_info = {0};
    if (nlo_vec_query_memory_info(state->backend, &mem_info) != NLO_VEC_STATUS_OK) {
        return NLO_MIN_DEVICE_RING_CAPACITY;
    }

    const double frac = (state->exec_options.device_heap_fraction > 0.0 &&
                         state->exec_options.device_heap_fraction <= 1.0)
                            ? state->exec_options.device_heap_fraction
                            : nlo_default_heap_fraction_for_device(mem_info.device_type);

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
    options.device_heap_fraction = 0.0;
    options.record_ring_target = 0u;
    options.forced_device_budget_bytes = 0u;
    return options;
}

sim_config* create_sim_config(size_t num_time_samples)
{
    if (num_time_samples == 0 || num_time_samples > NT_MAX) {
        return NULL;
    }

    sim_config* config = (sim_config*)calloc(1, sizeof(sim_config));
    if (config == NULL) {
        return NULL;
    }

    config->propagation.error_tolerance = 1e-6;
    config->spatial.nx = num_time_samples;
    config->spatial.ny = 1u;
    config->spatial.delta_x = 1.0;
    config->spatial.delta_y = 1.0;
    config->spatial.spatial_frequency_grid = NULL;
    config->spatial.potential_grid = NULL;
    config->runtime.dispersion_factor_expr = NULL;
    config->runtime.dispersion_expr = NULL;
    config->runtime.nonlinear_expr = NULL;
    config->runtime.num_constants = 0u;
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
    if (config->runtime.num_constants > NLO_RUNTIME_OPERATOR_CONSTANTS_MAX) {
        return NULL;
    }

    size_t spatial_nx = 0u;
    size_t spatial_ny = 0u;
    if (nlo_resolve_spatial_dimensions(config,
                                       num_time_samples,
                                       &spatial_nx,
                                       &spatial_ny) != 0) {
        return NULL;
    }
    (void)spatial_nx;
    (void)spatial_ny;

    simulation_state* state = (simulation_state*)calloc(1, sizeof(simulation_state));
    if (state == NULL) {
        return NULL;
    }

    state->config = config;
    state->exec_options = *exec_options;
    state->num_time_samples = num_time_samples;
    state->num_points_xy = num_time_samples;
    state->num_recorded_samples = compute_host_record_capacity(num_time_samples, num_recorded_samples);
    if (state->num_recorded_samples == 0u) {
        free_simulation_state(state);
        return NULL;
    }

    state->current_z = 0.0;
    state->current_step_size = config->propagation.starting_step_size;
    state->current_half_step_exp = 0.5 * state->current_step_size;
    state->dispersion_valid = 0;
    state->runtime_operator_stack_slots = 0u;

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
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.dispersion_operator_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.nonlinear_multiplier_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.potential_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.previous_field_vec) != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    nlo_vec_status status = nlo_vec_complex_fill(state->backend, state->current_field_vec, nlo_make(0.0, 0.0));
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    const nlo_complex* spectral_grid_source = config->spatial.spatial_frequency_grid;
    nlo_complex* generated_frequency_grid = NULL;
    if (spectral_grid_source == NULL) {
        const nlo_complex* temporal_frequency_grid = config->frequency.frequency_grid;
        const double delta_time = nlo_resolve_delta_time(config, num_time_samples);
        if (temporal_frequency_grid != NULL &&
            nlo_frequency_grid_matches_expected_unshifted(temporal_frequency_grid,
                                                          num_time_samples,
                                                          delta_time)) {
            spectral_grid_source = temporal_frequency_grid;
        } else {
            generated_frequency_grid = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
            if (generated_frequency_grid == NULL) {
                free_simulation_state(state);
                return NULL;
            }

            nlo_fill_default_omega_grid(generated_frequency_grid, num_time_samples, delta_time);
            spectral_grid_source = generated_frequency_grid;
        }
    }

    if (spectral_grid_source != NULL) {
        status = nlo_vec_upload(state->backend,
                                state->frequency_grid_vec,
                                spectral_grid_source,
                                num_time_samples * sizeof(nlo_complex));
    } else {
        status = nlo_vec_complex_fill(state->backend, state->frequency_grid_vec, nlo_make(0.0, 0.0));
    }
    if (generated_frequency_grid != NULL) {
        free(generated_frequency_grid);
        generated_frequency_grid = NULL;
    }
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    if (config->spatial.potential_grid != NULL) {
        status = nlo_vec_upload(state->backend,
                                state->working_vectors.potential_vec,
                                config->spatial.potential_grid,
                                num_time_samples * sizeof(nlo_complex));
    } else {
        status = nlo_vec_complex_fill(state->backend,
                                      state->working_vectors.potential_vec,
                                      nlo_make(0.0, 0.0));
    }
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    double resolved_runtime_constants[NLO_RUNTIME_OPERATOR_CONSTANTS_MAX];
    const double* runtime_constants = resolved_runtime_constants;
    size_t runtime_constant_count = nlo_resolve_runtime_constants(&config->runtime,
                                                                  resolved_runtime_constants);
    if (runtime_constant_count == 0u ||
        runtime_constant_count > NLO_RUNTIME_OPERATOR_CONSTANTS_MAX) {
        free_simulation_state(state);
        return NULL;
    }

    const char* dispersion_factor_expr = nlo_resolve_operator_expr(config->runtime.dispersion_factor_expr,
                                                                   NLO_DEFAULT_DISPERSION_FACTOR_EXPR);
    const char* dispersion_expr = nlo_resolve_operator_expr(config->runtime.dispersion_expr,
                                                            NLO_DEFAULT_DISPERSION_EXPR);
    const char* nonlinear_expr = nlo_resolve_operator_expr(config->runtime.nonlinear_expr,
                                                           NLO_DEFAULT_NONLINEAR_EXPR);

    status = nlo_operator_program_compile(dispersion_factor_expr,
                                          NLO_OPERATOR_CONTEXT_DISPERSION_FACTOR,
                                          runtime_constant_count,
                                          runtime_constants,
                                          &state->dispersion_factor_operator_program);
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    status = nlo_operator_program_compile(dispersion_expr,
                                          NLO_OPERATOR_CONTEXT_DISPERSION,
                                          runtime_constant_count,
                                          runtime_constants,
                                          &state->dispersion_operator_program);
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    status = nlo_operator_program_compile(nonlinear_expr,
                                          NLO_OPERATOR_CONTEXT_NONLINEAR,
                                          runtime_constant_count,
                                          runtime_constants,
                                          &state->nonlinear_operator_program);
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

    size_t required_stack_slots = state->dispersion_factor_operator_program.required_stack_slots;
    if (state->dispersion_operator_program.required_stack_slots > required_stack_slots) {
        required_stack_slots = state->dispersion_operator_program.required_stack_slots;
    }
    if (state->nonlinear_operator_program.required_stack_slots > required_stack_slots) {
        required_stack_slots = state->nonlinear_operator_program.required_stack_slots;
    }

    if (required_stack_slots == 0u ||
        required_stack_slots > NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS) {
        free_simulation_state(state);
        return NULL;
    }

    state->runtime_operator_stack_slots = required_stack_slots;
    for (size_t i = 0u; i < state->runtime_operator_stack_slots; ++i) {
        if (nlo_create_complex_vec(state->backend,
                                   num_time_samples,
                                   &state->runtime_operator_stack_vec[i]) != NLO_VEC_STATUS_OK) {
            free_simulation_state(state);
            return NULL;
        }
    }

    const nlo_operator_eval_context dispersion_factor_eval_ctx = {
        .frequency_grid = state->frequency_grid_vec,
        .field = state->current_field_vec,
        .dispersion_factor = NULL,
        .potential = state->working_vectors.potential_vec,
        .half_step_size = state->current_half_step_exp
    };
    status = nlo_operator_program_execute(state->backend,
                                          &state->dispersion_factor_operator_program,
                                          &dispersion_factor_eval_ctx,
                                          state->runtime_operator_stack_vec,
                                          state->runtime_operator_stack_slots,
                                          state->working_vectors.dispersion_factor_vec);
    if (status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return NULL;
    }

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
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.dispersion_operator_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.nonlinear_multiplier_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.potential_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.previous_field_vec);
        for (size_t i = 0u; i < NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS; ++i) {
            nlo_destroy_vec_if_set(state->backend, &state->runtime_operator_stack_vec[i]);
        }

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
