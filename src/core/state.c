/**
 * @file state.c
 * @brief State management for backend-resident simulation buffers.
 */

#include "core/state.h"
#include "fft/fft.h"
#include "io/snapshot_store.h"
#include "physics/operators.h"
#include "utility/state_debug.h"
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

#ifndef NLO_DEVICE_RING_BUDGET_HEADROOM_NUM
#define NLO_DEVICE_RING_BUDGET_HEADROOM_NUM 9u
#endif

#ifndef NLO_DEVICE_RING_BUDGET_HEADROOM_DEN
#define NLO_DEVICE_RING_BUDGET_HEADROOM_DEN 10u
#endif

#ifndef NLO_TWO_PI
#define NLO_TWO_PI 6.283185307179586476925286766559
#endif

#ifndef NLO_FREQ_GRID_REL_TOL
#define NLO_FREQ_GRID_REL_TOL 1e-9
#endif

static int checked_mul_size_t(size_t a, size_t b, size_t* out);
static int nlo_resolve_sim_dimensions(
    const sim_config* config,
    size_t total_samples,
    size_t* out_nt,
    size_t* out_nx,
    size_t* out_ny,
    int* out_explicit_nd
);
static size_t query_available_system_memory_bytes(void);
static size_t apply_memory_headroom(size_t available_bytes);
static size_t compute_host_record_capacity(size_t num_time_samples, size_t requested_records);
static int nlo_storage_options_enabled(const nlo_storage_options* options);
static double nlo_resolve_delta_time(const sim_config* config, size_t num_time_samples);
static void nlo_fill_default_omega_grid(nlo_complex* out_grid, size_t num_time_samples, double delta_time);
static void nlo_fill_default_k2_grid_xy(
    nlo_complex* out_grid_xy,
    size_t nx,
    size_t ny,
    double delta_x,
    double delta_y
);
static int nlo_expand_temporal_grid_to_volume(
    nlo_complex* out_volume,
    const nlo_complex* temporal_grid,
    size_t nt,
    size_t nx,
    size_t ny
);
static int nlo_expand_xy_grid_to_volume(
    nlo_complex* out_volume,
    const nlo_complex* xy_grid,
    size_t nt,
    size_t nx,
    size_t ny
);
static int nlo_frequency_grid_matches_expected_unshifted(
    const nlo_complex* grid,
    size_t num_time_samples,
    double delta_time
);
static nlo_vec_status nlo_snapshot_emit_record(simulation_state* state, size_t record_index, const nlo_complex* record);

#ifndef NLO_DEFAULT_DISPERSION_FACTOR_EXPR
#define NLO_DEFAULT_DISPERSION_FACTOR_EXPR "i*c0*w*w-c1"
#endif

#ifndef NLO_DEFAULT_DISPERSION_EXPR
#define NLO_DEFAULT_DISPERSION_EXPR "exp(h*D)"
#endif

#ifndef NLO_DEFAULT_NONLINEAR_EXPR
#define NLO_DEFAULT_NONLINEAR_EXPR "i*c2*I + i*V"
#endif

#ifndef NLO_DEFAULT_TRANSVERSE_FACTOR_EXPR
#define NLO_DEFAULT_TRANSVERSE_FACTOR_EXPR "0"
#endif

#ifndef NLO_DEFAULT_TRANSVERSE_EXPR
#define NLO_DEFAULT_TRANSVERSE_EXPR "exp(h*D)"
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

#ifndef NLO_DEFAULT_C3
#define NLO_DEFAULT_C3 0.0
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

static int nlo_resolve_sim_dimensions(
    const sim_config* config,
    size_t total_samples,
    size_t* out_nt,
    size_t* out_nx,
    size_t* out_ny,
    int* out_explicit_nd
)
{
    if (config == NULL ||
        out_nt == NULL ||
        out_nx == NULL ||
        out_ny == NULL ||
        out_explicit_nd == NULL ||
        total_samples == 0u) {
        return -1;
    }

    const size_t configured_nt = config->time.nt;
    size_t nx = config->spatial.nx;
    size_t ny = config->spatial.ny;

    if (configured_nt == 0u) {
        if (nx == 0u && ny == 0u) {
            *out_nt = total_samples;
            *out_nx = 1u;
            *out_ny = 1u;
            *out_explicit_nd = 0;
            return 0;
        }
        if (nx == 0u || ny == 0u) {
            return -1;
        }

        size_t total_points = 0u;
        if (checked_mul_size_t(nx, ny, &total_points) != 0 || total_points != total_samples) {
            return -1;
        }

        if (ny == 1u && nx == total_samples) {
            *out_nt = total_samples;
            *out_nx = 1u;
            *out_ny = 1u;
            *out_explicit_nd = 0;
            return 0;
        }

        *out_nt = 1u;
        *out_nx = nx;
        *out_ny = ny;
        *out_explicit_nd = 0;
        return 0;
    }

    if (nx == 0u) {
        nx = 1u;
    }
    if (ny == 0u) {
        ny = 1u;
    }

    size_t ntxy = 0u;
    if (checked_mul_size_t(configured_nt, nx, &ntxy) != 0) {
        return -1;
    }
    size_t resolved_total = 0u;
    if (checked_mul_size_t(ntxy, ny, &resolved_total) != 0 || resolved_total != total_samples) {
        return -1;
    }

    *out_nt = configured_nt;
    *out_nx = nx;
    *out_ny = ny;
    *out_explicit_nd = 1;
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

static void nlo_fill_default_k2_grid_xy(
    nlo_complex* out_grid_xy,
    size_t nx,
    size_t ny,
    double delta_x,
    double delta_y
)
{
    if (out_grid_xy == NULL || nx == 0u || ny == 0u) {
        return;
    }

    const double safe_dx = (delta_x > 0.0) ? delta_x : 1.0;
    const double safe_dy = (delta_y > 0.0) ? delta_y : 1.0;
    const double kx_step = NLO_TWO_PI / ((double)nx * safe_dx);
    const double ky_step = NLO_TWO_PI / ((double)ny * safe_dy);

    for (size_t y = 0u; y < ny; ++y) {
        const double ky = nlo_expected_omega_unshifted(y, ny, ky_step);
        for (size_t x = 0u; x < nx; ++x) {
            const double kx = nlo_expected_omega_unshifted(x, nx, kx_step);
            const double k2 = (kx * kx) + (ky * ky);
            out_grid_xy[(y * nx) + x] = nlo_make(k2, 0.0);
        }
    }
}

static int nlo_expand_temporal_grid_to_volume(
    nlo_complex* out_volume,
    const nlo_complex* temporal_grid,
    size_t nt,
    size_t nx,
    size_t ny
)
{
    if (out_volume == NULL || temporal_grid == NULL || nt == 0u || nx == 0u || ny == 0u) {
        return -1;
    }

    const size_t plane = nx * ny;
    for (size_t t = 0u; t < nt; ++t) {
        const nlo_complex omega = temporal_grid[t];
        const size_t base = t * plane;
        for (size_t i = 0u; i < plane; ++i) {
            out_volume[base + i] = omega;
        }
    }
    return 0;
}

static int nlo_expand_xy_grid_to_volume(
    nlo_complex* out_volume,
    const nlo_complex* xy_grid,
    size_t nt,
    size_t nx,
    size_t ny
)
{
    if (out_volume == NULL || xy_grid == NULL || nt == 0u || nx == 0u || ny == 0u) {
        return -1;
    }

    const size_t plane = nx * ny;
    for (size_t t = 0u; t < nt; ++t) {
        memcpy(out_volume + (t * plane), xy_grid, plane * sizeof(nlo_complex));
    }
    return 0;
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
    out_constants[3] = NLO_DEFAULT_C3;

    size_t count = 4u;
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
    budget_bytes =
        (budget_bytes / NLO_DEVICE_RING_BUDGET_HEADROOM_DEN) *
        NLO_DEVICE_RING_BUDGET_HEADROOM_NUM;
    if (budget_bytes == 0u) {
        return NLO_MIN_DEVICE_RING_CAPACITY;
    }

    size_t active_vec_count = 2u + NLO_WORK_VECTOR_COUNT;
    if (state->spatial_frequency_grid_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->transverse_factor_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->transverse_operator_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->runtime_operator_stack_slots > SIZE_MAX - active_vec_count) {
        return NLO_MIN_DEVICE_RING_CAPACITY;
    }
    active_vec_count += state->runtime_operator_stack_slots;

    size_t active_bytes = 0u;
    if (checked_mul_size_t(active_vec_count, per_record_bytes, &active_bytes) != 0) {
        return NLO_MIN_DEVICE_RING_CAPACITY;
    }

    if (active_bytes > SIZE_MAX - per_record_bytes) {
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

    nlo_state_debug_log_ring_capacity(requested_records,
                                      per_record_bytes,
                                      active_bytes,
                                      state->runtime_operator_stack_slots,
                                      budget_bytes,
                                      ring_capacity);

    return ring_capacity;
}

static int nlo_storage_options_enabled(const nlo_storage_options* options)
{
    return (options != NULL &&
            options->sqlite_path != NULL &&
            options->sqlite_path[0] != '\0');
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

nlo_storage_options nlo_storage_options_default(void)
{
    nlo_storage_options options;
    memset(&options, 0, sizeof(options));
    options.sqlite_path = NULL;
    options.run_id = NULL;
    options.sqlite_max_bytes = 0u;
    options.chunk_records = 0u;
    options.cap_policy = NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES;
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
    config->time.nt = 0u;
    config->spatial.nx = num_time_samples;
    config->spatial.ny = 1u;
    config->spatial.delta_x = 1.0;
    config->spatial.delta_y = 1.0;
    config->spatial.spatial_frequency_grid = NULL;
    config->spatial.potential_grid = NULL;
    config->runtime.dispersion_factor_expr = NULL;
    config->runtime.dispersion_expr = NULL;
    config->runtime.transverse_factor_expr = NULL;
    config->runtime.transverse_expr = NULL;
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
    return create_simulation_state_with_storage(config,
                                                num_time_samples,
                                                num_recorded_samples,
                                                exec_options,
                                                NULL);
}

simulation_state* create_simulation_state_with_storage(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options,
    const nlo_storage_options* storage_options
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

    size_t resolved_nt = 0u;
    size_t spatial_nx = 0u;
    size_t spatial_ny = 0u;
    int explicit_nd = 0;
    if (nlo_resolve_sim_dimensions(config,
                                   num_time_samples,
                                   &resolved_nt,
                                   &spatial_nx,
                                   &spatial_ny,
                                   &explicit_nd) != 0) {
        return NULL;
    }

    simulation_state* state = (simulation_state*)calloc(1, sizeof(simulation_state));
    if (state == NULL) {
        return NULL;
    }

    state->config = config;
    state->exec_options = *exec_options;
    state->nt = resolved_nt;
    state->nx = spatial_nx;
    state->ny = spatial_ny;
    state->num_time_samples = num_time_samples;
    state->num_points_xy = spatial_nx * spatial_ny;
    state->num_recorded_samples = num_recorded_samples;
    state->num_host_records = compute_host_record_capacity(num_time_samples, num_recorded_samples);
    const int storage_enabled = nlo_storage_options_enabled(storage_options);
    if (state->num_host_records == 0u && !storage_enabled) {
        nlo_state_debug_log_failure("host_record_capacity", NLO_VEC_STATUS_ALLOCATION_FAILED);
        free_simulation_state(state);
        return NULL;
    }
    if (!storage_enabled && state->num_host_records < num_recorded_samples) {
        nlo_state_debug_log_failure("host_record_capacity_insufficient", NLO_VEC_STATUS_ALLOCATION_FAILED);
        free_simulation_state(state);
        return NULL;
    }

    state->current_z = 0.0;
    state->current_step_size = config->propagation.starting_step_size;
    state->current_half_step_exp = 0.5 * state->current_step_size;
    state->dispersion_valid = 0;
    state->runtime_operator_stack_slots = 0u;
    state->snapshot_status = NLO_VEC_STATUS_OK;

    if (storage_enabled) {
        if (!nlo_snapshot_store_is_available()) {
            nlo_state_debug_log_failure("snapshot_store_unavailable", NLO_VEC_STATUS_UNSUPPORTED);
            free_simulation_state(state);
            return NULL;
        }

        nlo_snapshot_store_open_params store_params;
        memset(&store_params, 0, sizeof(store_params));
        store_params.config = config;
        store_params.exec_options = exec_options;
        store_params.storage_options = storage_options;
        store_params.num_time_samples = num_time_samples;
        store_params.num_recorded_samples = num_recorded_samples;
        state->snapshot_store = nlo_snapshot_store_open(&store_params);
        if (state->snapshot_store == NULL) {
            nlo_state_debug_log_failure("snapshot_store_open", NLO_VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
        nlo_snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
    }

    if (state->num_host_records > 0u) {
        size_t host_elements = 0u;
        if (checked_mul_size_t(state->num_time_samples, state->num_host_records, &host_elements) != 0) {
            nlo_state_debug_log_failure("host_elements_overflow", NLO_VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }

        state->field_buffer = (nlo_complex*)calloc(host_elements, sizeof(nlo_complex));
        if (state->field_buffer == NULL) {
            nlo_state_debug_log_failure("allocate_host_field_buffer", NLO_VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
    }

    if (storage_enabled && state->num_host_records < state->num_recorded_samples) {
        state->snapshot_scratch_record = (nlo_complex*)calloc(state->num_time_samples, sizeof(nlo_complex));
        if (state->snapshot_scratch_record == NULL) {
            nlo_state_debug_log_failure("allocate_snapshot_scratch_record", NLO_VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
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
        nlo_state_debug_log_failure("create_backend", NLO_VEC_STATUS_BACKEND_UNAVAILABLE);
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
        nlo_state_debug_log_failure("allocate_device_vectors", NLO_VEC_STATUS_ALLOCATION_FAILED);
        free_simulation_state(state);
        return NULL;
    }

    const int enable_transverse = (explicit_nd != 0) && (spatial_nx > 1u || spatial_ny > 1u);
    state->transverse_active = 0;
    if (enable_transverse) {
        if (nlo_create_complex_vec(state->backend, num_time_samples, &state->spatial_frequency_grid_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->transverse_factor_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->transverse_operator_vec) != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("allocate_transverse_vectors", NLO_VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
    }

    nlo_vec_status status = nlo_vec_complex_fill(state->backend, state->current_field_vec, nlo_make(0.0, 0.0));
    if (status != NLO_VEC_STATUS_OK) {
        nlo_state_debug_log_failure("clear_current_field", status);
        free_simulation_state(state);
        return NULL;
    }

    const double delta_time = nlo_resolve_delta_time(config, state->nt);
    if (explicit_nd != 0) {
        nlo_complex* temporal_line = (nlo_complex*)malloc(state->nt * sizeof(nlo_complex));
        nlo_complex* temporal_volume = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
        if (temporal_line == NULL || temporal_volume == NULL) {
            free(temporal_line);
            free(temporal_volume);
            nlo_state_debug_log_failure("allocate_temporal_frequency_grid", NLO_VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }

        if (config->frequency.frequency_grid != NULL) {
            memcpy(temporal_line, config->frequency.frequency_grid, state->nt * sizeof(nlo_complex));
        } else {
            nlo_fill_default_omega_grid(temporal_line, state->nt, delta_time);
        }

        if (nlo_expand_temporal_grid_to_volume(temporal_volume,
                                               temporal_line,
                                               state->nt,
                                               state->nx,
                                               state->ny) != 0) {
            free(temporal_line);
            free(temporal_volume);
            nlo_state_debug_log_failure("expand_temporal_frequency_grid", NLO_VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }

        status = nlo_vec_upload(state->backend,
                                state->frequency_grid_vec,
                                temporal_volume,
                                num_time_samples * sizeof(nlo_complex));
        free(temporal_line);
        free(temporal_volume);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("upload_frequency_grid", status);
            free_simulation_state(state);
            return NULL;
        }
    } else {
        const nlo_complex* spectral_grid_source = config->spatial.spatial_frequency_grid;
        nlo_complex* generated_frequency_grid = NULL;
        if (spectral_grid_source == NULL) {
            const nlo_complex* temporal_frequency_grid = config->frequency.frequency_grid;
            if (temporal_frequency_grid != NULL &&
                nlo_frequency_grid_matches_expected_unshifted(temporal_frequency_grid,
                                                              num_time_samples,
                                                              delta_time)) {
                spectral_grid_source = temporal_frequency_grid;
            } else {
                generated_frequency_grid = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
                if (generated_frequency_grid == NULL) {
                    nlo_state_debug_log_failure("allocate_generated_frequency_grid", NLO_VEC_STATUS_ALLOCATION_FAILED);
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
            nlo_state_debug_log_failure("upload_frequency_grid", status);
            free_simulation_state(state);
            return NULL;
        }
    }

    if (enable_transverse) {
        const size_t xy_points = state->num_points_xy;
        nlo_complex* k2_xy = (nlo_complex*)malloc(xy_points * sizeof(nlo_complex));
        nlo_complex* k2_volume = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
        nlo_complex* potential_volume = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
        if (k2_xy == NULL || k2_volume == NULL || potential_volume == NULL) {
            free(k2_xy);
            free(k2_volume);
            free(potential_volume);
            nlo_state_debug_log_failure("allocate_transverse_grids", NLO_VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }

        if (config->spatial.spatial_frequency_grid != NULL) {
            memcpy(k2_xy, config->spatial.spatial_frequency_grid, xy_points * sizeof(nlo_complex));
        } else {
            nlo_fill_default_k2_grid_xy(k2_xy,
                                        state->nx,
                                        state->ny,
                                        config->spatial.delta_x,
                                        config->spatial.delta_y);
        }
        if (nlo_expand_xy_grid_to_volume(k2_volume, k2_xy, state->nt, state->nx, state->ny) != 0) {
            free(k2_xy);
            free(k2_volume);
            free(potential_volume);
            nlo_state_debug_log_failure("expand_spatial_frequency_grid", NLO_VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }

        if (config->spatial.potential_grid != NULL) {
            if (nlo_expand_xy_grid_to_volume(potential_volume,
                                             config->spatial.potential_grid,
                                             state->nt,
                                             state->nx,
                                             state->ny) != 0) {
                free(k2_xy);
                free(k2_volume);
                free(potential_volume);
                nlo_state_debug_log_failure("expand_potential_grid", NLO_VEC_STATUS_INVALID_ARGUMENT);
                free_simulation_state(state);
                return NULL;
            }
        } else {
            for (size_t i = 0u; i < num_time_samples; ++i) {
                potential_volume[i] = nlo_make(0.0, 0.0);
            }
        }

        status = nlo_vec_upload(state->backend,
                                state->spatial_frequency_grid_vec,
                                k2_volume,
                                num_time_samples * sizeof(nlo_complex));
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_upload(state->backend,
                                    state->working_vectors.potential_vec,
                                    potential_volume,
                                    num_time_samples * sizeof(nlo_complex));
        }
        free(k2_xy);
        free(k2_volume);
        free(potential_volume);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("upload_transverse_grids", status);
            free_simulation_state(state);
            return NULL;
        }
    } else {
        if (config->spatial.potential_grid != NULL) {
            if (explicit_nd != 0) {
                nlo_complex* potential_volume = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
                if (potential_volume == NULL) {
                    nlo_state_debug_log_failure("allocate_potential_grid", NLO_VEC_STATUS_ALLOCATION_FAILED);
                    free_simulation_state(state);
                    return NULL;
                }
                for (size_t i = 0u; i < num_time_samples; ++i) {
                    potential_volume[i] = config->spatial.potential_grid[0];
                }
                status = nlo_vec_upload(state->backend,
                                        state->working_vectors.potential_vec,
                                        potential_volume,
                                        num_time_samples * sizeof(nlo_complex));
                free(potential_volume);
            } else {
                status = nlo_vec_upload(state->backend,
                                        state->working_vectors.potential_vec,
                                        config->spatial.potential_grid,
                                        num_time_samples * sizeof(nlo_complex));
            }
        } else {
            status = nlo_vec_complex_fill(state->backend,
                                          state->working_vectors.potential_vec,
                                          nlo_make(0.0, 0.0));
        }
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("upload_potential_grid", status);
            free_simulation_state(state);
            return NULL;
        }
    }

    double resolved_runtime_constants[NLO_RUNTIME_OPERATOR_CONSTANTS_MAX];
    const double* runtime_constants = resolved_runtime_constants;
    size_t runtime_constant_count = nlo_resolve_runtime_constants(&config->runtime,
                                                                  resolved_runtime_constants);
    if (runtime_constant_count == 0u ||
        runtime_constant_count > NLO_RUNTIME_OPERATOR_CONSTANTS_MAX) {
        nlo_state_debug_log_failure("resolve_runtime_constants", NLO_VEC_STATUS_INVALID_ARGUMENT);
        free_simulation_state(state);
        return NULL;
    }

    const char* dispersion_factor_expr = nlo_resolve_operator_expr(config->runtime.dispersion_factor_expr,
                                                                   NLO_DEFAULT_DISPERSION_FACTOR_EXPR);
    const char* dispersion_expr = nlo_resolve_operator_expr(config->runtime.dispersion_expr,
                                                            NLO_DEFAULT_DISPERSION_EXPR);
    const char* transverse_factor_expr = nlo_resolve_operator_expr(config->runtime.transverse_factor_expr,
                                                                   NLO_DEFAULT_TRANSVERSE_FACTOR_EXPR);
    const char* transverse_expr = nlo_resolve_operator_expr(config->runtime.transverse_expr,
                                                            NLO_DEFAULT_TRANSVERSE_EXPR);
    const char* nonlinear_expr = nlo_resolve_operator_expr(config->runtime.nonlinear_expr,
                                                           NLO_DEFAULT_NONLINEAR_EXPR);

    status = nlo_operator_program_compile(dispersion_factor_expr,
                                          NLO_OPERATOR_CONTEXT_DISPERSION_FACTOR,
                                          runtime_constant_count,
                                          runtime_constants,
                                          &state->dispersion_factor_operator_program);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_state_debug_log_failure("compile_dispersion_factor_program", status);
        free_simulation_state(state);
        return NULL;
    }

    status = nlo_operator_program_compile(dispersion_expr,
                                          NLO_OPERATOR_CONTEXT_DISPERSION,
                                          runtime_constant_count,
                                          runtime_constants,
                                          &state->dispersion_operator_program);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_state_debug_log_failure("compile_dispersion_program", status);
        free_simulation_state(state);
        return NULL;
    }

    if (enable_transverse) {
        status = nlo_operator_program_compile(transverse_factor_expr,
                                              NLO_OPERATOR_CONTEXT_DISPERSION_FACTOR,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->transverse_factor_operator_program);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("compile_transverse_factor_program", status);
            free_simulation_state(state);
            return NULL;
        }

        status = nlo_operator_program_compile(transverse_expr,
                                              NLO_OPERATOR_CONTEXT_DISPERSION,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->transverse_operator_program);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("compile_transverse_program", status);
            free_simulation_state(state);
            return NULL;
        }
    }

    status = nlo_operator_program_compile(nonlinear_expr,
                                          NLO_OPERATOR_CONTEXT_NONLINEAR,
                                          runtime_constant_count,
                                          runtime_constants,
                                          &state->nonlinear_operator_program);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_state_debug_log_failure("compile_nonlinear_program", status);
        free_simulation_state(state);
        return NULL;
    }

    size_t required_stack_slots = state->dispersion_factor_operator_program.required_stack_slots;
    if (state->dispersion_operator_program.required_stack_slots > required_stack_slots) {
        required_stack_slots = state->dispersion_operator_program.required_stack_slots;
    }
    if (state->transverse_factor_operator_program.required_stack_slots > required_stack_slots) {
        required_stack_slots = state->transverse_factor_operator_program.required_stack_slots;
    }
    if (state->transverse_operator_program.required_stack_slots > required_stack_slots) {
        required_stack_slots = state->transverse_operator_program.required_stack_slots;
    }
    if (state->nonlinear_operator_program.required_stack_slots > required_stack_slots) {
        required_stack_slots = state->nonlinear_operator_program.required_stack_slots;
    }

    if (required_stack_slots == 0u ||
        required_stack_slots > NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS) {
        nlo_state_debug_log_failure("resolve_runtime_stack_slots", NLO_VEC_STATUS_INVALID_ARGUMENT);
        free_simulation_state(state);
        return NULL;
    }

    state->runtime_operator_stack_slots = required_stack_slots;
    for (size_t i = 0u; i < state->runtime_operator_stack_slots; ++i) {
        if (nlo_create_complex_vec(state->backend,
                                   num_time_samples,
                                   &state->runtime_operator_stack_vec[i]) != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("allocate_runtime_stack_vectors", NLO_VEC_STATUS_ALLOCATION_FAILED);
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
        nlo_state_debug_log_failure("execute_dispersion_factor_program", status);
        free_simulation_state(state);
        return NULL;
    }

    if (enable_transverse) {
        const nlo_operator_eval_context transverse_factor_eval_ctx = {
            .frequency_grid = state->spatial_frequency_grid_vec,
            .field = state->current_field_vec,
            .dispersion_factor = NULL,
            .potential = state->working_vectors.potential_vec,
            .half_step_size = state->current_half_step_exp
        };
        status = nlo_operator_program_execute(state->backend,
                                              &state->transverse_factor_operator_program,
                                              &transverse_factor_eval_ctx,
                                              state->runtime_operator_stack_vec,
                                              state->runtime_operator_stack_slots,
                                              state->transverse_factor_vec);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("execute_transverse_factor_program", status);
            free_simulation_state(state);
            return NULL;
        }
        state->transverse_active = 1;
    }

    state->dispersion_valid = 1;

    nlo_vec_status fft_status = NLO_VEC_STATUS_UNSUPPORTED;
    if (explicit_nd != 0 && state->nt > 1u && state->nx > 1u && state->ny > 1u) {
        const nlo_fft_shape shape = {
            .rank = 3u,
            .dims = {state->nt, state->ny, state->nx}
        };
        fft_status = nlo_fft_plan_create_shaped_with_backend(state->backend,
                                                             &shape,
                                                             state->exec_options.fft_backend,
                                                             &state->fft_plan);
        if (fft_status != NLO_VEC_STATUS_OK) {
            const nlo_fft_shape fallback_shape = {
                .rank = 1u,
                .dims = {num_time_samples, 1u, 1u}
            };
            fft_status = nlo_fft_plan_create_shaped_with_backend(state->backend,
                                                                 &fallback_shape,
                                                                 state->exec_options.fft_backend,
                                                                 &state->fft_plan);
        }
    } else if (explicit_nd != 0 && state->nx > 1u && state->ny > 1u) {
        const nlo_fft_shape shape = {
            .rank = 2u,
            .dims = {state->ny, state->nx, 1u}
        };
        fft_status = nlo_fft_plan_create_shaped_with_backend(state->backend,
                                                             &shape,
                                                             state->exec_options.fft_backend,
                                                             &state->fft_plan);
        if (fft_status != NLO_VEC_STATUS_OK) {
            const nlo_fft_shape fallback_shape = {
                .rank = 1u,
                .dims = {num_time_samples, 1u, 1u}
            };
            fft_status = nlo_fft_plan_create_shaped_with_backend(state->backend,
                                                                 &fallback_shape,
                                                                 state->exec_options.fft_backend,
                                                                 &state->fft_plan);
        }
    } else {
        fft_status = nlo_fft_plan_create_with_backend(state->backend,
                                                      num_time_samples,
                                                      state->exec_options.fft_backend,
                                                      &state->fft_plan);
    }
    if (fft_status != NLO_VEC_STATUS_OK) {
        nlo_state_debug_log_failure("create_fft_plan", NLO_VEC_STATUS_BACKEND_UNAVAILABLE);
        free_simulation_state(state);
        return NULL;
    }

    state->record_ring_capacity = nlo_compute_device_ring_capacity(state, state->num_recorded_samples);
    if (state->record_ring_capacity > 0u) {
        state->record_ring_vec = (nlo_vec_buffer**)calloc(state->record_ring_capacity, sizeof(nlo_vec_buffer*));
        if (state->record_ring_vec == NULL) {
            nlo_state_debug_log_failure("allocate_record_ring_array", NLO_VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }

        for (size_t i = 0; i < state->record_ring_capacity; ++i) {
            if (nlo_create_complex_vec(state->backend, num_time_samples, &state->record_ring_vec[i]) != NLO_VEC_STATUS_OK) {
                nlo_state_debug_log_failure("allocate_record_ring_vectors", NLO_VEC_STATUS_ALLOCATION_FAILED);
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
        nlo_destroy_vec_if_set(state->backend, &state->spatial_frequency_grid_vec);
        nlo_destroy_vec_if_set(state->backend, &state->transverse_factor_vec);
        nlo_destroy_vec_if_set(state->backend, &state->transverse_operator_vec);
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
    nlo_snapshot_store_close(state->snapshot_store);
    state->snapshot_store = NULL;
    free(state->snapshot_scratch_record);
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

    if (state->num_recorded_samples > 1u) {
        status = nlo_snapshot_emit_record(state, 0u, field);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    state->current_record_index = (state->num_recorded_samples > 1u) ? 1u : 0u;
    state->record_ring_flushed_count = state->current_record_index;

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

static nlo_vec_status nlo_snapshot_emit_record(simulation_state* state, size_t record_index, const nlo_complex* record)
{
    if (state == NULL || record == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    int recorded_anywhere = 0;
    if (record_index < state->num_host_records) {
        nlo_complex* host_record = simulation_state_get_field_record(state, record_index);
        if (host_record == NULL) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }
        if (host_record != record) {
            memcpy(host_record, record, state->num_time_samples * sizeof(nlo_complex));
        }
        recorded_anywhere = 1;
    }

    if (state->snapshot_store != NULL) {
        const nlo_snapshot_store_status store_status =
            nlo_snapshot_store_write_record(state->snapshot_store,
                                           record_index,
                                           record,
                                           state->num_time_samples);
        nlo_snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
        if (store_status == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            state->snapshot_status = NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
            return state->snapshot_status;
        }
        recorded_anywhere = 1;
    }

    if (!recorded_anywhere) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    return NLO_VEC_STATUS_OK;
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
    nlo_complex* download_target = host_record;
    if (download_target == NULL) {
        download_target = state->snapshot_scratch_record;
    }
    if (download_target == NULL) {
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
                                             download_target,
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

    status = nlo_snapshot_emit_record(state, state->record_ring_flushed_count, download_target);
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

        status = nlo_snapshot_emit_record(state,
                                          state->current_record_index,
                                          (const nlo_complex*)host_src);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        state->current_record_index += 1u;
        return NLO_VEC_STATUS_OK;
    }

    if (state->record_ring_capacity == 0u) {
        nlo_complex* host_record = simulation_state_get_field_record(state, state->current_record_index);
        nlo_complex* download_target = host_record;
        if (download_target == NULL) {
            download_target = state->snapshot_scratch_record;
        }
        if (download_target == NULL) {
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
                                                 download_target,
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

        status = nlo_snapshot_emit_record(state, state->current_record_index, download_target);
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

    if (state->snapshot_store != NULL) {
        const nlo_snapshot_store_status store_status = nlo_snapshot_store_flush(state->snapshot_store);
        nlo_snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
        if (store_status == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            state->snapshot_status = NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
            return state->snapshot_status;
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
