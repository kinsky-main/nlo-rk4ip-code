/**
 * @file init_state_builder.c
 * @brief Simulation state construction and initialization internals.
 */

#include "core/state.h"
#include "core/init_internal.h"
#include "core/sim_dimensions_internal.h"
#include "fft/fft.h"
#include "io/log_sink.h"
#include "io/snapshot_store.h"
#include "physics/operators.h"
#include "utility/state_debug.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef MIN_DEVICE_RING_CAPACITY
#define MIN_DEVICE_RING_CAPACITY 1u
#endif

#ifndef MAX_DEVICE_RING_CAPACITY
#define MAX_DEVICE_RING_CAPACITY 4u
#endif

#ifndef DEVICE_RING_BUDGET_HEADROOM_NUM
#define DEVICE_RING_BUDGET_HEADROOM_NUM 9u
#endif

#ifndef DEVICE_RING_BUDGET_HEADROOM_DEN
#define DEVICE_RING_BUDGET_HEADROOM_DEN 10u
#endif

#ifndef DEVICE_RING_SAFETY_MIN_BYTES
#define DEVICE_RING_SAFETY_MIN_BYTES (256u * 1024u * 1024u)
#endif

#ifndef DEVICE_RING_SAFETY_TOTAL_DEN
#define DEVICE_RING_SAFETY_TOTAL_DEN 20u
#endif

#ifndef TWO_PI
#define TWO_PI 6.283185307179586476925286766559
#endif

#ifndef FREQ_GRID_REL_TOL
#define FREQ_GRID_REL_TOL 1e-9
#endif

#ifndef DEFAULT_DISPERSION_FACTOR_EXPR
#define DEFAULT_DISPERSION_FACTOR_EXPR "i*c0*w*w-c1"
#endif

#ifndef DEFAULT_LINEAR_FACTOR_EXPR
#define DEFAULT_LINEAR_FACTOR_EXPR "i*c0*wt*wt-c1"
#endif

#ifndef DEFAULT_DISPERSION_EXPR
#define DEFAULT_DISPERSION_EXPR "exp(h*D)"
#endif

#ifndef DEFAULT_LINEAR_EXPR
#define DEFAULT_LINEAR_EXPR "exp(h*D)"
#endif

#ifndef DEFAULT_POTENTIAL_EXPR
#define DEFAULT_POTENTIAL_EXPR "0"
#endif

#ifndef DEFAULT_NONLINEAR_EXPR
#define DEFAULT_NONLINEAR_EXPR "i*A*(c2*I + V)"
#endif

#ifndef DEFAULT_C0
#define DEFAULT_C0 -0.5
#endif

#ifndef DEFAULT_C1
#define DEFAULT_C1 0.0
#endif

#ifndef DEFAULT_C2
#define DEFAULT_C2 1.0
#endif

static void destroy_init_vec_if_set(vector_backend* backend, vec_buffer** vec)
{
    if (backend == NULL || vec == NULL || *vec == NULL) {
        return;
    }

    vec_destroy(backend, *vec);
    *vec = NULL;
}

static void release_init_vectors(simulation_state* state)
{
    if (state == NULL || state->backend == NULL) {
        return;
    }

    destroy_init_vec_if_set(state->backend, &state->init_vectors.wt_axis_vec);
    destroy_init_vec_if_set(state->backend, &state->init_vectors.kx_axis_vec);
    destroy_init_vec_if_set(state->backend, &state->init_vectors.ky_axis_vec);
    destroy_init_vec_if_set(state->backend, &state->init_vectors.t_axis_vec);
    destroy_init_vec_if_set(state->backend, &state->init_vectors.x_axis_vec);
    destroy_init_vec_if_set(state->backend, &state->init_vectors.y_axis_vec);
}

#ifndef DEFAULT_C3
#define DEFAULT_C3 0.0
#endif

#ifndef DEFAULT_RAMAN_TAU1
#define DEFAULT_RAMAN_TAU1 0.0122
#endif

#ifndef DEFAULT_RAMAN_TAU2
#define DEFAULT_RAMAN_TAU2 0.0320
#endif
static vec_status create_complex_vec(vector_backend* backend, size_t length, vec_buffer** out_vec)
{
    return vec_create(backend, VEC_KIND_COMPLEX64, length, out_vec);
}

static int operator_program_uses_opcode(
    const operator_program* program,
    operator_opcode opcode
)
{
    if (program == NULL || !program->active) {
        return 0;
    }

    for (size_t i = 0u; i < program->instruction_count; ++i) {
        if (program->instructions[i].opcode == opcode) {
            return 1;
        }
    }
    return 0;
}

static void warn_legacy_nonlinear_expression(
    const char* nonlinear_expr,
    const operator_program* nonlinear_program
)
{
    if (nonlinear_expr == NULL || nonlinear_expr[0] == '\0' || nonlinear_program == NULL) {
        return;
    }

    if (operator_program_uses_opcode(nonlinear_program, OPERATOR_OP_PUSH_SYMBOL_A)) {
        return;
    }
    if (!operator_program_uses_opcode(nonlinear_program, OPERATOR_OP_PUSH_SYMBOL_I) &&
        !operator_program_uses_opcode(nonlinear_program, OPERATOR_OP_PUSH_SYMBOL_V)) {
        return;
    }

    log_emit(LOG_LEVEL_WARN,
                 "[nlolib] nonlinear expression does not reference 'A'. "
                 "Nonlinear expressions now represent full RHS N(A), so legacy multiplier forms "
                 "must include A (for example: i*gamma*A*I). expression='%s'",
                 nonlinear_expr);
}

static double expected_omega_unshifted(size_t index, size_t num_time_samples, double omega_step)
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

static double resolve_delta_time(const sim_config* config, size_t num_time_samples)
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

static void fill_default_omega_grid(nlo_complex* out_grid, size_t num_time_samples, double delta_time)
{
    if (out_grid == NULL || num_time_samples == 0u) {
        return;
    }

    const double safe_delta_time = (delta_time > 0.0) ? delta_time : 1.0;
    const double omega_step = TWO_PI / ((double)num_time_samples * safe_delta_time);

    for (size_t i = 0u; i < num_time_samples; ++i) {
        out_grid[i] = make(expected_omega_unshifted(i, num_time_samples, omega_step), 0.0);
    }
}

static int expand_temporal_grid_to_volume(
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

static int frequency_grid_matches_expected_unshifted(
    const nlo_complex* grid,
    size_t num_time_samples,
    double delta_time
)
{
    if (grid == NULL || num_time_samples == 0u) {
        return 0;
    }

    const double safe_delta_time = (delta_time > 0.0) ? delta_time : 1.0;
    const double omega_step = TWO_PI / ((double)num_time_samples * safe_delta_time);

    for (size_t i = 0u; i < num_time_samples; ++i) {
        const double expected_real = expected_omega_unshifted(i, num_time_samples, omega_step);
        const double real_tol = FREQ_GRID_REL_TOL * fmax(1.0, fabs(expected_real));
        if (fabs(grid[i].re - expected_real) > real_tol) {
            return 0;
        }
        if (fabs(grid[i].im) > real_tol) {
            return 0;
        }
    }

    return 1;
}

static const char* resolve_operator_expr(const char* expr, const char* fallback)
{
    if (expr != NULL && expr[0] != '\0') {
        return expr;
    }

    return fallback;
}

static size_t resolve_runtime_constants(const runtime_operator_params* runtime, double out_constants[16])
{
    if (out_constants == NULL) {
        return 0u;
    }

    for (size_t i = 0u; i < RUNTIME_OPERATOR_CONSTANTS_MAX; ++i) {
        out_constants[i] = 0.0;
    }
    out_constants[0] = DEFAULT_C0;
    out_constants[1] = DEFAULT_C1;
    out_constants[2] = DEFAULT_C2;
    out_constants[3] = DEFAULT_C3;

    size_t count = 4u;
    if (runtime == NULL) {
        return count;
    }

    const size_t runtime_count = runtime->num_constants;
    if (runtime_count > RUNTIME_OPERATOR_CONSTANTS_MAX) {
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

static vec_status prepare_raman_response_host(
    const simulation_state* state,
    nlo_complex* out_response
)
{
    if (state == NULL || state->config == NULL || out_response == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    const runtime_operator_params* runtime = &state->config->runtime;
    const size_t n = state->num_time_samples;
    if (n == 0u) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    if (runtime->raman_response_time != NULL && runtime->raman_response_len > 0u) {
        if (runtime->raman_response_len != n) {
            log_emit(
                LOG_LEVEL_ERROR,
                "[nlolib] Raman response length mismatch: expected=%zu got=%zu",
                n,
                runtime->raman_response_len
            );
            return VEC_STATUS_INVALID_ARGUMENT;
        }
        memcpy(out_response, runtime->raman_response_time, n * sizeof(nlo_complex));
        return VEC_STATUS_OK;
    }

    const double dt = resolve_delta_time(state->config, state->nt);
    if (!(dt > 0.0)) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib] Raman model requires positive delta_time.");
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    const double tau1 = (runtime->raman_tau1 > 0.0) ? runtime->raman_tau1 : DEFAULT_RAMAN_TAU1;
    const double tau2 = (runtime->raman_tau2 > 0.0) ? runtime->raman_tau2 : DEFAULT_RAMAN_TAU2;
    if (!(tau1 > 0.0) || !(tau2 > 0.0)) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib] Raman tau parameters must be > 0.");
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    const double coef = (tau1 * tau1 + tau2 * tau2) / (tau1 * tau2 * tau2);
    double area = 0.0;
    for (size_t i = 0u; i < n; ++i) {
        const double t = (double)i * dt;
        const double val = coef * exp(-t / tau2) * sin(t / tau1);
        out_response[i] = make(val, 0.0);
        area += val;
    }
    area *= dt;
    if (!(area > 0.0) || !isfinite(area)) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib] Raman response normalization failed (area=%g).", area);
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    const double inv_area = 1.0 / area;
    for (size_t i = 0u; i < n; ++i) {
        out_response[i].re *= inv_area;
        out_response[i].im = 0.0;
    }
    return VEC_STATUS_OK;
}

static vec_status prepare_raman_state(simulation_state* state)
{
    if (state == NULL || state->backend == NULL || state->fft_plan == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (!state->nonlinear_raman_active) {
        return VEC_STATUS_OK;
    }

    nlo_complex* response_host = (nlo_complex*)malloc(state->num_time_samples * sizeof(nlo_complex));
    if (response_host == NULL) {
        return VEC_STATUS_ALLOCATION_FAILED;
    }

    vec_status status = prepare_raman_response_host(state, response_host);
    if (status == VEC_STATUS_OK) {
        status = vec_upload(
            state->backend,
            state->working_vectors.raman_mix_vec,
            response_host,
            state->num_time_samples * sizeof(nlo_complex)
        );
    }
    free(response_host);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    status = fft_forward_vec(
        state->fft_plan,
        state->working_vectors.raman_mix_vec,
        state->working_vectors.raman_response_fft_vec
    );
    if (status != VEC_STATUS_OK) {
        return status;
    }

    status = vec_complex_copy(
        state->backend,
        state->working_vectors.raman_derivative_factor_vec,
        state->frequency_grid_vec
    );
    if (status != VEC_STATUS_OK) {
        return status;
    }
    status = vec_complex_scalar_mul_inplace(
        state->backend,
        state->working_vectors.raman_derivative_factor_vec,
        make(0.0, 1.0)
    );
    if (status != VEC_STATUS_OK) {
        return status;
    }

    return VEC_STATUS_OK;
}

static void state_debug_log_backend_memory_stage(const simulation_state* state, const char* stage)
{
    if (state == NULL || state->backend == NULL) {
        state_debug_log_memory_checkpoint(stage, VEC_STATUS_INVALID_ARGUMENT, 0u, 0u, 0u);
        return;
    }

    vec_backend_memory_info mem_info = {0};
    const vec_status status = vec_query_memory_info(state->backend, &mem_info);
    state_debug_log_memory_checkpoint(stage,
                                          status,
                                          mem_info.device_local_total_bytes,
                                          mem_info.device_local_available_bytes,
                                          mem_info.max_storage_buffer_range);
}

static size_t count_full_volume_vectors(const simulation_state* state)
{
    if (state == NULL) {
        return 0u;
    }

    size_t count = 0u;
    if (state->current_field_vec != NULL) {
        count += 1u;
    }
    if (state->frequency_grid_vec != NULL) {
        count += 1u;
    }

    if (state->working_vectors.ip_field_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.field_working_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.field_freq_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.k_final_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.k_temp_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.dispersion_factor_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.dispersion_operator_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.potential_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.previous_field_vec != NULL) {
        count += 1u;
    }

    if (state->working_vectors.raman_intensity_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_delayed_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_spectrum_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_mix_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_polarization_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_derivative_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_response_fft_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.raman_derivative_factor_vec != NULL) {
        count += 1u;
    }

    if (state->working_vectors.wt_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.kx_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.ky_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.t_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.x_mesh_vec != NULL) {
        count += 1u;
    }
    if (state->working_vectors.y_mesh_vec != NULL) {
        count += 1u;
    }

    if (state->runtime_operator_stack_slots > SIZE_MAX - count) {
        return SIZE_MAX;
    }
    count += state->runtime_operator_stack_slots;

    if (state->fft_plan != NULL && vector_backend_get_type(state->backend) == VECTOR_BACKEND_VULKAN) {
        if (count == SIZE_MAX) {
            return SIZE_MAX;
        }
        count += 1u;
    }

    return count;
}

static int estimate_active_device_bytes(
    const simulation_state* state,
    size_t per_record_bytes,
    size_t* out_active_bytes
)
{
    if (out_active_bytes == NULL || state == NULL || per_record_bytes == 0u) {
        return -1;
    }

    const size_t full_volume_count = count_full_volume_vectors(state);
    if (full_volume_count == SIZE_MAX) {
        return -1;
    }

    size_t active_bytes = 0u;
    size_t full_volume_bytes = 0u;
    if (checked_mul_size_t(full_volume_count, per_record_bytes, &full_volume_bytes) != 0) {
        return -1;
    }
    active_bytes += full_volume_bytes;

    *out_active_bytes = active_bytes;
    return 0;
}

static size_t compute_device_ring_capacity(const simulation_state* state)
{
    if (state == NULL || state->backend == NULL || state->num_recorded_samples <= 1u) {
        return 0u;
    }

    if (vector_backend_get_type(state->backend) != VECTOR_BACKEND_VULKAN) {
        return 0u;
    }

    const double frac = (state->exec_options.device_heap_fraction > 0.0 &&
                         state->exec_options.device_heap_fraction <= 1.0)
                            ? state->exec_options.device_heap_fraction
                            : DEFAULT_DEVICE_HEAP_FRACTION;

    vec_backend_memory_info mem_info = {0};
    if (vec_query_memory_info(state->backend, &mem_info) != VEC_STATUS_OK) {
        return 0u;
    }

    const size_t per_record_bytes = state->num_time_samples * sizeof(nlo_complex);
    if (per_record_bytes == 0u) {
        return 0u;
    }

    size_t budget_bytes = state->exec_options.forced_device_budget_bytes;
    if (budget_bytes > 0u) {
        if (mem_info.device_local_available_bytes > 0u &&
            budget_bytes > mem_info.device_local_available_bytes) {
            budget_bytes = mem_info.device_local_available_bytes;
        }
    } else {
        budget_bytes = (size_t)((double)mem_info.device_local_available_bytes * frac);
    }
    budget_bytes =
        (budget_bytes / DEVICE_RING_BUDGET_HEADROOM_DEN) *
        DEVICE_RING_BUDGET_HEADROOM_NUM;
    if (budget_bytes == 0u) {
        return 0u;
    }

    size_t active_bytes = 0u;
    if (estimate_active_device_bytes(state, per_record_bytes, &active_bytes) != 0) {
        return 0u;
    }

    size_t conservative_vec_count = 2u + WORK_VECTOR_COUNT;
    if (state->working_vectors.wt_mesh_vec != NULL) {
        conservative_vec_count += 1u;
    }
    if (state->working_vectors.kx_mesh_vec != NULL) {
        conservative_vec_count += 1u;
    }
    if (state->working_vectors.ky_mesh_vec != NULL) {
        conservative_vec_count += 1u;
    }
    if (state->working_vectors.t_mesh_vec != NULL) {
        conservative_vec_count += 1u;
    }
    if (state->working_vectors.x_mesh_vec != NULL) {
        conservative_vec_count += 1u;
    }
    if (state->working_vectors.y_mesh_vec != NULL) {
        conservative_vec_count += 1u;
    }
    if (state->runtime_operator_stack_slots > SIZE_MAX - conservative_vec_count) {
        return 0u;
    }
    conservative_vec_count += state->runtime_operator_stack_slots;
    if (state->fft_plan != NULL) {
        if (conservative_vec_count == SIZE_MAX) {
            return 0u;
        }
        conservative_vec_count += 1u;
    }

    size_t conservative_active_bytes = 0u;
    if (checked_mul_size_t(conservative_vec_count, per_record_bytes, &conservative_active_bytes) == 0 &&
        conservative_active_bytes > active_bytes) {
        active_bytes = conservative_active_bytes;
    }

    size_t ring_capacity = budget_bytes / per_record_bytes;

    {
        const size_t ring_target = (state->exec_options.record_ring_target > 0u)
                                       ? state->exec_options.record_ring_target
                                       : (size_t)MAX_DEVICE_RING_CAPACITY;
        if (ring_capacity > ring_target) {
            ring_capacity = ring_target;
        }
    }

    size_t reserve_bytes = per_record_bytes;
    if (reserve_bytes < (size_t)DEVICE_RING_SAFETY_MIN_BYTES) {
        reserve_bytes = (size_t)DEVICE_RING_SAFETY_MIN_BYTES;
    }
    if (mem_info.device_local_total_bytes > 0u) {
        const size_t five_percent = mem_info.device_local_total_bytes / (size_t)DEVICE_RING_SAFETY_TOTAL_DEN;
        if (reserve_bytes < five_percent) {
            reserve_bytes = five_percent;
        }
    }

    size_t safe_capacity = 0u;
    if (mem_info.device_local_available_bytes > reserve_bytes) {
        safe_capacity = (mem_info.device_local_available_bytes - reserve_bytes) / per_record_bytes;
    }
    if (ring_capacity > safe_capacity) {
        ring_capacity = safe_capacity;
    }

    state_debug_log_ring_capacity(state->num_recorded_samples,
                                      per_record_bytes,
                                      active_bytes,
                                      state->runtime_operator_stack_slots,
                                      budget_bytes,
                                      ring_capacity);

    return ring_capacity;
}


static int storage_options_enabled(const storage_options* options)
{
    return (options != NULL &&
            options->sqlite_path != NULL &&
            options->sqlite_path[0] != '\0');
}
simulation_state* create_simulation_state(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const execution_options* exec_options
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
    const execution_options* exec_options,
    const storage_options* storage_options
)
{
    if (config == NULL || exec_options == NULL ||
        num_time_samples == 0 || num_recorded_samples == 0) {
        return NULL;
    }
    if (config->runtime.num_constants > RUNTIME_OPERATOR_CONSTANTS_MAX) {
        return NULL;
    }

    size_t resolved_nt = 0u;
    size_t spatial_nx = 0u;
    size_t spatial_ny = 0u;
    int explicit_nd = 0;
    int tensor_mode_active = 0;
    if (config->tensor.nt > 0u) {
        if (config->tensor.layout != TENSOR_LAYOUT_XYT_T_FAST ||
            config->tensor.nx == 0u ||
            config->tensor.ny == 0u) {
            return NULL;
        }
        size_t ntx = 0u;
        size_t resolved_total = 0u;
        if (checked_mul_size_t(config->tensor.nt, config->tensor.nx, &ntx) != 0 ||
            checked_mul_size_t(ntx, config->tensor.ny, &resolved_total) != 0 ||
            resolved_total != num_time_samples) {
            return NULL;
        }
        resolved_nt = config->tensor.nt;
        spatial_nx = config->tensor.nx;
        spatial_ny = config->tensor.ny;
        explicit_nd = 1;
        tensor_mode_active = 1;
    } else if (resolve_sim_dimensions_internal(config,
                                                   num_time_samples,
                                                   &resolved_nt,
                                                   &spatial_nx,
                                                   &spatial_ny,
                                                   &explicit_nd) != 0) {
        return NULL;
    }

    if (!tensor_mode_active && (spatial_nx > 1u || spatial_ny > 1u)) {
        log_emit(
            LOG_LEVEL_ERROR,
            "[nlolib] Coupled transverse runs require tensor descriptors (tensor.nt/tensor.nx/tensor.ny)."
        );
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
    state->tensor_layout = tensor_mode_active ? config->tensor.layout : TENSOR_LAYOUT_XYT_T_FAST;
    state->tensor_mode_active = tensor_mode_active;
    state->num_time_samples = num_time_samples;
    state->num_points_xy = spatial_nx * spatial_ny;
    state->num_recorded_samples = num_recorded_samples;
    const int coupled_mode = (spatial_nx > 1u || spatial_ny > 1u);
    const int storage_enabled = storage_options_enabled(storage_options);
    state->num_host_records = 0u;

    state->current_z = 0.0;
    state->current_step_size = config->propagation.starting_step_size;
    state->current_half_step_exp = 0.5 * state->current_step_size;
    state->dispersion_valid = 0;
    state->runtime_operator_stack_slots = 0u;
    state->snapshot_status = VEC_STATUS_OK;
    state->nonlinear_model = NONLINEAR_MODEL_EXPR;
    state->nonlinear_raman_active = 0;
    state->nonlinear_shock_active = 0;
    state->nonlinear_gamma = 0.0;
    state->raman_fraction = 0.0;
    state->shock_omega0 = 0.0;

    if (config->runtime.nonlinear_model == NONLINEAR_MODEL_KERR_RAMAN) {
        if (coupled_mode) {
            log_emit(
                LOG_LEVEL_ERROR,
                "[nlolib] nonlinear_model=kerr_raman is currently supported for temporal-only runs."
            );
            free_simulation_state(state);
            return NULL;
        }
        if (!isfinite(config->runtime.nonlinear_gamma)) {
            state_debug_log_failure("validate_raman_gamma", VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }
        if (!(config->runtime.raman_fraction >= 0.0) || !(config->runtime.raman_fraction <= 1.0)) {
            state_debug_log_failure("validate_raman_fraction", VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }
        if (config->runtime.shock_omega0 < 0.0 || !isfinite(config->runtime.shock_omega0)) {
            state_debug_log_failure("validate_shock_omega0", VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }
        state->nonlinear_model = NONLINEAR_MODEL_KERR_RAMAN;
        state->nonlinear_raman_active = 1;
        state->nonlinear_gamma = config->runtime.nonlinear_gamma;
        state->raman_fraction = config->runtime.raman_fraction;
        state->shock_omega0 = config->runtime.shock_omega0;
        state->nonlinear_shock_active = (config->runtime.shock_omega0 > 0.0) ? 1 : 0;
    }

    if (storage_enabled) {
        if (!snapshot_store_is_available()) {
            state_debug_log_failure("snapshot_store_unavailable", VEC_STATUS_UNSUPPORTED);
            free_simulation_state(state);
            return NULL;
        }

        snapshot_store_open_params store_params;
        memset(&store_params, 0, sizeof(store_params));
        store_params.config = config;
        store_params.exec_options = exec_options;
        store_params.storage_options = storage_options;
        store_params.num_time_samples = num_time_samples;
        store_params.num_recorded_samples = num_recorded_samples;
        state->snapshot_store = snapshot_store_open(&store_params);
        if (state->snapshot_store == NULL) {
            state_debug_log_failure("snapshot_store_open", VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
        snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
    }

    if (num_recorded_samples > 1u || storage_enabled) {
        state->snapshot_scratch_record = (nlo_complex*)calloc(state->num_time_samples, sizeof(nlo_complex));
        if (state->snapshot_scratch_record == NULL) {
            state_debug_log_failure("allocate_snapshot_scratch_record", VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
    }

    if (exec_options->backend_type == VECTOR_BACKEND_CPU) {
        state->backend = vector_backend_create_cpu();
    }
    else if (exec_options->backend_type == VECTOR_BACKEND_VULKAN) {
        state->backend = vector_backend_create_vulkan(&exec_options->vulkan);
    }
    else if (exec_options->backend_type == VECTOR_BACKEND_AUTO) {
        state->backend = vector_backend_create_vulkan(NULL);
    }
    else {
        state->backend = NULL;
    }

    if (state->backend == NULL) {
        state_debug_log_failure("create_backend", VEC_STATUS_BACKEND_UNAVAILABLE);
        free_simulation_state(state);
        return NULL;
    }
    state_debug_log_backend_memory_stage(state, "backend_created");

    if (create_complex_vec(state->backend, num_time_samples, &state->current_field_vec) != VEC_STATUS_OK ||
        (!state->tensor_mode_active &&
         create_complex_vec(state->backend, num_time_samples, &state->frequency_grid_vec) != VEC_STATUS_OK) ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.ip_field_vec) != VEC_STATUS_OK ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.field_working_vec) != VEC_STATUS_OK ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.field_freq_vec) != VEC_STATUS_OK ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.k_final_vec) != VEC_STATUS_OK ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.k_temp_vec) != VEC_STATUS_OK ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.dispersion_factor_vec) != VEC_STATUS_OK ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.dispersion_operator_vec) != VEC_STATUS_OK ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.potential_vec) != VEC_STATUS_OK ||
        create_complex_vec(state->backend, num_time_samples, &state->working_vectors.previous_field_vec) != VEC_STATUS_OK) {
        state_debug_log_failure("allocate_device_vectors", VEC_STATUS_ALLOCATION_FAILED);
        free_simulation_state(state);
        return NULL;
    }
    state_debug_log_backend_memory_stage(state, "base_vectors_allocated");
    if (state->nonlinear_raman_active) {
        if (create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_intensity_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_delayed_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_spectrum_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_mix_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_polarization_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_derivative_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_response_fft_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_derivative_factor_vec) != VEC_STATUS_OK) {
            state_debug_log_failure("allocate_raman_vectors", VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
        state_debug_log_backend_memory_stage(state, "raman_vectors_allocated");
    }

    if (state->tensor_mode_active) {
        if (create_complex_vec(state->backend, state->nt, &state->init_vectors.wt_axis_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, state->nx, &state->init_vectors.kx_axis_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, state->ny, &state->init_vectors.ky_axis_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, state->nt, &state->init_vectors.t_axis_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, state->nx, &state->init_vectors.x_axis_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, state->ny, &state->init_vectors.y_axis_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.wt_mesh_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.kx_mesh_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.ky_mesh_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.t_mesh_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.x_mesh_vec) != VEC_STATUS_OK ||
            create_complex_vec(state->backend, num_time_samples, &state->working_vectors.y_mesh_vec) != VEC_STATUS_OK) {
            state_debug_log_failure("allocate_tensor_mesh_vectors", VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
        state_debug_log_backend_memory_stage(state, "tensor_vectors_allocated");
    }

    vec_status status = vec_complex_fill(state->backend, state->current_field_vec, make(0.0, 0.0));
    if (status != VEC_STATUS_OK) {
        state_debug_log_failure("clear_current_field", status);
        free_simulation_state(state);
        return NULL;
    }

    const double delta_time = resolve_delta_time(config, state->nt);
    if (state->tensor_mode_active) {
        const double safe_dt = (delta_time > 0.0) ? delta_time : 1.0;
        const double safe_dx = (config->spatial.delta_x > 0.0) ? config->spatial.delta_x : 1.0;
        const double safe_dy = (config->spatial.delta_y > 0.0) ? config->spatial.delta_y : 1.0;

        if (config->time.wt_axis != NULL) {
            status = vec_upload(state->backend,
                                    state->init_vectors.wt_axis_vec,
                                    config->time.wt_axis,
                                    state->nt * sizeof(nlo_complex));
        } else {
            status = vec_complex_axis_unshifted_from_delta(state->backend,
                                                               state->init_vectors.wt_axis_vec,
                                                               safe_dt);
        }
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("prepare_tensor_wt_axis", status);
            free_simulation_state(state);
            return NULL;
        }

        if (config->spatial.kx_axis != NULL) {
            status = vec_upload(state->backend,
                                    state->init_vectors.kx_axis_vec,
                                    config->spatial.kx_axis,
                                    state->nx * sizeof(nlo_complex));
        } else {
            status = vec_complex_axis_unshifted_from_delta(state->backend,
                                                               state->init_vectors.kx_axis_vec,
                                                               safe_dx);
        }
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("prepare_tensor_kx_axis", status);
            free_simulation_state(state);
            return NULL;
        }

        if (config->spatial.ky_axis != NULL) {
            status = vec_upload(state->backend,
                                    state->init_vectors.ky_axis_vec,
                                    config->spatial.ky_axis,
                                    state->ny * sizeof(nlo_complex));
        } else {
            status = vec_complex_axis_unshifted_from_delta(state->backend,
                                                               state->init_vectors.ky_axis_vec,
                                                               safe_dy);
        }
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("prepare_tensor_ky_axis", status);
            free_simulation_state(state);
            return NULL;
        }

        status = vec_complex_axis_centered_from_delta(state->backend,
                                                          state->init_vectors.t_axis_vec,
                                                          safe_dt);
        if (status == VEC_STATUS_OK) {
            status = vec_complex_axis_centered_from_delta(state->backend,
                                                              state->init_vectors.x_axis_vec,
                                                              safe_dx);
        }
        if (status == VEC_STATUS_OK) {
            status = vec_complex_axis_centered_from_delta(state->backend,
                                                              state->init_vectors.y_axis_vec,
                                                              safe_dy);
        }
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("prepare_tensor_physical_axes", status);
            free_simulation_state(state);
            return NULL;
        }

        status = vec_complex_mesh_from_axis_tfast(state->backend,
                                                      state->working_vectors.wt_mesh_vec,
                                                      state->init_vectors.wt_axis_vec,
                                                      state->nt,
                                                      state->ny,
                                                      VEC_MESH_AXIS_T);
        if (status == VEC_STATUS_OK) {
            status = vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.kx_mesh_vec,
                                                          state->init_vectors.kx_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          VEC_MESH_AXIS_X);
        }
        if (status == VEC_STATUS_OK) {
            status = vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.ky_mesh_vec,
                                                          state->init_vectors.ky_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          VEC_MESH_AXIS_Y);
        }
        if (status == VEC_STATUS_OK) {
            status = vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.t_mesh_vec,
                                                          state->init_vectors.t_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          VEC_MESH_AXIS_T);
        }
        if (status == VEC_STATUS_OK) {
            status = vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.x_mesh_vec,
                                                          state->init_vectors.x_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          VEC_MESH_AXIS_X);
        }
        if (status == VEC_STATUS_OK) {
            status = vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.y_mesh_vec,
                                                          state->init_vectors.y_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          VEC_MESH_AXIS_Y);
        }
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("prepare_tensor_mesh_axes", status);
            free_simulation_state(state);
            return NULL;
        }

        release_init_vectors(state);
    } else if (explicit_nd != 0) {
        nlo_complex* temporal_line = (nlo_complex*)malloc(state->nt * sizeof(nlo_complex));
        nlo_complex* temporal_volume = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
        if (temporal_line == NULL || temporal_volume == NULL) {
            free(temporal_line);
            free(temporal_volume);
            state_debug_log_failure("allocate_temporal_frequency_grid", VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }

        if (config->frequency.frequency_grid != NULL) {
            memcpy(temporal_line, config->frequency.frequency_grid, state->nt * sizeof(nlo_complex));
        } else {
            fill_default_omega_grid(temporal_line, state->nt, delta_time);
        }

        if (expand_temporal_grid_to_volume(temporal_volume,
                                               temporal_line,
                                               state->nt,
                                               state->nx,
                                               state->ny) != 0) {
            free(temporal_line);
            free(temporal_volume);
            state_debug_log_failure("expand_temporal_frequency_grid", VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }

        status = vec_upload(state->backend,
                                state->frequency_grid_vec,
                                temporal_volume,
                                num_time_samples * sizeof(nlo_complex));
        free(temporal_line);
        free(temporal_volume);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("upload_frequency_grid", status);
            free_simulation_state(state);
            return NULL;
        }
    } else {
        const nlo_complex* spectral_grid_source = config->spatial.spatial_frequency_grid;
        nlo_complex* generated_frequency_grid = NULL;
        if (spectral_grid_source == NULL) {
            const nlo_complex* temporal_frequency_grid = config->frequency.frequency_grid;
            if (temporal_frequency_grid != NULL &&
                frequency_grid_matches_expected_unshifted(temporal_frequency_grid,
                                                              num_time_samples,
                                                              delta_time)) {
                spectral_grid_source = temporal_frequency_grid;
            } else {
                generated_frequency_grid = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
                if (generated_frequency_grid == NULL) {
                    state_debug_log_failure("allocate_generated_frequency_grid", VEC_STATUS_ALLOCATION_FAILED);
                    free_simulation_state(state);
                    return NULL;
                }

                fill_default_omega_grid(generated_frequency_grid, num_time_samples, delta_time);
                spectral_grid_source = generated_frequency_grid;
            }
        }

        if (spectral_grid_source != NULL) {
            status = vec_upload(state->backend,
                                    state->frequency_grid_vec,
                                    spectral_grid_source,
                                    num_time_samples * sizeof(nlo_complex));
        } else {
            status = vec_complex_fill(state->backend, state->frequency_grid_vec, make(0.0, 0.0));
        }
        if (generated_frequency_grid != NULL) {
            free(generated_frequency_grid);
            generated_frequency_grid = NULL;
        }
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("upload_frequency_grid", status);
            free_simulation_state(state);
            return NULL;
        }
    }

    if (!state->tensor_mode_active) {
        if (config->spatial.potential_grid != NULL) {
            if (explicit_nd != 0) {
                nlo_complex* potential_volume = (nlo_complex*)malloc(num_time_samples * sizeof(nlo_complex));
                if (potential_volume == NULL) {
                    state_debug_log_failure("allocate_potential_grid", VEC_STATUS_ALLOCATION_FAILED);
                    free_simulation_state(state);
                    return NULL;
                }
                for (size_t i = 0u; i < num_time_samples; ++i) {
                    potential_volume[i] = config->spatial.potential_grid[0];
                }
                status = vec_upload(state->backend,
                                        state->working_vectors.potential_vec,
                                        potential_volume,
                                        num_time_samples * sizeof(nlo_complex));
                free(potential_volume);
            } else {
                status = vec_upload(state->backend,
                                        state->working_vectors.potential_vec,
                                        config->spatial.potential_grid,
                                        num_time_samples * sizeof(nlo_complex));
            }
        } else {
            status = vec_complex_fill(state->backend,
                                          state->working_vectors.potential_vec,
                                          make(0.0, 0.0));
        }
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("upload_potential_grid", status);
            free_simulation_state(state);
            return NULL;
        }
    } else {
        if (config->spatial.potential_grid != NULL) {
            status = vec_upload(state->backend,
                                    state->working_vectors.potential_vec,
                                    config->spatial.potential_grid,
                                    num_time_samples * sizeof(nlo_complex));
        } else {
            status = vec_complex_fill(state->backend,
                                          state->working_vectors.potential_vec,
                                          make(0.0, 0.0));
        }
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("upload_potential_grid_tensor", status);
            free_simulation_state(state);
            return NULL;
        }
    }

    double resolved_runtime_constants[RUNTIME_OPERATOR_CONSTANTS_MAX];
    const double* runtime_constants = resolved_runtime_constants;
    size_t runtime_constant_count = resolve_runtime_constants(&config->runtime,
                                                                  resolved_runtime_constants);
    if (runtime_constant_count == 0u ||
        runtime_constant_count > RUNTIME_OPERATOR_CONSTANTS_MAX) {
        state_debug_log_failure("resolve_runtime_constants", VEC_STATUS_INVALID_ARGUMENT);
        free_simulation_state(state);
        return NULL;
    }

    const char* linear_factor_expr = resolve_operator_expr(config->runtime.linear_factor_expr,
                                                               resolve_operator_expr(
                                                                   config->runtime.dispersion_factor_expr,
                                                                   DEFAULT_LINEAR_FACTOR_EXPR));
    const char* linear_expr = resolve_operator_expr(config->runtime.linear_expr,
                                                        resolve_operator_expr(
                                                            config->runtime.dispersion_expr,
                                                            DEFAULT_LINEAR_EXPR));
    const char* potential_expr = resolve_operator_expr(config->runtime.potential_expr,
                                                           DEFAULT_POTENTIAL_EXPR);
    const char* dispersion_factor_expr = resolve_operator_expr(config->runtime.dispersion_factor_expr,
                                                                   DEFAULT_DISPERSION_FACTOR_EXPR);
    const char* dispersion_expr = resolve_operator_expr(config->runtime.dispersion_expr,
                                                            DEFAULT_DISPERSION_EXPR);
    const char* nonlinear_expr = resolve_operator_expr(config->runtime.nonlinear_expr,
                                                           DEFAULT_NONLINEAR_EXPR);

    if (state->tensor_mode_active) {
        status = operator_program_compile(potential_expr,
                                              OPERATOR_CONTEXT_POTENTIAL,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->potential_operator_program);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("compile_potential_program", status);
            free_simulation_state(state);
            return NULL;
        }

        status = operator_program_compile(linear_factor_expr,
                                              OPERATOR_CONTEXT_LINEAR_FACTOR,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->linear_factor_operator_program);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("compile_linear_factor_program", status);
            free_simulation_state(state);
            return NULL;
        }

        status = operator_program_compile(linear_expr,
                                              OPERATOR_CONTEXT_LINEAR,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->linear_operator_program);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("compile_linear_program", status);
            free_simulation_state(state);
            return NULL;
        }
    } else {
        status = operator_program_compile(dispersion_factor_expr,
                                              OPERATOR_CONTEXT_DISPERSION_FACTOR,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->dispersion_factor_operator_program);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("compile_dispersion_factor_program", status);
            free_simulation_state(state);
            return NULL;
        }

        status = operator_program_compile(dispersion_expr,
                                              OPERATOR_CONTEXT_DISPERSION,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->dispersion_operator_program);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("compile_dispersion_program", status);
            free_simulation_state(state);
            return NULL;
        }
    }

    status = operator_program_compile(nonlinear_expr,
                                          OPERATOR_CONTEXT_NONLINEAR,
                                          runtime_constant_count,
                                          runtime_constants,
                                          &state->nonlinear_operator_program);
    if (status != VEC_STATUS_OK) {
        state_debug_log_failure("compile_nonlinear_program", status);
        free_simulation_state(state);
        return NULL;
    }
    warn_legacy_nonlinear_expression(nonlinear_expr, &state->nonlinear_operator_program);

    size_t required_stack_slots = state->nonlinear_operator_program.required_stack_slots;
    if (state->tensor_mode_active) {
        if (state->potential_operator_program.required_stack_slots > required_stack_slots) {
            required_stack_slots = state->potential_operator_program.required_stack_slots;
        }
        if (state->linear_factor_operator_program.required_stack_slots > required_stack_slots) {
            required_stack_slots = state->linear_factor_operator_program.required_stack_slots;
        }
        if (state->linear_operator_program.required_stack_slots > required_stack_slots) {
            required_stack_slots = state->linear_operator_program.required_stack_slots;
        }
    } else {
        if (state->dispersion_factor_operator_program.required_stack_slots > required_stack_slots) {
            required_stack_slots = state->dispersion_factor_operator_program.required_stack_slots;
        }
        if (state->dispersion_operator_program.required_stack_slots > required_stack_slots) {
            required_stack_slots = state->dispersion_operator_program.required_stack_slots;
        }
    }

    if (required_stack_slots == 0u ||
        required_stack_slots > OPERATOR_PROGRAM_MAX_STACK_SLOTS) {
        state_debug_log_failure("resolve_runtime_stack_slots", VEC_STATUS_INVALID_ARGUMENT);
        free_simulation_state(state);
        return NULL;
    }

    state->runtime_operator_stack_slots = required_stack_slots;
    for (size_t i = 0u; i < state->runtime_operator_stack_slots; ++i) {
        if (create_complex_vec(state->backend,
                                   num_time_samples,
                                   &state->runtime_operator_stack_vec[i]) != VEC_STATUS_OK) {
            state_debug_log_failure("allocate_runtime_stack_vectors", VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
    }
    state_debug_log_backend_memory_stage(state, "runtime_stack_allocated");

    if (state->tensor_mode_active) {
        if (config->spatial.potential_grid == NULL) {
            const operator_eval_context potential_eval_ctx = {
                .frequency_grid = state_operator_frequency_grid(state),
                .wt_grid = state->working_vectors.wt_mesh_vec,
                .kx_grid = state->working_vectors.kx_mesh_vec,
                .ky_grid = state->working_vectors.ky_mesh_vec,
                .t_grid = state->working_vectors.t_mesh_vec,
                .x_grid = state->working_vectors.x_mesh_vec,
                .y_grid = state->working_vectors.y_mesh_vec,
                .field = state->current_field_vec,
                .dispersion_factor = NULL,
                .potential = NULL,
                .half_step_size = state->current_half_step_exp
            };
            status = operator_program_execute(state->backend,
                                                  &state->potential_operator_program,
                                                  &potential_eval_ctx,
                                                  state->runtime_operator_stack_vec,
                                                  state->runtime_operator_stack_slots,
                                                  state->working_vectors.potential_vec);
            if (status != VEC_STATUS_OK) {
                state_debug_log_failure("execute_potential_program", status);
                free_simulation_state(state);
                return NULL;
            }
        }

        const operator_eval_context linear_factor_eval_ctx = {
            .frequency_grid = state_operator_frequency_grid(state),
            .wt_grid = state->working_vectors.wt_mesh_vec,
            .kx_grid = state->working_vectors.kx_mesh_vec,
            .ky_grid = state->working_vectors.ky_mesh_vec,
            .t_grid = state->working_vectors.t_mesh_vec,
            .x_grid = state->working_vectors.x_mesh_vec,
            .y_grid = state->working_vectors.y_mesh_vec,
            .field = state->current_field_vec,
            .dispersion_factor = NULL,
            .potential = state->working_vectors.potential_vec,
            .half_step_size = state->current_half_step_exp
        };
        status = operator_program_execute(state->backend,
                                              &state->linear_factor_operator_program,
                                              &linear_factor_eval_ctx,
                                              state->runtime_operator_stack_vec,
                                              state->runtime_operator_stack_slots,
                                              state->working_vectors.dispersion_factor_vec);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("execute_linear_factor_program", status);
            free_simulation_state(state);
            return NULL;
        }
    } else {
        const operator_eval_context dispersion_factor_eval_ctx = {
            .frequency_grid = state_operator_frequency_grid(state),
            .field = state->current_field_vec,
            .dispersion_factor = NULL,
            .potential = state->working_vectors.potential_vec,
            .half_step_size = state->current_half_step_exp
        };
        status = operator_program_execute(state->backend,
                                              &state->dispersion_factor_operator_program,
                                              &dispersion_factor_eval_ctx,
                                              state->runtime_operator_stack_vec,
                                              state->runtime_operator_stack_slots,
                                              state->working_vectors.dispersion_factor_vec);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("execute_dispersion_factor_program", status);
            free_simulation_state(state);
            return NULL;
        }
    }

    state->dispersion_valid = 1;

    vec_status fft_status = VEC_STATUS_UNSUPPORTED;
    if (explicit_nd != 0 && state->nt > 1u && state->nx > 1u && state->ny > 1u) {
        const fft_shape shape = {
            .rank = 3u,
            .dims = {state->nx, state->ny, state->nt}
        };
        fft_status = fft_plan_create_shaped_with_backend(state->backend,
                                                             &shape,
                                                             state->exec_options.fft_backend,
                                                             &state->fft_plan);
        if (fft_status != VEC_STATUS_OK) {
            const fft_shape fallback_shape = {
                .rank = 1u,
                .dims = {num_time_samples, 1u, 1u}
            };
            fft_status = fft_plan_create_shaped_with_backend(state->backend,
                                                                 &fallback_shape,
                                                                 state->exec_options.fft_backend,
                                                                 &state->fft_plan);
        }
    } else if (explicit_nd != 0 && state->nx > 1u && state->ny > 1u) {
        const fft_shape shape = {
            .rank = 2u,
            .dims = {state->nx, state->ny, 1u}
        };
        fft_status = fft_plan_create_shaped_with_backend(state->backend,
                                                             &shape,
                                                             state->exec_options.fft_backend,
                                                             &state->fft_plan);
        if (fft_status != VEC_STATUS_OK) {
            const fft_shape fallback_shape = {
                .rank = 1u,
                .dims = {num_time_samples, 1u, 1u}
            };
            fft_status = fft_plan_create_shaped_with_backend(state->backend,
                                                                 &fallback_shape,
                                                                 state->exec_options.fft_backend,
                                                                 &state->fft_plan);
        }
    } else {
        fft_status = fft_plan_create_with_backend(state->backend,
                                                      num_time_samples,
                                                      state->exec_options.fft_backend,
                                                      &state->fft_plan);
    }
    if (fft_status != VEC_STATUS_OK) {
        state_debug_log_failure("create_fft_plan", VEC_STATUS_BACKEND_UNAVAILABLE);
        free_simulation_state(state);
        return NULL;
    }
    state_debug_log_backend_memory_stage(state, "fft_plan_created");

    if (state->nonlinear_raman_active) {
        status = prepare_raman_state(state);
        if (status != VEC_STATUS_OK) {
            state_debug_log_failure("prepare_raman_state", status);
            free_simulation_state(state);
            return NULL;
        }
        state_debug_log_backend_memory_stage(state, "raman_state_prepared");
    }

    state_debug_log_backend_memory_stage(state, "pre_ring_allocation");
    state->record_ring_capacity = compute_device_ring_capacity(state);
    if (state->record_ring_capacity > 0u) {
        state->record_ring_vec = (vec_buffer**)calloc(state->record_ring_capacity, sizeof(vec_buffer*));
        if (state->record_ring_vec == NULL) {
            state_debug_log_failure("allocate_record_ring_array", VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }

        const size_t per_record_bytes = num_time_samples * sizeof(nlo_complex);
        vec_backend_memory_info ring_mem_info = {0};
        (void)vec_query_memory_info(state->backend, &ring_mem_info);
        size_t ring_reserve_bytes = per_record_bytes;
        if (ring_reserve_bytes < (size_t)DEVICE_RING_SAFETY_MIN_BYTES) {
            ring_reserve_bytes = (size_t)DEVICE_RING_SAFETY_MIN_BYTES;
        }
        if (ring_mem_info.device_local_total_bytes > 0u) {
            const size_t five_percent =
                ring_mem_info.device_local_total_bytes / (size_t)DEVICE_RING_SAFETY_TOTAL_DEN;
            if (ring_reserve_bytes < five_percent) {
                ring_reserve_bytes = five_percent;
            }
        }

        for (size_t i = 0; i < state->record_ring_capacity; ++i) {
            if ((i & 7u) == 0u) {
                vec_backend_memory_info live_mem_info = {0};
                if (vec_query_memory_info(state->backend, &live_mem_info) == VEC_STATUS_OK &&
                    live_mem_info.device_local_available_bytes > 0u &&
                    live_mem_info.device_local_available_bytes <= ring_reserve_bytes + per_record_bytes) {
                    log_emit(
                        LOG_LEVEL_WARN,
                        "[nlolib] record-ring allocation stopped at %zu/%zu records (available VRAM reached safety reserve).",
                        i,
                        state->record_ring_capacity);
                    state->record_ring_capacity = i;
                    break;
                }
            }

            if (create_complex_vec(state->backend, num_time_samples, &state->record_ring_vec[i]) != VEC_STATUS_OK) {
                log_emit(
                    LOG_LEVEL_WARN,
                    "[nlolib] record-ring allocation truncated at %zu/%zu records; continuing with reduced ring.",
                    i,
                    state->record_ring_capacity);
                state_debug_log_failure("allocate_record_ring_vectors", VEC_STATUS_ALLOCATION_FAILED);
                if (i == 0u) {
                    free(state->record_ring_vec);
                    state->record_ring_vec = NULL;
                    state->record_ring_capacity = 0u;
                } else {
                    state->record_ring_capacity = i;
                }
                break;
            }
        }
    }
    state_debug_log_backend_memory_stage(state, "post_ring_allocation");

    return state;
}
