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

#ifndef NLO_DEFAULT_DISPERSION_FACTOR_EXPR
#define NLO_DEFAULT_DISPERSION_FACTOR_EXPR "i*c0*w*w-c1"
#endif

#ifndef NLO_DEFAULT_LINEAR_FACTOR_EXPR
#define NLO_DEFAULT_LINEAR_FACTOR_EXPR "i*c0*wt*wt-c1"
#endif

#ifndef NLO_DEFAULT_DISPERSION_EXPR
#define NLO_DEFAULT_DISPERSION_EXPR "exp(h*D)"
#endif

#ifndef NLO_DEFAULT_LINEAR_EXPR
#define NLO_DEFAULT_LINEAR_EXPR "exp(h*D)"
#endif

#ifndef NLO_DEFAULT_POTENTIAL_EXPR
#define NLO_DEFAULT_POTENTIAL_EXPR "0"
#endif

#ifndef NLO_DEFAULT_NONLINEAR_EXPR
#define NLO_DEFAULT_NONLINEAR_EXPR "i*A*(c2*I + V)"
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

#ifndef NLO_DEFAULT_RAMAN_TAU1
#define NLO_DEFAULT_RAMAN_TAU1 0.0122
#endif

#ifndef NLO_DEFAULT_RAMAN_TAU2
#define NLO_DEFAULT_RAMAN_TAU2 0.0320
#endif
static nlo_vec_status nlo_create_complex_vec(nlo_vector_backend* backend, size_t length, nlo_vec_buffer** out_vec)
{
    return nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, length, out_vec);
}

static int nlo_operator_program_uses_opcode(
    const nlo_operator_program* program,
    nlo_operator_opcode opcode
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

static void nlo_warn_legacy_nonlinear_expression(
    const char* nonlinear_expr,
    const nlo_operator_program* nonlinear_program
)
{
    if (nonlinear_expr == NULL || nonlinear_expr[0] == '\0' || nonlinear_program == NULL) {
        return;
    }

    if (nlo_operator_program_uses_opcode(nonlinear_program, NLO_OPERATOR_OP_PUSH_SYMBOL_A)) {
        return;
    }
    if (!nlo_operator_program_uses_opcode(nonlinear_program, NLO_OPERATOR_OP_PUSH_SYMBOL_I) &&
        !nlo_operator_program_uses_opcode(nonlinear_program, NLO_OPERATOR_OP_PUSH_SYMBOL_V)) {
        return;
    }

    nlo_log_emit(NLO_LOG_LEVEL_WARN,
                 "[nlolib] nonlinear expression does not reference 'A'. "
                 "Nonlinear expressions now represent full RHS N(A), so legacy multiplier forms "
                 "must include A (for example: i*gamma*A*I). expression='%s'",
                 nonlinear_expr);
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

static nlo_vec_status nlo_prepare_raman_response_host(
    const simulation_state* state,
    nlo_complex* out_response
)
{
    if (state == NULL || state->config == NULL || out_response == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    const runtime_operator_params* runtime = &state->config->runtime;
    const size_t n = state->num_time_samples;
    if (n == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (runtime->raman_response_time != NULL && runtime->raman_response_len > 0u) {
        if (runtime->raman_response_len != n) {
            nlo_log_emit(
                NLO_LOG_LEVEL_ERROR,
                "[nlolib] Raman response length mismatch: expected=%zu got=%zu",
                n,
                runtime->raman_response_len
            );
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }
        memcpy(out_response, runtime->raman_response_time, n * sizeof(nlo_complex));
        return NLO_VEC_STATUS_OK;
    }

    const double dt = nlo_resolve_delta_time(state->config, state->nt);
    if (!(dt > 0.0)) {
        nlo_log_emit(NLO_LOG_LEVEL_ERROR, "[nlolib] Raman model requires positive delta_time.");
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const double tau1 = (runtime->raman_tau1 > 0.0) ? runtime->raman_tau1 : NLO_DEFAULT_RAMAN_TAU1;
    const double tau2 = (runtime->raman_tau2 > 0.0) ? runtime->raman_tau2 : NLO_DEFAULT_RAMAN_TAU2;
    if (!(tau1 > 0.0) || !(tau2 > 0.0)) {
        nlo_log_emit(NLO_LOG_LEVEL_ERROR, "[nlolib] Raman tau parameters must be > 0.");
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const double coef = (tau1 * tau1 + tau2 * tau2) / (tau1 * tau2 * tau2);
    double area = 0.0;
    for (size_t i = 0u; i < n; ++i) {
        const double t = (double)i * dt;
        const double val = coef * exp(-t / tau2) * sin(t / tau1);
        out_response[i] = nlo_make(val, 0.0);
        area += val;
    }
    area *= dt;
    if (!(area > 0.0) || !isfinite(area)) {
        nlo_log_emit(NLO_LOG_LEVEL_ERROR, "[nlolib] Raman response normalization failed (area=%g).", area);
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    const double inv_area = 1.0 / area;
    for (size_t i = 0u; i < n; ++i) {
        out_response[i].re *= inv_area;
        out_response[i].im = 0.0;
    }
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_prepare_raman_state(simulation_state* state)
{
    if (state == NULL || state->backend == NULL || state->fft_plan == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (!state->nonlinear_raman_active) {
        return NLO_VEC_STATUS_OK;
    }

    nlo_complex* response_host = (nlo_complex*)malloc(state->num_time_samples * sizeof(nlo_complex));
    if (response_host == NULL) {
        return NLO_VEC_STATUS_ALLOCATION_FAILED;
    }

    nlo_vec_status status = nlo_prepare_raman_response_host(state, response_host);
    if (status == NLO_VEC_STATUS_OK) {
        status = nlo_vec_upload(
            state->backend,
            state->working_vectors.raman_mix_vec,
            response_host,
            state->num_time_samples * sizeof(nlo_complex)
        );
    }
    free(response_host);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_fft_forward_vec(
        state->fft_plan,
        state->working_vectors.raman_mix_vec,
        state->working_vectors.raman_response_fft_vec
    );
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vec_complex_copy(
        state->backend,
        state->working_vectors.raman_derivative_factor_vec,
        state->frequency_grid_vec
    );
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_vec_complex_scalar_mul_inplace(
        state->backend,
        state->working_vectors.raman_derivative_factor_vec,
        nlo_make(0.0, 1.0)
    );
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    return NLO_VEC_STATUS_OK;
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
    if (state->working_vectors.wt_axis_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.kx_axis_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.ky_axis_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.t_axis_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.x_axis_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.y_axis_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.wt_mesh_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.kx_mesh_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.ky_mesh_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.t_mesh_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.x_mesh_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->working_vectors.y_mesh_vec != NULL) {
        active_vec_count += 1u;
    }
    if (state->runtime_operator_stack_slots > SIZE_MAX - active_vec_count) {
        return NLO_MIN_DEVICE_RING_CAPACITY;
    }
    active_vec_count += state->runtime_operator_stack_slots;

    size_t active_bytes = 0u;
    if (nlo_checked_mul_size_t(active_vec_count, per_record_bytes, &active_bytes) != 0) {
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
        num_time_samples == 0 || num_recorded_samples == 0) {
        return NULL;
    }
    if (config->runtime.num_constants > NLO_RUNTIME_OPERATOR_CONSTANTS_MAX) {
        return NULL;
    }

    size_t resolved_nt = 0u;
    size_t spatial_nx = 0u;
    size_t spatial_ny = 0u;
    int explicit_nd = 0;
    int tensor_mode_active = 0;
    if (config->tensor.nt > 0u) {
        if (config->tensor.layout != NLO_TENSOR_LAYOUT_XYT_T_FAST ||
            config->tensor.nx == 0u ||
            config->tensor.ny == 0u) {
            return NULL;
        }
        size_t ntx = 0u;
        size_t resolved_total = 0u;
        if (nlo_checked_mul_size_t(config->tensor.nt, config->tensor.nx, &ntx) != 0 ||
            nlo_checked_mul_size_t(ntx, config->tensor.ny, &resolved_total) != 0 ||
            resolved_total != num_time_samples) {
            return NULL;
        }
        resolved_nt = config->tensor.nt;
        spatial_nx = config->tensor.nx;
        spatial_ny = config->tensor.ny;
        explicit_nd = 1;
        tensor_mode_active = 1;
    } else if (nlo_resolve_sim_dimensions_internal(config,
                                                   num_time_samples,
                                                   &resolved_nt,
                                                   &spatial_nx,
                                                   &spatial_ny,
                                                   &explicit_nd) != 0) {
        return NULL;
    }

    if (!tensor_mode_active && (spatial_nx > 1u || spatial_ny > 1u)) {
        nlo_log_emit(
            NLO_LOG_LEVEL_ERROR,
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
    state->tensor_layout = tensor_mode_active ? config->tensor.layout : NLO_TENSOR_LAYOUT_XYT_T_FAST;
    state->tensor_mode_active = tensor_mode_active;
    state->num_time_samples = num_time_samples;
    state->num_points_xy = spatial_nx * spatial_ny;
    state->num_recorded_samples = num_recorded_samples;
    const int coupled_mode = (spatial_nx > 1u || spatial_ny > 1u);
    state->num_host_records = nlo_compute_host_record_capacity(num_time_samples, num_recorded_samples);
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
    state->nonlinear_model = NLO_NONLINEAR_MODEL_EXPR;
    state->nonlinear_raman_active = 0;
    state->nonlinear_shock_active = 0;
    state->nonlinear_gamma = 0.0;
    state->raman_fraction = 0.0;
    state->shock_omega0 = 0.0;

    if (config->runtime.nonlinear_model == NLO_NONLINEAR_MODEL_KERR_RAMAN) {
        if (coupled_mode) {
            nlo_log_emit(
                NLO_LOG_LEVEL_ERROR,
                "[nlolib] nonlinear_model=kerr_raman is currently supported for temporal-only runs."
            );
            free_simulation_state(state);
            return NULL;
        }
        if (!isfinite(config->runtime.nonlinear_gamma)) {
            nlo_state_debug_log_failure("validate_raman_gamma", NLO_VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }
        if (!(config->runtime.raman_fraction >= 0.0) || !(config->runtime.raman_fraction <= 1.0)) {
            nlo_state_debug_log_failure("validate_raman_fraction", NLO_VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }
        if (config->runtime.shock_omega0 < 0.0 || !isfinite(config->runtime.shock_omega0)) {
            nlo_state_debug_log_failure("validate_shock_omega0", NLO_VEC_STATUS_INVALID_ARGUMENT);
            free_simulation_state(state);
            return NULL;
        }
        state->nonlinear_model = NLO_NONLINEAR_MODEL_KERR_RAMAN;
        state->nonlinear_raman_active = 1;
        state->nonlinear_gamma = config->runtime.nonlinear_gamma;
        state->raman_fraction = config->runtime.raman_fraction;
        state->shock_omega0 = config->runtime.shock_omega0;
        state->nonlinear_shock_active = (config->runtime.shock_omega0 > 0.0) ? 1 : 0;
    }

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
        if (nlo_checked_mul_size_t(state->num_time_samples, state->num_host_records, &host_elements) != 0) {
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
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.potential_vec) != NLO_VEC_STATUS_OK ||
        nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.previous_field_vec) != NLO_VEC_STATUS_OK) {
        nlo_state_debug_log_failure("allocate_device_vectors", NLO_VEC_STATUS_ALLOCATION_FAILED);
        free_simulation_state(state);
        return NULL;
    }
    if (state->nonlinear_raman_active) {
        if (nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_intensity_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_delayed_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_spectrum_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_mix_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_polarization_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_derivative_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_response_fft_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.raman_derivative_factor_vec) != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("allocate_raman_vectors", NLO_VEC_STATUS_ALLOCATION_FAILED);
            free_simulation_state(state);
            return NULL;
        }
    }

    if (state->tensor_mode_active) {
        if (nlo_create_complex_vec(state->backend, state->nt, &state->working_vectors.wt_axis_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, state->nx, &state->working_vectors.kx_axis_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, state->ny, &state->working_vectors.ky_axis_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, state->nt, &state->working_vectors.t_axis_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, state->nx, &state->working_vectors.x_axis_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, state->ny, &state->working_vectors.y_axis_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.wt_mesh_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.kx_mesh_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.ky_mesh_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.t_mesh_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.x_mesh_vec) != NLO_VEC_STATUS_OK ||
            nlo_create_complex_vec(state->backend, num_time_samples, &state->working_vectors.y_mesh_vec) != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("allocate_tensor_mesh_vectors", NLO_VEC_STATUS_ALLOCATION_FAILED);
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
    if (state->tensor_mode_active) {
        const double safe_dt = (delta_time > 0.0) ? delta_time : 1.0;
        const double safe_dx = (config->spatial.delta_x > 0.0) ? config->spatial.delta_x : 1.0;
        const double safe_dy = (config->spatial.delta_y > 0.0) ? config->spatial.delta_y : 1.0;

        if (config->time.wt_axis != NULL) {
            status = nlo_vec_upload(state->backend,
                                    state->working_vectors.wt_axis_vec,
                                    config->time.wt_axis,
                                    state->nt * sizeof(nlo_complex));
        } else {
            status = nlo_vec_complex_axis_unshifted_from_delta(state->backend,
                                                               state->working_vectors.wt_axis_vec,
                                                               safe_dt);
        }
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("prepare_tensor_wt_axis", status);
            free_simulation_state(state);
            return NULL;
        }

        if (config->spatial.kx_axis != NULL) {
            status = nlo_vec_upload(state->backend,
                                    state->working_vectors.kx_axis_vec,
                                    config->spatial.kx_axis,
                                    state->nx * sizeof(nlo_complex));
        } else {
            status = nlo_vec_complex_axis_unshifted_from_delta(state->backend,
                                                               state->working_vectors.kx_axis_vec,
                                                               safe_dx);
        }
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("prepare_tensor_kx_axis", status);
            free_simulation_state(state);
            return NULL;
        }

        if (config->spatial.ky_axis != NULL) {
            status = nlo_vec_upload(state->backend,
                                    state->working_vectors.ky_axis_vec,
                                    config->spatial.ky_axis,
                                    state->ny * sizeof(nlo_complex));
        } else {
            status = nlo_vec_complex_axis_unshifted_from_delta(state->backend,
                                                               state->working_vectors.ky_axis_vec,
                                                               safe_dy);
        }
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("prepare_tensor_ky_axis", status);
            free_simulation_state(state);
            return NULL;
        }

        status = nlo_vec_complex_axis_centered_from_delta(state->backend,
                                                          state->working_vectors.t_axis_vec,
                                                          safe_dt);
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_axis_centered_from_delta(state->backend,
                                                              state->working_vectors.x_axis_vec,
                                                              safe_dx);
        }
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_axis_centered_from_delta(state->backend,
                                                              state->working_vectors.y_axis_vec,
                                                              safe_dy);
        }
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("prepare_tensor_physical_axes", status);
            free_simulation_state(state);
            return NULL;
        }

        status = nlo_vec_complex_mesh_from_axis_tfast(state->backend,
                                                      state->working_vectors.wt_mesh_vec,
                                                      state->working_vectors.wt_axis_vec,
                                                      state->nt,
                                                      state->ny,
                                                      NLO_VEC_MESH_AXIS_T);
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.kx_mesh_vec,
                                                          state->working_vectors.kx_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          NLO_VEC_MESH_AXIS_X);
        }
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.ky_mesh_vec,
                                                          state->working_vectors.ky_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          NLO_VEC_MESH_AXIS_Y);
        }
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.t_mesh_vec,
                                                          state->working_vectors.t_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          NLO_VEC_MESH_AXIS_T);
        }
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.x_mesh_vec,
                                                          state->working_vectors.x_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          NLO_VEC_MESH_AXIS_X);
        }
        if (status == NLO_VEC_STATUS_OK) {
            status = nlo_vec_complex_mesh_from_axis_tfast(state->backend,
                                                          state->working_vectors.y_mesh_vec,
                                                          state->working_vectors.y_axis_vec,
                                                          state->nt,
                                                          state->ny,
                                                          NLO_VEC_MESH_AXIS_Y);
        }
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("prepare_tensor_mesh_axes", status);
            free_simulation_state(state);
            return NULL;
        }

        status = nlo_vec_complex_copy(state->backend,
                                      state->frequency_grid_vec,
                                      state->working_vectors.wt_mesh_vec);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("upload_frequency_grid_tensor", status);
            free_simulation_state(state);
            return NULL;
        }
    } else if (explicit_nd != 0) {
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

    if (!state->tensor_mode_active) {
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
    } else {
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
            nlo_state_debug_log_failure("upload_potential_grid_tensor", status);
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

    const char* linear_factor_expr = nlo_resolve_operator_expr(config->runtime.linear_factor_expr,
                                                               nlo_resolve_operator_expr(
                                                                   config->runtime.dispersion_factor_expr,
                                                                   NLO_DEFAULT_LINEAR_FACTOR_EXPR));
    const char* linear_expr = nlo_resolve_operator_expr(config->runtime.linear_expr,
                                                        nlo_resolve_operator_expr(
                                                            config->runtime.dispersion_expr,
                                                            NLO_DEFAULT_LINEAR_EXPR));
    const char* potential_expr = nlo_resolve_operator_expr(config->runtime.potential_expr,
                                                           NLO_DEFAULT_POTENTIAL_EXPR);
    const char* dispersion_factor_expr = nlo_resolve_operator_expr(config->runtime.dispersion_factor_expr,
                                                                   NLO_DEFAULT_DISPERSION_FACTOR_EXPR);
    const char* dispersion_expr = nlo_resolve_operator_expr(config->runtime.dispersion_expr,
                                                            NLO_DEFAULT_DISPERSION_EXPR);
    const char* nonlinear_expr = nlo_resolve_operator_expr(config->runtime.nonlinear_expr,
                                                           NLO_DEFAULT_NONLINEAR_EXPR);

    if (state->tensor_mode_active) {
        status = nlo_operator_program_compile(potential_expr,
                                              NLO_OPERATOR_CONTEXT_POTENTIAL,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->potential_operator_program);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("compile_potential_program", status);
            free_simulation_state(state);
            return NULL;
        }

        status = nlo_operator_program_compile(linear_factor_expr,
                                              NLO_OPERATOR_CONTEXT_LINEAR_FACTOR,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->linear_factor_operator_program);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("compile_linear_factor_program", status);
            free_simulation_state(state);
            return NULL;
        }

        status = nlo_operator_program_compile(linear_expr,
                                              NLO_OPERATOR_CONTEXT_LINEAR,
                                              runtime_constant_count,
                                              runtime_constants,
                                              &state->linear_operator_program);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("compile_linear_program", status);
            free_simulation_state(state);
            return NULL;
        }
    } else {
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
    nlo_warn_legacy_nonlinear_expression(nonlinear_expr, &state->nonlinear_operator_program);

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

    if (state->tensor_mode_active) {
        if (config->spatial.potential_grid == NULL) {
            const nlo_operator_eval_context potential_eval_ctx = {
                .frequency_grid = state->frequency_grid_vec,
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
            status = nlo_operator_program_execute(state->backend,
                                                  &state->potential_operator_program,
                                                  &potential_eval_ctx,
                                                  state->runtime_operator_stack_vec,
                                                  state->runtime_operator_stack_slots,
                                                  state->working_vectors.potential_vec);
            if (status != NLO_VEC_STATUS_OK) {
                nlo_state_debug_log_failure("execute_potential_program", status);
                free_simulation_state(state);
                return NULL;
            }
        }

        const nlo_operator_eval_context linear_factor_eval_ctx = {
            .frequency_grid = state->frequency_grid_vec,
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
        status = nlo_operator_program_execute(state->backend,
                                              &state->linear_factor_operator_program,
                                              &linear_factor_eval_ctx,
                                              state->runtime_operator_stack_vec,
                                              state->runtime_operator_stack_slots,
                                              state->working_vectors.dispersion_factor_vec);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("execute_linear_factor_program", status);
            free_simulation_state(state);
            return NULL;
        }
    } else {
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

    if (state->nonlinear_raman_active) {
        status = nlo_prepare_raman_state(state);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_state_debug_log_failure("prepare_raman_state", status);
            free_simulation_state(state);
            return NULL;
        }
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
