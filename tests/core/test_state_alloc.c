/**
 * @file test_state_alloc.c
 * @brief Unit tests for simulation state allocation sizing.
 */

#include "backend/nlo_complex.h"
#include "core/init.h"
#include "core/state.h"
#include "io/snapshot_store.h"
#include "nlolib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TEST_TWO_PI
#define TEST_TWO_PI 6.283185307179586476925286766559
#endif

#ifndef TEST_FREQ_EPS
#define TEST_FREQ_EPS 1e-9
#endif

/* Not assert(): Release builds define NDEBUG. */
#define STATE_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "%s:%d: STATE_CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static double test_expected_omega_unshifted(size_t index, size_t num_time_samples, double omega_step)
{
    const size_t positive_limit = (num_time_samples - 1u) / 2u;
    if (index <= positive_limit) {
        return (double)index * omega_step;
    }

    return -((double)num_time_samples - (double)index) * omega_step;
}

static void test_fill_expected_omega_grid(nlo_complex* out_grid, size_t num_time_samples, double delta_time)
{
    STATE_CHECK(out_grid != NULL);
    STATE_CHECK(num_time_samples > 0u);
    STATE_CHECK(delta_time > 0.0);

    const double omega_step = TEST_TWO_PI / ((double)num_time_samples * delta_time);
    for (size_t i = 0u; i < num_time_samples; ++i) {
        out_grid[i] = make(test_expected_omega_unshifted(i, num_time_samples, omega_step), 0.0);
    }
}

static void test_assert_grid_matches(const nlo_complex* actual, const nlo_complex* expected, size_t count)
{
    STATE_CHECK(actual != NULL);
    STATE_CHECK(expected != NULL);

    for (size_t i = 0u; i < count; ++i) {
        const double expected_real = expected[i].re;
        const double tol = TEST_FREQ_EPS * fmax(1.0, fabs(expected_real));
        STATE_CHECK(fabs(actual[i].re - expected_real) <= tol);
        STATE_CHECK(fabs(actual[i].im - expected[i].im) <= tol);
    }
}

static void test_init_state_success(void)
{
    const size_t num_time_samples = 1024;
    const size_t num_records = 64;

    sim_config* config = create_sim_config(num_time_samples);
    STATE_CHECK(config != NULL);

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);

    simulation_state* state = NULL;
    allocation_info info = {0};

    STATE_CHECK(init_simulation_state(config,
                                          num_time_samples,
                                          num_records,
                                          &exec_options,
                                          &info,
                                          &state) == 0);
    STATE_CHECK(state != NULL);

    STATE_CHECK(info.requested_records == num_records);
    STATE_CHECK(info.allocated_records <= 1u);
    STATE_CHECK(info.per_record_bytes == num_time_samples * sizeof(nlo_complex));
    STATE_CHECK(info.host_snapshot_bytes <= info.per_record_bytes);
    STATE_CHECK(info.working_vector_bytes >= info.per_record_bytes * 12u);
    STATE_CHECK(info.backend_type == VECTOR_BACKEND_CPU);

    free_simulation_state(state);
    free_sim_config(config);

    printf("test_init_state_success: validates allocation sizing info.\n");
}

static void test_record_count_does_not_change_init_snapshot_allocation(void)
{
    const size_t num_time_samples = 1024u;
    sim_config* config = create_sim_config(num_time_samples);
    STATE_CHECK(config != NULL);

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);

    simulation_state* state_small = NULL;
    simulation_state* state_large = NULL;
    allocation_info info_small = {0};
    allocation_info info_large = {0};

    STATE_CHECK(init_simulation_state(config,
                                          num_time_samples,
                                          2u,
                                          &exec_options,
                                          &info_small,
                                          &state_small) == 0);
    STATE_CHECK(init_simulation_state(config,
                                          num_time_samples,
                                          128u,
                                          &exec_options,
                                          &info_large,
                                          &state_large) == 0);
    STATE_CHECK(state_small != NULL);
    STATE_CHECK(state_large != NULL);

    STATE_CHECK(info_small.per_record_bytes == info_large.per_record_bytes);
    STATE_CHECK(info_small.host_snapshot_bytes == info_large.host_snapshot_bytes);
    STATE_CHECK(info_small.working_vector_bytes == info_large.working_vector_bytes);
    STATE_CHECK(info_small.allocated_records == info_large.allocated_records);

    free_simulation_state(state_large);
    free_simulation_state(state_small);
    free_sim_config(config);

    printf("test_record_count_does_not_change_init_snapshot_allocation: validates O(1) snapshot allocation.\n");
}

static void test_init_state_invalid_args(void)
{
    simulation_state* state = NULL;
    allocation_info info = {0};
    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);

    STATE_CHECK(init_simulation_state(NULL, 8, 1, &exec_options, &info, &state) != 0);
    STATE_CHECK(state == NULL);

    sim_config* config = create_sim_config(8);
    STATE_CHECK(config != NULL);

    STATE_CHECK(init_simulation_state(config, 0, 1, &exec_options, &info, &state) != 0);
    STATE_CHECK(state == NULL);

    STATE_CHECK(init_simulation_state(config, 8, 0, &exec_options, &info, &state) != 0);
    STATE_CHECK(state == NULL);

    config->spatial.nx = 3;
    config->spatial.ny = 3;
    STATE_CHECK(init_simulation_state(config, 8, 1, &exec_options, &info, &state) != 0);
    STATE_CHECK(state == NULL);

    free_sim_config(config);

    printf("test_init_state_invalid_args: validates argument checks.\n");
}

static void test_init_state_xy_shape_rejected_without_tensor(void)
{
    const size_t nx = 32;
    const size_t ny = 16;
    const size_t num_time_samples = nx * ny;
    const size_t num_records = 8;

    sim_config* config = create_sim_config(num_time_samples);
    STATE_CHECK(config != NULL);

    config->spatial.nx = nx;
    config->spatial.ny = ny;
    config->spatial.delta_x = 0.5;
    config->spatial.delta_y = 0.5;
    config->spatial.potential_grid = NULL;

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);

    simulation_state* state = NULL;
    allocation_info info = {0};
    STATE_CHECK(init_simulation_state(config,
                                          num_time_samples,
                                          num_records,
                                          &exec_options,
                                          &info,
                                          &state) != 0);
    STATE_CHECK(state == NULL);
    free_sim_config(config);

    printf("test_init_state_xy_shape_rejected_without_tensor: validates legacy flattened XY rejection.\n");
}

static void test_init_state_tensor_shape_rules(void)
{
    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);
    allocation_info info = {0};
    simulation_state* state = NULL;

    {
        const size_t nt = 8u;
        const size_t nx = 4u;
        const size_t ny = 2u;
        const size_t total_samples = nt * nx * ny;

        sim_config* config = create_sim_config(total_samples);
        STATE_CHECK(config != NULL);
        config->tensor.nt = nt;
        config->tensor.nx = nx;
        config->tensor.ny = ny;
        config->tensor.layout = TENSOR_LAYOUT_XYT_T_FAST;

        STATE_CHECK(init_simulation_state(config,
                                              total_samples,
                                              4u,
                                              &exec_options,
                                              &info,
                                              &state) == 0);
        STATE_CHECK(state != NULL);
        STATE_CHECK(state->nt == nt);
        STATE_CHECK(state->nx == nx);
        STATE_CHECK(state->ny == ny);
        STATE_CHECK(state->num_points_xy == nx * ny);
        STATE_CHECK(state->tensor_mode_active == 1);
        free_simulation_state(state);
        state = NULL;
        free_sim_config(config);
    }

    {
        const size_t nt = 6u;
        const size_t nx = 3u;
        const size_t ny = 1u;
        const size_t total_samples = nt * nx;

        sim_config* config = create_sim_config(total_samples);
        STATE_CHECK(config != NULL);
        config->tensor.nt = nt;
        config->tensor.nx = nx;
        config->tensor.ny = ny;
        config->tensor.layout = TENSOR_LAYOUT_XYT_T_FAST;

        STATE_CHECK(init_simulation_state(config,
                                              total_samples,
                                              2u,
                                              &exec_options,
                                              &info,
                                              &state) == 0);
        STATE_CHECK(state != NULL);
        STATE_CHECK(state->nt == nt);
        STATE_CHECK(state->nx == nx);
        STATE_CHECK(state->ny == 1u);
        free_simulation_state(state);
        state = NULL;
        free_sim_config(config);
    }

    {
        const size_t total_samples = 16u;
        sim_config* config = create_sim_config(total_samples);
        STATE_CHECK(config != NULL);
        config->tensor.nt = 4u;
        config->tensor.nx = 2u;
        config->tensor.ny = 3u;
        config->tensor.layout = TENSOR_LAYOUT_XYT_T_FAST;
        STATE_CHECK(init_simulation_state(config,
                                              total_samples,
                                              2u,
                                              &exec_options,
                                              &info,
                                              &state) != 0);
        STATE_CHECK(state == NULL);
        free_sim_config(config);
    }

    {
        sim_config* config = create_sim_config(16u);
        STATE_CHECK(config != NULL);
        config->tensor.nt = 4u;
        config->tensor.nx = 2u;
        config->tensor.ny = 2u;
        config->tensor.layout = 7;
        STATE_CHECK(init_simulation_state(config,
                                              16u,
                                              1u,
                                              &exec_options,
                                              &info,
                                              &state) != 0);
        STATE_CHECK(state == NULL);
        free_sim_config(config);
    }

    printf("test_init_state_tensor_shape_rules: validates tensor-dimension shape rules.\n");
}

static void test_default_frequency_grid_generated_when_invalid(void)
{
    const size_t num_time_samples = 16u;
    const double delta_time = 0.25;

    sim_config* config = create_sim_config(num_time_samples);
    STATE_CHECK(config != NULL);

    config->time.delta_time = delta_time;
    for (size_t i = 0u; i < num_time_samples; ++i) {
        config->frequency.frequency_grid[i] = make((double)i, 0.0);
    }

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);
    simulation_state* state = NULL;
    STATE_CHECK(init_simulation_state(config,
                                          num_time_samples,
                                          2u,
                                          &exec_options,
                                          NULL,
                                          &state) == 0);
    STATE_CHECK(state != NULL);

    nlo_complex* downloaded = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    nlo_complex* expected = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    STATE_CHECK(downloaded != NULL);
    STATE_CHECK(expected != NULL);

    test_fill_expected_omega_grid(expected, num_time_samples, delta_time);
    STATE_CHECK(vec_download(state->backend,
                                 state->frequency_grid_vec,
                                 downloaded,
                                 num_time_samples * sizeof(nlo_complex)) == VEC_STATUS_OK);
    test_assert_grid_matches(downloaded, expected, num_time_samples);

    free(expected);
    free(downloaded);
    free_simulation_state(state);
    free_sim_config(config);

    printf("test_default_frequency_grid_generated_when_invalid: validates temporal omega regeneration.\n");
}

static void test_frequency_grid_preserved_when_valid(void)
{
    const size_t num_time_samples = 17u;
    const double delta_time = 0.125;

    sim_config* config = create_sim_config(num_time_samples);
    STATE_CHECK(config != NULL);

    config->time.delta_time = delta_time;
    test_fill_expected_omega_grid(config->frequency.frequency_grid, num_time_samples, delta_time);

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);
    simulation_state* state = NULL;
    STATE_CHECK(init_simulation_state(config,
                                          num_time_samples,
                                          2u,
                                          &exec_options,
                                          NULL,
                                          &state) == 0);
    STATE_CHECK(state != NULL);

    nlo_complex* downloaded = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    STATE_CHECK(downloaded != NULL);
    STATE_CHECK(vec_download(state->backend,
                                 state->frequency_grid_vec,
                                 downloaded,
                                 num_time_samples * sizeof(nlo_complex)) == VEC_STATUS_OK);
    test_assert_grid_matches(downloaded, config->frequency.frequency_grid, num_time_samples);

    free(downloaded);
    free_simulation_state(state);
    free_sim_config(config);

    printf("test_frequency_grid_preserved_when_valid: validates valid temporal omega preservation.\n");
}

static void test_tensor_mode_frequency_mesh_generation(void)
{
    const size_t nt = 3u;
    const size_t nx = 2u;
    const size_t ny = 2u;
    const size_t total = nt * nx * ny;
    const double delta_time = 0.5;

    sim_config* config = create_sim_config(total);
    STATE_CHECK(config != NULL);
    config->tensor.nt = nt;
    config->tensor.nx = nx;
    config->tensor.ny = ny;
    config->tensor.layout = TENSOR_LAYOUT_XYT_T_FAST;
    config->time.delta_time = delta_time;
    config->spatial.delta_x = 1.0;
    config->spatial.delta_y = 1.0;
    config->runtime.linear_factor_expr = "i*(wt*wt + kx*kx + ky*ky)";
    config->runtime.linear_expr = "exp(h*D)";
    config->runtime.potential_expr = "x+y+t";

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);
    simulation_state* state = NULL;
    STATE_CHECK(init_simulation_state(config,
                                          total,
                                          2u,
                                          &exec_options,
                                          NULL,
                                          &state) == 0);
    STATE_CHECK(state != NULL);
    STATE_CHECK(state->tensor_mode_active == 1);
    STATE_CHECK(state->nt == nt);
    STATE_CHECK(state->nx == nx);
    STATE_CHECK(state->ny == ny);
    STATE_CHECK(state->init_vectors.wt_axis_vec == NULL);
    STATE_CHECK(state->init_vectors.kx_axis_vec == NULL);
    STATE_CHECK(state->init_vectors.ky_axis_vec == NULL);
    STATE_CHECK(state->init_vectors.t_axis_vec == NULL);
    STATE_CHECK(state->init_vectors.x_axis_vec == NULL);
    STATE_CHECK(state->init_vectors.y_axis_vec == NULL);
    STATE_CHECK(state->frequency_grid_vec == NULL);

    nlo_complex expected_axis[3] = {0};
    test_fill_expected_omega_grid(expected_axis, nt, delta_time);

    nlo_complex downloaded[12] = {0};
    STATE_CHECK(vec_download(state->backend,
                                 state->working_vectors.wt_mesh_vec,
                                 downloaded,
                                 sizeof(downloaded)) == VEC_STATUS_OK);
    for (size_t block = 0u; block < (nx * ny); ++block) {
        const size_t base = block * nt;
        for (size_t t = 0u; t < nt; ++t) {
            STATE_CHECK(fabs(downloaded[base + t].re - expected_axis[t].re) <= TEST_FREQ_EPS);
            STATE_CHECK(fabs(downloaded[base + t].im - expected_axis[t].im) <= TEST_FREQ_EPS);
        }
    }

    nlo_complex potential_downloaded[12] = {0};
    STATE_CHECK(vec_download(state->backend,
                                 state->working_vectors.potential_vec,
                                 potential_downloaded,
                                 sizeof(potential_downloaded)) == VEC_STATUS_OK);
    STATE_CHECK(fabs(potential_downloaded[0].re + 1.5) <= TEST_FREQ_EPS);

    free_simulation_state(state);
    free_sim_config(config);

    printf("test_tensor_mode_frequency_mesh_generation: validates tensor t-fast frequency mesh.\n");
}

static void test_tensor_mode_linear_factor_literal_power_vulkan(void)
{
#if !ENABLE_VULKAN_BACKEND
    printf("test_tensor_mode_linear_factor_literal_power_vulkan: skipped (Vulkan disabled at build).\n");
    return;
#else
    vector_backend* probe = vector_backend_create_vulkan(NULL);
    if (probe == NULL) {
        printf("test_tensor_mode_linear_factor_literal_power_vulkan: skipped (Vulkan unavailable).\n");
        return;
    }
    vector_backend_destroy(probe);

    const size_t nt = 8u;
    const size_t nx = 4u;
    const size_t ny = 4u;
    const size_t total = nt * nx * ny;

    sim_config* config = create_sim_config(total);
    STATE_CHECK(config != NULL);
    config->tensor.nt = nt;
    config->tensor.nx = nx;
    config->tensor.ny = ny;
    config->tensor.layout = TENSOR_LAYOUT_XYT_T_FAST;
    config->time.delta_time = 0.25;
    config->spatial.delta_x = 0.5;
    config->spatial.delta_y = 0.5;
    config->runtime.linear_factor_expr = "i*((0.5*c0)*(wt^2) + c1*((kx^2) + (ky^2)))";
    config->runtime.linear_expr = "exp(h*D)";
    config->runtime.potential_expr = "0";
    config->runtime.num_constants = 2u;
    config->runtime.constants[0] = 0.08;
    config->runtime.constants[1] = -0.2;

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_AUTO);
    simulation_state* state = NULL;
    STATE_CHECK(init_simulation_state(config,
                                          total,
                                          2u,
                                          &exec_options,
                                          NULL,
                                          &state) == 0);
    STATE_CHECK(state != NULL);
    STATE_CHECK(vector_backend_get_type(state->backend) == VECTOR_BACKEND_VULKAN);

    free_simulation_state(state);
    free_sim_config(config);

    printf("test_tensor_mode_linear_factor_literal_power_vulkan: validates Vulkan literal-power lowering.\n");
#endif
}

#if ENABLE_VULKAN_BACKEND
static void set_operator_jit_env(const char* value)
{
#if defined(_WIN32)
    STATE_CHECK(_putenv_s("NLO_OPERATOR_JIT", value) == 0);
#else
    STATE_CHECK(setenv("NLO_OPERATOR_JIT", value, 1) == 0);
#endif
}

/* Build a small tensor state (nx != ny, so x/y mix-ups show) and download its
 * potential and linear factor, which init evaluates from the coordinate grids. */
static simulation_state* create_compact_axis_test_state(
    const char* jit_env,
    nlo_complex* out_potential,
    nlo_complex* out_linear_factor,
    size_t total
)
{
    set_operator_jit_env(jit_env);
    sim_config* config = create_sim_config(total);
    STATE_CHECK(config != NULL);
    config->tensor.nt = 8u;
    config->tensor.nx = 4u;
    config->tensor.ny = 3u;
    config->tensor.layout = TENSOR_LAYOUT_XYT_T_FAST;
    config->time.delta_time = 0.25;
    config->spatial.delta_x = 0.5;
    config->spatial.delta_y = 0.7;
    config->runtime.linear_factor_expr = "i*(c0*(wt*wt) + c1*(kx*kx) + c2*(ky*ky))";
    config->runtime.linear_expr = "exp(h*D)";
    config->runtime.potential_expr = "c3*x + y*y + 0.1*t";
    config->runtime.num_constants = 4u;
    config->runtime.constants[0] = 0.08;
    config->runtime.constants[1] = -0.2;
    config->runtime.constants[2] = 0.3;
    config->runtime.constants[3] = 1.7;

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_VULKAN);
    simulation_state* state = NULL;
    STATE_CHECK(init_simulation_state(config, total, 2u, &exec_options, NULL, &state) == 0);
    STATE_CHECK(state != NULL);
    STATE_CHECK(vec_download(state->backend,
                             state->working_vectors.potential_vec,
                             out_potential,
                             total * sizeof(nlo_complex)) == VEC_STATUS_OK);
    STATE_CHECK(vec_download(state->backend,
                             state->working_vectors.dispersion_factor_vec,
                             out_linear_factor,
                             total * sizeof(nlo_complex)) == VEC_STATUS_OK);
    free_sim_config(config);
    return state;
}

static double max_abs_diff(const nlo_complex* a, const nlo_complex* b, size_t n)
{
    double diff = 0.0;
    for (size_t i = 0u; i < n; ++i) {
        diff = fmax(diff, hypot(RE(a[i]) - RE(b[i]), IM(a[i]) - IM(b[i])));
    }
    return diff;
}
#endif

static void test_tensor_mode_compact_axis_symbols_vulkan(void)
{
#if !ENABLE_VULKAN_BACKEND
    printf("test_tensor_mode_compact_axis_symbols_vulkan: skipped (Vulkan disabled at build).\n");
    return;
#else
    vector_backend* probe = vector_backend_create_vulkan(NULL);
    if (probe == NULL) {
        printf("test_tensor_mode_compact_axis_symbols_vulkan: skipped (Vulkan unavailable).\n");
        return;
    }
    vector_backend_destroy(probe);

    enum { TOTAL = 8 * 4 * 3 };
    static nlo_complex mesh_potential[TOTAL];
    static nlo_complex mesh_factor[TOTAL];
    static nlo_complex compact_potential[TOTAL];
    static nlo_complex compact_factor[TOTAL];

    /* Interpreter: full-grid meshes, as on main. */
    simulation_state* mesh_state =
        create_compact_axis_test_state("0", mesh_potential, mesh_factor, TOTAL);
    STATE_CHECK(!mesh_state->tensor_compact_axis_symbols);
    STATE_CHECK(mesh_state->working_vectors.wt_mesh_vec != NULL);
    STATE_CHECK(mesh_state->working_vectors.y_mesh_vec != NULL);
    STATE_CHECK(mesh_state->init_vectors.wt_axis_vec == NULL);
    free_simulation_state(mesh_state);

    /* JIT: no meshes, the 1-D axes stay alive for the kernels. */
    simulation_state* compact_state =
        create_compact_axis_test_state("1", compact_potential, compact_factor, TOTAL);
    STATE_CHECK(compact_state->tensor_compact_axis_symbols);
    STATE_CHECK(compact_state->potential_operator_program.vk_jit_active);
    STATE_CHECK(compact_state->linear_factor_operator_program.vk_jit_active);
    STATE_CHECK(compact_state->linear_operator_program.vk_jit_active);
    STATE_CHECK(compact_state->nonlinear_operator_program.vk_jit_active);
    STATE_CHECK(compact_state->working_vectors.wt_mesh_vec == NULL);
    STATE_CHECK(compact_state->working_vectors.kx_mesh_vec == NULL);
    STATE_CHECK(compact_state->working_vectors.ky_mesh_vec == NULL);
    STATE_CHECK(compact_state->working_vectors.t_mesh_vec == NULL);
    STATE_CHECK(compact_state->working_vectors.x_mesh_vec == NULL);
    STATE_CHECK(compact_state->working_vectors.y_mesh_vec == NULL);
    STATE_CHECK(compact_state->init_vectors.wt_axis_vec != NULL);
    STATE_CHECK(compact_state->init_vectors.kx_axis_vec != NULL);
    STATE_CHECK(compact_state->init_vectors.ky_axis_vec != NULL);
    STATE_CHECK(compact_state->init_vectors.t_axis_vec != NULL);
    STATE_CHECK(compact_state->init_vectors.x_axis_vec != NULL);
    STATE_CHECK(compact_state->init_vectors.y_axis_vec != NULL);
    free_simulation_state(compact_state);

    const double potential_diff = max_abs_diff(compact_potential, mesh_potential, TOTAL);
    const double factor_diff = max_abs_diff(compact_factor, mesh_factor, TOTAL);
    printf("test_tensor_mode_compact_axis_symbols_vulkan: potential diff %.2e, linear factor diff %.2e\n",
           potential_diff,
           factor_diff);
    STATE_CHECK(potential_diff <= 1e-12);
    STATE_CHECK(factor_diff <= 1e-12);
    /* The potential must actually vary along x and y for the check to mean anything. */
    STATE_CHECK(fabs(RE(mesh_potential[8]) - RE(mesh_potential[0])) > 0.1);
    STATE_CHECK(fabs(RE(mesh_potential[3 * 8]) - RE(mesh_potential[0])) > 0.1);

    set_operator_jit_env("1");
    printf("test_tensor_mode_compact_axis_symbols_vulkan: JIT drops the six tensor meshes.\n");
#endif
}

static void test_snapshot_store_dense_readback(void)
{
    if (!snapshot_store_is_available()) {
        printf("test_snapshot_store_dense_readback: skipped (storage unavailable).\n");
        return;
    }

    const size_t num_time_samples = 16u;
    const size_t num_records = 5u;
    const size_t total = num_time_samples * num_records;

    sim_config* config = create_sim_config(num_time_samples);
    STATE_CHECK(config != NULL);
    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);

    char db_path[256];
    (void)snprintf(db_path, sizeof(db_path), "test_snapshot_store_readback_%u.sqlite3", (unsigned)rand());

    storage_options storage_options = storage_options_default();
    storage_options.sqlite_path = db_path;
    storage_options.run_id = "test-snapshot-readback";
    storage_options.chunk_records = 2u;

    snapshot_store_open_params open_params;
    open_params.config = config;
    open_params.exec_options = &exec_options;
    open_params.storage_options = &storage_options;
    open_params.num_time_samples = num_time_samples;
    open_params.num_recorded_samples = num_records;

    snapshot_store* store = snapshot_store_open(&open_params);
    STATE_CHECK(store != NULL);

    nlo_complex* expected = (nlo_complex*)calloc(total, sizeof(nlo_complex));
    nlo_complex* restored = (nlo_complex*)calloc(total, sizeof(nlo_complex));
    STATE_CHECK(expected != NULL);
    STATE_CHECK(restored != NULL);

    for (size_t record = 0u; record < num_records; ++record) {
        for (size_t t = 0u; t < num_time_samples; ++t) {
            const size_t idx = (record * num_time_samples) + t;
            expected[idx] = make((double)(100u * record + t), (double)(-1.0 * (double)record));
        }
        STATE_CHECK(snapshot_store_write_record(
                        store,
                        record,
                        expected + (record * num_time_samples),
                        num_time_samples) != SNAPSHOT_STORE_STATUS_ERROR);
    }

    STATE_CHECK(snapshot_store_read_all_records(store, restored, num_records, num_time_samples) ==
                SNAPSHOT_STORE_STATUS_OK);
    for (size_t i = 0u; i < total; ++i) {
        STATE_CHECK(fabs(restored[i].re - expected[i].re) <= TEST_FREQ_EPS);
        STATE_CHECK(fabs(restored[i].im - expected[i].im) <= TEST_FREQ_EPS);
    }

    snapshot_store_close(store);
    free(expected);
    free(restored);
    free_sim_config(config);

    (void)remove(db_path);
    char db_wal_path[280];
    char db_shm_path[280];
    (void)snprintf(db_wal_path, sizeof(db_wal_path), "%s-wal", db_path);
    (void)snprintf(db_shm_path, sizeof(db_shm_path), "%s-shm", db_path);
    (void)remove(db_wal_path);
    (void)remove(db_shm_path);

    printf("test_snapshot_store_dense_readback: validates dense record reconstruction from chunks.\n");
}

static void test_dense_output_safety_cap_rejects_before_init(void)
{
    simulation_config config;
    memset(&config, 0, sizeof(config));
    config.propagation.propagation_distance = 0.1;

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);
    propagate_options options = nlolib_propagate_options_default();
    options.num_recorded_samples = 1u;
    options.output_mode = PROPAGATE_OUTPUT_FINAL_ONLY;
    options.return_records = 1;
    options.exec_options = &exec_options;

    nlo_complex input = make(1.0, 0.0);
    nlo_complex output = make(0.0, 0.0);
    size_t records_written = 99u;
    propagate_output out = nlolib_propagate_output_default();
    out.output_records = &output;
    out.output_record_capacity = 1u;
    out.records_written = &records_written;

    const size_t oversized_samples =
        (((size_t)2u * 1024u * 1024u * 1024u) / sizeof(nlo_complex)) + 1u;
    const nlolib_status status = nlolib_propagate(
        &config,
        NULL,
        oversized_samples,
        &input,
        &options,
        &out);

    STATE_CHECK(status == NLOLIB_STATUS_INVALID_ARGUMENT);
    STATE_CHECK(records_written == 0u);

    printf("test_dense_output_safety_cap_rejects_before_init: rejects oversized dense output.\n");
}

static void test_storage_only_propagate_without_dense_output(void)
{
    if (!snapshot_store_is_available()) {
        printf("test_storage_only_propagate_without_dense_output: skipped (storage unavailable).\n");
        return;
    }

    const size_t num_time_samples = 16u;
    sim_config* config = create_sim_config(num_time_samples);
    STATE_CHECK(config != NULL);
    config->propagation.propagation_distance = 0.02;
    config->propagation.starting_step_size = 0.01;
    config->propagation.max_step_size = 0.01;
    config->propagation.min_step_size = 0.001;
    config->propagation.error_tolerance = 1e-6;

    nlo_complex input[16];
    for (size_t i = 0u; i < num_time_samples; ++i) {
        input[i] = make(0.1 * (double)i, 0.0);
    }

    char db_path[256];
    (void)snprintf(db_path, sizeof(db_path), "test_storage_only_%u.sqlite3", (unsigned)rand());

    storage_options storage_options = storage_options_default();
    storage_options.sqlite_path = db_path;
    storage_options.run_id = "test-storage-only-propagate";
    storage_options.chunk_records = 2u;

    execution_options exec_options = execution_options_default(VECTOR_BACKEND_CPU);
    propagate_options options = nlolib_propagate_options_default();
    options.num_recorded_samples = 3u;
    options.return_records = 0;
    options.exec_options = &exec_options;
    options.storage_options = &storage_options;

    size_t records_written = 99u;
    storage_result storage_result;
    propagate_output out = nlolib_propagate_output_default();
    out.records_written = &records_written;
    out.storage_result = &storage_result;

    simulation_config sim_section;
    sim_section.propagation = config->propagation;
    sim_section.tensor = config->tensor;
    sim_section.time = config->time;
    sim_section.frequency = config->frequency;
    sim_section.spatial = config->spatial;
    physics_config physics_section = config->runtime;

    const nlolib_status status =
        nlolib_propagate(&sim_section, &physics_section, num_time_samples, input, &options, &out);
    STATE_CHECK(status == NLOLIB_STATUS_OK);
    STATE_CHECK(records_written == 0u);
    STATE_CHECK(storage_result.chunks_written == 2u);

    free_sim_config(config);
    (void)remove(db_path);

    printf("test_storage_only_propagate_without_dense_output: accepts storage-only propagation.\n");
}

int main(void)
{
    test_init_state_success();
    test_record_count_does_not_change_init_snapshot_allocation();
    test_init_state_invalid_args();
    test_init_state_xy_shape_rejected_without_tensor();
    test_init_state_tensor_shape_rules();
    test_default_frequency_grid_generated_when_invalid();
    test_frequency_grid_preserved_when_valid();
    test_tensor_mode_frequency_mesh_generation();
    test_tensor_mode_linear_factor_literal_power_vulkan();
    test_tensor_mode_compact_axis_symbols_vulkan();
    test_snapshot_store_dense_readback();
    test_dense_output_safety_cap_rejects_before_init();
    test_storage_only_propagate_without_dense_output();
    printf("test_core_state_alloc: all subtests completed.\n");
    return 0;
}
