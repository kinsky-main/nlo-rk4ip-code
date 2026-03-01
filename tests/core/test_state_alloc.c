/**
 * @file test_state_alloc.c
 * @brief Unit tests for simulation state allocation sizing.
 */

#include "backend/nlo_complex.h"
#include "core/init.h"
#include "core/state.h"
#include "io/snapshot_store.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NLO_TEST_TWO_PI
#define NLO_TEST_TWO_PI 6.283185307179586476925286766559
#endif

#ifndef NLO_TEST_FREQ_EPS
#define NLO_TEST_FREQ_EPS 1e-9
#endif

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
    assert(out_grid != NULL);
    assert(num_time_samples > 0u);
    assert(delta_time > 0.0);

    const double omega_step = NLO_TEST_TWO_PI / ((double)num_time_samples * delta_time);
    for (size_t i = 0u; i < num_time_samples; ++i) {
        out_grid[i] = nlo_make(test_expected_omega_unshifted(i, num_time_samples, omega_step), 0.0);
    }
}

static void test_assert_grid_matches(const nlo_complex* actual, const nlo_complex* expected, size_t count)
{
    assert(actual != NULL);
    assert(expected != NULL);

    for (size_t i = 0u; i < count; ++i) {
        const double expected_real = expected[i].re;
        const double tol = NLO_TEST_FREQ_EPS * fmax(1.0, fabs(expected_real));
        assert(fabs(actual[i].re - expected_real) <= tol);
        assert(fabs(actual[i].im - expected[i].im) <= tol);
    }
}

static void test_init_state_success(void)
{
    const size_t num_time_samples = 1024;
    const size_t num_records = 64;

    sim_config* config = create_sim_config(num_time_samples);
    assert(config != NULL);

    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);

    simulation_state* state = NULL;
    nlo_allocation_info info = {0};

    assert(nlo_init_simulation_state(config,
                                     num_time_samples,
                                     num_records,
                                     &exec_options,
                                     &info,
                                     &state) == 0);
    assert(state != NULL);

    assert(info.requested_records == num_records);
    assert(info.allocated_records == state->num_recorded_samples);
    assert(info.per_record_bytes == num_time_samples * sizeof(nlo_complex));
    assert(info.host_snapshot_bytes == info.per_record_bytes * state->num_recorded_samples);
    assert(info.working_vector_bytes == info.per_record_bytes * NLO_WORK_VECTOR_COUNT);
    assert(info.backend_type == NLO_VECTOR_BACKEND_CPU);

    free_simulation_state(state);
    free_sim_config(config);

    printf("test_init_state_success: validates allocation sizing info.\n");
}

static void test_init_state_invalid_args(void)
{
    simulation_state* state = NULL;
    nlo_allocation_info info = {0};
    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);

    assert(nlo_init_simulation_state(NULL, 8, 1, &exec_options, &info, &state) != 0);
    assert(state == NULL);

    sim_config* config = create_sim_config(8);
    assert(config != NULL);

    assert(nlo_init_simulation_state(config, 0, 1, &exec_options, &info, &state) != 0);
    assert(state == NULL);

    assert(nlo_init_simulation_state(config, 8, 0, &exec_options, &info, &state) != 0);
    assert(state == NULL);

    config->spatial.nx = 3;
    config->spatial.ny = 3;
    assert(nlo_init_simulation_state(config, 8, 1, &exec_options, &info, &state) != 0);
    assert(state == NULL);

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
    assert(config != NULL);

    config->spatial.nx = nx;
    config->spatial.ny = ny;
    config->spatial.delta_x = 0.5;
    config->spatial.delta_y = 0.5;
    config->spatial.potential_grid = NULL;

    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);

    simulation_state* state = NULL;
    nlo_allocation_info info = {0};
    assert(nlo_init_simulation_state(config,
                                     num_time_samples,
                                     num_records,
                                     &exec_options,
                                     &info,
                                     &state) != 0);
    assert(state == NULL);
    free_sim_config(config);

    printf("test_init_state_xy_shape_rejected_without_tensor: validates legacy flattened XY rejection.\n");
}

static void test_init_state_tensor_shape_rules(void)
{
    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    nlo_allocation_info info = {0};
    simulation_state* state = NULL;

    {
        const size_t nt = 8u;
        const size_t nx = 4u;
        const size_t ny = 2u;
        const size_t total_samples = nt * nx * ny;

        sim_config* config = create_sim_config(total_samples);
        assert(config != NULL);
        config->tensor.nt = nt;
        config->tensor.nx = nx;
        config->tensor.ny = ny;
        config->tensor.layout = NLO_TENSOR_LAYOUT_XYT_T_FAST;

        assert(nlo_init_simulation_state(config,
                                         total_samples,
                                         4u,
                                         &exec_options,
                                         &info,
                                         &state) == 0);
        assert(state != NULL);
        assert(state->nt == nt);
        assert(state->nx == nx);
        assert(state->ny == ny);
        assert(state->num_points_xy == nx * ny);
        assert(state->tensor_mode_active == 1);
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
        assert(config != NULL);
        config->tensor.nt = nt;
        config->tensor.nx = nx;
        config->tensor.ny = ny;
        config->tensor.layout = NLO_TENSOR_LAYOUT_XYT_T_FAST;

        assert(nlo_init_simulation_state(config,
                                         total_samples,
                                         2u,
                                         &exec_options,
                                         &info,
                                         &state) == 0);
        assert(state != NULL);
        assert(state->nt == nt);
        assert(state->nx == nx);
        assert(state->ny == 1u);
        free_simulation_state(state);
        state = NULL;
        free_sim_config(config);
    }

    {
        const size_t total_samples = 16u;
        sim_config* config = create_sim_config(total_samples);
        assert(config != NULL);
        config->tensor.nt = 4u;
        config->tensor.nx = 2u;
        config->tensor.ny = 3u;
        config->tensor.layout = NLO_TENSOR_LAYOUT_XYT_T_FAST;
        assert(nlo_init_simulation_state(config,
                                         total_samples,
                                         2u,
                                         &exec_options,
                                         &info,
                                         &state) != 0);
        assert(state == NULL);
        free_sim_config(config);
    }

    {
        sim_config* config = create_sim_config(16u);
        assert(config != NULL);
        config->tensor.nt = 4u;
        config->tensor.nx = 2u;
        config->tensor.ny = 2u;
        config->tensor.layout = 7;
        assert(nlo_init_simulation_state(config,
                                         16u,
                                         1u,
                                         &exec_options,
                                         &info,
                                         &state) != 0);
        assert(state == NULL);
        free_sim_config(config);
    }

    printf("test_init_state_tensor_shape_rules: validates tensor-dimension shape rules.\n");
}

static void test_default_frequency_grid_generated_when_invalid(void)
{
    const size_t num_time_samples = 16u;
    const double delta_time = 0.25;

    sim_config* config = create_sim_config(num_time_samples);
    assert(config != NULL);

    config->time.delta_time = delta_time;
    for (size_t i = 0u; i < num_time_samples; ++i) {
        config->frequency.frequency_grid[i] = nlo_make((double)i, 0.0);
    }

    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    simulation_state* state = NULL;
    assert(nlo_init_simulation_state(config,
                                     num_time_samples,
                                     2u,
                                     &exec_options,
                                     NULL,
                                     &state) == 0);
    assert(state != NULL);

    nlo_complex* downloaded = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    nlo_complex* expected = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    assert(downloaded != NULL);
    assert(expected != NULL);

    test_fill_expected_omega_grid(expected, num_time_samples, delta_time);
    assert(nlo_vec_download(state->backend,
                            state->frequency_grid_vec,
                            downloaded,
                            num_time_samples * sizeof(nlo_complex)) == NLO_VEC_STATUS_OK);
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
    assert(config != NULL);

    config->time.delta_time = delta_time;
    test_fill_expected_omega_grid(config->frequency.frequency_grid, num_time_samples, delta_time);

    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    simulation_state* state = NULL;
    assert(nlo_init_simulation_state(config,
                                     num_time_samples,
                                     2u,
                                     &exec_options,
                                     NULL,
                                     &state) == 0);
    assert(state != NULL);

    nlo_complex* downloaded = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    assert(downloaded != NULL);
    assert(nlo_vec_download(state->backend,
                            state->frequency_grid_vec,
                            downloaded,
                            num_time_samples * sizeof(nlo_complex)) == NLO_VEC_STATUS_OK);
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
    assert(config != NULL);
    config->tensor.nt = nt;
    config->tensor.nx = nx;
    config->tensor.ny = ny;
    config->tensor.layout = NLO_TENSOR_LAYOUT_XYT_T_FAST;
    config->time.delta_time = delta_time;
    config->spatial.delta_x = 1.0;
    config->spatial.delta_y = 1.0;
    config->runtime.linear_factor_expr = "i*(wt*wt + kx*kx + ky*ky)";
    config->runtime.linear_expr = "exp(h*D)";
    config->runtime.potential_expr = "x+y+t";

    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    simulation_state* state = NULL;
    assert(nlo_init_simulation_state(config,
                                     total,
                                     2u,
                                     &exec_options,
                                     NULL,
                                     &state) == 0);
    assert(state != NULL);
    assert(state->tensor_mode_active == 1);
    assert(state->nt == nt);
    assert(state->nx == nx);
    assert(state->ny == ny);

    nlo_complex expected_axis[3] = {0};
    test_fill_expected_omega_grid(expected_axis, nt, delta_time);

    nlo_complex downloaded[12] = {0};
    assert(nlo_vec_download(state->backend,
                            state->frequency_grid_vec,
                            downloaded,
                            sizeof(downloaded)) == NLO_VEC_STATUS_OK);
    for (size_t block = 0u; block < (nx * ny); ++block) {
        const size_t base = block * nt;
        for (size_t t = 0u; t < nt; ++t) {
            assert(fabs(downloaded[base + t].re - expected_axis[t].re) <= NLO_TEST_FREQ_EPS);
            assert(fabs(downloaded[base + t].im - expected_axis[t].im) <= NLO_TEST_FREQ_EPS);
        }
    }

    nlo_complex potential_downloaded[12] = {0};
    assert(nlo_vec_download(state->backend,
                            state->working_vectors.potential_vec,
                            potential_downloaded,
                            sizeof(potential_downloaded)) == NLO_VEC_STATUS_OK);
    assert(fabs(potential_downloaded[0].re + 1.5) <= NLO_TEST_FREQ_EPS);

    free_simulation_state(state);
    free_sim_config(config);

    printf("test_tensor_mode_frequency_mesh_generation: validates tensor t-fast frequency mesh.\n");
}

static void test_snapshot_store_dense_readback(void)
{
    if (!nlo_snapshot_store_is_available()) {
        printf("test_snapshot_store_dense_readback: skipped (storage unavailable).\n");
        return;
    }

    const size_t num_time_samples = 16u;
    const size_t num_records = 5u;
    const size_t total = num_time_samples * num_records;

    sim_config* config = create_sim_config(num_time_samples);
    assert(config != NULL);
    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);

    char db_path[256];
    (void)snprintf(db_path, sizeof(db_path), "test_snapshot_store_readback_%u.sqlite3", (unsigned)rand());

    nlo_storage_options storage_options = nlo_storage_options_default();
    storage_options.sqlite_path = db_path;
    storage_options.run_id = "test-snapshot-readback";
    storage_options.chunk_records = 2u;

    nlo_snapshot_store_open_params open_params;
    open_params.config = config;
    open_params.exec_options = &exec_options;
    open_params.storage_options = &storage_options;
    open_params.num_time_samples = num_time_samples;
    open_params.num_recorded_samples = num_records;

    nlo_snapshot_store* store = nlo_snapshot_store_open(&open_params);
    assert(store != NULL);

    nlo_complex* expected = (nlo_complex*)calloc(total, sizeof(nlo_complex));
    nlo_complex* restored = (nlo_complex*)calloc(total, sizeof(nlo_complex));
    assert(expected != NULL);
    assert(restored != NULL);

    for (size_t record = 0u; record < num_records; ++record) {
        for (size_t t = 0u; t < num_time_samples; ++t) {
            const size_t idx = (record * num_time_samples) + t;
            expected[idx] = nlo_make((double)(100u * record + t), (double)(-1.0 * (double)record));
        }
        assert(nlo_snapshot_store_write_record(
                   store,
                   record,
                   expected + (record * num_time_samples),
                   num_time_samples) != NLO_SNAPSHOT_STORE_STATUS_ERROR);
    }

    assert(nlo_snapshot_store_read_all_records(store, restored, num_records, num_time_samples) ==
           NLO_SNAPSHOT_STORE_STATUS_OK);
    for (size_t i = 0u; i < total; ++i) {
        assert(fabs(restored[i].re - expected[i].re) <= NLO_TEST_FREQ_EPS);
        assert(fabs(restored[i].im - expected[i].im) <= NLO_TEST_FREQ_EPS);
    }

    nlo_snapshot_store_close(store);
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

int main(void)
{
    test_init_state_success();
    test_init_state_invalid_args();
    test_init_state_xy_shape_rejected_without_tensor();
    test_init_state_tensor_shape_rules();
    test_default_frequency_grid_generated_when_invalid();
    test_frequency_grid_preserved_when_valid();
    test_tensor_mode_frequency_mesh_generation();
    test_snapshot_store_dense_readback();
    printf("test_core_state_alloc: all subtests completed.\n");
    return 0;
}
