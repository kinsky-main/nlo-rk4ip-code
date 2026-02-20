/**
 * @file test_state_alloc.c
 * @brief Unit tests for simulation state allocation sizing.
 */

#include "backend/nlo_complex.h"
#include "core/init.h"
#include "core/state.h"
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

static void test_init_state_xy_shape_success(void)
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
                                     &state) == 0);
    assert(state != NULL);
    assert(state->num_points_xy == num_time_samples);

    free_simulation_state(state);
    free_sim_config(config);

    printf("test_init_state_xy_shape_success: validates flattened XY shape handling.\n");
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

static void test_execution_options_default_heap_fraction_auto(void)
{
    nlo_execution_options options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    assert(options.device_heap_fraction == 0.0);
    assert(options.backend_type == NLO_VECTOR_BACKEND_CPU);
    assert(options.record_ring_target == 0u);
    assert(options.forced_device_budget_bytes == 0u);

    nlo_execution_options vk_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_VULKAN);
    assert(vk_options.device_heap_fraction == 0.0);
    assert(vk_options.backend_type == NLO_VECTOR_BACKEND_VULKAN);

    printf("test_execution_options_default_heap_fraction_auto: validates auto device_heap_fraction.\n");
}

static void test_explicit_heap_fraction_preserved(void)
{
    const size_t num_time_samples = 16;
    const size_t num_records = 4;

    sim_config* config = create_sim_config(num_time_samples);
    assert(config != NULL);

    nlo_execution_options exec_options = nlo_execution_options_default(NLO_VECTOR_BACKEND_CPU);
    exec_options.device_heap_fraction = 0.50;

    simulation_state* state = NULL;
    nlo_allocation_info info = {0};
    assert(nlo_init_simulation_state(config,
                                     num_time_samples,
                                     num_records,
                                     &exec_options,
                                     &info,
                                     &state) == 0);
    assert(state != NULL);
    assert(info.backend_type == NLO_VECTOR_BACKEND_CPU);
    assert(info.allocated_records == num_records);

    free_simulation_state(state);
    free_sim_config(config);

    printf("test_explicit_heap_fraction_preserved: validates explicit fraction override.\n");
}

int main(void)
{
    test_init_state_success();
    test_init_state_invalid_args();
    test_init_state_xy_shape_success();
    test_default_frequency_grid_generated_when_invalid();
    test_frequency_grid_preserved_when_valid();
    test_execution_options_default_heap_fraction_auto();
    test_explicit_heap_fraction_preserved();
    printf("test_core_state_alloc: all subtests completed.\n");
    return 0;
}
