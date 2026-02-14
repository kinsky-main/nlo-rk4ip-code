/**
 * @file test_state_alloc.c
 * @brief Unit tests for simulation state allocation sizing.
 */

#include "backend/nlo_complex.h"
#include "core/init.h"
#include "core/state.h"
#include <assert.h>
#include <stdio.h>

static void test_init_state_success(void)
{
    const size_t num_time_samples = 1024;
    const size_t num_records = 64;

    sim_config* config = create_sim_config(4, num_time_samples);
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

    sim_config* config = create_sim_config(1, 8);
    assert(config != NULL);

    assert(nlo_init_simulation_state(config, 0, 1, &exec_options, &info, &state) != 0);
    assert(state == NULL);

    assert(nlo_init_simulation_state(config, 8, 0, &exec_options, &info, &state) != 0);
    assert(state == NULL);

    free_sim_config(config);

    printf("test_init_state_invalid_args: validates argument checks.\n");
}

int main(void)
{
    test_init_state_success();
    test_init_state_invalid_args();
    printf("test_core_state_alloc: all subtests completed.\n");
    return 0;
}
