/**
 * @file test_state_alloc.c
 * @dir tests/core
 * @brief Unit tests for simulation state allocation sizing.
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */

#include "core/init.h"
#include "core/state.h"
#include "fft/nlo_complex.h"
#include <assert.h>
#include <stdio.h>

static void test_init_state_success(void)
{
    const size_t num_time_samples = 1024;
    const size_t num_records = 1000;

    sim_config *config = create_sim_config(1, num_time_samples);
    assert(config != NULL);

    simulation_state *state = NULL;
    nlo_allocation_info info = {0};

    assert(nlo_init_simulation_state(config,
                                     num_time_samples,
                                     num_records,
                                     &info,
                                     &state) == 0);
    assert(state != NULL);
    assert(info.requested_records == num_records);
    assert(info.records_per_block == state->num_recorded_samples);
    assert(info.records_per_block > 0);

    const size_t expected_blocks =
        (num_records / info.records_per_block) +
        ((num_records % info.records_per_block) ? 1u : 0u);

    assert(info.num_blocks == expected_blocks);
    assert(info.per_record_bytes == num_time_samples * sizeof(nlo_complex));
    assert(info.block_bytes == info.per_record_bytes * info.records_per_block + info.working_bytes);
    assert(info.total_bytes == info.block_bytes * info.num_blocks);

    free_simulation_state(state);
    free_sim_config(config);

    printf("test_init_state_success: validates allocation sizing info.\n");
}

static void test_init_state_invalid_args(void)
{
    simulation_state *state = NULL;
    nlo_allocation_info info = {0};

    assert(nlo_init_simulation_state(NULL, 8, 1, &info, &state) != 0);
    assert(state == NULL);

    sim_config *config = create_sim_config(1, 8);
    assert(config != NULL);

    assert(nlo_init_simulation_state(config, 0, 1, &info, &state) != 0);
    assert(state == NULL);

    assert(nlo_init_simulation_state(config, 8, 0, &info, &state) != 0);
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
