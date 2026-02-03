/**
 * @file init.c
 * @dir src/core
 * @brief Initialization helpers for simulation state setup.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "core/init.h"
#include "core/state.h"
#include "backend/nlo_complex.h"
#include <stdio.h>
#include <stdint.h>

static int checked_mul_size_t(size_t a, size_t b, size_t *out);
static int checked_add_size_t(size_t a, size_t b, size_t *out);
NLOLIB_API int nlo_init_simulation_state(const sim_config *config,
                              size_t num_time_samples,
                              size_t num_recorded_samples,
                              nlo_allocation_info *allocation_info,
                              simulation_state **out_state)
{
    simulation_state *state = NULL;
    nlo_allocation_info local_info = {0};
    size_t per_record_bytes = 0;
    size_t field_bytes = 0;
    size_t work_buffers = 0;
    size_t work_bytes = 0;
    size_t block_bytes = 0;
    size_t total_bytes = 0;
    size_t records_per_block = 0;
    size_t num_blocks = 0;

    if (out_state == NULL)
    {
        return -1;
    }

    *out_state = NULL;

    if (allocation_info != NULL)
    {
        *allocation_info = (nlo_allocation_info){0};
    }

    if (config == NULL || num_time_samples == 0 || num_recorded_samples == 0)
    {
        return -1;
    }

    state = create_simulation_state(config, num_time_samples, num_recorded_samples);
    if (state == NULL)
    {
        return -1;
    }

    records_per_block = state->num_recorded_samples;
    num_blocks = (records_per_block == 0) ? 0 :
        (num_recorded_samples / records_per_block) +
        ((num_recorded_samples % records_per_block) ? 1u : 0u);

    if (checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &per_record_bytes) != 0)
    {
        per_record_bytes = 0;
    }

    if (checked_mul_size_t(per_record_bytes, records_per_block, &field_bytes) != 0)
    {
        field_bytes = 0;
    }

    work_buffers = NLO_WORK_BUFFER_COUNT;
    if (checked_mul_size_t(per_record_bytes, work_buffers, &work_bytes) != 0)
    {
        work_bytes = 0;
    }

    if (checked_add_size_t(field_bytes, work_bytes, &block_bytes) != 0)
    {
        block_bytes = 0;
    }

    if (checked_mul_size_t(block_bytes, num_blocks, &total_bytes) != 0)
    {
        total_bytes = 0;
    }

    local_info.requested_records = num_recorded_samples;
    local_info.records_per_block = records_per_block;
    local_info.num_blocks = num_blocks;
    local_info.per_record_bytes = per_record_bytes;
    local_info.working_bytes = work_bytes;
    local_info.block_bytes = block_bytes;
    local_info.total_bytes = total_bytes;

    if (allocation_info != NULL)
    {
        *allocation_info = local_info;
    }

    if (block_bytes > 0 && num_blocks > 0)
    {
        printf("nlolib: allocation block size %zu bytes, blocks required %zu.\n",
               block_bytes,
               num_blocks);
    }

    *out_state = state;
    return 0;
}

static int checked_mul_size_t(size_t a, size_t b, size_t *out)
{
    if (out == NULL)
    {
        return -1;
    }

    if (a == 0 || b == 0)
    {
        *out = 0;
        return 0;
    }

    if (a > SIZE_MAX / b)
    {
        return -1;
    }

    *out = a * b;
    return 0;
}

static int checked_add_size_t(size_t a, size_t b, size_t *out)
{
    if (out == NULL)
    {
        return -1;
    }

    if (a > SIZE_MAX - b)
    {
        return -1;
    }

    *out = a + b;
    return 0;
}
