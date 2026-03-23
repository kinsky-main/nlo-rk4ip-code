/**
 * @file test_perf_profile.c
 * @brief Unit tests for internal performance profiling helpers.
 */

#include "physics/operator_program_jit.h"
#include "utility/perf_profile.h"

#include <assert.h>
#include <stdio.h>

static void test_event_accumulation(void)
{
    nlo_perf_profile_snapshot snapshot = {0};

    nlo_perf_profile_set_enabled(1);
    nlo_perf_profile_reset();
    nlo_perf_profile_add_event(NLO_PERF_EVENT_VEC_COPY, 1.25, 64u);
    nlo_perf_profile_add_event(NLO_PERF_EVENT_VEC_COPY, 0.75, 32u);
    nlo_perf_profile_add_event_gpu_time(NLO_PERF_EVENT_VEC_COPY, 0.50);
    nlo_perf_profile_snapshot_read(&snapshot);

    assert(snapshot.event_call_count[NLO_PERF_EVENT_VEC_COPY] == 2u);
    assert(snapshot.event_bytes[NLO_PERF_EVENT_VEC_COPY] == 96u);
    assert(snapshot.event_total_ms[NLO_PERF_EVENT_VEC_COPY] > 1.99);
    assert(snapshot.event_gpu_exec_ms[NLO_PERF_EVENT_VEC_COPY] > 0.49);
    nlo_perf_profile_set_enabled(0);
    printf("test_event_accumulation: passed.\n");
}

static void test_trace_overflow(void)
{
    nlo_perf_profile_trace_row rows[2];
    nlo_perf_profile_snapshot snapshot = {0};

    nlo_perf_profile_set_enabled(1);
    nlo_perf_profile_reset();
    nlo_perf_profile_trace_configure(rows, 2u);
    nlo_perf_profile_trace_set_run_metadata(11u, 2u, 3u, 4u);
    nlo_perf_profile_trace_set_step(5u, 1u);
    nlo_perf_profile_add_event(NLO_PERF_EVENT_FFT_FORWARD, 0.1, 16u);
    nlo_perf_profile_add_event(NLO_PERF_EVENT_FFT_INVERSE, 0.2, 32u);
    nlo_perf_profile_add_event(NLO_PERF_EVENT_VEC_ADD, 0.3, 48u);
    nlo_perf_profile_snapshot_read(&snapshot);

    assert(snapshot.trace_row_count == 2u);
    assert(snapshot.trace_dropped_count == 1u);
    assert(rows[0].run_id == 11u);
    assert(rows[0].step_index == 5u);
    assert(rows[0].event_id == (uint64_t)NLO_PERF_EVENT_FFT_FORWARD);
    nlo_perf_profile_trace_configure(NULL, 0u);
    nlo_perf_profile_set_enabled(0);
    printf("test_trace_overflow: passed.\n");
}

static void test_jit_mode_toggle(void)
{
    nlo_operator_program_set_jit_mode(NLO_OPERATOR_JIT_MODE_OFF);
    assert(nlo_operator_program_get_jit_mode() == NLO_OPERATOR_JIT_MODE_OFF);
    nlo_operator_program_set_jit_mode(NLO_OPERATOR_JIT_MODE_ON);
    assert(nlo_operator_program_get_jit_mode() == NLO_OPERATOR_JIT_MODE_ON);
    printf("test_jit_mode_toggle: passed.\n");
}

static void test_gpu_timestamp_mode_toggle(void)
{
    nlo_perf_profile_set_gpu_timestamp_mode(NLO_PERF_GPU_TIMESTAMPS_OFF);
    assert(nlo_perf_profile_get_gpu_timestamp_mode() == NLO_PERF_GPU_TIMESTAMPS_OFF);
    nlo_perf_profile_set_gpu_timestamp_mode(NLO_PERF_GPU_TIMESTAMPS_ON);
    assert(nlo_perf_profile_get_gpu_timestamp_mode() == NLO_PERF_GPU_TIMESTAMPS_ON);
    nlo_perf_profile_set_gpu_timestamp_mode(NLO_PERF_GPU_TIMESTAMPS_AUTO);
    assert(nlo_perf_profile_get_gpu_timestamp_mode() == NLO_PERF_GPU_TIMESTAMPS_AUTO);
    printf("test_gpu_timestamp_mode_toggle: passed.\n");
}

int main(void)
{
    test_event_accumulation();
    test_trace_overflow();
    test_jit_mode_toggle();
    test_gpu_timestamp_mode_toggle();
    return 0;
}
