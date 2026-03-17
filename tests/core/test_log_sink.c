/**
 * @file test_log_sink.c
 * @brief Unit tests for runtime log formatting and in-memory sink behavior.
 */

#include "io/log_format.h"
#include "io/propagate_log.h"
#include "io/log_sink.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_progress_callback_calls = 0;

nlo_execution_options nlo_execution_options_default(nlo_vector_backend_type backend_type)
{
    nlo_execution_options options;
    memset(&options, 0, sizeof(options));
    options.backend_type = backend_type;
    return options;
}

static int test_progress_callback(const nlo_progress_info* info, void* user_data)
{
    int* abort_after_first = (int*)user_data;
    assert(info != NULL);
    g_progress_callback_calls += 1;
    if (abort_after_first != NULL && *abort_after_first != 0) {
        *abort_after_first = 0;
        return 0;
    }
    return 1;
}

static void test_grouped_integer_format(void)
{
    char text[64];
    (void)nlo_log_format_u64_grouped(text, sizeof(text), 0u);
    assert(strcmp(text, "0") == 0);
    (void)nlo_log_format_u64_grouped(text, sizeof(text), 24u);
    assert(strcmp(text, "24") == 0);
    (void)nlo_log_format_u64_grouped(text, sizeof(text), 60u);
    assert(strcmp(text, "60") == 0);
    (void)nlo_log_format_u64_grouped(text, sizeof(text), 1000u);
    assert(strcmp(text, "1,000") == 0);
    (void)nlo_log_format_u64_grouped(text, sizeof(text), 16384u);
    assert(strcmp(text, "16,384") == 0);
    (void)nlo_log_format_u64_grouped(text, sizeof(text), 2621440u);
    assert(strcmp(text, "2,621,440") == 0);
    (void)nlo_log_format_u64_grouped(text, sizeof(text), 1234567890ull);
    assert(strcmp(text, "1,234,567,890") == 0);
    printf("test_grouped_integer_format: passed.\n");
}

static void test_size_format(void)
{
    char text[64];
    (void)nlo_log_format_bytes_human(text, sizeof(text), 512u);
    assert(strcmp(text, "512 B") == 0);
    (void)nlo_log_format_bytes_human(text, sizeof(text), 2048u);
    assert(strcmp(text, "2.0 KB") == 0);
    (void)nlo_log_format_bytes_human(text, sizeof(text), 16384u);
    assert(strcmp(text, "16.0 KB") == 0);
    (void)nlo_log_format_bytes_human(text, sizeof(text), 2621440u);
    assert(strcmp(text, "2.5 MB") == 0);
    (void)nlo_log_format_bytes_human(text, sizeof(text), (size_t)(3u * 1024u * 1024u));
    assert(strcmp(text, "3.0 MB") == 0);
    printf("test_size_format: passed.\n");
}

static void test_log_buffer_roundtrip(void)
{
    assert(nlo_log_set_level(NLO_LOG_LEVEL_INFO) == 0);
    assert(nlo_log_set_buffer(8192u) == 0);
    assert(nlo_log_clear_buffer() == 0);

    nlo_log_emit(NLO_LOG_LEVEL_INFO, "[test] hello %d", 12345);

    char out[1024];
    size_t written = 0u;
    assert(nlo_log_read_buffer(out, sizeof(out), &written, 0) == 0);
    assert(written > 0u);
    assert(strstr(out, "hello 12345") != NULL);

    assert(nlo_log_read_buffer(out, sizeof(out), &written, 1) == 0);
    assert(written > 0u);

    assert(nlo_log_read_buffer(out, sizeof(out), &written, 0) == 0);
    assert(written == 0u);
    printf("test_log_buffer_roundtrip: passed.\n");
}

static void test_progress_stream_options(void)
{
    assert(nlo_log_set_progress_stream(NLO_LOG_PROGRESS_STREAM_STDERR) == 0);
    assert(nlo_log_set_progress_stream(NLO_LOG_PROGRESS_STREAM_STDOUT) == 0);
    assert(nlo_log_set_progress_stream(NLO_LOG_PROGRESS_STREAM_BOTH) == 0);
    assert(nlo_log_set_progress_stream(-1) != 0);
    assert(nlo_log_set_progress_stream(3) != 0);
    assert(nlo_log_set_progress_stream(NLO_LOG_PROGRESS_STREAM_STDERR) == 0);
    printf("test_progress_stream_options: passed.\n");
}

static void test_progress_entries(void)
{
    assert(nlo_log_set_buffer(8192u) == 0);
    assert(nlo_log_clear_buffer() == 0);
    assert(nlo_log_set_level(NLO_LOG_LEVEL_INFO) == 0);
    assert(nlo_log_set_progress_options(1, 25, 1) == 0);

    nlo_log_progress_begin(0.0, 1.0);
    nlo_log_progress_step_rejected(0u, 0.0, 1.0, 0.1, 10.0, 0.02, 1u);
    nlo_log_progress_step_accepted(0u, 0.30, 1.0, 0.02, 0.5, 0.01);
    nlo_log_progress_finish(1.0, 1.0, 1);

    char out[4096];
    size_t written = 0u;
    assert(nlo_log_read_buffer(out, sizeof(out), &written, 1) == 0);
    assert(written > 0u);
    assert(strstr(out, "progress_summary") != NULL);
    assert(strstr(out, "elapsed_seconds") != NULL);
    assert(strstr(out, "state: complete") != NULL);
    printf("test_progress_entries: passed.\n");
}

static void test_progress_callback_abort(void)
{
    int abort_after_first = 1;
    g_progress_callback_calls = 0;

    assert(nlo_log_set_progress_options(0, 25, 0) == 0);
    assert(nlo_log_set_progress_callback(test_progress_callback, &abort_after_first) == 0);

    nlo_log_progress_begin(0.0, 1.0);
    nlo_log_progress_step_accepted(0u, 0.25, 1.0, 0.02, 0.0, 0.02);
    assert(g_progress_callback_calls == 1);
    assert(nlo_log_progress_abort_requested() != 0);
    nlo_log_progress_finish(0.25, 1.0, 0);

    assert(nlo_log_set_progress_callback(NULL, NULL) == 0);
    printf("test_progress_callback_abort: passed.\n");
}

static void test_allocation_summary_marks_non_vulkan_fields_not_applicable(void)
{
    nlo_allocation_info allocation = {0};
    allocation.per_record_bytes = 65536u;
    allocation.working_vector_bytes = 917504u;
    allocation.host_snapshot_bytes = 10485760u;

    assert(nlo_log_set_buffer(8192u) == 0);
    assert(nlo_log_clear_buffer() == 0);
    assert(nlo_log_set_level(NLO_LOG_LEVEL_INFO) == 0);

    nlo_log_propagate_allocation_summary(NLO_VECTOR_BACKEND_AUTO,
                                         NLO_VECTOR_BACKEND_CPU,
                                         &allocation,
                                         NULL);

    char out[4096];
    size_t written = 0u;
    assert(nlo_log_read_buffer(out, sizeof(out), &written, 1) == 0);
    assert(written > 0u);
    assert(strstr(out, "record_ring_capacity: n/a (non-Vulkan backend)") != NULL);
    assert(strstr(out, "record_ring_bytes: n/a (non-Vulkan backend)") != NULL);
    assert(strstr(out, "device_budget_bytes_effective: n/a (non-Vulkan backend)") != NULL);
    assert(strstr(out, "device_local_total_bytes: n/a (non-Vulkan backend)") != NULL);
    assert(strstr(out, "device_local_available_bytes: n/a (non-Vulkan backend)") != NULL);
    printf("test_allocation_summary_marks_non_vulkan_fields_not_applicable: passed.\n");
}

int main(void)
{
    test_grouped_integer_format();
    test_size_format();
    test_log_buffer_roundtrip();
    test_progress_stream_options();
    test_progress_entries();
    test_progress_callback_abort();
    test_allocation_summary_marks_non_vulkan_fields_not_applicable();

    (void)nlo_log_set_buffer(0u);
    (void)nlo_log_set_file(NULL, 0);
    printf("test_core_log_sink: all subtests completed.\n");
    return 0;
}
