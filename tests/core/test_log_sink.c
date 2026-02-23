/**
 * @file test_log_sink.c
 * @brief Unit tests for runtime log formatting and in-memory sink behavior.
 */

#include "io/log_format.h"
#include "io/log_sink.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    assert(strstr(out, "z_percent") != NULL);
    assert(strstr(out, "step_adjustment") != NULL);
    assert(strstr(out, "step_rejected") != NULL);
    printf("test_progress_entries: passed.\n");
}

int main(void)
{
    test_grouped_integer_format();
    test_size_format();
    test_log_buffer_roundtrip();
    test_progress_entries();

    (void)nlo_log_set_buffer(0u);
    (void)nlo_log_set_file(NULL, 0);
    printf("test_core_log_sink: all subtests completed.\n");
    return 0;
}
