/**
 * @file test_propagate_async.c
 * @brief Asynchronous propagation jobs: start, poll, cancel, finish.
 */

#include "nlolib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASYNC_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "%s:%d: ASYNC_CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define NUM_SAMPLES 256u
#define NUM_RECORDS 4u
#define TWO_PI 6.283185307179586476925286766559

typedef struct
{
    simulation_config sim;
    physics_config physics;
    execution_options exec;
    propagate_options options;
    propagate_output output;
    nlo_complex input[NUM_SAMPLES];
    nlo_complex frequency_grid[NUM_SAMPLES];
    nlo_complex records[NUM_SAMPLES * NUM_RECORDS];
    size_t records_written;
} run_fixture;

static void fixture_init(run_fixture* run, double distance, double max_step)
{
    const double delta_time = 0.02;
    memset(run, 0, sizeof(*run));

    for (size_t i = 0u; i < NUM_SAMPLES; ++i) {
        const double t = ((double)i - 0.5 * (double)NUM_SAMPLES) * delta_time;
        run->input[i].re = exp(-(t / 0.25) * (t / 0.25));
        const double k = (i < NUM_SAMPLES / 2u) ? (double)i : ((double)i - (double)NUM_SAMPLES);
        run->frequency_grid[i].re = TWO_PI * k / ((double)NUM_SAMPLES * delta_time);
    }

    run->sim.propagation.propagation_distance = distance;
    run->sim.propagation.starting_step_size = max_step;
    run->sim.propagation.max_step_size = max_step;
    run->sim.propagation.min_step_size = 1e-7;
    run->sim.propagation.error_tolerance = 1e-7;
    run->sim.time.pulse_period = (double)NUM_SAMPLES * delta_time;
    run->sim.time.delta_time = delta_time;
    run->sim.frequency.frequency_grid = run->frequency_grid;
    run->sim.spatial.nx = NUM_SAMPLES;
    run->sim.spatial.ny = 1u;
    run->sim.spatial.delta_x = 1.0;
    run->sim.spatial.delta_y = 1.0;

    run->physics.dispersion_factor_expr = "i*c0*w*w";
    run->physics.num_constants = 3u;
    run->physics.constants[0] = 0.025;
    run->physics.constants[2] = 0.01;

    run->exec.backend_type = VECTOR_BACKEND_CPU;
    run->exec.fft_backend = FFT_BACKEND_FFTW;
    run->exec.device_heap_fraction = 0.70;

    run->options = nlolib_propagate_options_default();
    run->options.exec_options = &run->exec;
    run->options.num_recorded_samples = NUM_RECORDS;

    run->output = nlolib_propagate_output_default();
    run->output.output_records = run->records;
    run->output.output_record_capacity = NUM_RECORDS;
    run->output.records_written = &run->records_written;
}

static void test_async_matches_sync(void)
{
    static run_fixture sync_run;
    static run_fixture async_run;
    fixture_init(&sync_run, 0.25, 5e-3);
    fixture_init(&async_run, 0.25, 5e-3);

    ASYNC_CHECK(nlolib_propagate(&sync_run.sim, &sync_run.physics, NUM_SAMPLES, sync_run.input,
                                 &sync_run.options, &sync_run.output) == NLOLIB_STATUS_OK);

    nlolib_propagate_job* job = NULL;
    ASYNC_CHECK(nlolib_propagate_start(&async_run.sim, &async_run.physics, NUM_SAMPLES, async_run.input,
                                       &async_run.options, &async_run.output, &job) == NLOLIB_STATUS_OK);
    ASYNC_CHECK(job != NULL);

    /* The job copied these; clobbering them must not reach the worker. */
    const propagate_options options_copy = async_run.options;
    memset(&async_run.options, 0xff, sizeof(async_run.options));
    memset(&async_run.exec, 0xff, sizeof(async_run.exec));

    int done = 0;
    progress_info info;
    memset(&info, 0, sizeof(info));
    for (int attempt = 0; attempt < 6000 && done == 0; ++attempt) {
        ASYNC_CHECK(nlolib_propagate_poll(job, 10u, &info, &done) == NLOLIB_STATUS_OK);
    }
    ASYNC_CHECK(done != 0);
    ASYNC_CHECK(info.event_type == PROGRESS_EVENT_FINISH);
    ASYNC_CHECK(info.percent >= 100.0 - 1e-9);
    ASYNC_CHECK(nlolib_propagate_finish(job) == NLOLIB_STATUS_OK);
    async_run.options = options_copy;

    ASYNC_CHECK(async_run.records_written == sync_run.records_written);
    ASYNC_CHECK(async_run.records_written == NUM_RECORDS);
    ASYNC_CHECK(memcmp(async_run.records, sync_run.records, sizeof(sync_run.records)) == 0);

    printf("test_async_matches_sync: an asynchronous job reproduces nlolib_propagate.\n");
}

static void test_cancel_and_single_job(void)
{
    static run_fixture run;
    static run_fixture second;
    /* 10^7 steps: far longer than the test takes to cancel it. */
    fixture_init(&run, 1.0e4, 1e-3);
    run.sim.propagation.min_step_size = 1e-3;
    run.options.num_recorded_samples = 1u;
    run.options.output_mode = PROPAGATE_OUTPUT_FINAL_ONLY;
    fixture_init(&second, 0.25, 5e-3);

    nlolib_propagate_job* job = NULL;
    ASYNC_CHECK(nlolib_propagate_start(&run.sim, &run.physics, NUM_SAMPLES, run.input,
                                       &run.options, &run.output, &job) == NLOLIB_STATUS_OK);

    nlolib_propagate_job* rejected = (nlolib_propagate_job*)&run;
    ASYNC_CHECK(nlolib_propagate_start(&second.sim, &second.physics, NUM_SAMPLES, second.input,
                                       &second.options, &second.output, &rejected) ==
                NLOLIB_STATUS_INVALID_ARGUMENT);
    ASYNC_CHECK(rejected == NULL);

    /* The log ring is read here while the worker appends to it. */
    ASYNC_CHECK(nlolib_set_log_buffer(8192u) == NLOLIB_STATUS_OK);
    int done = 0;
    progress_info info;
    memset(&info, 0, sizeof(info));
    for (int attempt = 0; attempt < 6000 && info.step_index < 8u; ++attempt) {
        char text[512];
        size_t written = 0u;
        ASYNC_CHECK(nlolib_read_log_buffer(text, sizeof(text), &written, 1) == NLOLIB_STATUS_OK);
        ASYNC_CHECK(nlolib_propagate_poll(job, 10u, &info, &done) == NLOLIB_STATUS_OK);
        ASYNC_CHECK(done == 0);
    }
    ASYNC_CHECK(info.step_index >= 8u);
    ASYNC_CHECK(info.percent < 100.0);

    ASYNC_CHECK(nlolib_propagate_cancel(job) == NLOLIB_STATUS_OK);
    ASYNC_CHECK(nlolib_propagate_finish(job) == NLOLIB_STATUS_ABORTED);
    ASYNC_CHECK(nlolib_set_log_buffer(0u) == NLOLIB_STATUS_OK);

    /* The slot is free again once the job is finished. */
    job = NULL;
    ASYNC_CHECK(nlolib_propagate_start(&second.sim, &second.physics, NUM_SAMPLES, second.input,
                                       &second.options, &second.output, &job) == NLOLIB_STATUS_OK);
    ASYNC_CHECK(nlolib_propagate_finish(job) == NLOLIB_STATUS_OK);
    ASYNC_CHECK(second.records_written == NUM_RECORDS);

    printf("test_cancel_and_single_job: cancel aborts the job; one job runs at a time.\n");
}

static void test_argument_errors(void)
{
    static run_fixture run;
    fixture_init(&run, 0.25, 5e-3);

    ASYNC_CHECK(nlolib_propagate_start(&run.sim, &run.physics, NUM_SAMPLES, run.input,
                                       &run.options, &run.output, NULL) == NLOLIB_STATUS_INVALID_ARGUMENT);
    ASYNC_CHECK(nlolib_propagate_poll(NULL, 0u, NULL, NULL) == NLOLIB_STATUS_INVALID_ARGUMENT);
    ASYNC_CHECK(nlolib_propagate_cancel(NULL) == NLOLIB_STATUS_INVALID_ARGUMENT);
    ASYNC_CHECK(nlolib_propagate_finish(NULL) == NLOLIB_STATUS_INVALID_ARGUMENT);

    /* An invalid propagation starts, and reports its error from finish. */
    nlolib_propagate_job* job = NULL;
    ASYNC_CHECK(nlolib_propagate_start(&run.sim, &run.physics, 0u, run.input,
                                       &run.options, &run.output, &job) == NLOLIB_STATUS_OK);
    ASYNC_CHECK(nlolib_propagate_finish(job) == NLOLIB_STATUS_INVALID_ARGUMENT);

    printf("test_argument_errors: null handles and invalid runs are reported.\n");
}

int main(void)
{
    ASYNC_CHECK(nlolib_set_progress_options(0, 5, 0) == NLOLIB_STATUS_OK);
    test_async_matches_sync();
    test_cancel_and_single_job();
    test_argument_errors();
    printf("test_core_propagate_async: all subtests completed.\n");
    return 0;
}
