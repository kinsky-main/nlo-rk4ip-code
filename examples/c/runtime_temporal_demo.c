#include "nlolib.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define DEMO_NUM_TIME_SAMPLES 512u
#define DEMO_NUM_RECORDED_SAMPLES 2u
#define DEMO_PI 3.14159265358979323846

static void fill_input_field(double delta_time, nlo_complex* out_field)
{
    const double center = 0.5 * (double)(DEMO_NUM_TIME_SAMPLES - 1u);
    for (size_t i = 0u; i < DEMO_NUM_TIME_SAMPLES; ++i) {
        const double t = ((double)i - center) * delta_time;
        const double envelope = exp(-pow(t / 0.25, 2.0));
        const double phase = -8.0 * t;
        out_field[i] = nlo_make(envelope * cos(phase), envelope * sin(phase));
    }
}

static void fill_frequency_grid(double delta_time, nlo_complex* out_grid)
{
    const double factor = 1.0 / ((double)DEMO_NUM_TIME_SAMPLES * delta_time);
    const size_t half = (DEMO_NUM_TIME_SAMPLES - 1u) / 2u;

    for (size_t i = 0u; i < DEMO_NUM_TIME_SAMPLES; ++i) {
        const double cycles =
            (i <= half)
                ? ((double)i * factor)
                : (-(double)(DEMO_NUM_TIME_SAMPLES - i) * factor);
        out_grid[i] = nlo_make(2.0 * DEMO_PI * cycles, 0.0);
    }
}

static double compute_power(const nlo_complex* field)
{
    double power = 0.0;
    for (size_t i = 0u; i < DEMO_NUM_TIME_SAMPLES; ++i) {
        const double re = field[i].re;
        const double im = field[i].im;
        power += (re * re) + (im * im);
    }
    return power;
}

static const char* status_to_string(nlolib_status status)
{
    switch (status) {
        case NLOLIB_STATUS_OK:
            return "NLOLIB_STATUS_OK";
        case NLOLIB_STATUS_INVALID_ARGUMENT:
            return "NLOLIB_STATUS_INVALID_ARGUMENT";
        case NLOLIB_STATUS_ALLOCATION_FAILED:
            return "NLOLIB_STATUS_ALLOCATION_FAILED";
        case NLOLIB_STATUS_NOT_IMPLEMENTED:
            return "NLOLIB_STATUS_NOT_IMPLEMENTED";
        default:
            return "NLOLIB_STATUS_UNKNOWN";
    }
}

int main(void)
{
    const double delta_time = 0.02;
    const double beta2 = 0.05;
    static const char* k_runtime_dispersion_factor_expr = "i*c0*w*w";

    static sim_config cfg;
    static nlo_execution_options exec_options;
    static nlo_complex input_field[DEMO_NUM_TIME_SAMPLES];
    static nlo_complex frequency_grid[DEMO_NUM_TIME_SAMPLES];
    static nlo_complex output_records[DEMO_NUM_TIME_SAMPLES * DEMO_NUM_RECORDED_SAMPLES];

    memset(&cfg, 0, sizeof(cfg));
    memset(&exec_options, 0, sizeof(exec_options));
    memset(output_records, 0, sizeof(output_records));

    fill_input_field(delta_time, input_field);
    fill_frequency_grid(delta_time, frequency_grid);

    cfg.propagation.propagation_distance = 0.25;
    cfg.propagation.starting_step_size = 1e-3;
    cfg.propagation.max_step_size = 5e-3;
    cfg.propagation.min_step_size = 1e-5;
    cfg.propagation.error_tolerance = 1e-7;

    cfg.time.pulse_period = (double)DEMO_NUM_TIME_SAMPLES * delta_time;
    cfg.time.delta_time = delta_time;

    cfg.frequency.frequency_grid = frequency_grid;

    cfg.spatial.nx = DEMO_NUM_TIME_SAMPLES;
    cfg.spatial.ny = 1u;
    cfg.spatial.delta_x = 1.0;
    cfg.spatial.delta_y = 1.0;
    cfg.spatial.spatial_frequency_grid = NULL;
    cfg.spatial.potential_grid = NULL;

    cfg.runtime.dispersion_factor_expr = k_runtime_dispersion_factor_expr;
    cfg.runtime.dispersion_expr = NULL;
    cfg.runtime.nonlinear_expr = NULL;
    cfg.runtime.num_constants = 3u;
    cfg.runtime.constants[0] = 0.5 * beta2;
    cfg.runtime.constants[1] = 0.0;
    cfg.runtime.constants[2] = 0.01;

    exec_options.backend_type = NLO_VECTOR_BACKEND_CPU;
    exec_options.fft_backend = NLO_FFT_BACKEND_FFTW;
    exec_options.device_heap_fraction = 0.0;
    exec_options.record_ring_target = 0u;
    exec_options.forced_device_budget_bytes = 0u;

    const nlolib_status status = nlolib_propagate(&cfg,
                                                  DEMO_NUM_TIME_SAMPLES,
                                                  input_field,
                                                  DEMO_NUM_RECORDED_SAMPLES,
                                                  output_records,
                                                  &exec_options);
    if (status != NLOLIB_STATUS_OK) {
        fprintf(stderr,
                "runtime_temporal_demo_c: nlolib_propagate failed with status=%d (%s)\n",
                (int)status,
                status_to_string(status));
        return 1;
    }

    const nlo_complex* final_field =
        output_records + ((DEMO_NUM_RECORDED_SAMPLES - 1u) * DEMO_NUM_TIME_SAMPLES);
    const double initial_power = compute_power(input_field);
    const double final_power = compute_power(final_field);

    printf("runtime_temporal_demo_c: propagated %zu samples.\n",
           (size_t)DEMO_NUM_TIME_SAMPLES);
    printf("initial power=%.6e final power=%.6e\n", initial_power, final_power);
    return 0;
}
