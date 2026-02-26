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

    static nlo_simulation_config sim_cfg;
    static nlo_physics_config physics_cfg;
    static nlo_execution_options exec_options;
    static nlo_propagate_options propagate_options;
    static nlo_propagate_output propagate_output;
    size_t records_written = 0u;
    static nlo_complex input_field[DEMO_NUM_TIME_SAMPLES];
    static nlo_complex frequency_grid[DEMO_NUM_TIME_SAMPLES];
    static nlo_complex output_records[DEMO_NUM_TIME_SAMPLES * DEMO_NUM_RECORDED_SAMPLES];

    memset(&sim_cfg, 0, sizeof(sim_cfg));
    memset(&physics_cfg, 0, sizeof(physics_cfg));
    memset(&exec_options, 0, sizeof(exec_options));
    memset(&propagate_options, 0, sizeof(propagate_options));
    memset(&propagate_output, 0, sizeof(propagate_output));
    memset(output_records, 0, sizeof(output_records));

    fill_input_field(delta_time, input_field);
    fill_frequency_grid(delta_time, frequency_grid);

    sim_cfg.propagation.propagation_distance = 0.25;
    sim_cfg.propagation.starting_step_size = 1e-3;
    sim_cfg.propagation.max_step_size = 5e-3;
    sim_cfg.propagation.min_step_size = 1e-5;
    sim_cfg.propagation.error_tolerance = 1e-7;

    sim_cfg.time.pulse_period = (double)DEMO_NUM_TIME_SAMPLES * delta_time;
    sim_cfg.time.delta_time = delta_time;

    sim_cfg.frequency.frequency_grid = frequency_grid;

    sim_cfg.spatial.nx = DEMO_NUM_TIME_SAMPLES;
    sim_cfg.spatial.ny = 1u;
    sim_cfg.spatial.delta_x = 1.0;
    sim_cfg.spatial.delta_y = 1.0;
    sim_cfg.spatial.spatial_frequency_grid = NULL;
    sim_cfg.spatial.potential_grid = NULL;

    physics_cfg.dispersion_factor_expr = k_runtime_dispersion_factor_expr;
    physics_cfg.dispersion_expr = NULL;
    physics_cfg.nonlinear_expr = NULL;
    physics_cfg.num_constants = 3u;
    physics_cfg.constants[0] = 0.5 * beta2;
    physics_cfg.constants[1] = 0.0;
    physics_cfg.constants[2] = 0.01;

    exec_options.backend_type = NLO_VECTOR_BACKEND_CPU;
    exec_options.fft_backend = NLO_FFT_BACKEND_FFTW;
    exec_options.device_heap_fraction = 0.70;
    exec_options.record_ring_target = 0u;
    exec_options.forced_device_budget_bytes = 0u;

    propagate_options = nlolib_propagate_options_default();
    propagate_options.exec_options = &exec_options;
    propagate_options.num_recorded_samples = DEMO_NUM_RECORDED_SAMPLES;

    propagate_output = nlolib_propagate_output_default();
    propagate_output.output_records = output_records;
    propagate_output.output_record_capacity = DEMO_NUM_RECORDED_SAMPLES;
    propagate_output.records_written = &records_written;

    const nlolib_status status = nlolib_propagate(&sim_cfg,
                                                  &physics_cfg,
                                                  DEMO_NUM_TIME_SAMPLES,
                                                  input_field,
                                                  &propagate_options,
                                                  &propagate_output);
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
    printf("records_written=%zu\n", records_written);
    printf("initial power=%.6e final power=%.6e\n", initial_power, final_power);
    return 0;
}
