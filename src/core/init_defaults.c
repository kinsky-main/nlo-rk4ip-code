/**
 * @file init_defaults.c
 * @brief Default option/config builders for simulation initialization.
 */

#include "core/state.h"
#include "io/snapshot_store.h"
#include <stdlib.h>
#include <string.h>

#ifndef NLO_DEFAULT_RAMAN_TAU1
#define NLO_DEFAULT_RAMAN_TAU1 0.0122
#endif

#ifndef NLO_DEFAULT_RAMAN_TAU2
#define NLO_DEFAULT_RAMAN_TAU2 0.0320
#endif

nlo_execution_options nlo_execution_options_default(nlo_vector_backend_type backend_type)
{
    nlo_execution_options options;
    memset(&options, 0, sizeof(options));
    options.backend_type = backend_type;
    options.fft_backend = NLO_FFT_BACKEND_AUTO;
    options.device_heap_fraction = NLO_DEFAULT_DEVICE_HEAP_FRACTION;
    options.record_ring_target = 0u;
    options.forced_device_budget_bytes = 0u;
    return options;
}

nlo_storage_options nlo_storage_options_default(void)
{
    nlo_storage_options options;
    memset(&options, 0, sizeof(options));
    options.sqlite_path = NULL;
    options.run_id = NULL;
    options.sqlite_max_bytes = 0u;
    options.chunk_records = 0u;
    options.cap_policy = NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES;
    options.log_final_output_field_to_db = 0;
    return options;
}

nlo_runtime_limits nlo_runtime_limits_default(void)
{
    nlo_runtime_limits limits;
    memset(&limits, 0, sizeof(limits));
    limits.max_num_time_samples_runtime = 0u;
    limits.max_num_recorded_samples_in_memory = 0u;
    limits.max_num_recorded_samples_with_storage =
        (SIZE_MAX < (size_t)9007199254740991ull)
            ? SIZE_MAX
            : (size_t)9007199254740991ull;
    limits.estimated_required_working_set_bytes = 0u;
    limits.estimated_device_budget_bytes = 0u;
    limits.storage_available = nlo_snapshot_store_is_available();
    return limits;
}

sim_config* create_sim_config(size_t num_time_samples)
{
    if (num_time_samples == 0) {
        return NULL;
    }

    sim_config* config = (sim_config*)calloc(1, sizeof(sim_config));
    if (config == NULL) {
        return NULL;
    }

    config->propagation.error_tolerance = 1e-6;
    config->tensor.nt = 0u;
    config->tensor.nx = 0u;
    config->tensor.ny = 0u;
    config->tensor.layout = NLO_TENSOR_LAYOUT_XYT_T_FAST;
    config->time.nt = 0u;
    config->time.wt_axis = NULL;
    config->spatial.nx = num_time_samples;
    config->spatial.ny = 1u;
    config->spatial.delta_x = 1.0;
    config->spatial.delta_y = 1.0;
    config->spatial.spatial_frequency_grid = NULL;
    config->spatial.kx_axis = NULL;
    config->spatial.ky_axis = NULL;
    config->spatial.potential_grid = NULL;
    config->runtime.linear_factor_expr = NULL;
    config->runtime.linear_expr = NULL;
    config->runtime.potential_expr = NULL;
    config->runtime.dispersion_factor_expr = NULL;
    config->runtime.dispersion_expr = NULL;
    config->runtime.nonlinear_expr = NULL;
    config->runtime.nonlinear_model = NLO_NONLINEAR_MODEL_EXPR;
    config->runtime.nonlinear_gamma = 0.0;
    config->runtime.raman_fraction = 0.0;
    config->runtime.raman_tau1 = NLO_DEFAULT_RAMAN_TAU1;
    config->runtime.raman_tau2 = NLO_DEFAULT_RAMAN_TAU2;
    config->runtime.shock_omega0 = 0.0;
    config->runtime.raman_response_time = NULL;
    config->runtime.raman_response_len = 0u;
    config->runtime.num_constants = 0u;
    config->frequency.frequency_grid = (nlo_complex*)calloc(num_time_samples, sizeof(nlo_complex));
    if (config->frequency.frequency_grid == NULL) {
        free(config);
        return NULL;
    }

    return config;
}

void free_sim_config(sim_config* config)
{
    if (config == NULL) {
        return;
    }

    free(config->frequency.frequency_grid);
    free(config);
}
