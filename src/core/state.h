/**
 * @file state.h
 * @brief Core state definitions and backend-resident buffer ownership.
 */
#pragma once

#include "backend/nlo_complex.h"
#include "backend/vector_backend.h"
#include "fft/fft_backend.h"
#include "physics/operator_program.h"
#include <stddef.h>

#ifndef NT_MAX
#define NT_MAX ((size_t)-1) /* Unbounded sentinel; runtime limits are query-driven. */
#endif

#ifndef NLO_WORK_VECTOR_COUNT
#define NLO_WORK_VECTOR_COUNT 14u
#endif

#ifndef NLO_DEFAULT_DEVICE_HEAP_FRACTION
#define NLO_DEFAULT_DEVICE_HEAP_FRACTION 0.70
#endif

#ifndef NLO_RUNTIME_OPERATOR_CONSTANTS_MAX
#define NLO_RUNTIME_OPERATOR_CONSTANTS_MAX 16u
#endif

#ifndef NLO_STORAGE_RUN_ID_MAX
#define NLO_STORAGE_RUN_ID_MAX 64u
#endif

typedef struct {
    double starting_step_size;
    double max_step_size;
    double min_step_size;
    double error_tolerance;
    double propagation_distance;
} propagation_params;

typedef struct {
    size_t nt;
    double pulse_period;
    double delta_time;
} time_grid;

typedef struct {
    nlo_complex* frequency_grid;
} frequency_grid;

typedef struct {
    size_t nx;
    size_t ny;
    double delta_x;
    double delta_y;
    nlo_complex* spatial_frequency_grid;
    nlo_complex* potential_grid;
} spatial_grid;

typedef struct {
    const char* dispersion_factor_expr;
    const char* dispersion_expr;
    const char* transverse_factor_expr;
    const char* transverse_expr;
    const char* nonlinear_expr;
    size_t num_constants;
    double constants[NLO_RUNTIME_OPERATOR_CONSTANTS_MAX];
} runtime_operator_params;

typedef struct {
    propagation_params propagation;
    time_grid time;
    frequency_grid frequency;
    spatial_grid spatial;
    runtime_operator_params runtime;
} sim_config;

typedef struct {
    nlo_vector_backend_type backend_type;
    nlo_fft_backend_type fft_backend;
    double device_heap_fraction;
    size_t record_ring_target;
    size_t forced_device_budget_bytes;
    nlo_vk_backend_config vulkan;
} nlo_execution_options;

typedef struct {
    size_t max_num_time_samples_runtime;
    size_t max_num_recorded_samples_in_memory;
    size_t max_num_recorded_samples_with_storage;
    size_t estimated_required_working_set_bytes;
    size_t estimated_device_budget_bytes;
    int storage_available;
} nlo_runtime_limits;

typedef enum {
    NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES = 0,
    NLO_STORAGE_DB_CAP_POLICY_FAIL = 1
} nlo_storage_db_cap_policy;

typedef struct {
    const char* sqlite_path;
    const char* run_id;
    size_t sqlite_max_bytes;
    size_t chunk_records;
    nlo_storage_db_cap_policy cap_policy;
    int log_final_output_field_to_db;
} nlo_storage_options;

typedef struct {
    char run_id[NLO_STORAGE_RUN_ID_MAX];
    size_t records_captured;
    size_t records_spilled;
    size_t chunks_written;
    size_t db_size_bytes;
    int truncated;
} nlo_storage_result;

typedef struct nlo_snapshot_store nlo_snapshot_store;

typedef struct {
    nlo_vec_buffer* ip_field_vec;
    nlo_vec_buffer* field_magnitude_vec;
    nlo_vec_buffer* field_working_vec;
    nlo_vec_buffer* field_freq_vec;
    nlo_vec_buffer* omega_power_vec;
    nlo_vec_buffer* k_1_vec;
    nlo_vec_buffer* k_2_vec;
    nlo_vec_buffer* k_3_vec;
    nlo_vec_buffer* k_4_vec;
    nlo_vec_buffer* dispersion_factor_vec;
    nlo_vec_buffer* dispersion_operator_vec;
    nlo_vec_buffer* nonlinear_multiplier_vec;
    nlo_vec_buffer* potential_vec;
    nlo_vec_buffer* previous_field_vec;
} simulation_working_vectors;

typedef struct nlo_fft_plan nlo_fft_plan;

typedef struct {
    const sim_config* config;
    nlo_execution_options exec_options;
    nlo_vector_backend* backend;

    size_t nt;
    size_t nx;
    size_t ny;
    size_t num_time_samples;
    size_t num_points_xy;
    size_t num_recorded_samples;
    size_t num_host_records;
    size_t current_record_index;

    nlo_complex* field_buffer;
    nlo_complex* snapshot_scratch_record;
    nlo_snapshot_store* snapshot_store;
    nlo_storage_result snapshot_result;
    nlo_vec_status snapshot_status;

    nlo_vec_buffer* current_field_vec;
    nlo_vec_buffer* frequency_grid_vec;
    simulation_working_vectors working_vectors;

    nlo_vec_buffer** record_ring_vec;
    size_t record_ring_capacity;
    size_t record_ring_head;
    size_t record_ring_size;
    size_t record_ring_flushed_count;

    nlo_fft_plan* fft_plan;

    double current_z;
    double current_step_size;
    double current_half_step_exp;
    int dispersion_valid;

    nlo_operator_program dispersion_factor_operator_program;
    nlo_operator_program dispersion_operator_program;
    nlo_operator_program transverse_factor_operator_program;
    nlo_operator_program transverse_operator_program;
    nlo_operator_program nonlinear_operator_program;
    int transverse_active;
    nlo_vec_buffer* spatial_frequency_grid_vec;
    nlo_vec_buffer* transverse_factor_vec;
    nlo_vec_buffer* transverse_operator_vec;
    size_t runtime_operator_stack_slots;
    nlo_vec_buffer* runtime_operator_stack_vec[NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS];
} simulation_state;

nlo_execution_options nlo_execution_options_default(nlo_vector_backend_type backend_type);
nlo_storage_options nlo_storage_options_default(void);
nlo_runtime_limits nlo_runtime_limits_default(void);

int nlo_query_runtime_limits_internal(
    const sim_config* config,
    const nlo_execution_options* exec_options,
    nlo_runtime_limits* out_limits
);

simulation_state* create_simulation_state(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options
);

simulation_state* create_simulation_state_with_storage(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options,
    const nlo_storage_options* storage_options
);

void free_simulation_state(simulation_state* state);

sim_config* create_sim_config(size_t num_time_samples);
void free_sim_config(sim_config* config);

nlo_vec_status simulation_state_upload_initial_field(simulation_state* state, const nlo_complex* field);
nlo_vec_status simulation_state_download_current_field(const simulation_state* state, nlo_complex* out_field);

nlo_vec_status simulation_state_capture_snapshot(simulation_state* state);
nlo_vec_status simulation_state_flush_snapshots(simulation_state* state);

static inline nlo_complex* simulation_state_get_field_record(simulation_state* state, size_t record_index)
{
    if (state == NULL || state->field_buffer == NULL || record_index >= state->num_host_records) {
        return NULL;
    }

    return state->field_buffer + (record_index * state->num_time_samples);
}
