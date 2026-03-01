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

/**
 * @brief Unbounded sample-count sentinel used by limit query APIs.
 */
#ifndef NT_MAX
#define NT_MAX ((size_t)-1) /* Unbounded sentinel; runtime limits are query-driven. */
#endif

/**
 * @brief Number of persistent work vectors reserved by the solver state.
 */
#ifndef NLO_WORK_VECTOR_COUNT
#define NLO_WORK_VECTOR_COUNT 21u
#endif

/**
 * @brief Default fraction of device-local memory considered usable.
 */
#ifndef NLO_DEFAULT_DEVICE_HEAP_FRACTION
#define NLO_DEFAULT_DEVICE_HEAP_FRACTION 0.70
#endif

/**
 * @brief Maximum number of runtime operator scalar constants.
 */
#ifndef NLO_RUNTIME_OPERATOR_CONSTANTS_MAX
#define NLO_RUNTIME_OPERATOR_CONSTANTS_MAX 16u
#endif

/**
 * @brief Maximum length (including terminator) for persisted run IDs.
 */
#ifndef NLO_STORAGE_RUN_ID_MAX
#define NLO_STORAGE_RUN_ID_MAX 64u
#endif

/**
 * @brief Propagation solver controls along the z dimension.
 */
typedef struct {
    double starting_step_size;
    double max_step_size;
    double min_step_size;
    double error_tolerance;
    double propagation_distance;
} propagation_params;

/**
 * @brief Temporal grid metadata.
 *
 * `nt` is the number of temporal samples in explicit ND mode.
 */
typedef struct {
    size_t nt;
    double pulse_period;
    double delta_time;
    nlo_complex* wt_axis;
} time_grid;

/**
 * @brief Optional precomputed temporal-frequency grid input.
 */
typedef struct {
    nlo_complex* frequency_grid;
} frequency_grid;

/**
 * @brief Spatial grid metadata and optional precomputed buffers.
 */
typedef struct {
    size_t nx;
    size_t ny;
    double delta_x;
    double delta_y;
    nlo_complex* spatial_frequency_grid;
    nlo_complex* kx_axis;
    nlo_complex* ky_axis;
    nlo_complex* potential_grid;
} spatial_grid;

/**
 * @brief Tensor memory layout selector for explicit 3D field descriptors.
 */
typedef enum {
    /** Flatten with temporal index fastest: idx = ((x * ny) + y) * nt + t. */
    NLO_TENSOR_LAYOUT_XYT_T_FAST = 0
} nlo_tensor_layout;

/**
 * @brief Explicit tensor descriptor for 3D runs.
 */
typedef struct {
    size_t nt;
    size_t nx;
    size_t ny;
    int layout;
} nlo_tensor3d_desc;

/**
 * @brief Nonlinear operator execution model selector.
 */
typedef enum {
    /** Evaluate compiled runtime expression `nonlinear_expr`. */
    NLO_NONLINEAR_MODEL_EXPR = 0,
    /** Use built-in Kerr + delayed Raman (+ optional shock) model. */
    NLO_NONLINEAR_MODEL_KERR_RAMAN = 1
} nlo_nonlinear_model;

/**
 * @brief Runtime expression settings for dispersion/nonlinearity operators.
 *
 * String expressions are compiled at runtime into operator programs.
 */
typedef struct {
    const char* linear_factor_expr;
    const char* linear_expr;
    const char* potential_expr;
    const char* dispersion_factor_expr;
    const char* dispersion_expr;
    const char* nonlinear_expr;
    int nonlinear_model;
    double nonlinear_gamma;
    double raman_fraction;
    double raman_tau1;
    double raman_tau2;
    double shock_omega0;
    nlo_complex* raman_response_time;
    size_t raman_response_len;
    size_t num_constants;
    double constants[NLO_RUNTIME_OPERATOR_CONSTANTS_MAX];
} runtime_operator_params;

/**
 * @brief Simulation-only input configuration (no runtime physics program).
 */
typedef struct {
    propagation_params propagation;
    nlo_tensor3d_desc tensor;
    time_grid time;
    frequency_grid frequency;
    spatial_grid spatial;
} nlo_simulation_config;

/**
 * @brief Physics/operator input configuration used by runtime evaluators.
 */
typedef runtime_operator_params nlo_physics_config;

/**
 * @brief Full internal simulation input configuration.
 *
 * This combines public simulation and physics sections for solver internals.
 */
typedef struct {
    propagation_params propagation;
    nlo_tensor3d_desc tensor;
    time_grid time;
    frequency_grid frequency;
    spatial_grid spatial;
    runtime_operator_params runtime;
} sim_config;

/**
 * @brief Runtime execution/backend selection and resource tuning options.
 */
typedef struct {
    nlo_vector_backend_type backend_type;
    nlo_fft_backend_type fft_backend;
    double device_heap_fraction;
    size_t record_ring_target;
    size_t forced_device_budget_bytes;
    nlo_vk_backend_config vulkan;
} nlo_execution_options;

/**
 * @brief Estimated runtime limits for current configuration/backend choices.
 */
typedef struct {
    size_t max_num_time_samples_runtime;
    size_t max_num_recorded_samples_in_memory;
    size_t max_num_recorded_samples_with_storage;
    size_t estimated_required_working_set_bytes;
    size_t estimated_device_budget_bytes;
    int storage_available;
} nlo_runtime_limits;

/**
 * @brief Database size-limit policy for snapshot storage.
 */
typedef enum {
    /** Stop writing new chunks once cap is reached; run continues. */
    NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES = 0,
    /** Treat cap violation as an error status. */
    NLO_STORAGE_DB_CAP_POLICY_FAIL = 1
} nlo_storage_db_cap_policy;

/**
 * @brief Snapshot persistence controls for SQLite-backed output chunking.
 */
typedef struct {
    const char* sqlite_path;
    const char* run_id;
    size_t sqlite_max_bytes;
    size_t chunk_records;
    nlo_storage_db_cap_policy cap_policy;
    int log_final_output_field_to_db;
} nlo_storage_options;

/**
 * @brief Summary of snapshot capture/storage results after a run.
 */
typedef struct {
    char run_id[NLO_STORAGE_RUN_ID_MAX];
    size_t records_captured;
    size_t records_spilled;
    size_t chunks_written;
    size_t db_size_bytes;
    int truncated;
} nlo_storage_result;

/**
 * @brief Per-step adaptive solver telemetry for accepted RK4 steps.
 */
typedef struct {
    size_t step_index;
    double z_current;
    double step_size;
    double next_step_size;
    double error;
} nlo_step_event;

/**
 * @brief Opaque snapshot store handle.
 */
typedef struct nlo_snapshot_store nlo_snapshot_store;

/**
 * @brief Internal set of preallocated working buffers used by RK4 propagation.
 */
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
    nlo_vec_buffer* potential_vec;
    nlo_vec_buffer* previous_field_vec;
    nlo_vec_buffer* raman_intensity_vec;
    nlo_vec_buffer* raman_delayed_vec;
    nlo_vec_buffer* raman_spectrum_vec;
    nlo_vec_buffer* raman_mix_vec;
    nlo_vec_buffer* raman_polarization_vec;
    nlo_vec_buffer* raman_derivative_vec;
    nlo_vec_buffer* raman_response_fft_vec;
    nlo_vec_buffer* raman_derivative_factor_vec;
    nlo_vec_buffer* wt_axis_vec;
    nlo_vec_buffer* kx_axis_vec;
    nlo_vec_buffer* ky_axis_vec;
    nlo_vec_buffer* t_axis_vec;
    nlo_vec_buffer* x_axis_vec;
    nlo_vec_buffer* y_axis_vec;
    nlo_vec_buffer* wt_mesh_vec;
    nlo_vec_buffer* kx_mesh_vec;
    nlo_vec_buffer* ky_mesh_vec;
    nlo_vec_buffer* t_mesh_vec;
    nlo_vec_buffer* x_mesh_vec;
    nlo_vec_buffer* y_mesh_vec;
} simulation_working_vectors;

typedef struct nlo_fft_plan nlo_fft_plan;

/**
 * @brief Internal mutable simulation runtime state.
 */
typedef struct {
    const sim_config* config;
    nlo_execution_options exec_options;
    nlo_vector_backend* backend;

    size_t nt;
    size_t nx;
    size_t ny;
    int tensor_layout;
    int tensor_mode_active;
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
    nlo_step_event* step_event_buffer;
    size_t step_event_capacity;
    size_t step_events_written;
    size_t step_events_dropped;

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

    nlo_operator_program linear_factor_operator_program;
    nlo_operator_program linear_operator_program;
    nlo_operator_program potential_operator_program;
    nlo_operator_program dispersion_factor_operator_program;
    nlo_operator_program dispersion_operator_program;
    nlo_operator_program nonlinear_operator_program;
    nlo_nonlinear_model nonlinear_model;
    int nonlinear_raman_active;
    int nonlinear_shock_active;
    double nonlinear_gamma;
    double raman_fraction;
    double shock_omega0;
    size_t runtime_operator_stack_slots;
    nlo_vec_buffer* runtime_operator_stack_vec[NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS];
} simulation_state;

/**
 * @brief Build default execution options for the selected backend family.
 *
 * @param backend_type Backend mode preference (CPU, Vulkan, or AUTO).
 * @return nlo_execution_options Initialized option block.
 */
nlo_execution_options nlo_execution_options_default(nlo_vector_backend_type backend_type);

/**
 * @brief Build default SQLite storage options.
 *
 * @return nlo_storage_options Initialized storage option block.
 */
nlo_storage_options nlo_storage_options_default(void);

/**
 * @brief Build default runtime limit descriptor values.
 *
 * @return nlo_runtime_limits Initialized runtime limits descriptor.
 */
nlo_runtime_limits nlo_runtime_limits_default(void);

/**
 * @brief Internal runtime-limit query helper used by public wrappers.
 *
 * @param config Optional simulation configuration.
 * @param exec_options Optional execution options.
 * @param out_limits Destination limits descriptor.
 * @return int 0 on success, nonzero on invalid inputs/backend failure.
 */
int nlo_query_runtime_limits_internal(
    const sim_config* config,
    const nlo_execution_options* exec_options,
    nlo_runtime_limits* out_limits
);

/**
 * @brief Allocate and initialize simulation state without storage persistence.
 *
 * @param config Simulation configuration.
 * @param num_time_samples Flattened sample count per record.
 * @param num_recorded_samples Number of records to capture.
 * @param exec_options Optional execution overrides (NULL for defaults).
 * @return simulation_state* Initialized state, or NULL on failure.
 */
simulation_state* create_simulation_state(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options
);

/**
 * @brief Allocate and initialize simulation state with optional storage support.
 *
 * @param config Simulation configuration.
 * @param num_time_samples Flattened sample count per record.
 * @param num_recorded_samples Number of records to capture.
 * @param exec_options Optional execution overrides (NULL for defaults).
 * @param storage_options Optional storage controls (NULL disables storage).
 * @return simulation_state* Initialized state, or NULL on failure.
 */
simulation_state* create_simulation_state_with_storage(
    const sim_config* config,
    size_t num_time_samples,
    size_t num_recorded_samples,
    const nlo_execution_options* exec_options,
    const nlo_storage_options* storage_options
);

/**
 * @brief Release all resources owned by a simulation state.
 *
 * @param state State to destroy (NULL is allowed).
 */
void free_simulation_state(simulation_state* state);

/**
 * @brief Allocate a default simulation configuration sized for a sample count.
 *
 * @param num_time_samples Initial temporal sample count hint.
 * @return sim_config* Newly allocated config, or NULL on failure.
 */
sim_config* create_sim_config(size_t num_time_samples);

/**
 * @brief Free a simulation configuration allocated by create_sim_config().
 *
 * @param config Configuration object to free.
 */
void free_sim_config(sim_config* config);

/**
 * @brief Upload the initial field into backend-resident simulation buffers.
 *
 * @param state Active simulation state.
 * @param field Host input field buffer with num_time_samples elements.
 * @return nlo_vec_status Upload/validation status.
 */
nlo_vec_status simulation_state_upload_initial_field(simulation_state* state, const nlo_complex* field);

/**
 * @brief Download the current backend field into host memory.
 *
 * @param state Active simulation state.
 * @param out_field Host output buffer with num_time_samples elements.
 * @return nlo_vec_status Download/validation status.
 */
nlo_vec_status simulation_state_download_current_field(const simulation_state* state, nlo_complex* out_field);

/**
 * @brief Capture one snapshot record into host ring/storage buffers.
 *
 * @param state Active simulation state.
 * @return nlo_vec_status Capture status.
 */
nlo_vec_status simulation_state_capture_snapshot(simulation_state* state);

/**
 * @brief Flush any pending snapshot data to host/storage sinks.
 *
 * @param state Active simulation state.
 * @return nlo_vec_status Flush status.
 */
nlo_vec_status simulation_state_flush_snapshots(simulation_state* state);

/**
 * @brief Return a pointer to a host-side record slot by index.
 *
 * @param state Active simulation state.
 * @param record_index Zero-based record index.
 * @return nlo_complex* Pointer to record start, or NULL when unavailable.
 */
static inline nlo_complex* simulation_state_get_field_record(simulation_state* state, size_t record_index)
{
    if (state == NULL || state->field_buffer == NULL || record_index >= state->num_host_records) {
        return NULL;
    }

    return state->field_buffer + (record_index * state->num_time_samples);
}
