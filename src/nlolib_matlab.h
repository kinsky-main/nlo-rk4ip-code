/**
 * @file nlolib_matlab.h
 * @brief Flattened, self-contained header for MATLAB loadlibrary().
 *
 * This header mirrors the public API surface of nlolib.h but removes all
 * transitive includes that MATLAB's C header parser cannot handle (e.g.
 * vulkan/vulkan.h).  It is intended ONLY for loadlibrary/calllib and must
 * be kept in sync with the canonical headers whenever the public ABI
 * changes.
 *
 * DO NOT include this header from C/C++ translation units that compile
 * against the real nlolib — it exists solely for the MATLAB FFI.
 *
 * @section matlab_install Installation (MATLAB Toolbox)
 *
 * <b>Option A — Install from .mltbx (recommended)</b>
 *
 * Download the latest @c nlolib.mltbx from the GitHub Releases page and
 * double-click it in MATLAB, or install programmatically:
 * @code{.m}
 *   matlab.addons.install('nlolib.mltbx');
 * @endcode
 *
 * <b>Option B — Build from source</b>
 *
 * @code{.sh}
 *   cmake -S . -B build
 *   cmake --build build --config Release --target matlab_stage
 * @endcode
 *
 * Then add the staging directory to the MATLAB path:
 * @code{.m}
 *   addpath('<repo>/build/matlab_toolbox');
 *   nlolib_setup();
 * @endcode
 *
 * The nlolib shared library (nlolib.dll / libnlolib.so) and this header
 * must be locatable at runtime.  The MATLAB wrapper searches the build
 * tree and common install locations automatically; set the environment
 * variable @c NLOLIB_LIBRARY to override.
 *
 * @section matlab_prereqs Prerequisites
 *
 * - MATLAB R2019b or later (loadlibrary with C99 header support).
 * - A GPU driver that ships the Vulkan loader (standard on NVIDIA, AMD,
 *   Intel desktop drivers).  No Vulkan SDK is needed at runtime.
 */
#ifndef NLOLIB_MATLAB_H
#define NLOLIB_MATLAB_H

#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------- */

/** @brief Unbounded sample-count sentinel for limit query APIs. */
#define NT_MAX ((size_t)-1) /* Unbounded sentinel; use nlolib_query_runtime_limits(). */
/** @brief Maximum runtime scalar constants accepted by expression programs. */
#define NLO_RUNTIME_OPERATOR_CONSTANTS_MAX 16
/** @brief Maximum persisted run-id length (including terminator). */
#define NLO_STORAGE_RUN_ID_MAX 64

/* -------------------------------------------------------------------
 * nlo_complex
 * ------------------------------------------------------------------- */

/**
 * @brief Complex scalar represented as two double values.
 */
typedef struct
{
    double re;
    double im;
} nlo_complex;

/* -------------------------------------------------------------------
 * Simulation configuration sub-structs
 * ------------------------------------------------------------------- */

/**
 * @brief Propagation solver controls along the z dimension.
 */
typedef struct
{
    double starting_step_size;
    double max_step_size;
    double min_step_size;
    double error_tolerance;
    double propagation_distance;
} propagation_params;

/**
 * @brief Temporal grid metadata.
 */
typedef struct
{
    size_t nt;
    double pulse_period;
    double delta_time;
} time_grid;

/**
 * @brief Optional precomputed temporal-frequency vector.
 */
typedef struct
{
    nlo_complex *frequency_grid;
} nlo_frequency_grid;

/**
 * @brief Spatial grid metadata and optional spatial buffers.
 */
typedef struct
{
    size_t nx;
    size_t ny;
    double delta_x;
    double delta_y;
    nlo_complex *spatial_frequency_grid;
    nlo_complex *potential_grid;
} spatial_grid;

/**
 * @brief Runtime expression settings for dispersion/nonlinearity operators.
 */
typedef struct
{
    const char *dispersion_factor_expr;
    const char *dispersion_expr;
    const char *transverse_factor_expr;
    const char *transverse_expr;
    const char *nonlinear_expr;
    size_t num_constants;
    double constants[NLO_RUNTIME_OPERATOR_CONSTANTS_MAX];
} runtime_operator_params;

/**
 * @brief Simulation-only input configuration.
 */
typedef struct
{
    propagation_params propagation;
    time_grid time;
    nlo_frequency_grid frequency;
    spatial_grid spatial;
} nlo_simulation_config;

/**
 * @brief Physics/operator input configuration.
 */
typedef runtime_operator_params nlo_physics_config;

/**
 * @brief Full internal simulation input configuration.
 */
typedef struct
{
    propagation_params propagation;
    time_grid time;
    nlo_frequency_grid frequency;
    spatial_grid spatial;
    runtime_operator_params runtime;
} sim_config;

/* -------------------------------------------------------------------
 * Backend / execution option enums
 * ------------------------------------------------------------------- */

/**
 * @brief Backend selection mode.
 */
typedef enum
{
    NLO_VECTOR_BACKEND_CPU = 0,
    NLO_VECTOR_BACKEND_VULKAN = 1,
    NLO_VECTOR_BACKEND_AUTO = 2
} nlo_vector_backend_type;

/**
 * @brief FFT implementation selector.
 */
typedef enum
{
    NLO_FFT_BACKEND_AUTO = 0,
    NLO_FFT_BACKEND_FFTW = 1,
    NLO_FFT_BACKEND_VKFFT = 2
} nlo_fft_backend_type;

/* -------------------------------------------------------------------
 * Vulkan backend config (Vulkan handles replaced with opaque stubs)
 * ------------------------------------------------------------------- */

typedef void *VkPhysicalDevice;
typedef void *VkDevice;
typedef void *VkQueue;
typedef void *VkCommandPool;

/**
 * @brief Vulkan backend configuration with opaque handle placeholders.
 */
typedef struct
{
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
    size_t descriptor_set_budget_bytes;
    uint32_t descriptor_set_count_override;
} nlo_vk_backend_config;

/* -------------------------------------------------------------------
 * Execution options
 * ------------------------------------------------------------------- */

/**
 * @brief Runtime backend/resource selection options.
 */
typedef struct
{
    nlo_vector_backend_type backend_type;
    nlo_fft_backend_type fft_backend;
    double device_heap_fraction;
    size_t record_ring_target;
    size_t forced_device_budget_bytes;
    nlo_vk_backend_config vulkan;
} nlo_execution_options;

/**
 * @brief Runtime limits/estimates for current backend and configuration.
 */
typedef struct
{
    size_t max_num_time_samples_runtime;
    size_t max_num_recorded_samples_in_memory;
    size_t max_num_recorded_samples_with_storage;
    size_t estimated_required_working_set_bytes;
    size_t estimated_device_budget_bytes;
    int storage_available;
} nlo_runtime_limits;

/* -------------------------------------------------------------------
 * Status codes
 * ------------------------------------------------------------------- */

/**
 * @brief Public status codes returned by nlolib APIs.
 */
typedef enum
{
    NLOLIB_STATUS_OK = 0,
    NLOLIB_STATUS_INVALID_ARGUMENT = 1,
    NLOLIB_STATUS_ALLOCATION_FAILED = 2,
    NLOLIB_STATUS_NOT_IMPLEMENTED = 3
} nlolib_status;

/**
 * @brief Public log-level thresholds for nlolib runtime logging.
 */
typedef enum
{
    NLOLIB_LOG_LEVEL_ERROR = 0,
    NLOLIB_LOG_LEVEL_WARN = 1,
    NLOLIB_LOG_LEVEL_INFO = 2,
    NLOLIB_LOG_LEVEL_DEBUG = 3
} nlolib_log_level;

/**
 * @brief Database size-limit behavior when snapshot storage reaches its cap.
 */
typedef enum
{
    NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES = 0,
    NLO_STORAGE_DB_CAP_POLICY_FAIL = 1
} nlo_storage_db_cap_policy;

/**
 * @brief SQLite snapshot storage controls.
 */
typedef struct
{
    const char *sqlite_path;
    const char *run_id;
    size_t sqlite_max_bytes;
    size_t chunk_records;
    nlo_storage_db_cap_policy cap_policy;
    int log_final_output_field_to_db;
} nlo_storage_options;

/**
 * @brief Summary of captured/spilled records from storage-enabled runs.
 */
typedef struct
{
    char run_id[NLO_STORAGE_RUN_ID_MAX];
    size_t records_captured;
    size_t records_spilled;
    size_t chunks_written;
    size_t db_size_bytes;
    int truncated;
} nlo_storage_result;

/**
 * @brief Propagation record output mode.
 */
typedef enum
{
    NLO_PROPAGATE_OUTPUT_DENSE = 0,
    NLO_PROPAGATE_OUTPUT_FINAL_ONLY = 1
} nlo_propagate_output_mode;

/**
 * @brief Unified propagation request options.
 */
typedef struct
{
    size_t num_recorded_samples;
    nlo_propagate_output_mode output_mode;
    int return_records;
    const nlo_execution_options *exec_options;
    const nlo_storage_options *storage_options;
} nlo_propagate_options;

/**
 * @brief Unified propagation output metadata and buffers.
 */
typedef struct
{
    nlo_complex *output_records;
    size_t output_record_capacity;
    size_t *records_written;
    nlo_storage_result *storage_result;
} nlo_propagate_output;

/* -------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------- */

/**
 * @brief MATLAB FFI mirror of nlolib_propagate().
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_propagate(
    const nlo_simulation_config *simulation_config,
    const nlo_physics_config *physics_config,
    size_t num_time_samples,
    const nlo_complex *input_field,
    const nlo_propagate_options *options,
    nlo_propagate_output *output);

/**
 * @brief MATLAB FFI mirror of nlolib_query_runtime_limits().
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_query_runtime_limits(
    const nlo_simulation_config *simulation_config,
    const nlo_physics_config *physics_config,
    const nlo_execution_options *exec_options,
    nlo_runtime_limits *out_limits);

/**
 * @brief Return whether SQLite snapshot storage is available in this build.
 *
 * See `nlolib.h` for full return-value semantics.
 */
int nlolib_storage_is_available(void);

/**
 * @brief Configure optional runtime log file output.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_set_log_file(const char *path_utf8, int append);

/**
 * @brief Configure optional in-memory runtime log ring buffer.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_set_log_buffer(size_t capacity_bytes);

/**
 * @brief Clear buffered in-memory runtime logs.
 *
 * See `nlolib.h` for full return-value semantics.
 */
nlolib_status nlolib_clear_log_buffer(void);

/**
 * @brief Read buffered runtime logs into caller memory.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_read_log_buffer(
    char *dst,
    size_t dst_bytes,
    size_t *out_written,
    int consume);

/**
 * @brief Set global runtime logging level threshold.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_set_log_level(int level);

/**
 * @brief Configure runtime progress logging behavior.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_set_progress_options(
    int enabled,
    int milestone_percent,
    int emit_on_step_adjust);

#endif /* NLOLIB_MATLAB_H */
