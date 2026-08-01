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
#define RUNTIME_OPERATOR_CONSTANTS_MAX 16
/** @brief Maximum persisted run-id length (including terminator). */
#define STORAGE_RUN_ID_MAX 64

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
    nlo_complex *wt_axis;
} time_grid;

/**
 * @brief Optional precomputed temporal-frequency vector.
 */
typedef struct
{
    nlo_complex *frequency_grid;
} frequency_grid;

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
    nlo_complex *kx_axis;
    nlo_complex *ky_axis;
    nlo_complex *potential_grid;
} spatial_grid;

/**
 * @brief Tensor memory layout selector for explicit 3D field descriptors.
 */
typedef enum
{
    TENSOR_LAYOUT_XYT_T_FAST = 0
} tensor_layout;

/**
 * @brief Explicit tensor descriptor for 3D runs.
 */
typedef struct
{
    size_t nt;
    size_t nx;
    size_t ny;
    int layout;
} tensor3d_desc;

/**
 * @brief Nonlinear operator execution model selector.
 *
 * Expression mode evaluates caller-supplied @c nonlinear_expr as the full
 * nonlinear RHS. The built-in Raman mode applies
 * \f[
 * N_R(A)=i\gamma A\left[(1-f_R)|A|^2+f_R\left(h_R \ast |A|^2\right)\right]
 * -\frac{\gamma}{\omega_0}\partial_t\!\left[
 * A\left((1-f_R)|A|^2+f_R\left(h_R \ast |A|^2\right)\right)\right]
 * \f]
 * when self-steepening is enabled, and drops the derivative term when
 * @c shock_omega0 is zero.
 */
typedef enum
{
    /** Evaluate compiled runtime expression `nonlinear_expr` as the full RHS, for example \f$N(A)=iA(c_2|A|^2+V)\f$. */
    NONLINEAR_MODEL_EXPR = 0,
    /** Use the built-in Kerr + delayed Raman (+ optional shock) model \f$N_R(A)\f$. */
    NONLINEAR_MODEL_KERR_RAMAN = 1
} nonlinear_model;

/**
 * @brief Runtime expression settings for dispersion/nonlinearity operators.
 *
 * Temporal dispersion defaults follow
 * \f[
 * D(\omega)=i c_0 \omega^2-c_1,\qquad
 * L_h(\omega)=\exp\!\left(hD(\omega)\right)
 * \f]
 * where @c dispersion_factor_expr provides \f$D\f$ and @c dispersion_expr
 * or @c linear_expr provides \f$L_h\f$.
 *
 * For the common quadratic GLSE form,
 * \f[
 * D(\omega)=i\frac{\beta_2}{2}\omega^2-\alpha_{\mathrm{amp}}
 * \f]
 * so the default runtime mapping is \f$c_0=\beta_2/2\f$,
 * \f$c_1=\alpha_{\mathrm{amp}}\f$, and \f$c_2=\gamma\f$. If a model uses
 * the common power-loss convention \f$-\alpha_{\mathrm{pow}}A/2\f$, pass
 * \f$c_1=\alpha_{\mathrm{pow}}/2\f$.
 *
 * Runtime constants are stored in @c constants[] and referenced as scalar
 * symbols @c c0, @c c1, @c c2, ... inside expressions. Higher-order
 * dispersion therefore uses successive constants, for example
 * \f[
 * D(\omega)=i\left(c_0\omega^2+c_1\omega^3+c_2\omega^4\right)-c_3,
 * \f]
 * rather than an array-valued \f$c_0\f$.
 *
 * Tensor linear expressions may depend on the symbol set
 * \f$\omega_t, k_x, k_y, t, x, y\f$ through the runtime names @c wt, @c kx,
 * @c ky, @c t, @c x, and @c y, and apply
 * \f[
 * L_h(\omega_t,k_x,k_y,t,x,y)=
 * \exp\!\left(hD(\omega_t,k_x,k_y,t,x,y)\right)
 * \f]
 * for example
 * \f[
 * D=i\left(\beta_{2,s}\omega_t^2+\beta_t(k_x^2+k_y^2)\right).
 * \f]
 *
 * The canonical expression-model nonlinear example is
 * \f[
 * N(A)=iA(c_2|A|^2+V).
 * \f]
 *
 * For @ref NONLINEAR_MODEL_KERR_RAMAN, the default delayed response is
 * normalized from
 * \f[
 * h_R(t)\propto e^{-t/\tau_2}\sin(t/\tau_1),\qquad
 * t\ge 0,\qquad \int_0^{\infty} h_R(t)\,dt = 1.
 * \f]
 */
typedef struct
{
    const char *linear_factor_expr;
    const char *linear_expr;
    const char *potential_expr;
    const char *dispersion_factor_expr;
    const char *dispersion_expr;
    const char *nonlinear_expr;
    int nonlinear_model;
    double nonlinear_gamma;
    double raman_fraction;
    double raman_tau1;
    double raman_tau2;
    double shock_omega0;
    nlo_complex *raman_response_time;
    size_t raman_response_len;
    size_t num_constants;
    double constants[RUNTIME_OPERATOR_CONSTANTS_MAX];
} runtime_operator_params;

/**
 * @brief Simulation-only input configuration.
 */
typedef struct
{
    propagation_params propagation;
    tensor3d_desc tensor;
    time_grid time;
    frequency_grid frequency;
    spatial_grid spatial;
} simulation_config;

/**
 * @brief Physics/operator input configuration.
 */
typedef runtime_operator_params physics_config;

/**
 * @brief Full internal simulation input configuration.
 */
typedef struct
{
    propagation_params propagation;
    tensor3d_desc tensor;
    time_grid time;
    frequency_grid frequency;
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
    VECTOR_BACKEND_CPU = 0,
    VECTOR_BACKEND_VULKAN = 1,
    VECTOR_BACKEND_AUTO = 2
} vector_backend_type;

/**
 * @brief FFT implementation selector.
 */
typedef enum
{
    FFT_BACKEND_AUTO = 0,
    FFT_BACKEND_FFTW = 1,
    FFT_BACKEND_VKFFT = 2
} fft_backend_type;

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
} vk_backend_config;

/* -------------------------------------------------------------------
 * Execution options
 * ------------------------------------------------------------------- */

/**
 * @brief Runtime backend/resource selection options.
 */
typedef struct
{
    vector_backend_type backend_type;
    fft_backend_type fft_backend;
    double device_heap_fraction;
    size_t record_ring_target;
    size_t forced_device_budget_bytes;
    vk_backend_config vulkan;
} execution_options;

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
} runtime_limits;

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
    NLOLIB_STATUS_NOT_IMPLEMENTED = 3,
    NLOLIB_STATUS_ABORTED = 4
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
 * @brief Output stream selection for runtime progress TUI rendering.
 */
typedef enum
{
    NLOLIB_PROGRESS_STREAM_STDERR = 0,
    NLOLIB_PROGRESS_STREAM_STDOUT = 1,
    NLOLIB_PROGRESS_STREAM_BOTH = 2
} nlolib_progress_stream_mode;

/**
 * @brief Snapshot of accumulated runtime performance counters.
 */
typedef struct
{
    double dispersion_ms;
    double nonlinear_ms;
    uint64_t dispersion_calls;
    uint64_t nonlinear_calls;
    uint64_t gpu_dispatch_count;
    uint64_t gpu_copy_count;
    uint64_t gpu_device_copy_count;
    uint64_t gpu_device_copy_bytes;
    uint64_t gpu_host_transfer_copy_count;
    uint64_t gpu_host_transfer_copy_bytes;
    uint64_t gpu_memory_pass_count;
    uint64_t gpu_memory_pass_bytes;
    uint64_t gpu_upload_count;
    uint64_t gpu_download_count;
    uint64_t gpu_upload_bytes;
    uint64_t gpu_download_bytes;
} nlo_perf_profile_snapshot;

/**
 * @brief Progress event class reported during propagation.
 */
typedef enum
{
    PROGRESS_EVENT_ACCEPTED = 0,
    PROGRESS_EVENT_REJECTED = 1,
    PROGRESS_EVENT_FINISH = 2
} progress_event_type;

/**
 * @brief Per-event propagation progress payload for caller callbacks.
 */
typedef struct
{
    progress_event_type event_type;
    size_t step_index;
    size_t reject_attempt;
    double z;
    double z_end;
    double percent;
    double step_size;
    double next_step_size;
    double error;
    double elapsed_seconds;
    double eta_seconds;
} progress_info;

/**
 * @brief Progress callback invoked during propagation.
 */
typedef int (*progress_callback)(const progress_info *info, void *user_data);

/**
 * @brief Database size-limit behavior when snapshot storage reaches its cap.
 */
typedef enum
{
    STORAGE_DB_CAP_POLICY_STOP_WRITES = 0,
    STORAGE_DB_CAP_POLICY_FAIL = 1
} storage_db_cap_policy;

/**
 * @brief SQLite snapshot storage controls.
 */
typedef struct
{
    const char *sqlite_path;
    const char *run_id;
    size_t sqlite_max_bytes;
    size_t chunk_records;
    storage_db_cap_policy cap_policy;
    int log_final_output_field_to_db;
} storage_options;

/**
 * @brief Summary of captured/spilled records from storage-enabled runs.
 */
typedef struct
{
    char run_id[STORAGE_RUN_ID_MAX];
    size_t records_captured;
    size_t records_spilled;
    size_t chunks_written;
    size_t db_size_bytes;
    int truncated;
} storage_result;

/**
 * @brief Per-step adaptive solver telemetry for accepted RK4 steps.
 */
typedef struct
{
    size_t step_index;
    double z_current;
    double step_size;
    double next_step_size;
    double error;
} step_event;

/**
 * @brief Propagation record output mode.
 */
typedef enum
{
    PROPAGATE_OUTPUT_DENSE = 0,
    PROPAGATE_OUTPUT_FINAL_ONLY = 1
} propagate_output_mode;

/**
 * @brief Unified propagation request options.
 */
typedef struct
{
    size_t num_recorded_samples;
    propagate_output_mode output_mode;
    int return_records;
    const execution_options *exec_options;
    const storage_options *storage_options;
    const double *explicit_record_z;
    size_t explicit_record_z_count;
    progress_callback progress_callback;
    void *progress_user_data;
} propagate_options;

/**
 * @brief Unified propagation output metadata and buffers.
 */
typedef struct
{
    double *output_records;
    size_t output_record_capacity;
    size_t *records_written;
    storage_result *storage_result;
    step_event *output_step_events;
    size_t output_step_event_capacity;
    size_t *step_events_written;
    size_t *step_events_dropped;
} propagate_output;

/* -------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------- */

/**
 * @brief MATLAB FFI mirror of nlolib_propagate().
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_propagate(
    const simulation_config *simulation_config,
    const physics_config *physics_config,
    size_t num_time_samples,
    const nlo_complex *input_field,
    const propagate_options *options,
    propagate_output *output);

/**
 * @brief MATLAB FFI mirror of nlolib_query_runtime_limits().
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_query_runtime_limits(
    const simulation_config *simulation_config,
    const physics_config *physics_config,
    const execution_options *exec_options,
    runtime_limits *out_limits);

/**
 * @brief Enable or disable runtime performance counter accumulation.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_perf_profile_set_enabled(int enabled);

/**
 * @brief Query whether runtime performance counters are enabled.
 *
 * See `nlolib.h` for full return-value semantics.
 */
int nlolib_perf_profile_is_enabled(void);

/**
 * @brief Reset all runtime performance counters.
 *
 * See `nlolib.h` for full return-value semantics.
 */
nlolib_status nlolib_perf_profile_reset(void);

/**
 * @brief Read current runtime performance counters into caller memory.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_perf_profile_read(nlo_perf_profile_snapshot *out_snapshot);

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
 * @brief Configure runtime progress TUI behavior.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_set_progress_options(
    int enabled,
    int milestone_percent,
    int emit_on_step_adjust);

/**
 * @brief Configure output stream selection for runtime progress TUI lines.
 *
 * See `nlolib.h` for full parameter and return-value semantics.
 */
nlolib_status nlolib_set_progress_stream(int stream_mode);

#endif /* NLOLIB_MATLAB_H */
