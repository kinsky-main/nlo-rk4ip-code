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

#define NT_MAX                              (1 << 20)
#define NLO_RUNTIME_OPERATOR_CONSTANTS_MAX  16

/* -------------------------------------------------------------------
 * nlo_complex
 * ------------------------------------------------------------------- */

typedef struct { double re; double im; } nlo_complex;

/* -------------------------------------------------------------------
 * Simulation configuration sub-structs
 * ------------------------------------------------------------------- */

typedef struct {
    double gamma;
} nonlinear_params;

typedef struct {
    size_t num_dispersion_terms;
    double betas[NT_MAX];
    double alpha;
} dispersion_params;

typedef struct {
    double starting_step_size;
    double max_step_size;
    double min_step_size;
    double error_tolerance;
    double propagation_distance;
} propagation_params;

typedef struct {
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
    double grin_gx;
    double grin_gy;
    nlo_complex* spatial_frequency_grid;
    nlo_complex* grin_potential_phase_grid;
} spatial_grid;

typedef struct {
    const char* dispersion_expr;
    const char* nonlinear_expr;
    size_t      num_constants;
    double      constants[NLO_RUNTIME_OPERATOR_CONSTANTS_MAX];
} runtime_operator_params;

typedef struct {
    nonlinear_params        nonlinear;
    dispersion_params       dispersion;
    propagation_params      propagation;
    time_grid               time;
    frequency_grid          frequency;
    spatial_grid            spatial;
    runtime_operator_params runtime;
} sim_config;

/* -------------------------------------------------------------------
 * Backend / execution option enums
 * ------------------------------------------------------------------- */

typedef enum {
    NLO_VECTOR_BACKEND_CPU    = 0,
    NLO_VECTOR_BACKEND_VULKAN = 1,
    NLO_VECTOR_BACKEND_AUTO   = 2
} nlo_vector_backend_type;

typedef enum {
    NLO_FFT_BACKEND_AUTO = 0,
    NLO_FFT_BACKEND_FFTW = 1,
    NLO_FFT_BACKEND_VKFFT = 2
} nlo_fft_backend_type;

/* -------------------------------------------------------------------
 * Vulkan backend config (Vulkan handles replaced with opaque stubs)
 * ------------------------------------------------------------------- */

typedef void*    VkPhysicalDevice;
typedef void*    VkDevice;
typedef void*    VkQueue;
typedef void*    VkCommandPool;

typedef struct {
    VkPhysicalDevice physical_device;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queue_family_index;
    VkCommandPool    command_pool;
    size_t           descriptor_set_budget_bytes;
    uint32_t         descriptor_set_count_override;
} nlo_vk_backend_config;

/* -------------------------------------------------------------------
 * Execution options
 * ------------------------------------------------------------------- */

typedef struct {
    nlo_vector_backend_type backend_type;
    nlo_fft_backend_type    fft_backend;
    double                  device_heap_fraction;
    size_t                  record_ring_target;
    size_t                  forced_device_budget_bytes;
    nlo_vk_backend_config   vulkan;
} nlo_execution_options;

/* -------------------------------------------------------------------
 * Status codes
 * ------------------------------------------------------------------- */

typedef enum {
    NLOLIB_STATUS_OK               = 0,
    NLOLIB_STATUS_INVALID_ARGUMENT = 1,
    NLOLIB_STATUS_ALLOCATION_FAILED = 2,
    NLOLIB_STATUS_NOT_IMPLEMENTED  = 3
} nlolib_status;

/* -------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------- */

nlolib_status nlolib_propagate(
    const sim_config*            config,
    size_t                       num_time_samples,
    const nlo_complex*           input_field,
    size_t                       num_recorded_samples,
    nlo_complex*                 output_records,
    const nlo_execution_options* exec_options
);

#endif /* NLOLIB_MATLAB_H */
