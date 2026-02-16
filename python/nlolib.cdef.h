// CFFI ABI declarations for NLOLib.

// MARK: Includes

// MARK: Const & Macros

// MARK: Typedefs

typedef struct { double re; double im; } nlo_complex;
typedef unsigned int uint32_t;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkCommandPool;

typedef enum {
    NLO_VECTOR_BACKEND_CPU = 0,
    NLO_VECTOR_BACKEND_VULKAN = 1
} nlo_vector_backend_type;

typedef enum {
    NLO_FFT_BACKEND_AUTO = 0,
    NLO_FFT_BACKEND_FFTW = 1,
    NLO_FFT_BACKEND_VKFFT = 2
} nlo_fft_backend_type;

typedef struct {
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
    size_t descriptor_set_budget_bytes;
    uint32_t descriptor_set_count_override;
} nlo_vk_backend_config;

typedef struct {
    double gamma;
} nonlinear_params;

typedef struct {
    size_t num_dispersion_terms;
    double betas[1048576];
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
    nonlinear_params nonlinear;
    dispersion_params dispersion;
    propagation_params propagation;
    time_grid time;
    frequency_grid frequency;
} sim_config;

typedef struct {
    nlo_vector_backend_type backend_type;
    nlo_fft_backend_type fft_backend;
    double device_heap_fraction;
    size_t record_ring_target;
    size_t forced_device_budget_bytes;
    nlo_vk_backend_config vulkan;
} nlo_execution_options;

typedef enum {
    NLOLIB_STATUS_OK = 0,
    NLOLIB_STATUS_INVALID_ARGUMENT = 1,
    NLOLIB_STATUS_ALLOCATION_FAILED = 2,
    NLOLIB_STATUS_NOT_IMPLEMENTED = 3
} nlolib_status;

nlolib_status nlolib_propagate(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options
);
