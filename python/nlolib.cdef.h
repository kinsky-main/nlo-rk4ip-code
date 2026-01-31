// CFFI ABI declarations for NLOLib.

// MARK: Includes

// MARK: Const & Macros

// MARK: Typedefs

typedef struct { double re; double im; } nlo_complex;

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

typedef enum {
    NLOLIB_STATUS_OK = 0,
    NLOLIB_STATUS_INVALID_ARGUMENT = 1,
    NLOLIB_STATUS_ALLOCATION_FAILED = 2,
    NLOLIB_STATUS_NOT_IMPLEMENTED = 3
} nlolib_status;

nlolib_status nlolib_propagate(const sim_config* config,
                               size_t num_time_samples,
                               const nlo_complex* input_field,
                               nlo_complex* output_field);
