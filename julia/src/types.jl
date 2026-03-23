const NLO_RUNTIME_OPERATOR_CONSTANTS_MAX = 16
const NLO_STORAGE_RUN_ID_MAX = 64

const NLO_TENSOR_LAYOUT_XYT_T_FAST = Cint(0)

const NLO_NONLINEAR_MODEL_EXPR = Cint(0)
const NLO_NONLINEAR_MODEL_KERR_RAMAN = Cint(1)

const NLO_VECTOR_BACKEND_CPU = Cint(0)
const NLO_VECTOR_BACKEND_VULKAN = Cint(1)
const NLO_VECTOR_BACKEND_CUDA = Cint(2)
const NLO_VECTOR_BACKEND_AUTO = Cint(3)

const NLO_FFT_BACKEND_AUTO = Cint(0)
const NLO_FFT_BACKEND_FFTW = Cint(1)
const NLO_FFT_BACKEND_VKFFT = Cint(2)
const NLO_FFT_BACKEND_CUFFT = Cint(3)
const NLO_FFT_BACKEND_CUFFT_XT = Cint(4)

const NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES = Cint(0)
const NLO_STORAGE_DB_CAP_POLICY_FAIL = Cint(1)

const NLO_PROPAGATE_OUTPUT_DENSE = Cint(0)
const NLO_PROPAGATE_OUTPUT_FINAL_ONLY = Cint(1)

const NLOLIB_STATUS_OK = Cint(0)
const NLOLIB_STATUS_INVALID_ARGUMENT = Cint(1)
const NLOLIB_STATUS_ALLOCATION_FAILED = Cint(2)
const NLOLIB_STATUS_NOT_IMPLEMENTED = Cint(3)
const NLOLIB_STATUS_ABORTED = Cint(4)

const NLOLIB_LOG_LEVEL_ERROR = Cint(0)
const NLOLIB_LOG_LEVEL_WARN = Cint(1)
const NLOLIB_LOG_LEVEL_INFO = Cint(2)
const NLOLIB_LOG_LEVEL_DEBUG = Cint(3)

const NLOLIB_PROGRESS_STREAM_STDERR = Cint(0)
const NLOLIB_PROGRESS_STREAM_STDOUT = Cint(1)
const NLOLIB_PROGRESS_STREAM_BOTH = Cint(2)

const NLO_PROGRESS_EVENT_ACCEPTED = Cint(0)
const NLO_PROGRESS_EVENT_REJECTED = Cint(1)
const NLO_PROGRESS_EVENT_FINISH = Cint(2)

struct NLOComplex
    re::Cdouble
    im::Cdouble
end

NLOComplex(z::Complex) = NLOComplex(real(z), imag(z))
Base.convert(::Type{ComplexF64}, z::NLOComplex) = ComplexF64(z.re, z.im)

struct PropagationParams
    starting_step_size::Cdouble
    max_step_size::Cdouble
    min_step_size::Cdouble
    error_tolerance::Cdouble
    propagation_distance::Cdouble
end

struct TimeGrid
    nt::Csize_t
    pulse_period::Cdouble
    delta_time::Cdouble
    wt_axis::Ptr{NLOComplex}
end

struct FrequencyGrid
    frequency_grid::Ptr{NLOComplex}
end

struct SpatialGrid
    nx::Csize_t
    ny::Csize_t
    delta_x::Cdouble
    delta_y::Cdouble
    spatial_frequency_grid::Ptr{NLOComplex}
    kx_axis::Ptr{NLOComplex}
    ky_axis::Ptr{NLOComplex}
    potential_grid::Ptr{NLOComplex}
end

struct Tensor3DDesc
    nt::Csize_t
    nx::Csize_t
    ny::Csize_t
    layout::Cint
end

struct RuntimeOperatorParams
    linear_factor_expr::Cstring
    linear_expr::Cstring
    potential_expr::Cstring
    dispersion_factor_expr::Cstring
    dispersion_expr::Cstring
    nonlinear_expr::Cstring
    nonlinear_model::Cint
    nonlinear_gamma::Cdouble
    raman_fraction::Cdouble
    raman_tau1::Cdouble
    raman_tau2::Cdouble
    shock_omega0::Cdouble
    raman_response_time::Ptr{NLOComplex}
    raman_response_len::Csize_t
    num_constants::Csize_t
    constants::NTuple{NLO_RUNTIME_OPERATOR_CONSTANTS_MAX, Cdouble}
end

const PhysicsConfig = RuntimeOperatorParams

struct SimulationConfig
    propagation::PropagationParams
    tensor::Tensor3DDesc
    time::TimeGrid
    frequency::FrequencyGrid
    spatial::SpatialGrid
end

struct VulkanBackendConfig
    physical_device::Ptr{Cvoid}
    device::Ptr{Cvoid}
    queue::Ptr{Cvoid}
    queue_family_index::UInt32
    command_pool::Ptr{Cvoid}
    descriptor_set_budget_bytes::Csize_t
    descriptor_set_count_override::UInt32
end

struct CudaBackendConfig
    device_ordinal::Cint
    enable_multi_gpu::Cint
    max_devices::UInt32
    enable_peer_access::Cint
    stream_count::UInt32
    pinned_staging_bytes::Csize_t
    graph_capture_enabled::Cint
    nvrtc_enabled::Cint
end

struct ExecutionOptions
    backend_type::Cint
    fft_backend::Cint
    device_heap_fraction::Cdouble
    record_ring_target::Csize_t
    forced_device_budget_bytes::Csize_t
    vulkan::VulkanBackendConfig
    cuda::CudaBackendConfig
end

struct RuntimeLimits
    max_num_time_samples_runtime::Csize_t
    max_num_recorded_samples_in_memory::Csize_t
    max_num_recorded_samples_with_storage::Csize_t
    estimated_required_working_set_bytes::Csize_t
    estimated_device_budget_bytes::Csize_t
    storage_available::Cint
end

struct StorageOptions
    sqlite_path::Cstring
    run_id::Cstring
    sqlite_max_bytes::Csize_t
    chunk_records::Csize_t
    cap_policy::Cint
    log_final_output_field_to_db::Cint
end

struct StorageResult
    run_id::NTuple{NLO_STORAGE_RUN_ID_MAX, Cchar}
    records_captured::Csize_t
    records_spilled::Csize_t
    chunks_written::Csize_t
    db_size_bytes::Csize_t
    truncated::Cint
end

struct StepEvent
    step_index::Csize_t
    z_current::Cdouble
    step_size::Cdouble
    next_step_size::Cdouble
    error::Cdouble
end

struct CProgressInfo
    event_type::Cint
    step_index::Csize_t
    reject_attempt::Csize_t
    z::Cdouble
    z_end::Cdouble
    percent::Cdouble
    step_size::Cdouble
    next_step_size::Cdouble
    error::Cdouble
    elapsed_seconds::Cdouble
    eta_seconds::Cdouble
end

struct PropagateOptions
    num_recorded_samples::Csize_t
    output_mode::Cint
    return_records::Cint
    exec_options::Ptr{ExecutionOptions}
    storage_options::Ptr{StorageOptions}
    explicit_record_z::Ptr{Cdouble}
    explicit_record_z_count::Csize_t
    progress_callback::Ptr{Cvoid}
    progress_user_data::Ptr{Cvoid}
end

struct PropagateOutput
    output_records::Ptr{NLOComplex}
    output_record_capacity::Csize_t
    records_written::Ptr{Csize_t}
    storage_result::Ptr{StorageResult}
    output_step_events::Ptr{StepEvent}
    output_step_event_capacity::Csize_t
    step_events_written::Ptr{Csize_t}
    step_events_dropped::Ptr{Csize_t}
end

struct PreparedValue{T}
    value::T
    keepalive::Tuple
end

_zero_chars(::Val{N}) where {N} = ntuple(_ -> Cchar(0), N)
_zero_constants() = ntuple(_ -> 0.0, NLO_RUNTIME_OPERATOR_CONSTANTS_MAX)

PropagationParams() = PropagationParams(0.0, 0.0, 0.0, 0.0, 0.0)
TimeGrid() = TimeGrid(0, 0.0, 0.0, C_NULL)
FrequencyGrid() = FrequencyGrid(C_NULL)
SpatialGrid() = SpatialGrid(0, 0, 1.0, 1.0, C_NULL, C_NULL, C_NULL, C_NULL)
Tensor3DDesc() = Tensor3DDesc(0, 0, 0, NLO_TENSOR_LAYOUT_XYT_T_FAST)
RuntimeOperatorParams() = RuntimeOperatorParams(
    C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
    NLO_NONLINEAR_MODEL_EXPR, 0.0, 0.0, 0.0, 0.0, 0.0, C_NULL, 0, 0, _zero_constants()
)
SimulationConfig() = SimulationConfig(PropagationParams(), Tensor3DDesc(), TimeGrid(), FrequencyGrid(), SpatialGrid())
VulkanBackendConfig() = VulkanBackendConfig(C_NULL, C_NULL, C_NULL, 0, C_NULL, 0, 0)
CudaBackendConfig() = CudaBackendConfig(-1, 1, UInt32(8), 1, UInt32(1), 0, 1, 1)
ExecutionOptions() = ExecutionOptions(NLO_VECTOR_BACKEND_AUTO,
                                      NLO_FFT_BACKEND_AUTO,
                                      0.70,
                                      0,
                                      0,
                                      VulkanBackendConfig(),
                                      CudaBackendConfig())
RuntimeLimits() = RuntimeLimits(0, 0, 0, 0, 0, 0)
StorageOptions() = StorageOptions(C_NULL, C_NULL, 0, 0, NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES, 0)
StorageResult() = StorageResult(_zero_chars(Val(NLO_STORAGE_RUN_ID_MAX)), 0, 0, 0, 0, 0)
StepEvent() = StepEvent(0, 0.0, 0.0, 0.0, 0.0)

default_execution_options(; backend_type::Integer = NLO_VECTOR_BACKEND_AUTO,
                            fft_backend::Integer = NLO_FFT_BACKEND_AUTO,
                            device_heap_fraction::Real = 0.70,
                            record_ring_target::Integer = 0,
                            forced_device_budget_bytes::Integer = 0,
                            vulkan::VulkanBackendConfig = VulkanBackendConfig(),
                            cuda::CudaBackendConfig = CudaBackendConfig()) =
    ExecutionOptions(
        Cint(backend_type),
        Cint(fft_backend),
        Cdouble(device_heap_fraction),
        Csize_t(record_ring_target),
        Csize_t(forced_device_budget_bytes),
        vulkan,
        cuda
    )

default_storage_options(; kwargs...) = storage_options(; kwargs...)
