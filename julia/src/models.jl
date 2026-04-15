Base.@kwdef struct RuntimeOperators
    linear_factor_expr::Union{Nothing, String} = nothing
    linear_expr::Union{Nothing, String} = nothing
    potential_expr::Union{Nothing, String} = nothing
    dispersion_factor_expr::Union{Nothing, String} = nothing
    dispersion_expr::Union{Nothing, String} = nothing
    nonlinear_expr::Union{Nothing, String} = nothing
    linear_factor_fn::Any = nothing
    linear_fn::Any = nothing
    potential_fn::Any = nothing
    dispersion_factor_fn::Any = nothing
    dispersion_fn::Any = nothing
    nonlinear_fn::Any = nothing
    nonlinear_model::Int = Int(NONLINEAR_MODEL_EXPR)
    nonlinear_gamma::Float64 = 0.0
    raman_fraction::Float64 = 0.0
    raman_tau1::Float64 = 0.0122
    raman_tau2::Float64 = 0.0320
    shock_omega0::Float64 = 0.0
    raman_response_time::Any = nothing
    constants::Vector{Float64} = Float64[]
    constant_bindings::Union{Nothing, Dict{String, Float64}} = nothing
    auto_capture_constants::Bool = true
end

Base.@kwdef mutable struct PulseSpec
    samples::Vector{ComplexF64}
    delta_time::Float64
    pulse_period::Union{Nothing, Float64} = nothing
    frequency_grid::Union{Nothing, Vector{ComplexF64}} = nothing
    tensor_nt::Union{Nothing, Int} = nothing
    tensor_nx::Union{Nothing, Int} = nothing
    tensor_ny::Union{Nothing, Int} = nothing
    tensor_layout::Int = Int(TENSOR_LAYOUT_XYT_T_FAST)
    delta_x::Float64 = 1.0
    delta_y::Float64 = 1.0
    spatial_frequency_grid::Union{Nothing, Vector{ComplexF64}} = nothing
    potential_grid::Union{Nothing, Vector{ComplexF64}} = nothing
end

Base.@kwdef struct OperatorSpec
    expr::Union{Nothing, String} = nothing
    fn::Any = nothing
    params::Any = nothing
end

struct ProgressInfo
    event_type::Int
    step_index::Int
    reject_attempt::Int
    z::Float64
    z_end::Float64
    percent::Float64
    step_size::Float64
    next_step_size::Float64
    error::Float64
    elapsed_seconds::Float64
    eta_seconds::Float64
end

struct PropagationResult
    records::Matrix{ComplexF64}
    z_axis::Vector{Float64}
    final::AbstractVector{ComplexF64}
    meta::Dict{String, Any}
end

struct PropagateResult
    records::Matrix{ComplexF64}
    z_axis::Vector{Float64}
    final::AbstractVector{ComplexF64}
    meta::Dict{String, Any}
    status::Int
    message::String
    t_events::Vector{Vector{Float64}}
    y_events::Vector{Matrix{ComplexF64}}
    sol::Any
end

struct PropagationAbortedError <: Exception
    result::PropagationResult
end

Base.showerror(io::IO, err::PropagationAbortedError) =
    print(io, "nlolib_propagate aborted by progress callback")
