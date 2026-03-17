const _LINEAR_OPERATOR_PRESETS = Dict(
    "none" => (expr = "0", params = Dict{String, Float64}()),
    "gvd" => (expr = "i*beta2*w*w-loss", params = Dict("beta2" => -0.5, "loss" => 0.0)),
)

const _NONLINEAR_OPERATOR_PRESETS = Dict(
    "none" => (expr = "0", params = Dict{String, Float64}()),
    "kerr" => (expr = "i*gamma*A*I", params = Dict("gamma" => 1.0)),
)

struct PreparedSimConfig
    simulation_config::SimulationConfig
    physics_config::PhysicsConfig
    keepalive::Vector{Any}
end

function _complex_vector(values)
    out = ComplexF64.(collect(values))
    isempty(out) && error("complex vector inputs must be non-empty")
    return out
end

function _complex_vector_or_nothing(values)
    values === nothing && return nothing
    return ComplexF64.(collect(values))
end

function _default_frequency_grid(num_time_samples::Integer, delta_time::Real)
    two_pi = 2.0 * π
    omega_step = two_pi / (Float64(num_time_samples) * Float64(delta_time))
    positive_limit = (Int(num_time_samples) - 1) ÷ 2
    out = Vector{ComplexF64}(undef, Int(num_time_samples))
    for i in eachindex(out)
        idx = i - 1
        omega = idx <= positive_limit ? Float64(idx) * omega_step : -(Float64(num_time_samples - idx) * omega_step)
        out[i] = ComplexF64(omega, 0.0)
    end
    return out
end

function _coerce_pulse_spec(pulse)
    pulse isa PulseSpec && return pulse
    if pulse isa NamedTuple || pulse isa AbstractDict
        getvalue(name, default=nothing) = pulse isa NamedTuple ? get(pulse, name, default) : get(pulse, String(name), get(pulse, name, default))
        samples = ComplexF64.(collect(getvalue(:samples, nothing)))
        delta_time = Float64(getvalue(:delta_time, throw(ArgumentError("pulse must define samples and delta_time"))))
        return PulseSpec(
            samples = samples,
            delta_time = delta_time,
            pulse_period = isnothing(getvalue(:pulse_period, nothing)) ? nothing : Float64(getvalue(:pulse_period)),
            frequency_grid = _complex_vector_or_nothing(getvalue(:frequency_grid, nothing)),
            tensor_nt = isnothing(getvalue(:tensor_nt, nothing)) ? nothing : Int(getvalue(:tensor_nt)),
            tensor_nx = isnothing(getvalue(:tensor_nx, nothing)) ? nothing : Int(getvalue(:tensor_nx)),
            tensor_ny = isnothing(getvalue(:tensor_ny, nothing)) ? nothing : Int(getvalue(:tensor_ny)),
            tensor_layout = Int(getvalue(:tensor_layout, NLO_TENSOR_LAYOUT_XYT_T_FAST)),
            delta_x = Float64(getvalue(:delta_x, 1.0)),
            delta_y = Float64(getvalue(:delta_y, 1.0)),
            spatial_frequency_grid = _complex_vector_or_nothing(getvalue(:spatial_frequency_grid, nothing)),
            potential_grid = _complex_vector_or_nothing(getvalue(:potential_grid, nothing)),
        )
    end
    throw(ArgumentError("pulse must be a PulseSpec, NamedTuple, or AbstractDict"))
end

function _normalize_pulse_spec(pulse)
    spec = _coerce_pulse_spec(pulse)
    isempty(spec.samples) && throw(ArgumentError("pulse.samples must be non-empty"))
    spec.delta_time > 0.0 || throw(ArgumentError("pulse.delta_time must be > 0"))
    if !isnothing(spec.tensor_nt)
        spec.tensor_nt > 0 || throw(ArgumentError("pulse.tensor_nt must be > 0"))
    end
    return spec
end

function _validate_coupled_pulse_spec(pulse::PulseSpec)
    isnothing(pulse.tensor_nt) && throw(ArgumentError("pulse.tensor_nt must be provided for tensor runs"))
    isnothing(pulse.tensor_nx) && throw(ArgumentError("pulse.tensor_nx must be provided for tensor runs"))
    isnothing(pulse.tensor_ny) && throw(ArgumentError("pulse.tensor_ny must be provided for tensor runs"))
    nt = Int(pulse.tensor_nt)
    nx = Int(pulse.tensor_nx)
    ny = Int(pulse.tensor_ny)
    nt > 0 || throw(ArgumentError("pulse.tensor_nt must be > 0"))
    nx > 0 || throw(ArgumentError("pulse.tensor_nx must be > 0"))
    ny > 0 || throw(ArgumentError("pulse.tensor_ny must be > 0"))
    length(pulse.samples) == nt * nx * ny ||
        throw(ArgumentError("length(pulse.samples) must equal tensor_nt*tensor_nx*tensor_ny"))
    xy = nx * ny
    if !isnothing(pulse.spatial_frequency_grid)
        length(pulse.spatial_frequency_grid) in (xy, nt * nx * ny) ||
            throw(ArgumentError("spatial_frequency_grid must have length nx*ny or nt*nx*ny"))
    end
    if !isnothing(pulse.potential_grid)
        length(pulse.potential_grid) in (xy, nt * nx * ny) ||
            throw(ArgumentError("potential_grid must have length nx*ny or nt*nx*ny"))
    end
end

function _solver_profile_defaults(profile::AbstractString, propagation_distance::Real)
    distance = Float64(propagation_distance)
    distance > 0.0 || throw(ArgumentError("propagation_distance must be > 0"))
    if profile == "balanced"
        return Dict(
            "starting_step_size" => distance / 200.0,
            "max_step_size" => distance / 25.0,
            "min_step_size" => distance / 20000.0,
            "error_tolerance" => 1e-6,
            "records" => 128,
        )
    elseif profile == "fast"
        return Dict(
            "starting_step_size" => distance / 120.0,
            "max_step_size" => distance / 12.0,
            "min_step_size" => distance / 4000.0,
            "error_tolerance" => 5e-6,
            "records" => 64,
        )
    elseif profile == "accuracy"
        return Dict(
            "starting_step_size" => distance / 400.0,
            "max_step_size" => distance / 50.0,
            "min_step_size" => distance / 80000.0,
            "error_tolerance" => 1e-7,
            "records" => 192,
        )
    end
    throw(ArgumentError("unsupported preset '$profile'"))
end

function _shift_constant_indices(expression::AbstractString, offset::Integer)
    offset == 0 && return String(expression)
    return replace(String(expression), r"\bc(\d+)\b" => s -> "c$(parse(Int, s.captures[1]) + offset)")
end

function _parameterize_expression(expression::AbstractString, params, offset::Integer)
    params === nothing && return (String(expression), Float64[])
    if params isa AbstractDict
        rewritten = String(expression)
        constants = Float64[]
        for (idx, pair) in enumerate(params)
            name = String(first(pair))
            occursin(r"^[A-Za-z_]\w*$", name) || throw(ArgumentError("invalid parameter name '$name'"))
            rewritten = replace(rewritten, Regex("\\b" * name * "\\b") => "c$(offset + idx - 1)")
            push!(constants, Float64(last(pair)))
        end
        return rewritten, constants
    end
    constants = Float64.(collect(params))
    return _shift_constant_indices(expression, offset), constants
end

function _coerce_operator_spec(operator)
    operator isa OperatorSpec && return operator
    operator isa AbstractString && return OperatorSpec(expr = String(operator))
    if operator isa NamedTuple || operator isa AbstractDict
        getvalue(name, default=nothing) = operator isa NamedTuple ? get(operator, name, default) : get(operator, String(name), get(operator, name, default))
        return OperatorSpec(
            expr = isnothing(getvalue(:expr, nothing)) ? nothing : String(getvalue(:expr)),
            fn = getvalue(:fn, nothing),
            params = getvalue(:params, nothing),
        )
    end
    throw(ArgumentError("operator must be a preset string, OperatorSpec, NamedTuple, or AbstractDict"))
end

function _unsupported_callable_operator(context::AbstractString)
    throw(ArgumentError(
        "$(context) callable operators are not implemented in the Julia high-level API; use expression strings or built-in presets instead"
    ))
end

function _resolve_operator_spec(context::AbstractString, operator, offset::Integer)
    presets = context == "linear" ? _LINEAR_OPERATOR_PRESETS : _NONLINEAR_OPERATOR_PRESETS
    spec = _coerce_operator_spec(operator)
    spec.expr !== nothing && spec.fn !== nothing &&
        throw(ArgumentError("$context operator cannot define both expr and fn"))
    expr = spec.expr
    params = spec.params
    if spec.fn !== nothing
        _unsupported_callable_operator(context)
    end
    if expr === nothing
        throw(ArgumentError("$context operator must define expr/fn or a known preset"))
    end
    if haskey(presets, expr)
        preset = presets[expr]
        expr = preset.expr
        params = preset.params
    end
    return _parameterize_expression(expr, params, offset)
end

function _storage_result_to_meta(storage_result::StorageResult)
    return Dict(
        "run_id" => _decode_run_id(storage_result.run_id),
        "records_captured" => Int(storage_result.records_captured),
        "records_spilled" => Int(storage_result.records_spilled),
        "chunks_written" => Int(storage_result.chunks_written),
        "db_size_bytes" => Int(storage_result.db_size_bytes),
        "truncated" => Bool(storage_result.truncated != 0),
    )
end

function _step_events_to_meta(step_events, count::Integer)
    n = max(0, Int(count))
    return Dict{String, Any}(
        "step_index" => [Int(step_events[i].step_index) for i in 1:n],
        "z" => [Float64(step_events[i].z_current) for i in 1:n],
        "step_size" => [Float64(step_events[i].step_size) for i in 1:n],
        "next_step_size" => [Float64(step_events[i].next_step_size) for i in 1:n],
        "error" => [Float64(step_events[i].error) for i in 1:n],
    )
end

function _progress_info_from_struct(info::CProgressInfo)
    return ProgressInfo(
        Int(info.event_type),
        Int(info.step_index),
        Int(info.reject_attempt),
        Float64(info.z),
        Float64(info.z_end),
        Float64(info.percent),
        Float64(info.step_size),
        Float64(info.next_step_size),
        Float64(info.error),
        Float64(info.elapsed_seconds),
        Float64(info.eta_seconds),
    )
end

function _validate_explicit_record_z(z_values, distance::Real)
    values = Float64.(collect(z_values))
    isempty(values) && throw(ArgumentError("t_eval must be non-empty"))
    prev = values[1]
    (0.0 <= prev <= Float64(distance)) || throw(ArgumentError("t_eval values must be within [0, propagation_distance]"))
    for current in Iterators.drop(values, 1)
        current >= prev || throw(ArgumentError("t_eval must be monotonic nondecreasing"))
        (0.0 <= current <= Float64(distance)) || throw(ArgumentError("t_eval values must be within [0, propagation_distance]"))
        prev = current
    end
    return values
end

function _cfg_complex_ptr(values::Vector{ComplexF64})
    return Base.unsafe_convert(Ptr{NLOComplex}, reinterpret(NLOComplex, values))
end

function _cfg_complex_ptr_or_null(values)
    values === nothing && return C_NULL
    return _cfg_complex_ptr(values)
end

_cfg_string_ptr(value::Union{Nothing, AbstractString}) = value === nothing ? C_NULL : Base.unsafe_convert(Cstring, value)

function _cfg_constants_tuple(values)
    constants = zeros(Float64, NLO_RUNTIME_OPERATOR_CONSTANTS_MAX)
    count = min(length(values), NLO_RUNTIME_OPERATOR_CONSTANTS_MAX)
    for idx in 1:count
        constants[idx] = Float64(values[idx])
    end
    return Tuple(constants)
end

function prepare_sim_config(num_time_samples::Integer;
                            propagation_distance::Real,
                            starting_step_size::Real,
                            max_step_size::Real,
                            min_step_size::Real,
                            error_tolerance::Real,
                            pulse_period::Real,
                            delta_time::Real,
                            tensor_nt = nothing,
                            tensor_nx = nothing,
                            tensor_ny = nothing,
                            tensor_layout::Integer = NLO_TENSOR_LAYOUT_XYT_T_FAST,
                            frequency_grid,
                            wt_axis = nothing,
                            delta_x::Real = 1.0,
                            delta_y::Real = 1.0,
                            kx_axis = nothing,
                            ky_axis = nothing,
                            spatial_frequency_grid = nothing,
                            potential_grid = nothing,
                            runtime::Union{Nothing, RuntimeOperators} = nothing)
    Int(num_time_samples) > 0 || throw(ArgumentError("num_time_samples must be > 0"))
    keepalive = Any[]
    total_samples = Int(num_time_samples)

    tensor_mode = tensor_nt !== nothing && Int(tensor_nt) > 0
    resolved_nt = total_samples
    resolved_nx = 1
    resolved_ny = 1
    if tensor_mode
        tensor_nx === nothing && throw(ArgumentError("tensor_nx is required when tensor_nt is set"))
        tensor_ny === nothing && throw(ArgumentError("tensor_ny is required when tensor_nt is set"))
        resolved_nt = Int(tensor_nt)
        resolved_nx = Int(tensor_nx)
        resolved_ny = Int(tensor_ny)
        resolved_nt * resolved_nx * resolved_ny == total_samples ||
            throw(ArgumentError("tensor_nt*tensor_nx*tensor_ny must match num_time_samples"))
    end

    freq_values = ComplexF64.(collect(frequency_grid))
    length(freq_values) in (resolved_nt, total_samples) ||
        throw(ArgumentError("frequency_grid length must match resolved nt or full sample count"))
    push!(keepalive, freq_values)

    wt_values = wt_axis === nothing ? nothing : ComplexF64.(collect(wt_axis))
    if wt_values !== nothing
        length(wt_values) == resolved_nt || throw(ArgumentError("wt_axis length must match resolved nt"))
        push!(keepalive, wt_values)
    end

    kx_values = kx_axis === nothing ? nothing : ComplexF64.(collect(kx_axis))
    if kx_values !== nothing
        length(kx_values) == resolved_nx || throw(ArgumentError("kx_axis length must match resolved nx"))
        push!(keepalive, kx_values)
    end

    ky_values = ky_axis === nothing ? nothing : ComplexF64.(collect(ky_axis))
    if ky_values !== nothing
        length(ky_values) == resolved_ny || throw(ArgumentError("ky_axis length must match resolved ny"))
        push!(keepalive, ky_values)
    end

    spatial_values = spatial_frequency_grid === nothing ? nothing : ComplexF64.(collect(spatial_frequency_grid))
    if spatial_values !== nothing
        valid_lengths = tensor_mode ? (resolved_nx * resolved_ny, total_samples) : (total_samples,)
        length(spatial_values) in valid_lengths ||
            throw(ArgumentError("spatial_frequency_grid length is inconsistent with the configured geometry"))
        push!(keepalive, spatial_values)
    end

    potential_values = potential_grid === nothing ? nothing : ComplexF64.(collect(potential_grid))
    if potential_values !== nothing
        if tensor_mode
            xy = resolved_nx * resolved_ny
            if length(potential_values) == xy
                potential_values = repeat(potential_values, resolved_nt)
            elseif length(potential_values) != total_samples
                throw(ArgumentError("potential_grid length must match nt*nx*ny or nx*ny for tensor runs"))
            end
        else
            length(potential_values) == total_samples ||
                throw(ArgumentError("potential_grid length must match num_time_samples for temporal runs"))
        end
        push!(keepalive, potential_values)
    end

    propagation = PropagationParams(
        Cdouble(starting_step_size),
        Cdouble(max_step_size),
        Cdouble(min_step_size),
        Cdouble(error_tolerance),
        Cdouble(propagation_distance),
    )
    tensor = tensor_mode ?
        Tensor3DDesc(Csize_t(resolved_nt), Csize_t(resolved_nx), Csize_t(resolved_ny), Cint(tensor_layout)) :
        Tensor3DDesc()
    time = TimeGrid(
        Csize_t(resolved_nt),
        Cdouble(pulse_period),
        Cdouble(delta_time),
        wt_values === nothing ? C_NULL : _cfg_complex_ptr(wt_values),
    )
    frequency = FrequencyGrid(_cfg_complex_ptr(freq_values))
    spatial = SpatialGrid(
        tensor_mode ? Csize_t(resolved_nx) : Csize_t(0),
        tensor_mode ? Csize_t(resolved_ny) : Csize_t(0),
        Cdouble(delta_x),
        Cdouble(delta_y),
        spatial_values === nothing ? C_NULL : _cfg_complex_ptr(spatial_values),
        kx_values === nothing ? C_NULL : _cfg_complex_ptr(kx_values),
        ky_values === nothing ? C_NULL : _cfg_complex_ptr(ky_values),
        potential_values === nothing ? C_NULL : _cfg_complex_ptr(potential_values),
    )

    physics = PhysicsConfig()
    if runtime !== nothing
        runtime.linear_factor_expr !== nothing && runtime.dispersion_factor_expr !== nothing &&
            throw(ArgumentError("linear_factor_expr and dispersion_factor_expr are aliases; provide only one"))
        runtime.linear_expr !== nothing && runtime.dispersion_expr !== nothing &&
            throw(ArgumentError("linear_expr and dispersion_expr are aliases; provide only one"))
        any(op !== nothing for op in (runtime.linear_factor_fn, runtime.linear_fn, runtime.potential_fn,
                                      runtime.dispersion_factor_fn, runtime.dispersion_fn, runtime.nonlinear_fn)) &&
            _unsupported_callable_operator("runtime")

        linear_factor_expr = isnothing(runtime.linear_factor_expr) ? runtime.dispersion_factor_expr : runtime.linear_factor_expr
        linear_expr = isnothing(runtime.linear_expr) ? runtime.dispersion_expr : runtime.linear_expr
        potential_expr = runtime.potential_expr
        nonlinear_expr = runtime.nonlinear_expr
        constants = Float64.(runtime.constants)
        length(constants) <= NLO_RUNTIME_OPERATOR_CONSTANTS_MAX ||
            throw(ArgumentError("runtime.constants exceeds NLO_RUNTIME_OPERATOR_CONSTANTS_MAX"))

        raman_response = runtime.raman_response_time === nothing ? nothing : ComplexF64.(collect(runtime.raman_response_time))
        if raman_response !== nothing
            push!(keepalive, raman_response)
        end

        for value in (linear_factor_expr, linear_expr, potential_expr, nonlinear_expr)
            value === nothing || push!(keepalive, String(value))
        end
        linear_factor_str = linear_factor_expr === nothing ? nothing : String(linear_factor_expr)
        linear_str = linear_expr === nothing ? nothing : String(linear_expr)
        potential_str = potential_expr === nothing ? nothing : String(potential_expr)
        nonlinear_str = nonlinear_expr === nothing ? nothing : String(nonlinear_expr)

        physics = PhysicsConfig(
            _cfg_string_ptr(linear_factor_str),
            _cfg_string_ptr(linear_str),
            _cfg_string_ptr(potential_str),
            _cfg_string_ptr(linear_factor_str),
            _cfg_string_ptr(linear_str),
            _cfg_string_ptr(nonlinear_str),
            Cint(runtime.nonlinear_model),
            Cdouble(runtime.nonlinear_gamma),
            Cdouble(runtime.raman_fraction),
            Cdouble(runtime.raman_tau1),
            Cdouble(runtime.raman_tau2),
            Cdouble(runtime.shock_omega0),
            raman_response === nothing ? C_NULL : _cfg_complex_ptr(raman_response),
            Csize_t(raman_response === nothing ? 0 : length(raman_response)),
            Csize_t(length(constants)),
            _cfg_constants_tuple(constants),
        )
    end

    simulation = SimulationConfig(propagation, tensor, time, frequency, spatial)
    return PreparedSimConfig(simulation, physics, keepalive)
end
