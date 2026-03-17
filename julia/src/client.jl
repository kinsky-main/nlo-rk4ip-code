struct NLolib
end

const _DEFAULT_API = Ref{Union{Nothing, NLolib}}(nothing)

function NLolib(path::AbstractString)
    load(path)
    return NLolib()
end

function NLolib(::Nothing)
    load()
    return NLolib()
end

function _api()
    if _DEFAULT_API[] === nothing
        _DEFAULT_API[] = NLolib(nothing)
    end
    return _DEFAULT_API[]
end

storage_is_available(::NLolib) = storage_is_available()
query_runtime_limits(::NLolib, args...; kwargs...) = query_runtime_limits(args...; kwargs...)
query_runtime_limits(config::PreparedSimConfig; exec_options = nothing) =
    query_runtime_limits(config.simulation_config, config.physics_config; exec_options = exec_options)
query_runtime_limits(::NLolib, config::PreparedSimConfig; exec_options = nothing) =
    query_runtime_limits(config; exec_options = exec_options)

function set_log_file(::NLolib, path::Union{Nothing, AbstractString}, append::Bool = false)
    status = ccall(_sym(:nlolib_set_log_file), Cint, (Cstring, Cint), path === nothing ? C_NULL : path, Cint(append))
    _check_status(status, "nlolib_set_log_file")
end

function set_log_buffer(::NLolib, capacity_bytes::Integer = 256 * 1024)
    status = ccall(_sym(:nlolib_set_log_buffer), Cint, (Csize_t,), Csize_t(capacity_bytes))
    _check_status(status, "nlolib_set_log_buffer")
end

function clear_log_buffer(::NLolib)
    status = ccall(_sym(:nlolib_clear_log_buffer), Cint, ())
    _check_status(status, "nlolib_clear_log_buffer")
end

function read_log_buffer(::NLolib; consume::Bool = true, max_bytes::Integer = 256 * 1024)
    max_bytes >= 2 || throw(ArgumentError("max_bytes must be >= 2"))
    buffer = Vector{UInt8}(undef, Int(max_bytes))
    written = Ref{Csize_t}(0)
    status = GC.@preserve buffer written begin
        ccall(
            _sym(:nlolib_read_log_buffer),
            Cint,
            (Ptr{UInt8}, Csize_t, Ptr{Csize_t}, Cint),
            pointer(buffer),
            Csize_t(max_bytes),
            Base.unsafe_convert(Ptr{Csize_t}, written),
            Cint(consume),
        )
    end
    _check_status(status, "nlolib_read_log_buffer")
    return String(buffer[1:Int(written[])])
end

function set_log_level(::NLolib, level::Integer = NLOLIB_LOG_LEVEL_INFO)
    status = ccall(_sym(:nlolib_set_log_level), Cint, (Cint,), Cint(level))
    _check_status(status, "nlolib_set_log_level")
end

function set_progress_options(::NLolib; enabled::Bool = true, milestone_percent::Integer = 5, emit_on_step_adjust::Bool = false)
    status = ccall(
        _sym(:nlolib_set_progress_options),
        Cint,
        (Cint, Cint, Cint),
        Cint(enabled),
        Cint(milestone_percent),
        Cint(emit_on_step_adjust),
    )
    _check_status(status, "nlolib_set_progress_options")
end

function set_progress_stream(::NLolib; stream_mode::Integer = NLOLIB_PROGRESS_STREAM_STDERR)
    status = ccall(_sym(:nlolib_set_progress_stream), Cint, (Cint,), Cint(stream_mode))
    _check_status(status, "nlolib_set_progress_stream")
end

set_log_file(path::Union{Nothing, AbstractString}, append::Bool = false) = set_log_file(_api(), path, append)
set_log_buffer(capacity_bytes::Integer = 256 * 1024) = set_log_buffer(_api(), capacity_bytes)
clear_log_buffer() = clear_log_buffer(_api())
read_log_buffer(; kwargs...) = read_log_buffer(_api(); kwargs...)
set_log_level(level::Integer = NLOLIB_LOG_LEVEL_INFO) = set_log_level(_api(), level)
set_progress_options(; kwargs...) = set_progress_options(_api(); kwargs...)
set_progress_stream(; kwargs...) = set_progress_stream(_api(); kwargs...)

function propagate(api::NLolib, config::Union{PreparedSimConfig, SimulationConfig}, input_field;
                   num_recorded_samples,
                   kwargs...)
    request = from_config(PropagateRequestBuilder(), config, input_field;
        num_recorded_samples = num_recorded_samples,
        kwargs...)
    return execute(request)
end

function _event_property(event, name::Symbol, default)
    if hasproperty(event, name)
        return getproperty(event, name)
    elseif event isa AbstractDict
        return get(event, name, get(event, String(name), default))
    end
    return default
end

function _event_crossings(event, z_axis::Vector{Float64}, records::Matrix{ComplexF64})
    t_hits = Float64[]
    y_hits = Vector{Vector{ComplexF64}}()
    direction = Float64(_event_property(event, :direction, 0.0))
    terminal = Bool(_event_property(event, :terminal, false))
    for i in 2:length(z_axis)
        z0 = z_axis[i - 1]
        z1 = z_axis[i]
        y0 = view(records, :, i - 1)
        y1 = view(records, :, i)
        f0 = Float64(event(z0, y0))
        f1 = Float64(event(z1, y1))
        crossed = (f0 == 0.0) || (f1 == 0.0) || ((f0 < 0.0 < f1) || (f1 < 0.0 < f0))
        crossed || continue
        slope = f1 - f0
        direction > 0.0 && slope <= 0.0 && continue
        direction < 0.0 && slope >= 0.0 && continue
        alpha = slope == 0.0 ? 0.0 : clamp((0.0 - f0) / slope, 0.0, 1.0)
        hit = ComplexF64.(y0 .+ (y1 .- y0) .* alpha)
        push!(t_hits, z0 + (z1 - z0) * alpha)
        push!(y_hits, hit)
        terminal && break
    end
    return t_hits, y_hits, terminal
end

function _evaluate_events(events, z_axis::Vector{Float64}, records::Matrix{ComplexF64})
    events === nothing && return Vector{Vector{Float64}}(), Vector{Matrix{ComplexF64}}(), false
    length(z_axis) < 2 && return Vector{Vector{Float64}}(), Vector{Matrix{ComplexF64}}(), false
    event_list = events isa Tuple || events isa AbstractVector ? collect(events) : Any[events]
    t_events = Vector{Vector{Float64}}(undef, length(event_list))
    y_events = Vector{Matrix{ComplexF64}}(undef, length(event_list))
    terminal_hit = false
    for idx in eachindex(event_list)
        hits, states, terminal = _event_crossings(event_list[idx], z_axis, records)
        t_events[idx] = hits
        y_events[idx] = isempty(states) ? zeros(ComplexF64, size(records, 1), 0) : hcat(states...)
        if terminal && !isempty(hits)
            terminal_hit = true
            break
        end
    end
    return t_events, y_events, terminal_hit
end

function _build_dense_sol(z_axis::Vector{Float64}, records::Matrix{ComplexF64})
    function sol(z::Real)
        isempty(z_axis) && return ComplexF64[]
        zf = Float64(z)
        zf <= z_axis[1] && return copy(view(records, :, 1))
        zf >= z_axis[end] && return copy(view(records, :, size(records, 2)))
        for i in 2:length(z_axis)
            z0 = z_axis[i - 1]
            z1 = z_axis[i]
            if zf <= z1
                alpha = z1 > z0 ? (zf - z0) / (z1 - z0) : 0.0
                return ComplexF64.(view(records, :, i - 1) .+ (view(records, :, i) .- view(records, :, i - 1)) .* alpha)
            end
        end
        return copy(view(records, :, size(records, 2)))
    end
    return sol
end

function _truncate_to_terminal_event(z_axis::Vector{Float64}, records::Matrix{ComplexF64},
                                     t_events::Vector{Vector{Float64}}, y_events::Vector{Matrix{ComplexF64}})
    event_z = nothing
    event_state = nothing
    for idx in eachindex(t_events)
        isempty(t_events[idx]) && continue
        candidate = t_events[idx][1]
        if event_z === nothing || candidate < event_z
            event_z = candidate
            event_state = view(y_events[idx], :, 1)
        end
    end
    event_z === nothing && return z_axis, records
    keep = findall(z -> z <= event_z, z_axis)
    kept_z = z_axis[keep]
    kept_records = records[:, keep]
    if isempty(kept_z) || abs(kept_z[end] - event_z) > 1e-15
        kept_z = vcat(kept_z, Float64(event_z))
        kept_records = hcat(kept_records, ComplexF64.(event_state))
    end
    return kept_z, kept_records
end

function propagate(api::NLolib, pulse, linear_operator = "gvd", nonlinear_operator = "kerr"; kwargs...)
    request = from_pulse(PropagateRequestBuilder(), pulse, linear_operator, nonlinear_operator; kwargs...)
    dense_output = Bool(get(Dict{Symbol, Any}(pairs(kwargs)), :dense_output, false))
    events = get(Dict{Symbol, Any}(pairs(kwargs)), :events, nothing)
    low_level = execute(request)
    t_events, y_events, terminal_hit = _evaluate_events(events, low_level.z_axis, low_level.records)
    records = low_level.records
    z_axis = low_level.z_axis
    final = low_level.final
    status = Int(get(low_level.meta, "status", NLOLIB_STATUS_OK))
    message = String(get(low_level.meta, "message", "propagate completed"))
    if terminal_hit
        z_axis, records = _truncate_to_terminal_event(z_axis, records, t_events, y_events)
        final = view(records, :, size(records, 2))
        status = 1
        message = "A termination event occurred."
    end
    sol = dense_output ? _build_dense_sol(z_axis, records) : nothing
    return PropagateResult(records, z_axis, final, low_level.meta, status, message, t_events, y_events, sol)
end

propagate(config::Union{PreparedSimConfig, SimulationConfig}, input_field; kwargs...) =
    propagate(_api(), config, input_field; kwargs...)

propagate(pulse, linear_operator = "gvd", nonlinear_operator = "kerr"; kwargs...) =
    propagate(_api(), pulse, linear_operator, nonlinear_operator; kwargs...)
