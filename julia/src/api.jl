_raw(x) = x isa PreparedValue ? x.value : x
_keepalive(x) = x isa PreparedValue ? x.keepalive : ()

_string_ptr(value::Union{Nothing, AbstractString}) =
    value === nothing ? C_NULL : Base.unsafe_convert(Cstring, value)

function _constants_tuple(values)
    constants = zeros(Float64, NLO_RUNTIME_OPERATOR_CONSTANTS_MAX)
    count = min(length(values), NLO_RUNTIME_OPERATOR_CONSTANTS_MAX)
    for i in 1:count
        constants[i] = Float64(values[i])
    end
    return Tuple(constants)
end

function _complex_ptr_or_null(values)
    values === nothing && return C_NULL
    return _complex_ptr(values)
end

_complex_storage(values::StridedArray{ComplexF64}) = reinterpret(NLOComplex, values)
_complex_storage(values::StridedArray{NLOComplex}) = values

function _complex_ptr(values)
    return Base.unsafe_convert(Ptr{NLOComplex}, _complex_storage(values))
end

function _record_capacity(output::AbstractVector)
    return 1
end

function _record_capacity(output::AbstractMatrix)
    return size(output, 2)
end

function _num_time_samples(output::AbstractVector)
    return length(output)
end

function _num_time_samples(output::AbstractMatrix)
    return size(output, 1)
end

function _decode_run_id(chars::NTuple{NLO_STORAGE_RUN_ID_MAX, Cchar})
    bytes = UInt8[UInt8(Int(c) & 0xff) for c in chars]
    stop = findfirst(==(0x00), bytes)
    data = stop === nothing ? bytes : bytes[1:stop - 1]
    return String(data)
end

function physics_config(;
    linear_factor_expr::Union{Nothing, AbstractString} = nothing,
    linear_expr::Union{Nothing, AbstractString} = nothing,
    potential_expr::Union{Nothing, AbstractString} = nothing,
    dispersion_factor_expr::Union{Nothing, AbstractString} = nothing,
    dispersion_expr::Union{Nothing, AbstractString} = nothing,
    nonlinear_expr::Union{Nothing, AbstractString} = nothing,
    nonlinear_model::Integer = NLO_NONLINEAR_MODEL_EXPR,
    nonlinear_gamma::Real = 0.0,
    raman_fraction::Real = 0.0,
    raman_tau1::Real = 0.0,
    raman_tau2::Real = 0.0,
    shock_omega0::Real = 0.0,
    raman_response_time = nothing,
    constants = ()
)
    keepalive = (
        linear_factor_expr,
        linear_expr,
        potential_expr,
        dispersion_factor_expr,
        dispersion_expr,
        nonlinear_expr,
        raman_response_time
    )
    value = PhysicsConfig(
        _string_ptr(linear_factor_expr),
        _string_ptr(linear_expr),
        _string_ptr(potential_expr),
        _string_ptr(dispersion_factor_expr),
        _string_ptr(dispersion_expr),
        _string_ptr(nonlinear_expr),
        Cint(nonlinear_model),
        Cdouble(nonlinear_gamma),
        Cdouble(raman_fraction),
        Cdouble(raman_tau1),
        Cdouble(raman_tau2),
        Cdouble(shock_omega0),
        _complex_ptr_or_null(raman_response_time),
        Csize_t(raman_response_time === nothing ? 0 : length(raman_response_time)),
        Csize_t(min(length(constants), NLO_RUNTIME_OPERATOR_CONSTANTS_MAX)),
        _constants_tuple(constants)
    )
    return PreparedValue(value, keepalive)
end

function storage_options(;
    sqlite_path::AbstractString,
    run_id::Union{Nothing, AbstractString} = nothing,
    sqlite_max_bytes::Integer = 0,
    chunk_records::Integer = 0,
    cap_policy::Integer = NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES,
    log_final_output_field_to_db::Bool = false
)
    keepalive = (sqlite_path, run_id)
    value = StorageOptions(
        _string_ptr(sqlite_path),
        _string_ptr(run_id),
        Csize_t(sqlite_max_bytes),
        Csize_t(chunk_records),
        Cint(cap_policy),
        Cint(log_final_output_field_to_db)
    )
    return PreparedValue(value, keepalive)
end

function storage_is_available()
    status = ccall(_sym(:nlolib_storage_is_available), Cint, ())
    return status != 0
end

function _status_name(status::Integer)
    status == NLOLIB_STATUS_OK && return "NLOLIB_STATUS_OK"
    status == NLOLIB_STATUS_INVALID_ARGUMENT && return "NLOLIB_STATUS_INVALID_ARGUMENT"
    status == NLOLIB_STATUS_ALLOCATION_FAILED && return "NLOLIB_STATUS_ALLOCATION_FAILED"
    status == NLOLIB_STATUS_NOT_IMPLEMENTED && return "NLOLIB_STATUS_NOT_IMPLEMENTED"
    status == NLOLIB_STATUS_ABORTED && return "NLOLIB_STATUS_ABORTED"
    return "NLOLIB_STATUS_UNKNOWN"
end

function _check_status(status::Integer, context::AbstractString)
    status == NLOLIB_STATUS_OK && return
    error("$(context) failed with status=$(Int(status)) ($( _status_name(status) ))")
end

function query_runtime_limits(simulation_config = nothing,
                              physics_config_value = nothing;
                              exec_options = nothing)
    sim_raw = simulation_config === nothing ? nothing : _raw(simulation_config)
    phys_raw = physics_config_value === nothing ? nothing : _raw(physics_config_value)
    exec_raw = exec_options === nothing ? nothing : _raw(exec_options)

    limits_ref = Ref(RuntimeLimits())
    sim_ref = sim_raw === nothing ? nothing : Ref(sim_raw)
    phys_ref = phys_raw === nothing ? nothing : Ref(phys_raw)
    exec_ref = exec_raw === nothing ? nothing : Ref(exec_raw)
    sim_ptr = sim_ref === nothing ? Ptr{SimulationConfig}(C_NULL) : Base.unsafe_convert(Ptr{SimulationConfig}, sim_ref)
    phys_ptr = phys_ref === nothing ? Ptr{PhysicsConfig}(C_NULL) : Base.unsafe_convert(Ptr{PhysicsConfig}, phys_ref)
    exec_ptr = exec_ref === nothing ? Ptr{ExecutionOptions}(C_NULL) : Base.unsafe_convert(Ptr{ExecutionOptions}, exec_ref)
    keepalive = (simulation_config, physics_config_value, exec_options)

    status = GC.@preserve keepalive limits_ref sim_ref phys_ref exec_ref begin
        ccall(
            _sym(:nlolib_query_runtime_limits),
            Cint,
            (Ptr{SimulationConfig}, Ptr{PhysicsConfig}, Ptr{ExecutionOptions}, Ref{RuntimeLimits}),
            sim_ptr,
            phys_ptr,
            exec_ptr,
            limits_ref
        )
    end
    _check_status(status, "nlolib_query_runtime_limits")
    return limits_ref[]
end

function wrap_records(ptr::Ptr{NLOComplex}, num_time_samples::Integer, num_records::Integer; own::Bool = false)
    raw = unsafe_wrap(Array, ptr, (Int(num_time_samples), Int(num_records)); own = own)
    return reinterpret(ComplexF64, raw)
end

final_record(records::AbstractMatrix) = view(records, :, size(records, 2))

function tensor_record_view(record::AbstractVector, nt::Integer, nx::Integer, ny::Integer)
    length(record) == nt * nx * ny || error("record length does not match nt*nx*ny")
    return reshape(record, Int(nt), Int(ny), Int(nx))
end

function tensor_record_view(records::AbstractMatrix, index::Integer, nt::Integer, nx::Integer, ny::Integer)
    return tensor_record_view(view(records, :, index), nt, nx, ny)
end

function propagate!(output,
                    simulation_config,
                    physics_config_value,
                    input_field;
                    num_recorded_samples = nothing,
                    output_mode = nothing,
                    return_records = nothing,
                    exec_options = nothing,
                    storage_options_value = nothing,
                    explicit_record_z = nothing,
                    step_events = nothing)
    sim_raw = _raw(simulation_config)
    phys_raw = _raw(physics_config_value)
    exec_raw = exec_options === nothing ? nothing : _raw(exec_options)
    storage_raw = storage_options_value === nothing ? nothing : _raw(storage_options_value)

    n = length(input_field)
    record_capacity = output === nothing ? 0 : _record_capacity(output)
    num_records = num_recorded_samples === nothing ? max(record_capacity, 1) : Int(num_recorded_samples)
    records_enabled = return_records === nothing ? output !== nothing : Bool(return_records)

    records_enabled || output === nothing || error("output buffer provided while return_records=false")
    if records_enabled && output === nothing && storage_raw === nothing
        error("return_records=true requires an output buffer or storage options")
    end
    if output !== nothing && _num_time_samples(output) != n
        error("output buffer first dimension must match input field length")
    end
    if output !== nothing && record_capacity < num_records
        error("output buffer capacity is smaller than num_recorded_samples")
    end

    mode = output_mode === nothing ? (num_records == 1 ? NLO_PROPAGATE_OUTPUT_FINAL_ONLY : NLO_PROPAGATE_OUTPUT_DENSE) : Cint(output_mode)
    sim_ref = Ref(sim_raw)
    phys_ref = Ref(phys_raw)
    exec_ref = exec_raw === nothing ? nothing : Ref(exec_raw)
    storage_ref = storage_raw === nothing ? nothing : Ref(storage_raw)
    sim_ptr = Base.unsafe_convert(Ptr{SimulationConfig}, sim_ref)
    phys_ptr = Base.unsafe_convert(Ptr{PhysicsConfig}, phys_ref)
    exec_ptr = exec_ref === nothing ? Ptr{ExecutionOptions}(C_NULL) : Base.unsafe_convert(Ptr{ExecutionOptions}, exec_ref)
    storage_ptr = storage_ref === nothing ? Ptr{StorageOptions}(C_NULL) : Base.unsafe_convert(Ptr{StorageOptions}, storage_ref)

    records_written_ref = Ref{Csize_t}(0)
    step_events_written_ref = Ref{Csize_t}(0)
    step_events_dropped_ref = Ref{Csize_t}(0)
    storage_result_ref = storage_raw === nothing ? nothing : Ref(StorageResult())

    output_ptr = output === nothing ? C_NULL : _complex_ptr(output)
    input_storage = _complex_storage(input_field)
    output_storage = output === nothing ? nothing : _complex_storage(output)
    input_ptr = Base.unsafe_convert(Ptr{NLOComplex}, input_storage)
    explicit_z_ptr = explicit_record_z === nothing ? C_NULL : pointer(explicit_record_z)
    step_events_ptr = step_events === nothing ? C_NULL : pointer(step_events)

    options = PropagateOptions(
        Csize_t(num_records),
        mode,
        Cint(records_enabled),
        exec_ptr,
        storage_ptr,
        explicit_z_ptr,
        Csize_t(explicit_record_z === nothing ? 0 : length(explicit_record_z)),
        C_NULL,
        C_NULL
    )
    options_ref = Ref(options)
    output_info = PropagateOutput(
        output_ptr,
        Csize_t(record_capacity),
        Base.unsafe_convert(Ptr{Csize_t}, records_written_ref),
        storage_result_ref === nothing ? C_NULL : Base.unsafe_convert(Ptr{StorageResult}, storage_result_ref),
        step_events_ptr,
        Csize_t(step_events === nothing ? 0 : length(step_events)),
        Base.unsafe_convert(Ptr{Csize_t}, step_events_written_ref),
        Base.unsafe_convert(Ptr{Csize_t}, step_events_dropped_ref)
    )
    output_info_ref = Ref(output_info)

    keepalive = (
        input_field,
        input_storage,
        output,
        output_storage,
        explicit_record_z,
        step_events,
        _keepalive(simulation_config),
        _keepalive(physics_config_value),
        _keepalive(exec_options),
        _keepalive(storage_options_value)
    )

    status = GC.@preserve keepalive sim_ref phys_ref exec_ref storage_ref records_written_ref step_events_written_ref step_events_dropped_ref storage_result_ref options_ref output_info_ref begin
        ccall(
            _sym(:nlolib_propagate),
            Cint,
            (Ptr{SimulationConfig}, Ptr{PhysicsConfig}, Csize_t, Ptr{NLOComplex}, Ref{PropagateOptions}, Ref{PropagateOutput}),
            sim_ptr,
            phys_ptr,
            Csize_t(n),
            input_ptr,
            options_ref,
            output_info_ref
        )
    end
    _check_status(status, "nlolib_propagate")

    return (
        records_written = Int(records_written_ref[]),
        storage_result = storage_result_ref === nothing ? nothing : storage_result_ref[],
        storage_run_id = storage_result_ref === nothing ? nothing : _decode_run_id(storage_result_ref[].run_id),
        step_events_written = Int(step_events_written_ref[]),
        step_events_dropped = Int(step_events_dropped_ref[])
    )
end

function propagate(simulation_config,
                   physics_config_value,
                   input_field;
                   num_recorded_samples::Integer = 2,
                   output_mode = nothing,
                   exec_options = nothing,
                   storage_options_value = nothing,
                   explicit_record_z = nothing,
                   capture_step_history::Bool = false,
                   step_history_capacity::Integer = 0)
    n = length(input_field)
    raw_output = Matrix{NLOComplex}(undef, n, Int(num_recorded_samples))
    step_events = capture_step_history ? Vector{StepEvent}(undef, Int(step_history_capacity)) : nothing

    meta = propagate!(
        raw_output,
        simulation_config,
        physics_config_value,
        input_field;
        num_recorded_samples = num_recorded_samples,
        output_mode = output_mode,
        return_records = true,
        exec_options = exec_options,
        storage_options_value = storage_options_value,
        explicit_record_z = explicit_record_z,
        step_events = step_events
    )

    records = reinterpret(ComplexF64, raw_output)
    written = meta.records_written
    if written < size(records, 2)
        records = @view records[:, 1:written]
    end

    return (
        records = records,
        final = written > 0 ? view(records, :, written) : view(records, :, 1:0),
        meta = meta,
        step_events = step_events === nothing ? nothing : @view step_events[1:meta.step_events_written]
    )
end
