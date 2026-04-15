const _PROGRESS_CALLBACK_REF = Ref{Any}(nothing)
const _PROGRESS_CALLBACK_ERROR = Ref{Any}(nothing)

function _progress_trampoline(info_ptr::Ptr{CProgressInfo}, ::Ptr{Cvoid})::Cint
    info_ptr == C_NULL && return Cint(1)
    callback = _PROGRESS_CALLBACK_REF[]
    callback === nothing && return Cint(1)
    info = unsafe_load(info_ptr)
    try
        result = callback(_progress_info_from_struct(info))
        return Cint(result === nothing ? 1 : Bool(result))
    catch err
        _PROGRESS_CALLBACK_ERROR[] = err
        return Cint(0)
    end
end

const _PROGRESS_CALLBACK_C = @cfunction(_progress_trampoline, Cint, (Ptr{CProgressInfo}, Ptr{Cvoid}))

function _build_z_axis(distance::Real, num_records::Integer)
    count = Int(num_records)
    count <= 0 && return Float64[]
    count == 1 && return [Float64(distance)]
    return collect(range(0.0, Float64(distance), length = count))
end

function _build_low_level_result(request::NormalizedPropagateRequest,
                                 records::AbstractMatrix{ComplexF64},
                                 records_written::Integer,
                                 storage_result,
                                 step_events,
                                 step_events_written::Integer,
                                 step_events_dropped::Integer,
                                 status::Integer)
    written = Int(records_written)
    if written < size(records, 2)
        records = records[:, 1:written]
    end
    z_axis = request.explicit_record_z === nothing ?
        _build_z_axis(request.sim_cfg.propagation.propagation_distance, written) :
        copy(request.explicit_record_z[1:written])
    meta = Dict{String, Any}(
        "output" => request.output_label,
        "records" => request.num_records,
        "records_requested" => request.num_records,
        "records_written" => written,
        "storage_enabled" => request.sqlite_path !== nothing,
        "records_returned" => written > 0,
        "backend_requested" => request.exec_options === nothing ? Int(VECTOR_BACKEND_AUTO) : Int(request.exec_options.backend_type),
        "coupled" => Bool(Int(request.sim_cfg.spatial.nx) > 1 || Int(request.sim_cfg.spatial.ny) > 1),
        "status" => Int(status),
        "message" => status == NLOLIB_STATUS_ABORTED ? "propagate aborted by progress callback" : "propagate completed",
    )
    storage_result !== nothing && (meta["storage_result"] = _storage_result_to_meta(storage_result))
    if step_events !== nothing
        step_history = _step_events_to_meta(step_events, step_events_written)
        step_history["dropped"] = Int(step_events_dropped)
        step_history["capacity"] = request.step_history_capacity
        meta["step_history"] = step_history
    end
    merge!(meta, request.meta_overrides)
    final = written > 0 ? view(records, :, written) : view(records, :, 1:0)
    return PropagationResult(records, z_axis, final, meta)
end

function execute(request::NormalizedPropagateRequest)
    request.sqlite_path !== nothing && !storage_is_available() &&
        throw(ArgumentError("SQLite storage is not available in this nlolib build"))

    sim_ref = Ref(request.sim_cfg)
    phys_ref = Ref(request.phys_cfg)
    exec_ref = request.exec_options === nothing ? nothing : Ref(request.exec_options)

    input_field = ComplexF64.(request.input_seq)
    input_storage = reinterpret(NLOComplex, input_field)
    input_ptr = Base.unsafe_convert(Ptr{NLOComplex}, input_storage)

    raw_output = request.return_records ? Matrix{NLOComplex}(undef, length(input_field), request.num_records) : nothing
    output_ptr = raw_output === nothing ? C_NULL : Base.unsafe_convert(Ptr{NLOComplex}, raw_output)
    output_records = raw_output === nothing ? zeros(ComplexF64, length(input_field), 0) : reinterpret(ComplexF64, raw_output)

    explicit_record_z = request.explicit_record_z
    storage_prepared = request.sqlite_path === nothing ? nothing : storage_options(
        sqlite_path = request.sqlite_path,
        run_id = request.run_id,
        sqlite_max_bytes = request.sqlite_max_bytes,
        chunk_records = request.chunk_records,
        cap_policy = request.cap_policy,
        log_final_output_field_to_db = request.log_final_output_field_to_db,
    )
    storage_ref = storage_prepared === nothing ? nothing : Ref(_raw(storage_prepared))
    step_events = request.capture_step_history && request.step_history_capacity > 0 ?
        Vector{StepEvent}(undef, request.step_history_capacity) : nothing

    records_written_ref = Ref{Csize_t}(0)
    step_events_written_ref = Ref{Csize_t}(0)
    step_events_dropped_ref = Ref{Csize_t}(0)
    storage_result_ref = Ref(StorageResult())

    options = PropagateOptions(
        Csize_t(request.num_records),
        request.num_records == 1 ? PROPAGATE_OUTPUT_FINAL_ONLY : PROPAGATE_OUTPUT_DENSE,
        Cint(request.return_records),
        exec_ref === nothing ? C_NULL : Base.unsafe_convert(Ptr{ExecutionOptions}, exec_ref),
        storage_ref === nothing ? C_NULL : Base.unsafe_convert(Ptr{StorageOptions}, storage_ref),
        explicit_record_z === nothing ? C_NULL : pointer(explicit_record_z),
        Csize_t(explicit_record_z === nothing ? 0 : length(explicit_record_z)),
        request.progress_callback === nothing ? C_NULL : _PROGRESS_CALLBACK_C,
        C_NULL,
    )
    output = PropagateOutput(
        output_ptr,
        Csize_t(raw_output === nothing ? 0 : size(raw_output, 2)),
        Base.unsafe_convert(Ptr{Csize_t}, records_written_ref),
        request.sqlite_path === nothing ? C_NULL : Base.unsafe_convert(Ptr{StorageResult}, storage_result_ref),
        step_events === nothing ? C_NULL : pointer(step_events),
        Csize_t(step_events === nothing ? 0 : length(step_events)),
        Base.unsafe_convert(Ptr{Csize_t}, step_events_written_ref),
        Base.unsafe_convert(Ptr{Csize_t}, step_events_dropped_ref),
    )
    options_ref = Ref(options)
    output_ref = Ref(output)

    _PROGRESS_CALLBACK_REF[] = request.progress_callback
    _PROGRESS_CALLBACK_ERROR[] = nothing
    keepalive = (
        request.keepalive,
        input_field,
        raw_output,
        explicit_record_z,
        step_events,
        _keepalive(storage_prepared),
    )

    status = GC.@preserve keepalive sim_ref phys_ref exec_ref storage_ref options_ref output_ref records_written_ref step_events_written_ref step_events_dropped_ref storage_result_ref begin
        ccall(
            _sym(:nlolib_propagate),
            Cint,
            (Ptr{SimulationConfig}, Ptr{PhysicsConfig}, Csize_t, Ptr{NLOComplex}, Ref{PropagateOptions}, Ref{PropagateOutput}),
            Base.unsafe_convert(Ptr{SimulationConfig}, sim_ref),
            Base.unsafe_convert(Ptr{PhysicsConfig}, phys_ref),
            Csize_t(length(input_field)),
            input_ptr,
            options_ref,
            output_ref
        )
    end
    callback_error = _PROGRESS_CALLBACK_ERROR[]
    _PROGRESS_CALLBACK_REF[] = nothing
    _PROGRESS_CALLBACK_ERROR[] = nothing

    status in (NLOLIB_STATUS_OK, NLOLIB_STATUS_ABORTED) ||
        _check_status(status, "nlolib_propagate")

    result = _build_low_level_result(
        request,
        output_records,
        records_written_ref[],
        request.sqlite_path === nothing ? nothing : storage_result_ref[],
        step_events,
        step_events_written_ref[],
        step_events_dropped_ref[],
        status,
    )
    callback_error !== nothing && throw(callback_error)
    status == NLOLIB_STATUS_ABORTED && throw(PropagationAbortedError(result))
    return result
end
