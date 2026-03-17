struct NormalizedPropagateRequest
    sim_cfg::SimulationConfig
    phys_cfg::PhysicsConfig
    input_seq::Vector{ComplexF64}
    keepalive::Vector{Any}
    num_records::Int
    exec_options::Union{Nothing, ExecutionOptions}
    sqlite_path::Union{Nothing, String}
    run_id::Union{Nothing, String}
    sqlite_max_bytes::Int
    chunk_records::Int
    cap_policy::Int
    log_final_output_field_to_db::Bool
    return_records::Bool
    capture_step_history::Bool
    step_history_capacity::Int
    output_label::String
    explicit_record_z::Union{Nothing, Vector{Float64}}
    progress_callback::Any
    meta_overrides::Dict{String, Any}
end

struct PropagateRequestBuilder
end

function _normalize_solver_aliases(kwargs::Base.Iterators.Pairs)
    out = Dict{Symbol, Any}(pairs(kwargs))
    if haskey(out, :t_span)
        t_span = out[:t_span]
        length(t_span) == 2 || throw(ArgumentError("t_span must have length 2"))
        Float64(t_span[1]) == 0.0 || throw(ArgumentError("t_span start must be 0.0"))
        out[:propagation_distance] = Float64(t_span[2]) - Float64(t_span[1])
        delete!(out, :t_span)
    end
    haskey(out, :first_step) && !haskey(out, :starting_step_size) && (out[:starting_step_size] = Float64(pop!(out, :first_step)))
    haskey(out, :max_step) && !haskey(out, :max_step_size) && (out[:max_step_size] = Float64(pop!(out, :max_step)))
    haskey(out, :min_step) && !haskey(out, :min_step_size) && (out[:min_step_size] = Float64(pop!(out, :min_step)))
    haskey(out, :rtol) && !haskey(out, :error_tolerance) && (out[:error_tolerance] = Float64(pop!(out, :rtol)))
    haskey(out, :output_mode) && !haskey(out, :output) && (out[:output] = pop!(out, :output_mode))
    return out
end

function from_config(::PropagateRequestBuilder, config, input_field;
                     num_recorded_samples,
                     physics_config = nothing,
                     exec_options = nothing,
                     sqlite_path = nothing,
                     run_id = nothing,
                     sqlite_max_bytes::Integer = 0,
                     chunk_records::Integer = 0,
                     cap_policy::Integer = NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES,
                     log_final_output_field_to_db::Bool = false,
                     return_records::Bool = true,
                     capture_step_history::Bool = false,
                     step_history_capacity::Integer = (capture_step_history ? 200000 : 0),
                     progress_callback = nothing,
                     t_eval = nothing)
    num_records = Int(num_recorded_samples)
    num_records > 0 || throw(ArgumentError("num_recorded_samples must be > 0"))
    input_seq = ComplexF64.(collect(input_field))
    isempty(input_seq) && throw(ArgumentError("input_field must be non-empty"))
    keepalive = Any[]
    sim_cfg = config isa PreparedSimConfig ? config.simulation_config : config
    phys_cfg = config isa PreparedSimConfig ? config.physics_config : (physics_config === nothing ? PhysicsConfig() : physics_config)
    config isa PreparedSimConfig && append!(keepalive, config.keepalive)
    physics_config !== nothing && (phys_cfg = physics_config)
    explicit_record_z = t_eval === nothing ? nothing : _validate_explicit_record_z(t_eval, sim_cfg.propagation.propagation_distance)
    explicit_record_z !== nothing && (num_records = length(explicit_record_z))
    return NormalizedPropagateRequest(
        sim_cfg,
        phys_cfg,
        input_seq,
        keepalive,
        num_records,
        exec_options,
        isnothing(sqlite_path) ? nothing : String(sqlite_path),
        isnothing(run_id) ? nothing : String(run_id),
        Int(sqlite_max_bytes),
        Int(chunk_records),
        Int(cap_policy),
        Bool(log_final_output_field_to_db),
        Bool(return_records),
        Bool(capture_step_history),
        Int(step_history_capacity),
        num_records == 1 ? "final" : "dense",
        explicit_record_z,
        progress_callback,
        Dict{String, Any}(),
    )
end

function from_pulse(::PropagateRequestBuilder, pulse, linear_operator = "gvd", nonlinear_operator = "kerr"; kwargs...)
    options = _normalize_solver_aliases(kwargs)
    haskey(options, :propagation_distance) || throw(ArgumentError("high-level propagate requires propagation_distance"))
    pulse_spec = _normalize_pulse_spec(pulse)
    if haskey(options, :pulse_period)
        pulse_spec.pulse_period = Float64(pop!(options, :pulse_period))
    end
    haskey(options, :frequency_grid) && (pulse_spec.frequency_grid = _complex_vector_or_nothing(pop!(options, :frequency_grid)))
    haskey(options, :tensor_nt) && (pulse_spec.tensor_nt = Int(pop!(options, :tensor_nt)))
    haskey(options, :tensor_nx) && (pulse_spec.tensor_nx = Int(pop!(options, :tensor_nx)))
    haskey(options, :tensor_ny) && (pulse_spec.tensor_ny = Int(pop!(options, :tensor_ny)))
    haskey(options, :tensor_layout) && (pulse_spec.tensor_layout = Int(pop!(options, :tensor_layout)))
    haskey(options, :delta_x) && (pulse_spec.delta_x = Float64(pop!(options, :delta_x)))
    haskey(options, :delta_y) && (pulse_spec.delta_y = Float64(pop!(options, :delta_y)))
    haskey(options, :spatial_frequency_grid) && (pulse_spec.spatial_frequency_grid = _complex_vector_or_nothing(pop!(options, :spatial_frequency_grid)))
    haskey(options, :potential_grid) && (pulse_spec.potential_grid = _complex_vector_or_nothing(pop!(options, :potential_grid)))

    propagation_distance = Float64(pop!(options, :propagation_distance))
    preset = String(get(options, :preset, "balanced"))
    profile = _solver_profile_defaults(preset, propagation_distance)
    start_step = Float64(get(options, :starting_step_size, profile["starting_step_size"]))
    max_step = Float64(get(options, :max_step_size, profile["max_step_size"]))
    min_step = Float64(get(options, :min_step_size, profile["min_step_size"]))
    tol = Float64(get(options, :error_tolerance, profile["error_tolerance"]))
    output = String(get(options, :output, "dense"))
    num_records = output == "final" ? 1 : Int(get(options, :records, profile["records"]))
    output in ("dense", "final") || throw(ArgumentError("output must be 'dense' or 'final'"))
    num_records > 0 || throw(ArgumentError("records must be > 0"))

    tensor_mode = !isnothing(pulse_spec.tensor_nt) && !isnothing(pulse_spec.tensor_nx) && !isnothing(pulse_spec.tensor_ny)
    tensor_mode && _validate_coupled_pulse_spec(pulse_spec)
    temporal_samples = isnothing(pulse_spec.tensor_nt) ? length(pulse_spec.samples) : Int(pulse_spec.tensor_nt)
    pulse_period = isnothing(pulse_spec.pulse_period) ? pulse_spec.delta_time * temporal_samples : Float64(pulse_spec.pulse_period)
    frequency_grid = isnothing(pulse_spec.frequency_grid) ? _default_frequency_grid(temporal_samples, pulse_spec.delta_time) : pulse_spec.frequency_grid

    linear_expr, linear_constants = _resolve_operator_spec("linear", linear_operator, 0)
    nonlinear_expr, nonlinear_constants = _resolve_operator_spec("nonlinear", nonlinear_operator, length(linear_constants))
    runtime = RuntimeOperators(
        linear_factor_expr = linear_expr,
        nonlinear_expr = nonlinear_expr,
        nonlinear_model = Int(get(options, :nonlinear_model, NLO_NONLINEAR_MODEL_EXPR)),
        nonlinear_gamma = Float64(get(options, :nonlinear_gamma, 0.0)),
        raman_fraction = Float64(get(options, :raman_fraction, 0.0)),
        raman_tau1 = Float64(get(options, :raman_tau1, 0.0122)),
        raman_tau2 = Float64(get(options, :raman_tau2, 0.0320)),
        shock_omega0 = Float64(get(options, :shock_omega0, 0.0)),
        raman_response_time = get(options, :raman_response_time, nothing),
        constants = vcat(linear_constants, nonlinear_constants),
        constant_bindings = nothing,
        auto_capture_constants = false,
    )

    prepared = prepare_sim_config(length(pulse_spec.samples);
        propagation_distance = propagation_distance,
        starting_step_size = start_step,
        max_step_size = max_step,
        min_step_size = min_step,
        error_tolerance = tol,
        pulse_period = pulse_period,
        delta_time = pulse_spec.delta_time,
        tensor_nt = pulse_spec.tensor_nt,
        tensor_nx = pulse_spec.tensor_nx,
        tensor_ny = pulse_spec.tensor_ny,
        tensor_layout = pulse_spec.tensor_layout,
        frequency_grid = frequency_grid,
        delta_x = pulse_spec.delta_x,
        delta_y = pulse_spec.delta_y,
        spatial_frequency_grid = pulse_spec.spatial_frequency_grid,
        potential_grid = pulse_spec.potential_grid,
        runtime = runtime
    )

    explicit_record_z = haskey(options, :t_eval) ? _validate_explicit_record_z(options[:t_eval], propagation_distance) : nothing
    explicit_record_z !== nothing && (num_records = length(explicit_record_z))
    capture_step_history = Bool(get(options, :capture_step_history, false))
    step_history_capacity = Int(get(options, :step_history_capacity, capture_step_history ? 200000 : 0))

    return NormalizedPropagateRequest(
        prepared.simulation_config,
        prepared.physics_config,
        pulse_spec.samples,
        copy(prepared.keepalive),
        num_records,
        get(options, :exec_options, nothing),
        isnothing(get(options, :sqlite_path, nothing)) ? nothing : String(options[:sqlite_path]),
        isnothing(get(options, :run_id, nothing)) ? nothing : String(options[:run_id]),
        Int(get(options, :sqlite_max_bytes, 0)),
        Int(get(options, :chunk_records, 0)),
        Int(get(options, :cap_policy, NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES)),
        Bool(get(options, :log_final_output_field_to_db, false)),
        Bool(get(options, :return_records, true)),
        capture_step_history,
        step_history_capacity,
        output,
        explicit_record_z,
        get(options, :progress_callback, nothing),
        Dict(
            "preset" => preset,
            "output" => output,
            "coupled" => Bool(tensor_mode && ((Int(pulse_spec.tensor_nx) > 1) || (Int(pulse_spec.tensor_ny) > 1))),
        ),
    )
end
