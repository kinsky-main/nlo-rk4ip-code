using NLOLibExamples
pushfirst!(LOAD_PATH, nlo_package_root_from(@__FILE__))

using CairoMakie
using FFTW
using NLOLib

function sech(x)
    return 1.0 ./ cosh.(x)
end

to_dimensionless_time(T, t0) = Float64.(T) ./ Float64(t0)
to_normalized_envelope(A, z, p0, alpha) = ComplexF64.(A) .* exp(0.5 * alpha * z) ./ sqrt(p0)
to_physical_envelope(U, z, p0, alpha) = ComplexF64.(U) .* (sqrt(p0) * exp(-0.5 * alpha * z))

function filter_record_clipped_steps(telemetry, z_axis)
    telemetry === nothing && return telemetry
    haskey(telemetry, "z") || return telemetry
    haskey(telemetry, "step_size") || return telemetry
    haskey(telemetry, "next_step_size") || return telemetry

    z_samples = Float64.(z_axis)
    length(z_samples) > 1 || return telemetry
    spacing = (z_samples[end] - z_samples[1]) / Float64(length(z_samples) - 1)
    (!isfinite(spacing) || spacing <= 0.0) && return telemetry

    z = Float64.(telemetry["z"])
    accepted = Float64.(telemetry["step_size"])
    proposed = Float64.(telemetry["next_step_size"])
    n = min(length(z), length(accepted), length(proposed))
    n > 0 || return telemetry

    z = z[1:n]
    accepted = accepted[1:n]
    proposed = proposed[1:n]
    expected_boundaries = z_samples[2:end-1]
    isempty(expected_boundaries) && return telemetry

    proximity_eps = max(64.0 * eps(max(1.0, abs(z_samples[end]))), spacing * 0.25)
    distance_to_expected = [minimum(abs.(value .- expected_boundaries)) for value in z]
    clipped = distance_to_expected .<= proximity_eps
    any(clipped) || return telemetry

    keep = .!clipped
    kept = count(keep)
    total = length(keep)
    if kept <= 0
        return telemetry
    end

    min_keep = max(16, ceil(Int, 0.35 * total))
    if kept < min_keep
        return telemetry
    end

    z_all_span = total > 1 ? (z[end] - z[1]) : 0.0
    z_filtered = z[keep]
    z_filtered_span = kept > 1 ? (z_filtered[end] - z_filtered[1]) : 0.0
    if z_all_span > 0.0 && z_filtered_span < (0.80 * z_all_span)
        return telemetry
    end

    return Dict(
        "z" => z_filtered,
        "step_size" => accepted[keep],
        "next_step_size" => proposed[keep],
        "dropped" => get(telemetry, "dropped", 0),
    )
end

function save_soliton_plots(output_dir, t, z_axis, z_axis_norm, numerical_records, reference_records, telemetry)
    mkpath(output_dir)

    fig1 = styled_figure()
    ax1 = Axis(fig1[1, 1], xlabel = "t / T0", ylabel = "z / Z0", title = "Numerical intensity")
    hm1 = heatmap!(
        ax1,
        t,
        z_axis_norm,
        permutedims(normalized_plot_data(abs2.(numerical_records)), (2, 1)),
        colormap = nlolib_hdr_colormap(),
        colorrange = (0.0, 1.0),
    )
    Colorbar(fig1[1, 2], hm1, label = "Normalized intensity")
    save_example_figure(joinpath(output_dir, "soliton_intensity_map.png"), fig1)

    fig2 = styled_figure()
    ax2 = Axis(fig2[1, 1], xlabel = "t / T0", ylabel = "Intensity", title = "Final intensity")
    lines!(ax2, t, abs2.(reference_records[end, :]), label = "Analytical")
    lines!(ax2, t, abs2.(numerical_records[end, :]), label = "Numerical")
    axislegend(ax2, position = :rt)
    save_example_figure(joinpath(output_dir, "soliton_final_intensity.png"), fig2)

    fig3 = styled_figure()
    ax3 = Axis(fig3[1, 1], xlabel = "z / Z0", ylabel = "Relative L2 intensity error", title = "Propagation error")
    lines!(ax3, z_axis_norm, relative_l2_intensity_error_curve(numerical_records, reference_records))
    save_example_figure(joinpath(output_dir, "soliton_error_curve.png"), fig3)

    telemetry_plot = filter_record_clipped_steps(telemetry, z_axis)
    if telemetry_plot !== nothing && !isempty(get(telemetry_plot, "z", Float64[]))
        fig4 = styled_figure()
        ax4 = Axis(fig4[1, 1], xlabel = "z (m)", ylabel = "Step size (m)", title = "Adaptive step history")
        lines!(ax4, telemetry_plot["z"], telemetry_plot["step_size"], label = "accepted")
        lines!(ax4, telemetry_plot["z"], telemetry_plot["next_step_size"], label = "next")
        axislegend(ax4, position = :rt)
        save_example_figure(joinpath(output_dir, "soliton_step_history.png"), fig4)
    end
end

function main(argv = ARGS)
    args = parse_example_args("second_order_soliton", "Second-order soliton validation with DB-backed run/replot.", argv)
    activate_example_theme!()
    NLOLib.set_progress_options(enabled = true, milestone_percent = 2, emit_on_step_adjust = true)
    NLOLib.set_log_level(NLOLIB_LOG_LEVEL_INFO)
    db = ExampleRunDB(args[:db_path])
    example_name = "second_order_soliton_rk4ip"
    case_key = "default"

    beta2 = -0.01
    gamma = 0.01
    alpha = 0.0
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * log(1.0 + sqrt(2.0)))
    p0 = abs(beta2) / (gamma * t0 * t0)
    z_final = second_order_soliton_period(beta2, t0)
    n = 2^12
    dt = (40.0 * t0) / n
    T = centered_time_grid(n, dt)
    t = to_dimensionless_time(T, t0)
    omega = 2.0 * π .* FFTW.fftfreq(n, 1.0 / dt)
    field0 = to_physical_envelope(2.0 .* sech(t), 0.0, p0, alpha)
    num_records = 160

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        loaded = load_case(db; example_name = example_name, run_group = run_group, case_key = case_key)
        records = loaded.records
        z_axis = loaded.z_axis
        telemetry = load_step_history(db; run_id = loaded.run_id)
    else
        run_group = begin_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        pulse = PulseSpec(
            samples = field0,
            delta_time = dt,
            pulse_period = n * dt,
            frequency_grid = ComplexF64.(omega),
        )
        linear = OperatorSpec(expr = "i*beta2*w*w-loss", params = Dict("beta2" => 0.5 * beta2, "loss" => 0.5 * alpha))
        nonlinear = OperatorSpec(expr = "i*gamma*A*I", params = Dict("gamma" => gamma))
        exec = default_execution_options(backend_type = NLO_VECTOR_BACKEND_CPU, fft_backend = NLO_FFT_BACKEND_FFTW)
        storage = storage_kwargs(db;
            example_name = example_name,
            run_group = run_group,
            case_key = case_key,
            chunk_records = 8)
        result = NLOLib.propagate(
            pulse,
            linear,
            nonlinear;
            t_span = (0.0, z_final),
            t_eval = collect(range(0.0, z_final, length = num_records)),
            first_step = 1e-4,
            max_step = 0.01,
            min_step = 1e-9,
            rtol = 1e-10,
            exec_options = exec,
            capture_step_history = true,
            step_history_capacity = 200000,
            sqlite_path = storage.sqlite_path,
            run_id = storage.run_id,
            chunk_records = storage.chunk_records,
            sqlite_max_bytes = storage.sqlite_max_bytes,
            log_final_output_field_to_db = storage.log_final_output_field_to_db,
        )
        records = permutedims(result.records, (2, 1))
        z_axis = result.z_axis
        telemetry = get(result.meta, "step_history", nothing)
        save_case_from_solver_meta!(db;
            example_name = example_name,
            run_group = run_group,
            case_key = case_key,
            solver_meta = result.meta,
            meta = Dict(
                "beta2" => beta2,
                "gamma" => gamma,
                "alpha" => alpha,
                "t0" => t0,
                "p0" => p0,
                "n" => n,
                "dt" => dt,
                "z_final" => z_final,
            ),
            save_step_history = true)
    end

    normalized_records = Array{ComplexF64}(undef, size(records))
    for idx in axes(records, 1)
        normalized_records[idx, :] = to_normalized_envelope(records[idx, :], z_axis[idx], p0, alpha)
    end
    reference_records = second_order_soliton_normalized_records(t, z_axis, beta2, t0)
    epsilon = relative_l2_intensity_error(normalized_records[end, :], reference_records[end, :])

    z_axis_norm = z_axis ./ z_final
    save_soliton_plots(args[:output_dir], t, z_axis, z_axis_norm, normalized_records, reference_records, telemetry)

    println("second-order soliton summary")
    println("  run_group = $(run_group)")
    println("  epsilon = $(epsilon)")
    println("  peak numerical intensity = $(maximum(abs2.(normalized_records)))")
    println("  peak analytical intensity = $(maximum(abs2.(reference_records)))")
    return epsilon
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
