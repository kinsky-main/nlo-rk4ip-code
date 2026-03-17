include(joinpath(@__DIR__, "backend", "common.jl"))
include(joinpath(@__DIR__, "backend", "metrics.jl"))
include(joinpath(@__DIR__, "backend", "reference.jl"))
include(joinpath(@__DIR__, "backend", "storage.jl"))

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

function save_soliton_plots(output_dir, t, z_axis_norm, numerical_records, reference_records, telemetry)
    mkpath(output_dir)

    fig1 = Figure(size = (900, 600))
    ax1 = Axis(fig1[1, 1], xlabel = "t / T0", ylabel = "z / Z0", title = "Numerical intensity")
    hm1 = heatmap!(ax1, t, z_axis_norm, abs2.(numerical_records))
    Colorbar(fig1[1, 2], hm1)
    save(joinpath(output_dir, "soliton_intensity_map.png"), fig1)

    fig2 = Figure(size = (900, 600))
    ax2 = Axis(fig2[1, 1], xlabel = "t / T0", ylabel = "Intensity", title = "Final intensity")
    lines!(ax2, t, abs2.(reference_records[end, :]), label = "Analytical")
    lines!(ax2, t, abs2.(numerical_records[end, :]), label = "Numerical")
    axislegend(ax2, position = :rt)
    save(joinpath(output_dir, "soliton_final_intensity.png"), fig2)

    fig3 = Figure(size = (900, 600))
    ax3 = Axis(fig3[1, 1], xlabel = "z / Z0", ylabel = "Relative L2 intensity error", title = "Propagation error")
    lines!(ax3, z_axis_norm, relative_l2_intensity_error_curve(numerical_records, reference_records))
    save(joinpath(output_dir, "soliton_error_curve.png"), fig3)

    if telemetry !== nothing && !isempty(get(telemetry, "z", Float64[]))
        fig4 = Figure(size = (900, 600))
        ax4 = Axis(fig4[1, 1], xlabel = "z (m)", ylabel = "Step size (m)", title = "Adaptive step history")
        lines!(ax4, telemetry["z"], telemetry["step_size"], label = "accepted")
        lines!(ax4, telemetry["z"], telemetry["next_step_size"], label = "next")
        axislegend(ax4, position = :rt)
        save(joinpath(output_dir, "soliton_step_history.png"), fig4)
    end
end

function main(argv = ARGS)
    args = parse_example_args("second_order_soliton", "Second-order soliton validation with DB-backed run/replot.", argv)
    NLOLib.set_progress_options(enabled = false)
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
    save_soliton_plots(args[:output_dir], t, z_axis_norm, normalized_records, reference_records, telemetry)

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
