using NLOLibExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using CairoMakie
using FFTW
using LinearAlgebra
using NLOLib

function guided_spatial_mode(x_axis, y_axis; mode_width, chirp)
    field = Array{ComplexF64}(undef, length(y_axis), length(x_axis))
    width_sq = mode_width * mode_width
    for yi in eachindex(y_axis), xi in eachindex(x_axis)
        x = Float64(x_axis[xi])
        y = Float64(y_axis[yi])
        r2 = (x * x) + (y * y)
        amplitude = exp(-r2 / (2.0 * width_sq))
        field[yi, xi] = amplitude * exp(1.0im * chirp * r2)
    end
    return field
end

sech(x) = 1.0 ./ cosh.(Float64.(x))
dispersion_length(beta2, temporal_width) = (Float64(temporal_width)^2) / abs(Float64(beta2))
soliton_period(beta2, temporal_width) = 0.5 * π * dispersion_length(beta2, temporal_width)
fundamental_soliton_power(beta2, gamma, temporal_width) = abs(Float64(beta2)) / (Float64(gamma) * Float64(temporal_width)^2)
diffraction_length(beta_t, mode_width) = (Float64(mode_width)^2) / abs(Float64(beta_t))

function temporal_envelope(t_axis; temporal_width, beta2, gamma, soliton_order)
    p1 = fundamental_soliton_power(beta2, gamma, temporal_width)
    return ComplexF64.(Float64(soliton_order) * sqrt(p1) .* sech(Float64.(t_axis) ./ temporal_width))
end

function grin_launch_field(t_axis, x_axis, y_axis; temporal_width, beta2, gamma, soliton_order, mode_width, spatial_chirp)
    temporal = temporal_envelope(t_axis; temporal_width = temporal_width, beta2 = beta2, gamma = gamma, soliton_order = soliton_order)
    transverse = guided_spatial_mode(x_axis, y_axis; mode_width = mode_width, chirp = spatial_chirp)
    return reshape(temporal, length(temporal), 1, 1) .* reshape(transverse, 1, size(transverse, 1), size(transverse, 2))
end

function flatten_tfast(field_tyx)
    field = ComplexF64.(field_tyx)
    nt, ny, nx = size(field)
    out = Vector{ComplexF64}(undef, nt * ny * nx)
    idx = 1
    for x in 1:nx, y in 1:ny, t in 1:nt
        out[idx] = field[t, y, x]
        idx += 1
    end
    return out
end

function unflatten_tfast_records(records_flat, nt, ny, nx)
    flat = ComplexF64.(records_flat)
    out = Array{ComplexF64}(undef, size(flat, 1), nt, ny, nx)
    for ridx in axes(flat, 1)
        idx = 1
        for x in 1:nx, y in 1:ny, t in 1:nt
            out[ridx, t, y, x] = flat[ridx, idx]
            idx += 1
        end
    end
    return out
end

function grin_potential_grid(x_axis, y_axis, grin_strength)
    potential = Array{ComplexF64}(undef, length(y_axis), length(x_axis))
    for yi in eachindex(y_axis), xi in eachindex(x_axis)
        x = Float64(x_axis[xi])
        y = Float64(y_axis[yi])
        potential[yi, xi] = ComplexF64(grin_strength * ((x * x) + (y * y)))
    end
    return potential
end

function overlap_fidelity_curve(records_tyx, launch_tyx)
    reference = vec(ComplexF64.(launch_tyx))
    ref_norm = max(norm(reference), 1e-30)
    out = Vector{Float64}(undef, size(records_tyx, 1))
    for ridx in axes(records_tyx, 1)
        record = vec(ComplexF64.(records_tyx[ridx, :, :, :]))
        denom = max(ref_norm * norm(record), 1e-30)
        out[ridx] = abs(dot(reference, record)) / denom
    end
    return out
end

function rms_radius_curve(records_tyx, x_axis, y_axis)
    r2 = Array{Float64}(undef, length(y_axis), length(x_axis))
    for yi in eachindex(y_axis), xi in eachindex(x_axis)
        x = Float64(x_axis[xi])
        y = Float64(y_axis[yi])
        r2[yi, xi] = (x * x) + (y * y)
    end
    out = Vector{Float64}(undef, size(records_tyx, 1))
    for ridx in axes(records_tyx, 1)
        intensity = dropdims(sum(abs2.(ComplexF64.(records_tyx[ridx, :, :, :])), dims = 1), dims = 1)
        total = max(sum(intensity), 1e-30)
        out[ridx] = sqrt(sum(r2 .* intensity) / total)
    end
    return out
end

function rms_temporal_width_curve(records_tyx, t_axis)
    t = Float64.(t_axis)
    out = Vector{Float64}(undef, size(records_tyx, 1))
    for ridx in axes(records_tyx, 1)
        weights = vec(sum(abs2.(ComplexF64.(records_tyx[ridx, :, :, :])), dims = (2, 3)))
        total = max(sum(weights), 1e-30)
        center = sum(t .* weights) / total
        out[ridx] = sqrt(sum(((t .- center) .^ 2) .* weights) / total)
    end
    return out
end

total_power_curve(records_tyx) = [sum(abs2.(ComplexF64.(records_tyx[ridx, :, :, :]))) for ridx in axes(records_tyx, 1)]
relative_power_drift_curve(power_curve) = abs.(Float64.(power_curve) .- Float64(power_curve[1])) ./ max(Float64(power_curve[1]), 1e-30)
peak_transverse_intensity_curve(records_xy) = [maximum(Float64.(records_xy[ridx, :, :])) for ridx in axes(records_xy, 1)]

function centerline_intensity_map(records_tyx)
    center_row = Int(cld(size(records_tyx, 3), 2))
    out = Array{Float64}(undef, size(records_tyx, 1), size(records_tyx, 4))
    for ridx in axes(records_tyx, 1)
        out[ridx, :] .= vec(sum(abs2.(ComplexF64.(records_tyx[ridx, :, center_row, :])), dims = 1))
    end
    return out
end

function time_integrated_xy_records(records_tyx)
    out = Array{Float64}(undef, size(records_tyx, 1), size(records_tyx, 3), size(records_tyx, 4))
    for ridx in axes(records_tyx, 1)
        out[ridx, :, :] .= dropdims(sum(abs2.(ComplexF64.(records_tyx[ridx, :, :, :])), dims = 1), dims = 1)
    end
    return out
end

function temporal_marginal_curve(records_tyx)
    out = Array{Float64}(undef, size(records_tyx, 1), size(records_tyx, 2))
    for ridx in axes(records_tyx, 1)
        out[ridx, :] .= vec(sum(abs2.(ComplexF64.(records_tyx[ridx, :, :, :])), dims = (2, 3)))
    end
    return out
end

function save_grin_plots(output_dir, z_axis, t_axis, x_axis, y_axis, field0_tyx, nonlinear_records, linear_records, nonlinear_radius, linear_radius, nonlinear_temporal_width, linear_temporal_width, nonlinear_peak, linear_peak, nonlinear_overlap, linear_overlap, nonlinear_power_drift, linear_power_drift)
    mkpath(output_dir)

    nonlinear_xy = time_integrated_xy_records(nonlinear_records)
    linear_xy = time_integrated_xy_records(linear_records)
    nonlinear_centerline = centerline_intensity_map(nonlinear_records)
    linear_centerline = centerline_intensity_map(linear_records)
    nonlinear_temporal = temporal_marginal_curve(nonlinear_records)
    linear_temporal = temporal_marginal_curve(linear_records)
    launch_centerline = vec(sum(abs2.(ComplexF64.(field0_tyx[:, Int(cld(size(field0_tyx, 2), 2)), :])), dims = 1))
    launch_temporal = vec(sum(abs2.(ComplexF64.(field0_tyx)), dims = (2, 3)))

    fig1 = styled_figure()
    ax1 = Axis(fig1[1, 1], xlabel = "x / w0", ylabel = "z / L_D", title = "Linear baseline center-line intensity")
    hm1 = heatmap!(ax1, x_axis, z_axis, permutedims(normalized_plot_data(linear_centerline), (2, 1)); colorrange = (0.0, 1.0))
    Colorbar(fig1[1, 2], hm1, label = "Normalized center-line intensity")
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_linear_centerline_map.png"), fig1)

    fig2 = styled_figure()
    ax2 = Axis(fig2[1, 1], xlabel = "x / w0", ylabel = "z / L_D", title = "Nonlinear center-line intensity")
    hm2 = heatmap!(ax2, x_axis, z_axis, permutedims(normalized_plot_data(nonlinear_centerline), (2, 1)); colorrange = (0.0, 1.0))
    Colorbar(fig2[1, 2], hm2, label = "Normalized center-line intensity")
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_nonlinear_centerline_map.png"), fig2)

    fig3 = styled_figure()
    ax3 = Axis(fig3[1, 1], xlabel = "z / L_D", ylabel = "RMS transverse radius", title = "Transverse radius")
    lines!(ax3, z_axis, nonlinear_radius, label = "Nonlinear")
    lines!(ax3, z_axis, linear_radius, label = "Linear baseline")
    axislegend(ax3, position = :rt)
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_rms_radius.png"), fig3)

    fig4 = styled_figure()
    ax4 = Axis(fig4[1, 1], xlabel = "z / L_D", ylabel = "Peak transverse intensity", title = "Peak transverse intensity")
    lines!(ax4, z_axis, nonlinear_peak, label = "Nonlinear")
    lines!(ax4, z_axis, linear_peak, label = "Linear baseline")
    axislegend(ax4, position = :rt)
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_peak_intensity.png"), fig4)

    fig5 = styled_figure()
    ax5 = Axis(fig5[1, 1], xlabel = "z / L_D", ylabel = "RMS temporal width", title = "Temporal broadening")
    lines!(ax5, z_axis, nonlinear_temporal_width, label = "Nonlinear")
    lines!(ax5, z_axis, linear_temporal_width, label = "Linear baseline")
    axislegend(ax5, position = :rt)
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_rms_temporal_width.png"), fig5)

    fig6 = styled_figure()
    ax6 = Axis(fig6[1, 1], xlabel = "z / L_D", ylabel = "Overlap fidelity", title = "Launch-mode overlap")
    lines!(ax6, z_axis, nonlinear_overlap, label = "Nonlinear")
    lines!(ax6, z_axis, linear_overlap, label = "Linear baseline")
    axislegend(ax6, position = :rb)
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_overlap_fidelity.png"), fig6)

    fig7 = styled_figure()
    ax7 = Axis(fig7[1, 1], xlabel = "z / L_D", ylabel = "Relative power drift", title = "Power drift")
    lines!(ax7, z_axis, nonlinear_power_drift, label = "Nonlinear")
    lines!(ax7, z_axis, linear_power_drift, label = "Linear baseline")
    axislegend(ax7, position = :rt)
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_power_drift.png"), fig7)

    fig8 = styled_figure()
    ax8 = Axis(fig8[1, 1], xlabel = "x / w0", ylabel = "Center-line intensity", title = "Final center-line comparison")
    lines!(ax8, x_axis, launch_centerline, label = "Launch")
    lines!(ax8, x_axis, linear_centerline[end, :], label = "Linear final")
    lines!(ax8, x_axis, nonlinear_centerline[end, :], label = "Nonlinear final")
    axislegend(ax8, position = :rt)
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_final_centerline_comparison.png"), fig8)

    fig9 = styled_figure()
    ax9 = Axis(fig9[1, 1], xlabel = "t / T0", ylabel = "Temporal marginal intensity", title = "Final temporal comparison")
    lines!(ax9, t_axis, launch_temporal, label = "Launch")
    lines!(ax9, t_axis, linear_temporal[end, :], label = "Linear final")
    lines!(ax9, t_axis, nonlinear_temporal[end, :], label = "Nonlinear final")
    axislegend(ax9, position = :rt)
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_final_temporal_comparison.png"), fig9)

    fig10 = styled_figure()
    ax10 = Axis(fig10[1, 1], xlabel = "t / T0", ylabel = "z / L_D", title = "Nonlinear temporal marginal")
    hm10 = heatmap!(ax10, t_axis, z_axis, permutedims(normalized_plot_data(nonlinear_temporal), (2, 1)); colorrange = (0.0, 1.0))
    Colorbar(fig10[1, 2], hm10, label = "Normalized temporal marginal intensity")
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_nonlinear_temporal_map.png"), fig10)

    fig11 = styled_figure()
    ax11 = Axis(fig11[1, 1], xlabel = "x / w0", ylabel = "y / w0", title = "Nonlinear final transverse intensity")
    hm11 = heatmap!(ax11, x_axis, y_axis, permutedims(normalized_plot_data(nonlinear_xy[end, :, :]), (2, 1)); colorrange = (0.0, 1.0))
    Colorbar(fig11[1, 2], hm11, label = "Normalized final intensity")
    save_example_figure(joinpath(output_dir, "high_order_grin_soliton_nonlinear_final_xy_map.png"), fig11)

    plot_3d_intensity_contours_propagation(
        x_axis,
        y_axis,
        z_axis,
        linear_xy,
        joinpath(output_dir, "high_order_grin_soliton_linear_3d_intensity_contour_surfaces.png");
        input_is_intensity = true,
        z_label = "z / L_D",
    )

    plot_3d_intensity_contours_propagation(
        x_axis,
        y_axis,
        z_axis,
        nonlinear_xy,
        joinpath(output_dir, "high_order_grin_soliton_nonlinear_3d_intensity_contour_surfaces.png");
        input_is_intensity = true,
        z_label = "z / L_D",
        xy_crop_inset = 0.5,
    )
end

function main(argv = ARGS)
    args = parse_example_args("high_order_grin_soliton", "High-order GRIN-guided nonlinear tensor propagation.", argv)
    activate_example_theme!()
    NLOLib.set_progress_options(enabled = true, milestone_percent = 2, emit_on_step_adjust = true)
    NLOLib.set_log_level(NLOLIB_LOG_LEVEL_INFO)
    db = ExampleRunDB(args[:db_path])
    example_name = "high_order_grin_soliton_rk4ip"
    nonlinear_case_key = "nonlinear"
    linear_case_key = "linear_baseline"

    nt = 1024
    nx = 64
    ny = 64
    dt = 0.06
    dx = 0.24
    dy = 0.24
    temporal_width = 0.30
    soliton_order = 1.0
    mode_width = 1.0
    spatial_chirp = 0.0
    beta2 = -0.08
    beta_t = -0.08
    grin_strength = 1.5e-5
    gamma_nonlinear = 1.0
    z_period = soliton_period(beta2, temporal_width)
    z_final = 6.0 * z_period
    num_records = 120

    t_axis = centered_time_grid(nt, dt)
    x_axis = centered_spatial_grid(nx, dx)
    y_axis = centered_spatial_grid(ny, dy)
    omega = 2.0 * π .* FFTW.fftfreq(nt, 1.0 / dt)
    field0_tyx = ComplexF64.(grin_launch_field(
        t_axis,
        x_axis,
        y_axis;
        temporal_width = temporal_width,
        beta2 = beta2,
        gamma = gamma_nonlinear,
        soliton_order = soliton_order,
        mode_width = mode_width,
        spatial_chirp = spatial_chirp,
    ))
    potential = vec(grin_potential_grid(x_axis, y_axis, grin_strength))

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        loaded_nonlinear = load_case(db; example_name = example_name, run_group = run_group, case_key = nonlinear_case_key)
        loaded_linear = load_case(db; example_name = example_name, run_group = run_group, case_key = linear_case_key)
        meta = loaded_nonlinear.meta
        nt = Int(meta["nt"])
        nx = Int(meta["nx"])
        ny = Int(meta["ny"])
        dt = Float64(meta["dt"])
        dx = Float64(meta["dx"])
        dy = Float64(meta["dy"])
        temporal_width = Float64(meta["temporal_width"])
        soliton_order = Float64(meta["soliton_order"])
        mode_width = Float64(meta["mode_width"])
        spatial_chirp = Float64(meta["spatial_chirp"])
        beta2 = Float64(meta["beta2"])
        beta_t = Float64(meta["beta_t"])
        grin_strength = Float64(meta["grin_strength"])
        gamma_nonlinear = Float64(meta["gamma_nonlinear"])
        t_axis = centered_time_grid(nt, dt)
        x_axis = centered_spatial_grid(nx, dx)
        y_axis = centered_spatial_grid(ny, dy)
        field0_tyx = ComplexF64.(grin_launch_field(
            t_axis,
            x_axis,
            y_axis;
            temporal_width = temporal_width,
            beta2 = beta2,
            gamma = gamma_nonlinear,
            soliton_order = soliton_order,
            mode_width = mode_width,
            spatial_chirp = spatial_chirp,
        ))
        z_axis = loaded_nonlinear.z_axis
        nonlinear_records = unflatten_tfast_records(loaded_nonlinear.records, nt, ny, nx)
        linear_records = unflatten_tfast_records(loaded_linear.records, nt, ny, nx)
    else
        run_group = begin_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        pulse = PulseSpec(
            samples = flatten_tfast(field0_tyx),
            delta_time = dt,
            pulse_period = nt * dt,
            frequency_grid = ComplexF64.(omega),
            tensor_nt = nt,
            tensor_nx = nx,
            tensor_ny = ny,
            delta_x = dx,
            delta_y = dy,
            potential_grid = ComplexF64.(potential),
        )
        linear = OperatorSpec(
            expr = "i*(beta2*(wt*wt) + beta_t*((kx*kx)+(ky*ky)))",
            params = Dict("beta2" => 0.5 * beta2, "beta_t" => beta_t),
        )
        nonlinear_full = OperatorSpec(expr = "i*A*(gamma*I + V)", params = Dict("gamma" => gamma_nonlinear))
        nonlinear_linear = OperatorSpec(expr = "i*A*(gamma*I + V)", params = Dict("gamma" => 0.0))
        exec = default_execution_options(backend_type = VECTOR_BACKEND_AUTO, fft_backend = FFT_BACKEND_AUTO)
        storage_nonlinear = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = nonlinear_case_key, chunk_records = 4)
        storage_linear = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = linear_case_key, chunk_records = 4)

        result_nonlinear = NLOLib.propagate(
            pulse,
            linear,
            nonlinear_full;
            t_span = (0.0, z_final),
            t_eval = collect(range(0.0, z_final, length = num_records)),
            first_step = 5e-3,
            max_step = 2e-1,
            min_step = 1e-5,
            rtol = 1e-6,
            exec_options = exec,
            sqlite_path = storage_nonlinear.sqlite_path,
            run_id = storage_nonlinear.run_id,
            chunk_records = storage_nonlinear.chunk_records,
        )
        result_linear = NLOLib.propagate(
            pulse,
            linear,
            nonlinear_linear;
            t_span = (0.0, z_final),
            t_eval = collect(range(0.0, z_final, length = num_records)),
            first_step = 5e-3,
            max_step = 2e-1,
            min_step = 1e-5,
            rtol = 1e-6,
            exec_options = exec,
            sqlite_path = storage_linear.sqlite_path,
            run_id = storage_linear.run_id,
            chunk_records = storage_linear.chunk_records,
        )

        z_axis = result_nonlinear.z_axis
        nonlinear_records = unflatten_tfast_records(permutedims(result_nonlinear.records, (2, 1)), nt, ny, nx)
        linear_records = unflatten_tfast_records(permutedims(result_linear.records, (2, 1)), nt, ny, nx)
        meta = Dict(
            "nt" => nt,
            "nx" => nx,
            "ny" => ny,
            "dt" => dt,
            "dx" => dx,
            "dy" => dy,
            "temporal_width" => temporal_width,
            "soliton_order" => soliton_order,
            "mode_width" => mode_width,
            "spatial_chirp" => spatial_chirp,
            "beta2" => beta2,
            "beta_t" => beta_t,
            "grin_strength" => grin_strength,
            "gamma_nonlinear" => gamma_nonlinear,
        )
        save_case_from_solver_meta!(db; example_name = example_name, run_group = run_group, case_key = nonlinear_case_key, solver_meta = result_nonlinear.meta, meta = meta)
        save_case_from_solver_meta!(db; example_name = example_name, run_group = run_group, case_key = linear_case_key, solver_meta = result_linear.meta, meta = meta)
    end

    nonlinear_xy = time_integrated_xy_records(nonlinear_records)
    linear_xy = time_integrated_xy_records(linear_records)
    nonlinear_radius = rms_radius_curve(nonlinear_records, x_axis, y_axis)
    linear_radius = rms_radius_curve(linear_records, x_axis, y_axis)
    nonlinear_temporal_width = rms_temporal_width_curve(nonlinear_records, t_axis)
    linear_temporal_width = rms_temporal_width_curve(linear_records, t_axis)
    nonlinear_peak = peak_transverse_intensity_curve(nonlinear_xy)
    linear_peak = peak_transverse_intensity_curve(linear_xy)
    nonlinear_overlap = overlap_fidelity_curve(nonlinear_records, field0_tyx)
    linear_overlap = overlap_fidelity_curve(linear_records, field0_tyx)
    nonlinear_power = total_power_curve(nonlinear_records)
    linear_power = total_power_curve(linear_records)
    nonlinear_power_drift_curve = relative_power_drift_curve(nonlinear_power)
    linear_power_drift_curve = relative_power_drift_curve(linear_power)
    z_period = soliton_period(beta2, temporal_width)
    ld = dispersion_length(beta2, temporal_width)
    lnl = 1.0 / (gamma_nonlinear * fundamental_soliton_power(beta2, gamma_nonlinear, temporal_width))
    ldiff = diffraction_length(beta_t, mode_width)
    z_final = isempty(z_axis) ? z_final : z_axis[end]
    t_scaled = t_axis ./ temporal_width
    x_scaled = x_axis ./ mode_width
    y_scaled = y_axis ./ mode_width
    z_scaled = z_axis ./ ld

    save_grin_plots(
        args[:output_dir],
        z_scaled,
        t_scaled,
        x_scaled,
        y_scaled,
        field0_tyx,
        nonlinear_records,
        linear_records,
        nonlinear_radius,
        linear_radius,
        nonlinear_temporal_width,
        linear_temporal_width,
        nonlinear_peak,
        linear_peak,
        nonlinear_overlap,
        linear_overlap,
        nonlinear_power_drift_curve,
        linear_power_drift_curve,
    )

    nonlinear_power_drift = nonlinear_power_drift_curve[end]
    linear_power_drift = linear_power_drift_curve[end]

    println("high-order GRIN soliton summary")
    println("  nonlinear final radius / launch radius = $(nonlinear_radius[end] / nonlinear_radius[1])")
    println("  linear final radius / launch radius = $(linear_radius[end] / linear_radius[1])")
    println("  nonlinear radius excursion = $(maximum(nonlinear_radius) / nonlinear_radius[1])")
    println("  linear radius excursion = $(maximum(linear_radius) / linear_radius[1])")
    println("  nonlinear final temporal width / launch width = $(nonlinear_temporal_width[end] / nonlinear_temporal_width[1])")
    println("  linear final temporal width / launch width = $(linear_temporal_width[end] / linear_temporal_width[1])")
    println("  nonlinear min overlap fidelity = $(minimum(nonlinear_overlap))")
    println("  linear min overlap fidelity = $(minimum(linear_overlap))")
    println("  nonlinear power drift = $(nonlinear_power_drift)")
    println("  linear power drift = $(linear_power_drift)")
    println("  nonlinear max power drift = $(maximum(nonlinear_power_drift_curve))")
    println("  linear max power drift = $(maximum(linear_power_drift_curve))")
    println("  L_D = $(ld)")
    println("  L_NL(fundamental) = $(lnl)")
    println("  L_diff = $(ldiff)")
    println("  soliton period / L_D = $(z_period / ld)")
    println("  periods covered = $(z_final / z_period)")
    return max(maximum(nonlinear_power_drift_curve), maximum(linear_power_drift_curve))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
