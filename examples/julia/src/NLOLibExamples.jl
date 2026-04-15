module NLOLibExamples

using PrecompileTools: @setup_workload, @compile_workload

const _BACKEND_DIR = normpath(joinpath(@__DIR__, "..", "backend"))

include(joinpath(_BACKEND_DIR, "common.jl"))
include(joinpath(_BACKEND_DIR, "metrics.jl"))
include(joinpath(_BACKEND_DIR, "plotting.jl"))
include(joinpath(_BACKEND_DIR, "reference.jl"))
include(joinpath(_BACKEND_DIR, "storage.jl"))

export centered_time_grid
export centered_spatial_grid
export repo_root_from
export package_root_from
export build_example_parser
export parse_example_args
export relative_l2_intensity_error
export relative_l2_intensity_error_curve
export nlolib_hdr_colormap
export activate_example_theme!
export styled_figure
export save_example_figure
export normalized_plot_data
export plot_3d_intensity_volume_propagation
export plot_3d_intensity_contours_propagation
export second_order_soliton_period
export second_order_soliton_normalized_envelope
export second_order_soliton_normalized_records
export exact_linear_tensor3d_records
export LoadedCase
export ExampleRunDB
export begin_group
export latest_run_group
export nth_latest_run_group
export resolve_replot_group
export storage_kwargs
export save_case!
export save_case_from_solver_meta!
export load_case
export load_step_history
export save_step_history!

@setup_workload begin
    activate_example_theme!()

    t_axis = centered_time_grid(64, 0.01)
    x_axis = centered_spatial_grid(24, 0.12)
    y_axis = centered_spatial_grid(20, 0.14)
    z_axis = collect(range(0.0, 0.5, length = 8))
    pulse = ComplexF64.(exp.(-(t_axis .^ 2)))
    records = repeat(reshape(pulse, 1, :), length(z_axis), 1)
    reference = copy(records)

    field0_tyx = Array{ComplexF64}(undef, length(t_axis), length(y_axis), length(x_axis))
    for ti in eachindex(t_axis), yi in eachindex(y_axis), xi in eachindex(x_axis)
        field0_tyx[ti, yi, xi] = exp(-((t_axis[ti] / 0.2)^2 + (x_axis[xi] / 0.5)^2 + (y_axis[yi] / 0.6)^2))
    end
    omega = 2.0 * π .* FFTW.fftfreq(length(t_axis), 1.0 / 0.01)
    kx = 2.0 * π .* FFTW.fftfreq(length(x_axis), 1.0 / 0.12)
    ky = 2.0 * π .* FFTW.fftfreq(length(y_axis), 1.0 / 0.14)
    intensity_volume = Array{Float64}(undef, length(z_axis), length(y_axis), length(x_axis))
    for zi in eachindex(z_axis), yi in eachindex(y_axis), xi in eachindex(x_axis)
        sigma = 0.25 + 0.35 * z_axis[zi]
        intensity_volume[zi, yi, xi] = exp(-((x_axis[xi] / sigma)^2 + (y_axis[yi] / (0.85 * sigma))^2))
    end

    temp_root = mktempdir()

    @compile_workload begin
        parse_example_args("precompile_example", "Warmup example", String[])
        relative_l2_intensity_error(records[end, :], reference[end, :])
        relative_l2_intensity_error_curve(records, reference)
        second_order_soliton_period(-0.01, 0.08)
        second_order_soliton_normalized_records(t_axis, z_axis, -0.01, 0.08)
        exact_linear_tensor3d_records(field0_tyx, z_axis, omega, kx, ky, -0.005, -0.02)

        fig = styled_figure()
        ax = Axis(fig[1, 1], xlabel = "t", ylabel = "z")
        hm = heatmap!(ax, t_axis, z_axis, permutedims(normalized_plot_data(abs2.(records)), (2, 1)); colorrange = (0.0, 1.0))
        Colorbar(fig[1, 2], hm, label = "Normalized intensity")
        save_example_figure(joinpath(temp_root, "warmup_heatmap.png"), fig)

        plot_3d_intensity_contours_propagation(
            x_axis,
            y_axis,
            z_axis,
            intensity_volume,
            joinpath(temp_root, "warmup_contours.png");
            input_is_intensity = true,
            max_x_samples = 24,
            max_y_samples = 20,
            max_z_samples = 8,
        )
    end
end

end
