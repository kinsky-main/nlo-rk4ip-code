include(joinpath(@__DIR__, "backend", "common.jl"))
include(joinpath(@__DIR__, "backend", "metrics.jl"))
include(joinpath(@__DIR__, "backend", "reference.jl"))
include(joinpath(@__DIR__, "backend", "storage.jl"))

pushfirst!(LOAD_PATH, nlo_package_root_from(@__FILE__))

using CairoMakie
using FFTW
using LinearAlgebra
using NLOLib

function gaussian_tensor_field(t_axis, x_axis, y_axis; temporal_width, x_width, y_width)
    temporal = exp.(-((t_axis ./ temporal_width) .^ 2))
    transverse = [exp(-((x / x_width)^2) - ((y / y_width)^2)) for y in y_axis, x in x_axis]
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
    for idx in axes(flat, 1)
        cursor = 1
        for x in 1:nx, y in 1:ny, t in 1:nt
            out[idx, t, y, x] = flat[idx, cursor]
            cursor += 1
        end
    end
    return out
end

function marginal_intensity_profiles(field_tyx)
    intensity = abs2.(ComplexF64.(field_tyx))
    temporal = vec(sum(intensity, dims = (2, 3)))
    x_profile = vec(sum(intensity, dims = (1, 2)))
    y_profile = vec(sum(intensity, dims = (1, 3)))
    return temporal, x_profile, y_profile
end

function relative_l2_real(prediction, reference)
    pred = Float64.(prediction)
    ref = Float64.(reference)
    return norm(pred .- ref) / max(norm(ref), 1e-30)
end

function save_tensor_plots(output_dir, z_axis, t_axis, x_axis, y_axis, records_tyx, reference_records, field_error_curve, intensity_error_curve)
    mkpath(output_dir)
    final_temporal_ref, final_x_ref, final_y_ref = marginal_intensity_profiles(reference_records[end, :, :, :])
    final_temporal_num, final_x_num, final_y_num = marginal_intensity_profiles(records_tyx[end, :, :, :])

    fig1 = Figure(size = (900, 600))
    ax1 = Axis(fig1[1, 1], xlabel = "z", ylabel = "Relative error", title = "Tensor propagation error")
    lines!(ax1, z_axis, field_error_curve, label = "field")
    lines!(ax1, z_axis, intensity_error_curve, label = "intensity")
    axislegend(ax1, position = :rt)
    save(joinpath(output_dir, "tensor_error_curve.png"), fig1)

    fig2 = Figure(size = (900, 600))
    ax2 = Axis(fig2[1, 1], xlabel = "time", ylabel = "Marginal intensity", title = "Final temporal marginal")
    lines!(ax2, t_axis, final_temporal_ref, label = "reference")
    lines!(ax2, t_axis, final_temporal_num, label = "numerical")
    axislegend(ax2, position = :rt)
    save(joinpath(output_dir, "tensor_temporal_marginal.png"), fig2)

    fig3 = Figure(size = (900, 600))
    ax3 = Axis(fig3[1, 1], xlabel = "x", ylabel = "Marginal intensity", title = "Final x marginal")
    lines!(ax3, x_axis, final_x_ref, label = "reference")
    lines!(ax3, x_axis, final_x_num, label = "numerical")
    axislegend(ax3, position = :rt)
    save(joinpath(output_dir, "tensor_x_marginal.png"), fig3)

    fig4 = Figure(size = (900, 600))
    ax4 = Axis(fig4[1, 1], xlabel = "y", ylabel = "Marginal intensity", title = "Final y marginal")
    lines!(ax4, y_axis, final_y_ref, label = "reference")
    lines!(ax4, y_axis, final_y_num, label = "numerical")
    axislegend(ax4, position = :rt)
    save(joinpath(output_dir, "tensor_y_marginal.png"), fig4)
end

function main(argv = ARGS)
    args = parse_example_args("tensor_dispersion_3d", "Tensor 3D dispersion/diffraction with DB-backed run/replot.", argv)
    NLOLib.set_progress_options(enabled = false)
    db = ExampleRunDB(args[:db_path])
    example_name = "tensor_dispersion_3d_rk4ip"
    case_key = "default"

    nt = 128
    nx = 64
    ny = 64
    dt = 0.04
    dx = 0.15
    dy = 0.15
    temporal_width = 0.24
    x_width = 0.60
    y_width = 0.70
    beta2 = 0.08
    beta_t = -0.20
    z_final = 0.80
    num_records = 100

    t_axis = centered_time_grid(nt, dt)
    x_axis = centered_spatial_grid(nx, dx)
    y_axis = centered_spatial_grid(ny, dy)
    omega = 2.0 * π .* FFTW.fftfreq(nt, 1.0 / dt)
    kx = 2.0 * π .* FFTW.fftfreq(nx, 1.0 / dx)
    ky = 2.0 * π .* FFTW.fftfreq(ny, 1.0 / dy)
    field0_tyx = ComplexF64.(gaussian_tensor_field(t_axis, x_axis, y_axis; temporal_width = temporal_width, x_width = x_width, y_width = y_width))

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        loaded = load_case(db; example_name = example_name, run_group = run_group, case_key = case_key)
        z_axis = loaded.z_axis
        records_flat = loaded.records
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
        )
        linear = OperatorSpec(
            expr = "i*(beta2*wt*wt + beta_t*((kx*kx)+(ky*ky)))",
            params = Dict("beta2" => 0.5 * beta2, "beta_t" => beta_t),
        )
        exec = default_execution_options(backend_type = NLO_VECTOR_BACKEND_CPU, fft_backend = NLO_FFT_BACKEND_FFTW)
        storage = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = case_key, chunk_records = 2)
        result = NLOLib.propagate(
            pulse,
            linear,
            "none";
            t_span = (0.0, z_final),
            t_eval = collect(range(0.0, z_final, length = num_records)),
            first_step = 0.02,
            max_step = 0.08,
            min_step = 1e-5,
            rtol = 1e-7,
            exec_options = exec,
            sqlite_path = storage.sqlite_path,
            run_id = storage.run_id,
            chunk_records = storage.chunk_records,
        )
        z_axis = result.z_axis
        records_flat = permutedims(result.records, (2, 1))
        save_case_from_solver_meta!(db;
            example_name = example_name,
            run_group = run_group,
            case_key = case_key,
            solver_meta = result.meta,
            meta = Dict("nt" => nt, "nx" => nx, "ny" => ny, "dt" => dt, "dx" => dx, "dy" => dy, "temporal_width" => temporal_width, "x_width" => x_width, "y_width" => y_width, "beta2" => beta2, "beta_t" => beta_t))
    end

    records_tyx = unflatten_tfast_records(records_flat, nt, ny, nx)
    reference_records = exact_linear_tensor3d_records(field0_tyx, z_axis, omega, kx, ky, 0.5 * beta2, beta_t)
    field_error_curve = [norm(vec(records_tyx[i, :, :, :] .- reference_records[i, :, :, :])) / max(norm(vec(reference_records[i, :, :, :])), 1e-30) for i in axes(records_tyx, 1)]
    intensity_error_curve = [relative_l2_intensity_error(records_tyx[i, :, :, :], reference_records[i, :, :, :]) for i in axes(records_tyx, 1)]

    final_temporal_ref, final_x_ref, final_y_ref = marginal_intensity_profiles(reference_records[end, :, :, :])
    final_temporal_num, final_x_num, final_y_num = marginal_intensity_profiles(records_tyx[end, :, :, :])
    temporal_profile_error = relative_l2_real(final_temporal_num, final_temporal_ref)
    x_profile_error = relative_l2_real(final_x_num, final_x_ref)
    y_profile_error = relative_l2_real(final_y_num, final_y_ref)

    save_tensor_plots(args[:output_dir], z_axis, t_axis, x_axis, y_axis, records_tyx, reference_records, field_error_curve, intensity_error_curve)

    println("tensor 3D dispersion summary")
    println("  final field error = $(field_error_curve[end])")
    println("  final intensity error = $(intensity_error_curve[end])")
    println("  marginal profile errors = t=$(temporal_profile_error), x=$(x_profile_error), y=$(y_profile_error)")
    return maximum((intensity_error_curve[end], temporal_profile_error, x_profile_error, y_profile_error))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
