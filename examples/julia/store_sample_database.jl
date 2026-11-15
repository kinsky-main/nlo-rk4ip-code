# Write a small shared run database for readers of the Nlolib run store:
# one time-only run (Raman soliton fission) and one (t, x, y) run (GRIN-guided
# soliton), both with the default reduced maps and final field.
#
#   julia --project=julia examples/julia/store_sample_database.jl [path/to/nlolib.sqlite]
#
# Layout of the result: docs/nlolib_run_database.md.

using NLOLib

function centered_axis(n, step)
    return [(i - (n - 1) / 2) * step for i in 0:(n - 1)]
end

fftfreq_omega(n, dt) = [2π * (i <= (n - 1) ÷ 2 ? i : i - n) / (n * dt) for i in 0:(n - 1)]

function main(path)
    NLOLib.set_progress_options(enabled = false)
    NLOLib.set_log_level(NLOLIB_LOG_LEVEL_WARN)
    mkpath(dirname(path))
    exec = default_execution_options(backend_type = VECTOR_BACKEND_CPU, fft_backend = FFT_BACKEND_FFTW)
    beta2 = -0.08
    width = 0.30
    p1 = abs(beta2) / width^2                      # fundamental soliton power for gamma = 1
    soliton_period = 0.5 * π * width^2 / abs(beta2)

    # 1D: third-order soliton with delayed Raman response.
    nt, dt = 1024, 0.02
    t = centered_axis(nt, dt)
    pulse_1d = PulseSpec(samples = ComplexF64.(3.0 * sqrt(p1) ./ cosh.(t ./ width)), delta_time = dt)
    linear_1d = OperatorSpec(expr = "i*beta2*w*w", params = Dict("beta2" => 0.5 * beta2))
    result_1d = NLOLib.propagate(pulse_1d, linear_1d, "kerr";
        propagation_distance = soliton_period, records = 64, exec_options = exec, sqlite_path = path,
        nonlinear_model = NONLINEAR_MODEL_KERR_RAMAN, nonlinear_gamma = 1.0, raman_fraction = 0.18,
        max_step = 5e-3, rtol = 1e-7)

    # 3D: GRIN-guided fundamental soliton.
    nt3, nx, ny, dt3, dx = 128, 32, 32, 0.06, 0.25
    t3 = centered_axis(nt3, dt3)
    x = centered_axis(nx, dx)
    y = centered_axis(ny, dx)
    field = Vector{ComplexF64}(undef, nt3 * nx * ny)
    potential = similar(field)
    for xi in 1:nx, yi in 1:ny
        r2 = x[xi]^2 + y[yi]^2
        for ti in 1:nt3
            idx = (((xi - 1) * ny) + (yi - 1)) * nt3 + ti      # t fastest
            field[idx] = sqrt(p1) / cosh(t3[ti] / width) * exp(-r2 / 2)
            potential[idx] = 0.02 * r2
        end
    end
    pulse_3d = PulseSpec(samples = field, delta_time = dt3, pulse_period = nt3 * dt3,
                         frequency_grid = ComplexF64.(fftfreq_omega(nt3, dt3)),
                         tensor_nt = nt3, tensor_nx = nx, tensor_ny = ny, delta_x = dx, delta_y = dx,
                         potential_grid = potential)
    linear_3d = OperatorSpec(expr = "i*(beta2*(wt*wt) + beta_t*((kx*kx)+(ky*ky)))",
                             params = Dict("beta2" => 0.5 * beta2, "beta_t" => -0.08))
    nonlinear_3d = OperatorSpec(expr = "i*A*(gamma*I + V)", params = Dict("gamma" => 1.0))
    result_3d = NLOLib.propagate(pulse_3d, linear_3d, nonlinear_3d;
        t_span = (0.0, soliton_period), t_eval = collect(range(0.0, soliton_period, length = 32)),
        first_step = 5e-3, max_step = 5e-2, min_step = 1e-6, rtol = 1e-6, exec_options = exec,
        sqlite_path = path)

    staging = joinpath(dirname(path), "staging")
    isdir(staging) && isempty(readdir(staging)) && rm(staging)
    for (label, result) in (("1D", result_1d), ("3D", result_3d))
        stored = result.meta["storage_result"]
        println(label, " run: key ", stored["key"], ", merged ", stored["merged"])
    end
    println("wrote ", path)
end

main(isempty(ARGS) ? joinpath(homedir(), "nlolib-sample", "nlolib.sqlite") : ARGS[1])
