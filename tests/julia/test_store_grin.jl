# Stored GRIN soliton runs through the Julia binding: staging -> merge into a
# shared database, a `runs` row that satisfies the ModelStore index contract,
# and reduced maps that match the same quantities computed from the raw
# records, in 1D (Raman soliton fission) and 3D (GRIN-guided soliton).
#
# Runs in the examples project (FFTW, SQLite), reusing the GRIN example's
# field and potential builders on a reduced grid.

include(joinpath(@__DIR__, "..", "..", "examples", "julia", "high_order_grin_soliton_rk4ip.jl"))

using Test
using FFTW
using SQLite
using DBInterface

const KEY_RE = r"^[0-9a-f]{16}$"
const TIME_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?$"

# SQLite.jl rows are views of the live statement; copy each one out.
query_rows(db, sql) = [NamedTuple{Tuple(propertynames(row))}(Tuple(getproperty(row, n) for n in propertynames(row)))
                       for row in DBInterface.execute(db, sql)]

function check_runs_contract(db_path)
    db = SQLite.DB(db_path)
    try
        @test only(query_rows(db, "PRAGMA journal_mode;"))[1] == "delete"
        info = query_rows(db, "PRAGMA table_info(runs);")
        fixed = [("key", "TEXT", 0, 1), ("kind", "TEXT", 1, 0), ("file", "TEXT", 1, 0),
                 ("created_at", "TEXT", 0, 0), ("git_commit", "TEXT", 0, 0), ("julia_version", "TEXT", 0, 0)]
        for (i, (name, type, notnull, pk)) in enumerate(fixed)
            @test info[i].name == name
            @test info[i].type == type
            @test info[i].notnull == notnull
            @test info[i].pk == pk
        end
        params = info[length(fixed) + 1:end]
        @test !isempty(params)
        names = [String(c.name) for c in info]
        @test length(unique(lowercase.(names))) == length(names)
        for c in params
            @test startswith(c.name, "nlolib_")
            @test c.name == lowercase(c.name)
            @test 1 <= ncodeunits(c.name) <= 63
            @test c.type in ("INTEGER", "REAL", "TEXT")
        end
        declared = Dict(String(c.name) => String(c.type) for c in params)
        typeof_sql = "SELECT key, kind, file, created_at, git_commit, julia_version, typeof(key) AS tk, " *
                     join(["typeof(\"$(c.name)\") AS \"t_$(c.name)\"" for c in params], ", ") * " FROM runs;"
        rows = query_rows(db, typeof_sql)
        @test !isempty(rows)
        for row in rows
            @test row.tk == "text"
            @test occursin(KEY_RE, row.key)
            @test row.kind == "nlolib"
            @test row.file == "nlolib/nlolib.sqlite"
            @test occursin(TIME_RE, row.created_at)
            @test !ismissing(row.git_commit) && !isempty(row.git_commit)
            @test ismissing(row.julia_version)
            for (name, type) in declared
                t = getproperty(row, Symbol("t_" * name))
                @test t != "blob"
                t == "null" || @test uppercase(t == "real" ? "REAL" : t) == type
            end
        end
        binding = only(query_rows(db, "SELECT DISTINCT nlolib_binding FROM runs;"))[1]
        @test binding == "julia"
        @test only(query_rows(db, "PRAGMA integrity_check;"))[1] == "ok"
    finally
        SQLite.close(db)
    end
end

# Reduced maps recomputed from the returned records (columns = records, t fastest).
function maps_from_records(records, nt, nx, ny)
    nrec = size(records, 2)
    power = zeros(nt, nrec)
    spectrum = zeros(nt, nrec)
    fx = zeros(nx, nrec)
    fy = zeros(ny, nrec)
    for r in 1:nrec
        a = reshape(records[:, r], nt, ny, nx)
        intensity = abs2.(a)
        power[:, r] = vec(sum(intensity, dims = (2, 3)))
        spectrum[:, r] = fftshift(vec(sum(abs2.(fft(a, 1)), dims = (2, 3))))
        fx[:, r] = vec(sum(intensity, dims = (1, 2)))
        fy[:, r] = vec(sum(intensity, dims = (1, 3)))
    end
    return power, spectrum, fx, fy
end

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1e-300)

function check_stored_maps(db_path, result, nt, nx, ny, dt)
    key = result.meta["storage_result"]["key"]
    @test occursin(KEY_RE, key)
    @test result.meta["storage_result"]["merged"] == true
    @test storage_has_run(db_path, key)
    stored = read_stored_run(db_path, key)
    @test (stored.nt, stored.nx, stored.ny) == (nt, nx, ny)
    @test stored.record_mode == :reduced_and_raw
    @test stored.z ≈ result.z_axis
    power, spectrum, fx, fy = maps_from_records(result.records, nt, nx, ny)
    @test relerr(stored.power, power) < 1e-12
    @test relerr(stored.spectrum, spectrum) < 1e-11
    @test relerr(stored.fluence_x, fx) < 1e-12
    @test relerr(stored.fluence_y, fy) < 1e-12
    @test stored.omega ≈ fftshift(2π .* fftfreq(nt, 1 / dt))
    @test stored.t ≈ ((0:nt - 1) .- (nt - 1) / 2) .* dt
    @test vec(stored.raw) == vec(result.records)
    @test vec(stored.final_field) == result.records[:, end]
    return key
end

@testset "stored GRIN soliton runs" begin
    root = mktempdir()
    db_path = joinpath(root, "nlolib", "nlolib.sqlite")
    exec = default_execution_options(backend_type = VECTOR_BACKEND_CPU, fft_backend = FFT_BACKEND_FFTW)
    beta2 = -0.08
    temporal_width = 0.30
    gamma_nonlinear = 1.0
    dt = 0.06

    @testset "3D GRIN soliton" begin
        nt, nx, ny = 128, 12, 12
        dx = dy = 0.45
        t_axis = centered_time_grid(nt, dt)
        x_axis = centered_spatial_grid(nx, dx)
        y_axis = centered_spatial_grid(ny, dy)
        field0 = grin_launch_field(t_axis, x_axis, y_axis; temporal_width = temporal_width, beta2 = beta2,
                                   gamma = gamma_nonlinear, soliton_order = 1.0, mode_width = 1.0, spatial_chirp = 0.0)
        potential = vec(grin_potential_grid(x_axis, y_axis, 1.5e-2))
        pulse = PulseSpec(samples = flatten_tfast(field0), delta_time = dt, pulse_period = nt * dt,
                          frequency_grid = ComplexF64.(2π .* fftfreq(nt, 1 / dt)),
                          tensor_nt = nt, tensor_nx = nx, tensor_ny = ny, delta_x = dx, delta_y = dy,
                          potential_grid = ComplexF64.(repeat(potential, inner = nt)))
        linear = OperatorSpec(expr = "i*(beta2*(wt*wt) + beta_t*((kx*kx)+(ky*ky)))",
                              params = Dict("beta2" => 0.5 * beta2, "beta_t" => -0.08))
        nonlinear = OperatorSpec(expr = "i*A*(gamma*I + V)", params = Dict("gamma" => gamma_nonlinear))
        z_final = soliton_period(beta2, temporal_width)
        kwargs = (t_span = (0.0, z_final), t_eval = collect(range(0.0, z_final, length = 6)),
                  first_step = 5e-3, max_step = 5e-2, min_step = 1e-6, rtol = 1e-6, exec_options = exec,
                  sqlite_path = db_path, record_mode = :reduced_and_raw)
        expected_key = storage_run_key(pulse, linear, nonlinear; kwargs...)
        @test !storage_has_run(db_path, expected_key)
        result = NLOLib.propagate(pulse, linear, nonlinear; kwargs...)
        @test check_stored_maps(db_path, result, nt, nx, ny, dt) == expected_key
        @test isempty(filter(f -> endswith(f, ".sqlite"), readdir(joinpath(root, "nlolib", "staging"))))
    end

    @testset "1D Raman soliton fission" begin
        nt = 512
        t_axis = centered_time_grid(nt, dt / 2)
        pulse = PulseSpec(samples = temporal_envelope(t_axis; temporal_width = temporal_width, beta2 = beta2,
                                                      gamma = gamma_nonlinear, soliton_order = 3.0),
                          delta_time = dt / 2)
        z_final = 0.5 * soliton_period(beta2, temporal_width)
        linear = OperatorSpec(expr = "i*beta2*w*w", params = Dict("beta2" => 0.5 * beta2))
        kwargs = (propagation_distance = z_final, records = 9, exec_options = exec, sqlite_path = db_path,
                  record_mode = :reduced_and_raw, nonlinear_model = NONLINEAR_MODEL_KERR_RAMAN,
                  nonlinear_gamma = gamma_nonlinear, raman_fraction = 0.18,
                  max_step = 1e-2, rtol = 1e-7)
        result = NLOLib.propagate(pulse, linear, "kerr"; kwargs...)
        key = check_stored_maps(db_path, result, nt, 1, 1, dt / 2)
        @test key == storage_run_key(pulse, linear, "kerr"; kwargs...)
    end

    check_runs_contract(db_path)
end
