using Test
using NLOLib

function zero_operator_case()
    nt = 64
    dt = 0.02
    zmax = 0.0

    propagation = PropagationParams(1e-3, 1e-2, 1e-6, 1e-8, zmax)
    sim = SimulationConfig(
        propagation,
        Tensor3DDesc(),
        TimeGrid(nt, nt * dt, dt, C_NULL),
        FrequencyGrid(C_NULL),
        SpatialGrid()
    )
    phys = physics_config(
        linear_factor_expr = "0",
        linear_expr = "exp(h*D)",
        potential_expr = "0",
        nonlinear_expr = "0"
    )
    exec = default_execution_options(
        backend_type = NLO_VECTOR_BACKEND_CPU,
        fft_backend = NLO_FFT_BACKEND_FFTW
    )
    field = ComplexF64.(exp.(-((collect(0:nt - 1) .- nt / 2) ./ 8) .^ 2))

    limits = query_runtime_limits(sim, phys; exec_options = exec)
    @test limits.max_num_time_samples_runtime > 0

    result = propagate(sim, phys, field;
        num_recorded_samples = 1,
        output_mode = NLO_PROPAGATE_OUTPUT_FINAL_ONLY,
        exec_options = exec
    )
    @test size(result.records) == (nt, 1)
    @test result.meta.records_written == 1
    @test result.meta.step_events_written == 0
    @test result.meta.step_events_dropped == 0
    @test result.records[:, 1] ≈ field atol = 1e-12 rtol = 1e-12
    @test final_record(result.records) ≈ field atol = 1e-12 rtol = 1e-12

    raw_output = Matrix{NLOComplex}(undef, nt, 1)
    meta = propagate!(
        raw_output,
        sim,
        phys,
        reinterpret(NLOComplex, field);
        num_recorded_samples = 1,
        output_mode = NLO_PROPAGATE_OUTPUT_FINAL_ONLY,
        exec_options = exec
    )
    @test meta.records_written == 1

    wrapped = wrap_records(pointer(raw_output), nt, 1)
    @test wrapped[:, 1] ≈ field atol = 1e-12 rtol = 1e-12

    tensor = tensor_record_view(wrapped, 1, nt, 1, 1)
    @test size(tensor) == (nt, 1, 1)
end

@testset "NLOLib Julia wrapper" begin
    load()
    @test !isempty(loaded_library_path())
    @test sizeof(NLOComplex) == 16
    zero_operator_case()
end
