import math

from nlolib_ctypes import (
    NLO_VECTOR_BACKEND_AUTO,
    NLO_VECTOR_BACKEND_CPU,
    NLolib,
    OperatorSpec,
    PulseSpec,
    RuntimeOperators,
    default_execution_options,
    prepare_sim_config,
)


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _l2_norm(field):
    return sum((v.real * v.real) + (v.imag * v.imag) for v in field) ** 0.5


def _base_config(n: int):
    return prepare_sim_config(
        n,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.1,
        min_step_size=0.001,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=0.001,
        frequency_grid=[0j] * n,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0]),
    )


def main():
    api = NLolib()
    api.set_log_buffer(128 * 1024)
    api.set_log_level(2)
    api.set_progress_options(enabled=True, milestone_percent=20, emit_on_step_adjust=True)
    api.clear_log_buffer()
    print("test_python_api_smoke: runtime log API configured.")

    cpu_opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)
    limits = api.query_runtime_limits(exec_options=cpu_opts)
    assert int(limits.max_num_time_samples_runtime) > 0
    assert int(limits.max_num_recorded_samples_with_storage) > 0

    n = 128
    cfg = _base_config(n)
    input_field = [0j] * n

    out_records = api.propagate(cfg, input_field, 4, cpu_opts).records
    assert len(out_records) == 4
    assert len(out_records[0]) == n
    log_text = api.read_log_buffer(consume=True)
    assert "propagate request" in log_text
    assert "field_size" in log_text
    print("test_python_api_smoke: CPU 1D propagation returned expected record-major shape.")

    auto_opts = default_execution_options(NLO_VECTOR_BACKEND_AUTO)
    identity_cfg = prepare_sim_config(
        n,
        propagation_distance=0.02,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=1.0,
        delta_time=0.001,
        frequency_grid=[0j] * n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="0",
            constants=[0.0, 0.0, 0.0, 0.0],
        ),
    )
    gaussian = [complex(math.exp(-((i - 0.5 * n) / 18.0) ** 2), 0.0) for i in range(n)]
    identity_records = api.propagate(identity_cfg, gaussian, 3, auto_opts).records
    baseline_norm = _l2_norm(identity_records[0])
    final_norm = _l2_norm(identity_records[-1])
    rel_drift = abs(final_norm - baseline_norm) / max(baseline_norm, 1e-12)
    assert rel_drift <= 1e-6, f"AUTO identity propagation drift too large: {rel_drift}"
    print("test_python_api_smoke: AUTO identity propagation preserved field norm.")

    step_history_result = api.propagate(
        identity_cfg,
        gaussian,
        3,
        auto_opts,
        capture_step_history=True,
        step_history_capacity=20000,
    )
    step_history = step_history_result.meta.get("step_history")
    assert isinstance(step_history, dict)
    assert isinstance(step_history.get("z"), list)
    assert isinstance(step_history.get("step_size"), list)
    assert isinstance(step_history.get("next_step_size"), list)
    assert isinstance(step_history.get("dropped"), int)
    assert len(step_history["z"]) > 0
    assert step_history["dropped"] >= 0
    print("test_python_api_smoke: optional step history capture returned structured telemetry.")

    nx = 16
    ny = 8
    nxy = nx * ny
    cfg_2d = prepare_sim_config(
        nxy,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.1,
        min_step_size=0.001,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=0.001,
        frequency_grid=[0j] * nxy,
        spatial_nx=nx,
        spatial_ny=ny,
        spatial_frequency_grid=[0j] * nxy,
        potential_grid=[1 + 0j] * nxy,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0]),
    )
    out_2d = api.propagate(cfg_2d, [0j] * nxy, 1, cpu_opts).records
    assert len(out_2d) == 1
    assert len(out_2d[0]) == nxy
    print("test_python_api_smoke: flattened 2D propagation returned expected shape.")

    nt = 4
    nx3 = 4
    ny3 = 2
    n3 = nt * nx3 * ny3
    cfg_3d = prepare_sim_config(
        n3,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.1,
        min_step_size=0.001,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=0.001,
        time_nt=nt,
        frequency_grid=[0j] * nt,
        spatial_nx=nx3,
        spatial_ny=ny3,
        spatial_frequency_grid=[0j] * (nx3 * ny3),
        potential_grid=[0j] * (nx3 * ny3),
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0, 0.0]),
    )
    out_3d = api.propagate(cfg_3d, [0j] * n3, 1, cpu_opts).records
    assert len(out_3d) == 1
    assert len(out_3d[0]) == n3
    print("test_python_api_smoke: explicit 3+1D layout propagation returned expected shape.")

    bad_cfg_2d = prepare_sim_config(
        nxy,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.1,
        min_step_size=0.001,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=0.001,
        frequency_grid=[0j] * nxy,
        spatial_nx=nx + 1,
        spatial_ny=ny,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0]),
    )
    try:
        api.propagate(bad_cfg_2d, [0j] * nxy, 1, cpu_opts)
        raise AssertionError("expected invalid flattened shape to fail")
    except RuntimeError as exc:
        assert "status=1" in str(exc)
    print("test_python_api_smoke: invalid flattened XY shape rejected as expected.")

    pulse = PulseSpec(
        samples=input_field,
        delta_time=0.001,
    )
    sim_result = api.propagate(
        pulse,
        "gvd",
        "kerr",
        propagation_distance=0.02,
        preset="balanced",
        exec_options=cpu_opts,
    )
    assert len(sim_result.records) == 128
    assert len(sim_result.records[0]) == n
    assert len(sim_result.z_axis) == 128
    assert _max_abs_diff(sim_result.final, sim_result.records[-1]) == 0.0
    print("test_python_api_smoke: high-level propagate facade returned dense output.")

    explicit_result = api.propagate(
        pulse,
        OperatorSpec(expr="i*beta2*w*w", params={"beta2": -0.5}),
        OperatorSpec(expr="i*gamma*A*I", params={"gamma": 1.0}),
        propagation_distance=0.02,
        records=16,
        exec_options=cpu_opts,
    )
    assert len(explicit_result.records) == 16
    assert len(explicit_result.records[0]) == n
    print("test_python_api_smoke: explicit operator override path returned expected shape.")

    nt_c = 4
    nx_c = 4
    ny_c = 2
    n_c = nt_c * nx_c * ny_c
    pulse_coupled = PulseSpec(
        samples=[0j] * n_c,
        delta_time=0.001,
        pulse_period=1.0,
        time_nt=nt_c,
        frequency_grid=[0j] * nt_c,
        spatial_nx=nx_c,
        spatial_ny=ny_c,
        spatial_frequency_grid=[0j] * (nx_c * ny_c),
        potential_grid=[0j] * (nx_c * ny_c),
    )
    coupled_result = api.propagate(
        pulse_coupled,
        OperatorSpec(expr="i*beta2*w*w-loss", params={"beta2": 0.0, "loss": 0.0}),
        OperatorSpec(expr="i*A*(gamma*I + V)", params={"gamma": 0.0}),
        transverse_operator=OperatorSpec(expr="i*beta_t*w", params={"beta_t": 0.0}),
        propagation_distance=0.01,
        records=2,
        exec_options=cpu_opts,
    )
    assert len(coupled_result.records) == 2
    assert len(coupled_result.records[0]) == n_c
    assert coupled_result.meta["coupled"] is True
    print("test_python_api_smoke: coupled transverse propagate returned expected shape.")

    try:
        api.propagate(
            pulse,
            OperatorSpec(expr="0", fn=lambda A, w: 0.0),  # noqa: E731
            "kerr",
            propagation_distance=0.02,
            exec_options=cpu_opts,
        )
        raise AssertionError("expected mixed expr/fn operator to fail")
    except ValueError as exc:
        assert "cannot define both expr and fn" in str(exc)
    print("test_python_api_smoke: invalid mixed operator spec rejected as expected.")


if __name__ == "__main__":
    main()
