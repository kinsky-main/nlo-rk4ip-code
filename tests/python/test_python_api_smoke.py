import math

from nlolib import (
    NLOLIB_PROGRESS_STREAM_BOTH,
    NLOLIB_PROGRESS_STREAM_STDERR,
    NLOLIB_STATUS_ABORTED,
    VECTOR_BACKEND_AUTO,
    VECTOR_BACKEND_CPU,
    VECTOR_BACKEND_VULKAN,
    TENSOR_LAYOUT_XYT_T_FAST,
    NLolib,
    OperatorSpec,
    PropagationAbortedError,
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
    api.set_progress_stream(NLOLIB_PROGRESS_STREAM_BOTH)
    api.set_progress_stream(NLOLIB_PROGRESS_STREAM_STDERR)
    api.clear_log_buffer()
    print("test_python_api_smoke: runtime log API configured.")

    cpu_opts = default_execution_options(VECTOR_BACKEND_CPU)
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

    auto_opts = default_execution_options(VECTOR_BACKEND_AUTO)
    identity_distance = 0.02
    identity_cfg = prepare_sim_config(
        n,
        propagation_distance=identity_distance,
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

    vulkan_opts = default_execution_options(VECTOR_BACKEND_VULKAN)
    explicit_vulkan_records = api.propagate(identity_cfg, gaussian, 3, vulkan_opts).records
    assert len(explicit_vulkan_records) == 3
    assert len(explicit_vulkan_records[0]) == n
    print("test_python_api_smoke: explicit VULKAN defaults now resolve a usable backend.")

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

    callback_calls = {"count": 0}

    def abort_on_first_progress(info):
        callback_calls["count"] += 1
        return False

    try:
        api.propagate(
            identity_cfg,
            gaussian,
            3,
            auto_opts,
            capture_step_history=True,
            step_history_capacity=128,
            progress_callback=abort_on_first_progress,
        )
        raise AssertionError("expected propagate to abort via progress callback")
    except PropagationAbortedError as exc:
        assert callback_calls["count"] > 0
        assert int(exc.result.meta.get("status", -1)) == NLOLIB_STATUS_ABORTED
        assert int(exc.result.meta.get("records_written", 0)) >= 0
    print("test_python_api_smoke: progress callback can abort propagation cleanly.")

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
        tensor_nt=1,
        tensor_nx=nx,
        tensor_ny=ny,
        tensor_layout=TENSOR_LAYOUT_XYT_T_FAST,
        frequency_grid=[0j],
        potential_grid=[1 + 0j] * nxy,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0]),
    )
    out_2d = api.propagate(cfg_2d, [0j] * nxy, 1, cpu_opts).records
    assert len(out_2d) == 1
    assert len(out_2d[0]) == nxy
    print("test_python_api_smoke: tensor 2D propagation returned expected shape.")

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
        tensor_nt=nt,
        tensor_nx=nx3,
        tensor_ny=ny3,
        tensor_layout=TENSOR_LAYOUT_XYT_T_FAST,
        frequency_grid=[0j] * nt,
        potential_grid=[0j] * n3,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0, 0.0]),
    )
    out_3d = api.propagate(cfg_3d, [0j] * n3, 1, cpu_opts).records
    assert len(out_3d) == 1
    assert len(out_3d[0]) == n3
    print("test_python_api_smoke: explicit 3+1D layout propagation returned expected shape.")

    try:
        _ = prepare_sim_config(
            nxy,
            propagation_distance=0.0,
            starting_step_size=0.01,
            max_step_size=0.1,
            min_step_size=0.001,
            error_tolerance=1e-6,
            pulse_period=1.0,
            delta_time=0.001,
            tensor_nt=2,
            tensor_nx=nx,
            tensor_ny=ny,
            frequency_grid=[0j, 0j],
            runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0]),
        )
        raise AssertionError("expected inconsistent tensor shape to fail")
    except ValueError as exc:
        assert "tensor_nt * tensor_nx * tensor_ny must match num_time_samples" in str(exc)
    print("test_python_api_smoke: inconsistent tensor shape rejected as expected.")

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
        tensor_nt=nt_c,
        tensor_nx=nx_c,
        tensor_ny=ny_c,
        tensor_layout=TENSOR_LAYOUT_XYT_T_FAST,
        frequency_grid=[0j] * nt_c,
        potential_grid=[0j] * n_c,
    )
    coupled_result = api.propagate(
        pulse_coupled,
        OperatorSpec(expr="i*(beta2*wt*wt + beta_t*(kx*kx + ky*ky))", params={"beta2": 0.0, "beta_t": 0.0}),
        OperatorSpec(expr="i*A*(gamma*I + V)", params={"gamma": 0.0}),
        propagation_distance=0.01,
        records=2,
        exec_options=cpu_opts,
    )
    assert len(coupled_result.records) == 2
    assert len(coupled_result.records[0]) == n_c
    assert coupled_result.meta["coupled"] is True
    print("test_python_api_smoke: coupled tensor propagate returned expected shape.")

    dense_record_count = 257
    dense_history_result = api.propagate(
        identity_cfg,
        gaussian,
        dense_record_count,
        cpu_opts,
        capture_step_history=True,
        step_history_capacity=20000,
    )
    dense_history = dense_history_result.meta.get("step_history")
    assert isinstance(dense_history, dict)
    dense_steps = [float(v) for v in dense_history.get("step_size", [])]
    assert len(dense_steps) > 0
    dense_spacing = float(identity_distance) / float(dense_record_count - 1)
    assert max(dense_steps) > (4.0 * dense_spacing)
    assert len(dense_history.get("z", [])) < dense_record_count
    assert len(dense_history_result.records) == dense_record_count
    print("test_python_api_smoke: adaptive step sizes are no longer capped by record spacing.")

    fixed_identity_cfg = prepare_sim_config(
        n,
        propagation_distance=0.01,
        starting_step_size=1e-3,
        max_step_size=1e-3,
        min_step_size=1e-3,
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
    fixed_aligned = api.propagate(
        fixed_identity_cfg,
        gaussian,
        11,
        cpu_opts,
        capture_step_history=True,
        step_history_capacity=128,
    )
    assert len(fixed_aligned.records) == 11
    assert len(fixed_aligned.z_axis) == 11
    assert int(fixed_aligned.meta.get("records_written", -1)) == 11
    fixed_history = fixed_aligned.meta.get("step_history")
    assert isinstance(fixed_history, dict)
    fixed_step_sizes = [float(v) for v in fixed_history.get("step_size", [])]
    fixed_next_sizes = [float(v) for v in fixed_history.get("next_step_size", [])]
    fixed_errors = [float(v) for v in fixed_history.get("error", [])]
    assert len(fixed_step_sizes) > 0
    assert all(abs(v) <= 1e-15 for v in fixed_errors)
    count = min(len(fixed_step_sizes), len(fixed_next_sizes))
    for i in range(count):
        assert abs(fixed_step_sizes[i] - fixed_next_sizes[i]) <= 1e-15

    fixed_oversubscribed = api.propagate(fixed_identity_cfg, gaussian, 64, cpu_opts)
    assert len(fixed_oversubscribed.records) == 11
    assert len(fixed_oversubscribed.z_axis) == 11
    assert int(fixed_oversubscribed.meta.get("records_requested", -1)) == 64
    assert int(fixed_oversubscribed.meta.get("records_written", -1)) == 11
    print("test_python_api_smoke: fixed-step oversubscribed records are clamped to step-aligned samples.")

    try:
        api.propagate(
            pulse,
            "gvd",
            "kerr",
            transverse_operator=OperatorSpec(expr="i*beta_t*w", params={"beta_t": 0.0}),
            propagation_distance=0.01,
            records=2,
            exec_options=cpu_opts,
        )
        raise AssertionError("expected removed transverse_operator kwarg to fail")
    except TypeError as exc:
        assert "transverse_operator has been removed" in str(exc)
    print("test_python_api_smoke: removed transverse_operator kwarg rejected as expected.")

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
