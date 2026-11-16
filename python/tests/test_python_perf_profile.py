import math

import nlolib


def main() -> None:
    api = nlolib.NLolib()
    api.perf_profile_reset()
    api.perf_profile_set_enabled(True)
    assert api.perf_profile_is_enabled() is True

    n = 64
    dt = 0.01
    omega = [complex(v, 0.0) for v in (2.0 * math.pi * i / (n * dt) for i in range(n))]
    cfg = nlolib.prepare_sim_config(
        n,
        propagation_distance=0.02,
        starting_step_size=0.01,
        max_step_size=0.01,
        min_step_size=0.01,
        error_tolerance=1e-9,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=omega,
        runtime=nlolib.RuntimeOperators(
            linear_factor_expr="i*c0*wt*wt",
            nonlinear_expr="i*A*(c1*I)",
            constants=[0.02, 0.01],
        ),
    )
    field0 = [complex(math.exp(-((idx - 0.5 * n) / 10.0) ** 2), 0.0) for idx in range(n)]
    exec_options = nlolib.default_execution_options(nlolib.NLO_VECTOR_BACKEND_CPU)
    api.propagate(cfg, field0, 1, exec_options)
    snapshot = api.perf_profile_read()
    api.perf_profile_set_enabled(False)

    assert snapshot.dispersion_ms >= 0.0
    assert snapshot.nonlinear_ms >= 0.0
    assert snapshot.dispersion_calls > 0
    assert snapshot.nonlinear_calls > 0
    assert math.isfinite(snapshot.dispersion_ms)
    assert math.isfinite(snapshot.nonlinear_ms)
    print("test_python_perf_profile: perf profile APIs returned finite counters after propagation.")


if __name__ == "__main__":
    main()
