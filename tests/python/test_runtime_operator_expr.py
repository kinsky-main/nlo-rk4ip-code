import math
import random

from nlolib_ctypes import (
    NLO_VECTOR_BACKEND_CPU,
    NLolib,
    RuntimeOperators,
    default_execution_options,
    prepare_sim_config,
)


def _omega_grid_unshifted(n, dt):
    two_pi = 2.0 * math.pi
    return [
        two_pi
        * (
            float(i) / (float(n) * dt)
            if i <= (n - 1) // 2
            else -float(n - i) / (float(n) * dt)
        )
        for i in range(n)
    ]


def _random_input_field(n, seed):
    rng = random.Random(seed)
    return [complex(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)) for _ in range(n)]


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def test_dispersion_expression_matches_legacy(api, opts):
    n = 256
    dt = 0.02
    beta2 = -0.04
    omega = _omega_grid_unshifted(n, dt)
    input_field = _random_input_field(n, seed=7)

    common = dict(
        propagation_distance=0.2,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
    )

    legacy_cfg = prepare_sim_config(
        n,
        gamma=0.0,
        betas=[0.0, 0.0, beta2],
        alpha=0.0,
        **common,
    )
    runtime_cfg = prepare_sim_config(
        n,
        gamma=0.0,
        betas=[],
        alpha=0.0,
        runtime=RuntimeOperators(
            dispersion_expr="exp(i*c0*w*w)",
            nonlinear_expr=None,
            constants=[beta2 / 2.0],
        ),
        **common,
    )

    legacy_final = api.propagate(legacy_cfg, input_field, 2, opts)[1]
    runtime_final = api.propagate(runtime_cfg, input_field, 2, opts)[1]
    err = _max_abs_diff(legacy_final, runtime_final)
    assert err <= 2e-8, f"dispersion runtime mismatch: err={err}"


def test_nonlinear_expression_matches_legacy(api, opts):
    n = 256
    dt = 0.01
    gamma = 0.6
    input_field = _random_input_field(n, seed=11)

    common = dict(
        propagation_distance=0.08,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[0j] * n,
    )

    legacy_cfg = prepare_sim_config(
        n,
        gamma=gamma,
        betas=[],
        alpha=0.0,
        **common,
    )
    runtime_cfg = prepare_sim_config(
        n,
        gamma=0.0,
        betas=[],
        alpha=0.0,
        runtime=RuntimeOperators(
            dispersion_expr=None,
            nonlinear_expr="i*c0*I",
            constants=[gamma],
        ),
        **common,
    )

    legacy_final = api.propagate(legacy_cfg, input_field, 2, opts)[1]
    runtime_final = api.propagate(runtime_cfg, input_field, 2, opts)[1]
    err = _max_abs_diff(legacy_final, runtime_final)
    assert err <= 2e-8, f"nonlinear runtime mismatch: err={err}"


def test_dispersion_callable_matches_string(api, opts):
    n = 256
    dt = 0.02
    beta2 = -0.04
    scale = beta2 / 2.0
    omega = _omega_grid_unshifted(n, dt)
    input_field = _random_input_field(n, seed=17)

    common = dict(
        propagation_distance=0.2,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
    )

    string_cfg = prepare_sim_config(
        n,
        gamma=0.0,
        betas=[],
        alpha=0.0,
        runtime=RuntimeOperators(
            dispersion_expr="exp(i*c0*w*w)",
            constants=[scale],
        ),
        **common,
    )

    dispersion_fn = lambda w: math.exp((1j * scale) * (w**2))  # noqa: E731
    callable_cfg = prepare_sim_config(
        n,
        gamma=0.0,
        betas=[],
        alpha=0.0,
        runtime=RuntimeOperators(
            dispersion_fn=dispersion_fn,
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts)[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts)[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"dispersion callable mismatch: err={err}"


def test_nonlinear_callable_matches_string(api, opts):
    n = 256
    dt = 0.01
    gamma = 0.6
    input_field = _random_input_field(n, seed=23)

    common = dict(
        propagation_distance=0.08,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[0j] * n,
    )

    string_cfg = prepare_sim_config(
        n,
        gamma=0.0,
        betas=[],
        alpha=0.0,
        runtime=RuntimeOperators(
            nonlinear_expr="i*c0*I",
            constants=[gamma],
        ),
        **common,
    )

    nonlinear_fn = lambda A, I: (1j * gamma) * I  # noqa: E731
    callable_cfg = prepare_sim_config(
        n,
        gamma=0.0,
        betas=[],
        alpha=0.0,
        runtime=RuntimeOperators(
            nonlinear_fn=nonlinear_fn,
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts)[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts)[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"nonlinear callable mismatch: err={err}"


def test_extended_runtime_operators_execute(api, opts):
    n = 128
    dt = 0.02
    omega = _omega_grid_unshifted(n, dt)
    input_field = _random_input_field(n, seed=31)

    cfg = prepare_sim_config(
        n,
        gamma=0.0,
        betas=[],
        alpha=0.0,
        propagation_distance=0.03,
        starting_step_size=5e-4,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-8,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
        runtime=RuntimeOperators(
            dispersion_expr="exp(i*c0*(w^2)/(c1+1.0))",
            nonlinear_expr="sin(A)+cos(A)+log(I+1.0)+sqrt(I+1.0)",
            constants=[0.01, 2.0],
        ),
    )

    records = api.propagate(cfg, input_field, 2, opts)
    assert len(records) == 2
    assert len(records[0]) == n


def main():
    api = NLolib()
    opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)
    test_dispersion_expression_matches_legacy(api, opts)
    test_nonlinear_expression_matches_legacy(api, opts)
    test_dispersion_callable_matches_string(api, opts)
    test_nonlinear_callable_matches_string(api, opts)
    test_extended_runtime_operators_execute(api, opts)
    print("test_runtime_operator_expr: runtime expressions match legacy operators.")


if __name__ == "__main__":
    main()
