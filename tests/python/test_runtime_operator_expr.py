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


def test_default_runtime_matches_explicit_defaults(api, opts):
    n = 256
    dt = 0.02
    omega = _omega_grid_unshifted(n, dt)
    input_field = _random_input_field(n, seed=5)

    common = dict(
        propagation_distance=0.1,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
    )

    default_cfg = prepare_sim_config(n, **common)
    explicit_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(constants=[-0.5, 0.0, 1.0]),
        **common,
    )

    default_final = api.propagate(default_cfg, input_field, 2, opts)[1]
    explicit_final = api.propagate(explicit_cfg, input_field, 2, opts)[1]
    err = _max_abs_diff(default_final, explicit_final)
    assert err <= 2e-8, f"default runtime mismatch: err={err}"


def test_dispersion_factor_callable_matches_string(api, opts):
    n = 256
    dt = 0.02
    scale = -0.02
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
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w",
            constants=[scale, 0.0, 0.0],
        ),
        **common,
    )

    dispersion_factor_fn = lambda A, w: (1j * scale) * (w**2)  # noqa: E731
    callable_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_fn=dispersion_factor_fn,
            constants=[0.0, 0.0, 0.0],
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts)[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts)[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"dispersion factor callable mismatch: err={err}"


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
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="i*c0*I",
            constants=[gamma, 0.0, 0.0],
        ),
        **common,
    )

    nonlinear_fn = lambda A, I: (1j * gamma) * I  # noqa: E731
    callable_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_fn=nonlinear_fn,
            constants=[0.0, 0.0, 0.0],
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts)[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts)[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"nonlinear callable mismatch: err={err}"


def test_field_first_callable_signature_enforced():
    n = 64
    dt = 0.05
    omega = _omega_grid_unshifted(n, dt)
    try:
        prepare_sim_config(
            n,
            propagation_distance=0.01,
            starting_step_size=1e-3,
            max_step_size=2e-3,
            min_step_size=1e-5,
            error_tolerance=1e-8,
            pulse_period=float(n) * dt,
            delta_time=dt,
            frequency_grid=[complex(w, 0.0) for w in omega],
            runtime=RuntimeOperators(dispersion_factor_fn=lambda: 1.0),  # noqa: E731
        )
        raise AssertionError("expected field-first callable signature check to fail")
    except ValueError as exc:
        assert "dispersion_factor callable" in str(exc)


def test_extended_runtime_operators_execute(api, opts):
    n = 128
    dt = 0.02
    omega = _omega_grid_unshifted(n, dt)
    input_field = _random_input_field(n, seed=31)

    cfg = prepare_sim_config(
        n,
        propagation_distance=0.03,
        starting_step_size=5e-4,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-8,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*(w^2)/(c1+1.0)",
            dispersion_expr="exp(h*D)",
            nonlinear_expr="sin(A)+cos(A)+log(I+1.0)+sqrt(I+1.0)",
            constants=[0.01, 2.0, 0.05],
        ),
    )

    records = api.propagate(cfg, input_field, 2, opts)
    assert len(records) == 2
    assert len(records[0]) == n


def main():
    api = NLolib()
    opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)
    test_default_runtime_matches_explicit_defaults(api, opts)
    test_dispersion_factor_callable_matches_string(api, opts)
    test_nonlinear_callable_matches_string(api, opts)
    test_field_first_callable_signature_enforced()
    test_extended_runtime_operators_execute(api, opts)
    print("test_runtime_operator_expr: operator-only runtime expressions validated.")


if __name__ == "__main__":
    main()
