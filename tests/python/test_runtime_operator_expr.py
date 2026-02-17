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


def main():
    api = NLolib()
    opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)
    test_dispersion_expression_matches_legacy(api, opts)
    test_nonlinear_expression_matches_legacy(api, opts)
    print("test_runtime_operator_expr: runtime expressions match legacy operators.")


if __name__ == "__main__":
    main()
