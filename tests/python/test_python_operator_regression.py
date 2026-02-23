import cmath
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


def _centered_time_grid(n, dt):
    mid = n // 2
    return [float(i - mid) * dt for i in range(n)]


def _random_input_field(n, seed):
    rng = random.Random(seed)
    return [complex(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)) for _ in range(n)]


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _gaussian_with_phase(t, sigma, d):
    out = [0j] * len(t)
    for i, ti in enumerate(t):
        amp = math.exp(-((ti / sigma) * (ti / sigma)))
        phase = -d * ti
        out[i] = amp * complex(math.cos(phase), math.sin(phase))
    return out


def _intensity_centroid(t, field):
    weighted = 0.0
    total = 0.0
    for ti, ai in zip(t, field):
        ii = (ai.real * ai.real) + (ai.imag * ai.imag)
        weighted += ti * ii
        total += ii

    if total <= 0.0:
        return 0.0
    return weighted / total


def _sech(x):
    return 1.0 / math.cosh(x)


def _second_order_soliton_analytic(t_value, z, beta2, t0):
    ld = (t0 * t0) / abs(beta2)
    xi = z / ld
    numerator = 4.0 * (
        math.cosh(3.0 * t_value) + 3.0 * cmath.exp(4.0j * xi) * math.cosh(t_value)
    ) * cmath.exp(0.5j * xi)
    denominator = math.cosh(4.0 * t_value) + 4.0 * math.cosh(2.0 * t_value) + 3.0 * math.cos(4.0 * xi)
    return numerator / denominator


def test_dispersion_factor_callable_matches_string(api, opts):
    n = 192
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

    string_final = api.propagate(string_cfg, input_field, 2, opts).records[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts).records[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"dispersion factor callable mismatch: err={err}"


def test_nonlinear_callable_matches_string(api, opts):
    n = 192
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

    string_final = api.propagate(string_cfg, input_field, 2, opts).records[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts).records[1]
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

    records = api.propagate(cfg, input_field, 2, opts).records
    assert len(records) == 2
    assert len(records[0]) == n


def test_transverse_runtime_callable_matches_string(api, opts):
    nt = 4
    nx = 6
    ny = 4
    n = nt * nx * ny
    coef = -0.015
    input_field = _random_input_field(n, seed=91)
    k2 = [complex(float(i % (nx * ny)), 0.0) for i in range(nx * ny)]

    common = dict(
        propagation_distance=0.01,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(nt) * 0.02,
        delta_time=0.02,
        time_nt=nt,
        frequency_grid=[0j] * nt,
        spatial_nx=nx,
        spatial_ny=ny,
        spatial_frequency_grid=k2,
        potential_grid=[0j] * (nx * ny),
    )

    string_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            transverse_factor_expr="i*c3*w",
            transverse_expr="exp(h*D)",
            nonlinear_expr="0",
            constants=[0.0, 0.0, 0.0, coef],
        ),
        **common,
    )
    callable_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            transverse_factor_fn=lambda A, w: (1j * coef) * w,  # noqa: E731
            transverse_fn=lambda A, D, h: cmath.exp(h * D),  # noqa: E731
            nonlinear_expr="0",
            constants=[0.0, 0.0, 0.0, 0.0],
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts).records[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts).records[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"transverse callable mismatch: err={err}"


def test_beta_sum_callable_matches_string(api, opts):
    n = 192
    dt = 0.02
    beta2 = 0.04
    beta3 = -0.003
    beta4 = 0.0002
    omega = _omega_grid_unshifted(n, dt)
    input_field = _random_input_field(n, seed=71)

    common = dict(
        propagation_distance=0.08,
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
            dispersion_factor_expr="i*(c0*(w^2)+c1*(w^3)+c2*(w^4))",
            constants=[beta2, beta3, beta4, 0.0],
        ),
        **common,
    )

    beta_sum_fn = lambda A, w: (1j * (beta2 * (w**2) + beta3 * (w**3) + beta4 * (w**4)))  # noqa: E731
    callable_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_fn=beta_sum_fn,
            constants=[0.0, 0.0, 0.0, 0.0],
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts).records[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts).records[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-6, f"beta-sum callable mismatch: err={err}"


def test_diffraction_callable_matches_string(api, opts):
    nt = 4
    nx = 6
    ny = 4
    n = nt * nx * ny
    beta_t = -0.018
    input_field = _random_input_field(n, seed=101)
    k2 = [complex(float((i % (nx * ny)) + 1), 0.0) for i in range(nx * ny)]

    common = dict(
        propagation_distance=0.01,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(nt) * 0.02,
        delta_time=0.02,
        time_nt=nt,
        frequency_grid=[0j] * nt,
        spatial_nx=nx,
        spatial_ny=ny,
        spatial_frequency_grid=k2,
        potential_grid=[0j] * (nx * ny),
    )

    string_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            transverse_factor_expr="i*c3*w",
            transverse_expr="exp(h*D)",
            nonlinear_expr="0",
            constants=[0.0, 0.0, 0.0, beta_t],
        ),
        **common,
    )
    callable_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            transverse_factor_fn=lambda A, w: (1j * beta_t) * w,  # noqa: E731
            transverse_fn=lambda A, D, h: cmath.exp(h * D),  # noqa: E731
            nonlinear_expr="0",
            constants=[0.0, 0.0, 0.0, 0.0],
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts).records[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts).records[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"diffraction callable mismatch: err={err}"


def test_raman_like_nonlinear_callable_matches_string(api, opts):
    nt = 4
    nx = 4
    ny = 4
    n = nt * nx * ny
    gamma = 0.015
    f_r = 0.18
    beta_t = -0.01
    input_field = _random_input_field(n, seed=131)
    k2 = [complex(float((i % (nx * ny)) + 1), 0.0) for i in range(nx * ny)]
    potential = [complex(0.02 * float(i + 1), 0.0) for i in range(nx * ny)]

    common = dict(
        propagation_distance=0.008,
        starting_step_size=8e-4,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(nt) * 0.02,
        delta_time=0.02,
        time_nt=nt,
        frequency_grid=[0j] * nt,
        spatial_nx=nx,
        spatial_ny=ny,
        spatial_frequency_grid=k2,
        potential_grid=potential,
    )

    string_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            transverse_factor_expr="i*c3*w",
            transverse_expr="exp(h*D)",
            nonlinear_expr="i*A*(c0*(1.0-c1)*I + c0*c1*V)",
            constants=[gamma, f_r, 0.0, beta_t],
        ),
        **common,
    )
    callable_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            transverse_factor_fn=lambda A, w: (1j * beta_t) * w,  # noqa: E731
            transverse_fn=lambda A, D, h: cmath.exp(h * D),  # noqa: E731
            nonlinear_fn=lambda A, I, V: (1j * A) * (gamma * (1.0 - f_r) * I + gamma * f_r * V),  # noqa: E731
            constants=[0.0, 0.0, 0.0, 0.0],
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts).records[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts).records[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"Raman-like callable mismatch: err={err}"


def test_linear_drift_signed_prediction(api, opts):
    n = 384
    dt = 0.01
    sigma = 0.20
    beta2 = 0.05
    d = 12.0
    z_final = 0.5

    t = _centered_time_grid(n, dt)
    omega = _omega_grid_unshifted(n, dt)

    cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=5e-3,
        max_step_size=2e-2,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
        runtime=RuntimeOperators(constants=[0.5 * beta2, 0.0, 0.0]),
    )

    pulse_pos = _gaussian_with_phase(t, sigma, d)
    pulse_neg = _gaussian_with_phase(t, sigma, -d)

    final_pos = api.propagate(cfg, pulse_pos, 2, opts).records[1]
    final_neg = api.propagate(cfg, pulse_neg, 2, opts).records[1]

    shift_pos = _intensity_centroid(t, final_pos) - _intensity_centroid(t, pulse_pos)
    shift_neg = _intensity_centroid(t, final_neg) - _intensity_centroid(t, pulse_neg)

    expected_pos = beta2 * d * z_final
    expected_neg = beta2 * (-d) * z_final
    assert abs(shift_pos) > 0.05
    assert abs(shift_neg) > 0.05
    assert shift_pos * shift_neg < 0.0

    rel_err_pos = abs(shift_pos - expected_pos) / max(abs(expected_pos), 1e-12)
    rel_err_neg = abs(shift_neg - expected_neg) / max(abs(expected_neg), 1e-12)
    assert rel_err_pos <= 0.30
    assert rel_err_neg <= 0.30

    mag_sym = abs(abs(shift_pos) - abs(shift_neg)) / max(abs(shift_pos), abs(shift_neg))
    assert mag_sym <= 0.20


def test_second_order_soliton_intensity_error(api, opts):
    beta2 = -0.01
    gamma = 0.01
    alpha = 0.0
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    p0 = (2**2) * abs(beta2) / (gamma * t0 * t0)
    z_final = 0.506

    n = 224
    tmax = 8.0 * t0
    times = [(-tmax) + (2.0 * tmax) * float(i) / float(n - 1) for i in range(n)]
    t_dimless = [ti / t0 for ti in times]
    dt = times[1] - times[0]
    omega = _omega_grid_unshifted(n, dt)

    u0 = [2.0 * _sech(ti) for ti in t_dimless]
    a0 = [complex(math.sqrt(p0) * ui, 0.0) for ui in u0]

    cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=2e-4,
        max_step_size=1e-2,
        min_step_size=1e-7,
        error_tolerance=1e-5,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
        runtime=RuntimeOperators(constants=[0.5 * beta2, 0.0, gamma]),
    )

    records = api.propagate(cfg, a0, 2, opts).records
    final_field = records[1]
    assert all(math.isfinite(v.real) and math.isfinite(v.imag) for v in final_field)

    u_num = [val / math.sqrt(p0) for val in final_field]
    u_true = [_second_order_soliton_analytic(ti, z_final, beta2, t0) for ti in t_dimless]
    intensity_num = [abs(v) ** 2 for v in u_num]
    intensity_true = [abs(v) ** 2 for v in u_true]
    epsilon = sum(abs(a - b) for a, b in zip(intensity_num, intensity_true)) / float(len(intensity_num))
    epsilon /= max(intensity_true)
    assert epsilon <= 1e-2, f"second-order soliton intensity error too high: {epsilon}"


def main():
    api = NLolib()
    opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)
    test_dispersion_factor_callable_matches_string(api, opts)
    test_nonlinear_callable_matches_string(api, opts)
    test_field_first_callable_signature_enforced()
    test_extended_runtime_operators_execute(api, opts)
    test_transverse_runtime_callable_matches_string(api, opts)
    test_beta_sum_callable_matches_string(api, opts)
    test_diffraction_callable_matches_string(api, opts)
    test_raman_like_nonlinear_callable_matches_string(api, opts)
    test_linear_drift_signed_prediction(api, opts)
    test_second_order_soliton_intensity_error(api, opts)
    print("test_python_operator_regression: runtime-operator propagation regressions validated.")


if __name__ == "__main__":
    main()
