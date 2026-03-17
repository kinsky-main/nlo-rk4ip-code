import cmath
import math
import random
import numpy as np

from nlolib import (
    NLO_NONLINEAR_MODEL_EXPR,
    NLO_NONLINEAR_MODEL_KERR_RAMAN,
    NLOLIB_LOG_LEVEL_ERROR,
    NLOLIB_LOG_LEVEL_WARN,
    NLO_VECTOR_BACKEND_AUTO,
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


def _l2_norm(values):
    acc = 0.0
    for value in values:
        acc += (value.real * value.real) + (value.imag * value.imag)
    return math.sqrt(acc)


def _relative_l2_error(a, b):
    diff = [x - y for x, y in zip(a, b)]
    denom = max(_l2_norm(b), 1e-15)
    return _l2_norm(diff) / denom


def _filtered_relative_l2_error(a, b, min_relative_intensity=1.0e-8):
    if len(a) != len(b):
        raise ValueError("a and b must have the same length.")
    if min_relative_intensity < 0.0:
        raise ValueError("min_relative_intensity must be >= 0.")

    intensities = [(v.real * v.real) + (v.imag * v.imag) for v in b]
    peak_intensity = max(intensities) if intensities else 0.0
    if not (peak_intensity > 0.0):
        return _relative_l2_error(a, b)

    threshold = peak_intensity * float(min_relative_intensity)
    filtered_a = []
    filtered_b = []
    for av, bv, intensity in zip(a, b, intensities):
        if intensity >= threshold:
            filtered_a.append(av)
            filtered_b.append(bv)
    if len(filtered_b) <= 0:
        return _relative_l2_error(a, b)
    return _relative_l2_error(filtered_a, filtered_b)


def _fit_loglog_slope(x_values, y_values):
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if len(x_values) < 2:
        raise ValueError("at least two samples are required to fit a slope.")

    lx = [math.log(x) for x in x_values]
    ly = [math.log(y) for y in y_values]
    mean_x = sum(lx) / float(len(lx))
    mean_y = sum(ly) / float(len(ly))

    var_x = 0.0
    cov_xy = 0.0
    for vx, vy in zip(lx, ly):
        dx = vx - mean_x
        dy = vy - mean_y
        var_x += dx * dx
        cov_xy += dx * dy

    if var_x <= 0.0:
        raise ValueError("cannot fit slope with zero variance in x_values.")
    return cov_xy / var_x


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


def _second_order_soliton_case(n, window_multiple_t0):
    beta2 = -0.01
    gamma = 0.01
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    p0 = abs(beta2) / (gamma * t0 * t0)
    ld = (t0 * t0) / abs(beta2)
    z_final = 0.5 * math.pi * ld

    dt = (float(window_multiple_t0) * t0) / float(n)
    times = _centered_time_grid(n, dt)
    tau = [ti / t0 for ti in times]
    omega = _omega_grid_unshifted(n, dt)
    a0 = [complex(math.sqrt(p0) * (2.0 * _sech(ti)), 0.0) for ti in tau]
    u_true = [_second_order_soliton_analytic(ti, z_final, beta2, t0) for ti in tau]
    return {
        "beta2": beta2,
        "gamma": gamma,
        "t0": t0,
        "p0": p0,
        "ld": ld,
        "z_final": z_final,
        "dt": dt,
        "tau": tau,
        "omega": omega,
        "a0": a0,
        "u_true": u_true,
    }


def _second_order_intensity_error(final_field, p0, reference_u):
    u_num = [val / math.sqrt(p0) for val in final_field]
    intensity_num = [abs(v) ** 2 for v in u_num]
    intensity_true = [abs(v) ** 2 for v in reference_u]
    diff_sq = 0.0
    ref_sq = 0.0
    for a, b in zip(intensity_num, intensity_true):
        d = a - b
        diff_sq += d * d
        ref_sq += b * b
    if ref_sq <= 0.0:
        return float("nan")
    return math.sqrt(diff_sq / ref_sq)


def _exact_linear_temporal_propagation(initial_field, omega, c0, c1, z):
    spectrum = np.fft.fft(np.asarray(initial_field, dtype=np.complex128))
    phase = np.exp(((1.0j * float(c0) * (np.asarray(omega, dtype=np.float64) ** 2)) - float(c1)) * float(z))
    return np.fft.ifft(spectrum * phase)


def _exact_nonlinear_phase_rotation(initial_field, gamma, z):
    field = np.asarray(initial_field, dtype=np.complex128)
    intensity = np.abs(field) ** 2
    return field * np.exp(1.0j * float(gamma) * intensity * float(z))


def _default_raman_response(n, dt, tau1, tau2):
    coef = (tau1 * tau1 + tau2 * tau2) / (tau1 * tau2 * tau2)
    values = [0.0] * n
    area = 0.0
    for i in range(n):
        t = float(i) * dt
        v = coef * math.exp(-t / tau2) * math.sin(t / tau1)
        values[i] = v
        area += v
    area *= dt
    if not (area > 0.0):
        raise ValueError("invalid Raman response normalization area")
    inv_area = 1.0 / area
    return [complex(v * inv_area, 0.0) for v in values]


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
            nonlinear_expr="i*c0*A*I",
            constants=[gamma, 0.0, 0.0],
        ),
        **common,
    )

    nonlinear_fn = lambda A, I: (1j * gamma) * A * I  # noqa: E731
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


def test_nonlinear_legacy_multiplier_warns(api, opts):
    n = 96
    dt = 0.01
    input_field = _random_input_field(n, seed=29)

    cfg = prepare_sim_config(
        n,
        propagation_distance=0.02,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[0j] * n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="i*c0*I",
            constants=[0.3, 0.0, 0.0],
        ),
    )

    api.set_log_buffer(64 * 1024)
    api.set_log_level(1)
    api.clear_log_buffer()
    _ = api.propagate(cfg, input_field, 2, opts)
    log_text = api.read_log_buffer(consume=True)
    assert "does not reference 'A'" in log_text


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


def test_linear_dispersion_aliases_are_mutually_exclusive():
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
            runtime=RuntimeOperators(
                linear_factor_expr="i*0.1*w*w",
                dispersion_factor_expr="i*0.1*w*w",
            ),
        )
        raise AssertionError("expected linear/dispersive alias exclusivity check to fail")
    except ValueError as exc:
        assert "aliases; provide only one" in str(exc)


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


def test_tensor_linear_factor_alias_matches(api, opts):
    nt = 4
    nx = 6
    ny = 4
    n = nt * nx * ny
    coef = -0.015
    input_field = _random_input_field(n, seed=91)

    common = dict(
        propagation_distance=0.01,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(nt) * 0.02,
        delta_time=0.02,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        frequency_grid=[0j] * nt,
        potential_grid=[0j] * n,
    )

    explicit_tensor_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            linear_factor_expr="i*c0*(kx*kx + ky*ky)",
            linear_expr="exp(h*D)",
            nonlinear_expr="0",
            constants=[coef, 0.0, 0.0, 0.0],
        ),
        **common,
    )
    alias_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*(kx*kx + ky*ky)",
            dispersion_expr="exp(h*D)",
            nonlinear_expr="0",
            constants=[coef, 0.0, 0.0, 0.0],
        ),
        **common,
    )

    explicit_final = api.propagate(explicit_tensor_cfg, input_field, 2, opts).records[1]
    alias_final = api.propagate(alias_cfg, input_field, 2, opts).records[1]
    err = _max_abs_diff(explicit_final, alias_final)
    assert err <= 2e-8, f"tensor linear factor alias mismatch: err={err}"


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


def test_tensor_diffraction_matches_fft_reference(api, opts):
    nt = 1
    nx = 20
    ny = 18
    n = nt * nx * ny
    beta_t = -0.018
    input_field = _random_input_field(n, seed=101)

    explicit_cfg = prepare_sim_config(
        n,
        propagation_distance=0.01,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=1.0,
        delta_time=1.0,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        frequency_grid=[0j] * nt,
        runtime=RuntimeOperators(
            linear_factor_expr="i*c0*(kx*kx + ky*ky)",
            linear_expr="exp(h*D)",
            nonlinear_expr="0",
            constants=[beta_t, 0.0, 0.0, 0.0],
        ),
    )
    split_cfg = prepare_sim_config(
        n,
        propagation_distance=0.01,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=1.0,
        delta_time=1.0,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        frequency_grid=[0j] * nt,
        runtime=RuntimeOperators(
            linear_factor_expr="i*(c0*kx*kx + c1*ky*ky)",
            linear_expr="exp(h*D)",
            nonlinear_expr="0",
            constants=[beta_t, beta_t, 0.0, 0.0],
        ),
    )

    explicit_final = api.propagate(explicit_cfg, input_field, 2, opts).records[1]
    split_final = api.propagate(split_cfg, input_field, 2, opts).records[1]
    rel = _relative_l2_error(explicit_final, split_final)
    assert rel <= 2e-8, f"tensor diffraction expression mismatch: rel={rel}"


def test_tensor_wt_only_matches_transverse_tiling(api, opts):
    nt = 64
    nx = 6
    ny = 4
    dt = 0.02
    beta2 = -0.018
    omega = _omega_grid_unshifted(nt, dt)
    t = _centered_time_grid(nt, dt)
    pulse = _gaussian_with_phase(t, sigma=0.18, d=5.0)

    common = dict(
        propagation_distance=0.03,
        starting_step_size=1e-3,
        max_step_size=3e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(nt) * dt,
        delta_time=dt,
        runtime=RuntimeOperators(
            linear_factor_expr="i*c0*wt*wt",
            linear_expr="exp(h*D)",
            nonlinear_expr="0",
            constants=[0.5 * beta2, 0.0, 0.0, 0.0],
        ),
    )

    ref_cfg = prepare_sim_config(
        nt,
        tensor_nt=nt,
        tensor_nx=1,
        tensor_ny=1,
        frequency_grid=[complex(w, 0.0) for w in omega],
        potential_grid=[0j] * nt,
        **common,
    )
    ref_final = api.propagate(ref_cfg, pulse, 2, opts).records[1]

    tiled_input = []
    for _ in range(nx * ny):
        tiled_input.extend(pulse)

    tiled_cfg = prepare_sim_config(
        nt * nx * ny,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        frequency_grid=[complex(w, 0.0) for w in omega],
        potential_grid=[0j] * (nt * nx * ny),
        **common,
    )
    tiled_final = api.propagate(tiled_cfg, tiled_input, 2, opts).records[1]

    tiled = [complex(v) for v in tiled_final]
    ref = [complex(v) for v in ref_final]
    max_rel = 0.0
    for xi in range(nx):
        for yi in range(ny):
            base = ((xi * ny) + yi) * nt
            slab = tiled[base : base + nt]
            rel = _relative_l2_error(slab, ref)
            if rel > max_rel:
                max_rel = rel

    assert max_rel <= 2e-6, f"wt-only tensor tiling mismatch: max_rel={max_rel}"


def test_raman_like_nonlinear_callable_matches_string(api, opts):
    nt = 4
    nx = 4
    ny = 4
    n = nt * nx * ny
    gamma = 0.015
    f_r = 0.18
    input_field = _random_input_field(n, seed=131)
    potential_xy = [complex(0.02 * float(i + 1), 0.0) for i in range(nx * ny)]
    potential = potential_xy * nt

    common = dict(
        propagation_distance=0.008,
        starting_step_size=8e-4,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(nt) * 0.02,
        delta_time=0.02,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        frequency_grid=[0j] * nt,
        potential_grid=potential,
    )

    string_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            linear_factor_expr="0",
            linear_expr="exp(h*D)",
            nonlinear_expr="i*A*(c0*(1.0-c1)*I + c0*c1*V)",
            constants=[gamma, f_r, 0.0, 0.0],
        ),
        **common,
    )
    callable_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            linear_factor_expr="0",
            linear_expr="exp(h*D)",
            nonlinear_fn=lambda A, I, V: (1j * A) * (gamma * (1.0 - f_r) * I + gamma * f_r * V),  # noqa: E731
            constants=[0.0, 0.0, 0.0, 0.0],
        ),
        **common,
    )

    string_final = api.propagate(string_cfg, input_field, 2, opts).records[1]
    callable_final = api.propagate(callable_cfg, input_field, 2, opts).records[1]
    err = _max_abs_diff(string_final, callable_final)
    assert err <= 2e-8, f"Raman-like callable mismatch: err={err}"


def test_kerr_raman_model_reduces_to_kerr_when_fraction_zero(api, opts):
    n = 256
    dt = 0.01
    beta2 = 0.02
    gamma = 0.7
    omega = _omega_grid_unshifted(n, dt)
    t = _centered_time_grid(n, dt)
    input_field = _gaussian_with_phase(t, sigma=0.24, d=6.0)

    common = dict(
        propagation_distance=0.06,
        starting_step_size=1e-3,
        max_step_size=3e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
    )

    kerr_expr_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w",
            nonlinear_expr="i*c1*A*I",
            constants=[0.5 * beta2, gamma, 0.0, 0.0],
            nonlinear_model=NLO_NONLINEAR_MODEL_EXPR,
        ),
        **common,
    )
    kerr_raman_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w",
            nonlinear_expr="0",
            constants=[0.5 * beta2, 0.0, 0.0, 0.0],
            nonlinear_model=NLO_NONLINEAR_MODEL_KERR_RAMAN,
            nonlinear_gamma=gamma,
            raman_fraction=0.0,
            shock_omega0=0.0,
        ),
        **common,
    )

    expr_final = api.propagate(kerr_expr_cfg, input_field, 2, opts).records[1]
    raman_final = api.propagate(kerr_raman_cfg, input_field, 2, opts).records[1]
    rel = _relative_l2_error(raman_final, expr_final)
    assert rel <= 2e-7, f"kerr_raman(f_r=0) mismatch vs Kerr expression: rel={rel}"


def test_kerr_raman_custom_response_matches_generated_default(api, opts):
    n = 256
    dt = 0.004
    beta2 = -0.015
    gamma = 1.1
    f_r = 0.18
    tau1 = 0.0122
    tau2 = 0.0320
    omega = _omega_grid_unshifted(n, dt)
    t = _centered_time_grid(n, dt)
    input_field = _gaussian_with_phase(t, sigma=0.14, d=3.0)
    response = _default_raman_response(n, dt, tau1, tau2)

    common = dict(
        propagation_distance=0.05,
        starting_step_size=8e-4,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
    )

    generated_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w",
            nonlinear_expr="0",
            constants=[0.5 * beta2, 0.0, 0.0, 0.0],
            nonlinear_model=NLO_NONLINEAR_MODEL_KERR_RAMAN,
            nonlinear_gamma=gamma,
            raman_fraction=f_r,
            raman_tau1=tau1,
            raman_tau2=tau2,
            shock_omega0=0.0,
        ),
        **common,
    )
    custom_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w",
            nonlinear_expr="0",
            constants=[0.5 * beta2, 0.0, 0.0, 0.0],
            nonlinear_model=NLO_NONLINEAR_MODEL_KERR_RAMAN,
            nonlinear_gamma=gamma,
            raman_fraction=f_r,
            raman_tau1=tau1,
            raman_tau2=tau2,
            shock_omega0=0.0,
            raman_response_time=response,
        ),
        **common,
    )

    generated_final = api.propagate(generated_cfg, input_field, 2, opts).records[1]
    custom_final = api.propagate(custom_cfg, input_field, 2, opts).records[1]
    rel = _relative_l2_error(custom_final, generated_final)
    assert rel <= 2e-7, f"custom Raman response mismatch vs generated default: rel={rel}"


def test_kerr_raman_rejects_coupled_mode(api, opts):
    nt = 8
    nx = 4
    ny = 2
    n = nt * nx * ny
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.02,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(nt) * 0.02,
        delta_time=0.02,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        frequency_grid=[0j] * nt,
        potential_grid=[0j] * n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="0",
            constants=[0.0, 0.0, 0.0, 0.0],
            nonlinear_model=NLO_NONLINEAR_MODEL_KERR_RAMAN,
            nonlinear_gamma=0.8,
            raman_fraction=0.18,
            raman_tau1=0.0122,
            raman_tau2=0.0320,
            shock_omega0=0.0,
        ),
    )

    field = _random_input_field(n, seed=211)
    try:
        api.propagate(cfg, field, 2, opts)
        raise AssertionError("expected coupled kerr_raman propagation to fail")
    except RuntimeError as exc:
        msg = str(exc)
        assert ("status=1" in msg) or ("status=2" in msg)


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


def test_adaptive_embedded_error_estimator_remains_stable(api, opts):
    n = 192
    dt = 0.004
    z_final = 0.06
    t = _centered_time_grid(n, dt)
    omega = _omega_grid_unshifted(n, dt)
    a0 = [complex(math.exp(-((ti / 0.10) ** 2)), 0.0) for ti in t]

    adaptive_cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=1e-3,
        max_step_size=8e-3,
        min_step_size=1e-5,
        error_tolerance=1e-6,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="i*c0*A*I",
            constants=[6.0, 0.0, 0.0, 0.0],
        ),
    )
    adaptive_result = api.propagate(
        adaptive_cfg,
        a0,
        2,
        opts,
        capture_step_history=True,
        step_history_capacity=4096,
    )
    adaptive_final = adaptive_result.records[1]
    assert all(math.isfinite(v.real) and math.isfinite(v.imag) for v in adaptive_final)

    step_history = adaptive_result.meta.get("step_history")
    assert isinstance(step_history, dict)
    step_sizes = [float(v) for v in step_history.get("step_size", [])]
    errors = [float(v) for v in step_history.get("error", [])]
    assert len(step_sizes) > 0
    assert all(math.isfinite(v) and v >= 0.0 for v in errors)

    reference_cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=2.5e-4,
        max_step_size=2.5e-4,
        min_step_size=2.5e-4,
        error_tolerance=1e-6,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="i*c0*A*I",
            constants=[6.0, 0.0, 0.0, 0.0],
        ),
    )
    reference_final = api.propagate(reference_cfg, a0, 2, opts).records[1]
    rel_err = _relative_l2_error(adaptive_final, reference_final)
    assert rel_err <= 0.15, f"adaptive h-dependent nonlinear run deviated too much: {rel_err}"


def test_adaptive_relative_error_is_amplitude_scale_invariant(api, opts):
    n = 160
    dt = 0.005
    z_final = 0.06
    t = _centered_time_grid(n, dt)
    omega = _omega_grid_unshifted(n, dt)
    base_field = [complex(math.exp(-((ti / 0.11) ** 2)), 0.0) for ti in t]
    scaled_field = [complex(1.0e3 * v.real, 1.0e3 * v.imag) for v in base_field]

    cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=1e-3,
        max_step_size=8e-3,
        min_step_size=1e-5,
        error_tolerance=1e-6,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="i*c0*A",
            constants=[12.0, 0.0, 0.0, 0.0],
        ),
    )

    base_result = api.propagate(
        cfg,
        base_field,
        2,
        opts,
        capture_step_history=True,
        step_history_capacity=4096,
    )
    scaled_result = api.propagate(
        cfg,
        scaled_field,
        2,
        opts,
        capture_step_history=True,
        step_history_capacity=4096,
    )

    base_history = base_result.meta.get("step_history")
    scaled_history = scaled_result.meta.get("step_history")
    assert isinstance(base_history, dict)
    assert isinstance(scaled_history, dict)

    base_steps = [float(v) for v in base_history.get("step_size", [])]
    scaled_steps = [float(v) for v in scaled_history.get("step_size", [])]
    base_errors = [float(v) for v in base_history.get("error", [])]
    scaled_errors = [float(v) for v in scaled_history.get("error", [])]
    assert len(base_steps) > 0
    assert len(base_errors) > 0
    assert len(scaled_steps) > 0
    assert len(scaled_errors) > 0
    assert all(math.isfinite(v) and v >= 0.0 for v in base_errors)
    assert all(math.isfinite(v) and v >= 0.0 for v in scaled_errors)

    base_errors_sorted = sorted(base_errors)
    scaled_errors_sorted = sorted(scaled_errors)
    base_med = base_errors_sorted[len(base_errors_sorted) // 2]
    scaled_med = scaled_errors_sorted[len(scaled_errors_sorted) // 2]
    med_ratio = scaled_med / max(base_med, 1e-20)
    assert 0.10 <= med_ratio <= 10.0, (
        f"relative adaptive error changed by more than one order of magnitude under amplitude scaling: "
        f"median ratio={med_ratio}"
    )
    steps_ratio = float(len(scaled_steps)) / float(len(base_steps))
    assert 0.25 <= steps_ratio <= 4.0, (
        f"adaptive step-count change under amplitude scaling is unexpectedly large: ratio={steps_ratio}"
    )


def test_adaptive_tighter_tolerance_reduces_final_error(api, opts):
    n = 192
    dt = 0.004
    z_final = 0.06
    t = _centered_time_grid(n, dt)
    omega = _omega_grid_unshifted(n, dt)
    a0 = [complex(math.exp(-((ti / 0.10) ** 2)), 0.0) for ti in t]

    common = dict(
        propagation_distance=z_final,
        starting_step_size=1e-3,
        max_step_size=8e-3,
        min_step_size=1e-5,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="i*c0*A*I",
            constants=[6.0, 0.0, 0.0, 0.0],
        ),
    )

    loose_cfg = prepare_sim_config(n, error_tolerance=1e-5, **common)
    tight_cfg = prepare_sim_config(n, error_tolerance=1e-7, **common)
    reference_cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=2.5e-4,
        max_step_size=2.5e-4,
        min_step_size=2.5e-4,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="i*c0*A*I",
            constants=[6.0, 0.0, 0.0, 0.0],
        ),
    )

    loose_result = api.propagate(
        loose_cfg,
        a0,
        2,
        opts,
        capture_step_history=True,
        step_history_capacity=4096,
    )
    tight_result = api.propagate(
        tight_cfg,
        a0,
        2,
        opts,
        capture_step_history=True,
        step_history_capacity=4096,
    )
    reference_final = api.propagate(reference_cfg, a0, 2, opts).records[1]

    loose_final = loose_result.records[1]
    tight_final = tight_result.records[1]
    loose_rel_err = _relative_l2_error(loose_final, reference_final)
    tight_rel_err = _relative_l2_error(tight_final, reference_final)
    assert tight_rel_err <= loose_rel_err, (
        f"tight tolerance should not worsen final error: loose={loose_rel_err}, tight={tight_rel_err}"
    )

    loose_history = loose_result.meta.get("step_history")
    tight_history = tight_result.meta.get("step_history")
    assert isinstance(loose_history, dict)
    assert isinstance(tight_history, dict)
    loose_steps = [float(v) for v in loose_history.get("step_size", [])]
    tight_steps = [float(v) for v in tight_history.get("step_size", [])]
    assert len(loose_steps) > 0
    assert len(tight_steps) > 0
    assert len(tight_steps) >= len(loose_steps), (
        f"tight tolerance should require at least as many accepted steps: loose={len(loose_steps)}, "
        f"tight={len(tight_steps)}"
    )


def test_adaptive_min_step_out_of_tolerance_emits_warning(api, opts):
    n = 96
    dt = 0.01
    z_final = 0.04
    min_step_size = 1e-4
    t = _centered_time_grid(n, dt)
    omega = _omega_grid_unshifted(n, dt)
    a0 = [complex(math.exp(-((ti / 0.09) ** 2)), 0.0) for ti in t]
    tol = 1e-30

    cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=min_step_size,
        error_tolerance=tol,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="i*c0*A*I",
            constants=[20.0, 0.0, 0.0, 0.0],
        ),
    )

    api.set_log_level(NLOLIB_LOG_LEVEL_WARN)
    api.set_log_buffer(256 * 1024)
    api.clear_log_buffer()
    result = api.propagate(
        cfg,
        a0,
        2,
        opts,
        capture_step_history=True,
        step_history_capacity=4096,
    )
    logs = api.read_log_buffer(consume=True, max_bytes=256 * 1024)

    history = result.meta.get("step_history")
    assert isinstance(history, dict)
    step_sizes = [float(v) for v in history.get("step_size", [])]
    errors = [float(v) for v in history.get("error", [])]
    assert len(step_sizes) > 0
    assert len(errors) > 0

    hit_min_step = any(v <= (min_step_size * (1.0 + 1e-12)) for v in step_sizes)
    exceeded_tol = any(v > tol for v in errors)
    assert hit_min_step and exceeded_tol, "test setup did not reach min-step out-of-tolerance regime"
    assert "adaptive solver reached min_step_size while local relative error remains above tolerance" in logs


def test_second_order_soliton_intensity_error(api, opts):
    case = _second_order_soliton_case(n=224, window_multiple_t0=16.0)

    cfg = prepare_sim_config(
        224,
        propagation_distance=case["z_final"],
        starting_step_size=2e-4,
        max_step_size=1e-2,
        min_step_size=1e-7,
        error_tolerance=1e-5,
        pulse_period=224.0 * case["dt"],
        delta_time=case["dt"],
        frequency_grid=[complex(om, 0.0) for om in case["omega"]],
        runtime=RuntimeOperators(constants=[0.5 * case["beta2"], 0.0, case["gamma"]]),
    )

    records = api.propagate(cfg, case["a0"], 2, opts).records
    final_field = records[1]
    assert all(math.isfinite(v.real) and math.isfinite(v.imag) for v in final_field)

    epsilon = _second_order_intensity_error(final_field, case["p0"], case["u_true"])
    assert epsilon <= 1e-2, f"second-order soliton intensity error too high: {epsilon}"


def test_second_order_soliton_adaptive_vs_fixed_reference_auto_gpu(api, opts):
    if not _auto_backend_resolves_to_vulkan(api):
        print("test_second_order_soliton_adaptive_vs_fixed_reference_auto_gpu: AUTO did not resolve to Vulkan, skipping.")
        return

    case = _second_order_soliton_case(n=1024, window_multiple_t0=40.0)
    runtime = RuntimeOperators(constants=[0.5 * case["beta2"], 0.0, case["gamma"]])
    gpu_opts = default_execution_options(NLO_VECTOR_BACKEND_AUTO)

    adaptive_cfg = prepare_sim_config(
        1024,
        propagation_distance=case["z_final"],
        starting_step_size=1e-4,
        max_step_size=1e-2,
        min_step_size=1e-9,
        error_tolerance=1e-10,
        pulse_period=1024.0 * case["dt"],
        delta_time=case["dt"],
        frequency_grid=[complex(om, 0.0) for om in case["omega"]],
        runtime=runtime,
    )
    fixed_cfg = prepare_sim_config(
        1024,
        propagation_distance=case["z_final"],
        starting_step_size=case["z_final"] / 1024.0,
        max_step_size=case["z_final"] / 1024.0,
        min_step_size=case["z_final"] / 1024.0,
        error_tolerance=1e-10,
        pulse_period=1024.0 * case["dt"],
        delta_time=case["dt"],
        frequency_grid=[complex(om, 0.0) for om in case["omega"]],
        runtime=runtime,
    )

    api.set_log_level(NLOLIB_LOG_LEVEL_ERROR)
    try:
        adaptive_result = api.propagate(
            adaptive_cfg,
            case["a0"],
            2,
            gpu_opts,
            capture_step_history=True,
            step_history_capacity=32768,
        )
        fixed_result = api.propagate(fixed_cfg, case["a0"], 2, gpu_opts)
    finally:
        api.set_log_level(NLOLIB_LOG_LEVEL_WARN)

    adaptive_final = adaptive_result.records[1]
    fixed_final = fixed_result.records[1]
    adaptive_error = _second_order_intensity_error(adaptive_final, case["p0"], case["u_true"])
    fixed_error = _second_order_intensity_error(fixed_final, case["p0"], case["u_true"])

    adaptive_history = adaptive_result.meta.get("step_history")
    assert isinstance(adaptive_history, dict)
    adaptive_steps = [float(v) for v in adaptive_history.get("step_size", [])]
    assert len(adaptive_steps) > 0
    assert all(math.isfinite(v) and v > 0.0 for v in adaptive_steps)
    assert math.isfinite(adaptive_error)
    assert math.isfinite(fixed_error)
    assert fixed_error < adaptive_error, (
        f"fixed-step reference should outperform adaptive second-order run on the same GPU path: "
        f"fixed={fixed_error}, adaptive={adaptive_error}"
    )
    # Diagnostic guard: adaptive should stay in the same broad error band as a
    # fine fixed-step reference, but current solver work still leaves a clear
    # gap here for the second-order case.
    assert adaptive_error <= (10.0 * fixed_error), (
        f"adaptive second-order error drifted too far above fixed-step reference: "
        f"fixed={fixed_error}, adaptive={adaptive_error}"
    )


def test_fixed_step_linear_only_exact_cpu_vs_auto_gpu(api, opts):
    n = 512
    dt = 0.02
    z_final = 0.35
    c0 = -0.5
    c1 = 0.0
    times = _centered_time_grid(n, dt)
    omega = _omega_grid_unshifted(n, dt)
    a0 = [complex(math.exp(-((ti / 0.7) ** 2)), 0.0) for ti in times]
    exact = _exact_linear_temporal_propagation(a0, omega, c0, c1, z_final)
    step_counts = [1, 2, 4, 8, 16]

    cpu_errors = []
    for step_count in step_counts:
        step = z_final / float(step_count)
        cfg = prepare_sim_config(
            n,
            propagation_distance=z_final,
            starting_step_size=step,
            max_step_size=step,
            min_step_size=step,
            error_tolerance=1e-12,
            pulse_period=float(n) * dt,
            delta_time=dt,
            frequency_grid=[complex(om, 0.0) for om in omega],
            runtime=RuntimeOperators(constants=[c0, c1, 0.0]),
        )
        cpu_final = api.propagate(cfg, a0, 2, default_execution_options(NLO_VECTOR_BACKEND_CPU)).records[1]
        cpu_errors.append(_relative_l2_error(cpu_final, exact))

    assert max(cpu_errors) <= 5e-12, f"CPU linear-only fixed-step propagation drifted from exact solution: {cpu_errors}"

    if not _auto_backend_resolves_to_vulkan(api):
        print("test_fixed_step_linear_only_exact_cpu_vs_auto_gpu: AUTO did not resolve to Vulkan, skipping.")
        return

    gpu_errors = []
    for step_count in step_counts:
        step = z_final / float(step_count)
        cfg = prepare_sim_config(
            n,
            propagation_distance=z_final,
            starting_step_size=step,
            max_step_size=step,
            min_step_size=step,
            error_tolerance=1e-12,
            pulse_period=float(n) * dt,
            delta_time=dt,
            frequency_grid=[complex(om, 0.0) for om in omega],
            runtime=RuntimeOperators(constants=[c0, c1, 0.0]),
        )
        auto_final = api.propagate(cfg, a0, 2, default_execution_options(NLO_VECTOR_BACKEND_AUTO)).records[1]
        gpu_errors.append(_relative_l2_error(auto_final, exact))

    assert max(gpu_errors) <= 5e-6, f"AUTO/Vulkan linear-only fixed-step error exceeded diagnostic bound: {gpu_errors}"


def test_fixed_step_nonlinear_only_order_auto_gpu(api, opts):
    if not _auto_backend_resolves_to_vulkan(api):
        print("test_fixed_step_nonlinear_only_order_auto_gpu: AUTO did not resolve to Vulkan, skipping.")
        return

    n = 256
    dt = 0.05
    z_final = 0.2
    gamma = 1.3
    times = _centered_time_grid(n, dt)
    a0 = [complex(0.7 * math.exp(-((ti / 0.6) ** 2)), 0.0) for ti in times]
    exact = _exact_nonlinear_phase_rotation(a0, gamma, z_final)
    step_counts = [4, 8, 16, 32, 64]
    errors = []

    for step_count in step_counts:
        step = z_final / float(step_count)
        cfg = prepare_sim_config(
            n,
            propagation_distance=z_final,
            starting_step_size=step,
            max_step_size=step,
            min_step_size=step,
            error_tolerance=1e-12,
            pulse_period=float(n) * dt,
            delta_time=dt,
            frequency_grid=[complex(0.0, 0.0) for _ in range(n)],
            runtime=RuntimeOperators(
                dispersion_factor_expr="0",
                nonlinear_expr="i*c2*A*I",
                constants=[0.0, 0.0, gamma],
            ),
        )
        final_field = api.propagate(cfg, a0, 2, default_execution_options(NLO_VECTOR_BACKEND_AUTO)).records[1]
        errors.append(_relative_l2_error(final_field, exact))

    local_orders = [
        math.log(errors[i - 1] / errors[i], 2.0)
        for i in range(1, len(errors))
        if errors[i] > 0.0
    ]
    assert min(local_orders) >= 3.8, (
        f"nonlinear-only fixed-step path lost fourth-order convergence on AUTO/Vulkan: "
        f"errors={errors}, local_orders={local_orders}"
    )


def test_expr_mode_ignores_raman_parameters(api, opts):
    n = 160
    dt = 0.01
    z_final = 0.04
    omega = _omega_grid_unshifted(n, dt)
    t = _centered_time_grid(n, dt)
    a0 = [complex(math.exp(-((ti / 0.11) ** 2)), 0.0) for ti in t]

    common = dict(
        propagation_distance=z_final,
        starting_step_size=1e-3,
        max_step_size=4e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
    )
    expr_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w",
            nonlinear_expr="i*c1*A*I",
            constants=[0.01, 0.7, 0.0, 0.0],
            nonlinear_model=NLO_NONLINEAR_MODEL_EXPR,
        ),
        **common,
    )
    expr_with_raman_params_cfg = prepare_sim_config(
        n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w",
            nonlinear_expr="i*c1*A*I",
            constants=[0.01, 0.7, 0.0, 0.0],
            nonlinear_model=NLO_NONLINEAR_MODEL_EXPR,
            nonlinear_gamma=123.0,
            raman_fraction=0.35,
            raman_tau1=0.02,
            raman_tau2=0.05,
            shock_omega0=17.0,
            raman_response_time=[complex(1.0 if i == 0 else 0.0, 0.0) for i in range(n)],
        ),
        **common,
    )

    expr_final = api.propagate(expr_cfg, a0, 2, opts).records[1]
    expr_with_raman_params_final = api.propagate(expr_with_raman_params_cfg, a0, 2, opts).records[1]
    rel = _relative_l2_error(expr_final, expr_with_raman_params_final)
    assert rel <= 2e-8, f"expr-mode propagation changed when Raman-only parameters were set: rel={rel}"


def test_fixed_step_fundamental_soliton_order(api, opts):
    slope, errors = _fixed_step_fundamental_soliton_order_stats(api, opts)
    assert slope >= 3.0, f"fixed-step fundamental soliton slope unexpectedly low: {slope}"
    assert errors[0] > errors[-1], f"fixed-step refinement did not reduce final error: {errors}"


def _fixed_step_fundamental_soliton_order_stats(api, opts):
    beta2 = -0.01
    gamma = 0.01
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    p0 = abs(beta2) / (gamma * t0 * t0)
    ld = (t0 * t0) / abs(beta2)
    z_final = 0.5 * math.pi * ld

    n = 1024
    dt = (32.0 * t0) / float(n)
    t = _centered_time_grid(n, dt)
    tau = [ti / t0 for ti in t]
    omega = _omega_grid_unshifted(n, dt)
    a0 = [complex(math.sqrt(p0) / math.cosh(xi), 0.0) for xi in tau]

    step_counts = [1, 2, 4, 8, 16]
    step_sizes = []
    errors = []
    for step_count in step_counts:
        dz = z_final / float(step_count)
        cfg = prepare_sim_config(
            n,
            propagation_distance=z_final,
            starting_step_size=dz,
            max_step_size=dz,
            min_step_size=dz,
            error_tolerance=1e-6,
            pulse_period=float(n) * dt,
            delta_time=dt,
            frequency_grid=[complex(om, 0.0) for om in omega],
            runtime=RuntimeOperators(constants=[0.5 * beta2, 0.0, gamma]),
        )
        records = api.propagate(cfg, a0, 2, opts).records
        final_field = records[1]
        assert all(math.isfinite(v.real) and math.isfinite(v.imag) for v in final_field)

        a_true = [
            complex((math.sqrt(p0) / math.cosh(xi)) * math.cos(0.5 * (z_final / ld)),
                    (math.sqrt(p0) / math.cosh(xi)) * math.sin(0.5 * (z_final / ld)))
            for xi in tau
        ]
        errors.append(_filtered_relative_l2_error(final_field, a_true))
        step_sizes.append(dz)

    slope = _fit_loglog_slope(step_sizes, errors)
    return slope, errors


def _auto_backend_resolves_to_vulkan(api):
    n = 32
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.01,
        min_step_size=0.01,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=1.0 / float(n),
        frequency_grid=[0j] * n,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 0.0]),
    )
    api.set_log_level(2)
    api.set_log_buffer(64 * 1024)
    api.clear_log_buffer()
    api.propagate(cfg, [0j] * n, 1, default_execution_options(NLO_VECTOR_BACKEND_AUTO))
    logs = api.read_log_buffer(consume=True, max_bytes=64 * 1024)
    return "actual: VULKAN" in logs


def test_fixed_step_fundamental_soliton_order_auto_gpu(api, opts):
    if not _auto_backend_resolves_to_vulkan(api):
        print("test_fixed_step_fundamental_soliton_order_auto_gpu: AUTO did not resolve to Vulkan, skipping.")
        return

    cpu_slope, cpu_errors = _fixed_step_fundamental_soliton_order_stats(
        api,
        default_execution_options(NLO_VECTOR_BACKEND_CPU),
    )
    auto_slope, auto_errors = _fixed_step_fundamental_soliton_order_stats(
        api,
        default_execution_options(NLO_VECTOR_BACKEND_AUTO),
    )
    assert auto_slope >= 3.0, f"fixed-step AUTO/Vulkan soliton slope unexpectedly low: {auto_slope}"
    assert abs(auto_slope - cpu_slope) <= 0.2, (
        f"AUTO/Vulkan slope drifted too far from CPU: cpu={cpu_slope}, auto={auto_slope}"
    )
    assert auto_errors[0] > auto_errors[-1], f"AUTO/Vulkan refinement did not reduce final error: {auto_errors}"


def main():
    api = NLolib()
    opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)
    test_dispersion_factor_callable_matches_string(api, opts)
    test_nonlinear_callable_matches_string(api, opts)
    test_nonlinear_legacy_multiplier_warns(api, opts)
    test_field_first_callable_signature_enforced()
    test_linear_dispersion_aliases_are_mutually_exclusive()
    test_extended_runtime_operators_execute(api, opts)
    test_tensor_linear_factor_alias_matches(api, opts)
    test_beta_sum_callable_matches_string(api, opts)
    test_tensor_diffraction_matches_fft_reference(api, opts)
    test_tensor_wt_only_matches_transverse_tiling(api, opts)
    test_raman_like_nonlinear_callable_matches_string(api, opts)
    test_kerr_raman_model_reduces_to_kerr_when_fraction_zero(api, opts)
    test_kerr_raman_custom_response_matches_generated_default(api, opts)
    test_kerr_raman_rejects_coupled_mode(api, opts)
    test_linear_drift_signed_prediction(api, opts)
    test_adaptive_embedded_error_estimator_remains_stable(api, opts)
    test_adaptive_relative_error_is_amplitude_scale_invariant(api, opts)
    test_adaptive_tighter_tolerance_reduces_final_error(api, opts)
    test_adaptive_min_step_out_of_tolerance_emits_warning(api, opts)
    test_second_order_soliton_intensity_error(api, opts)
    test_second_order_soliton_adaptive_vs_fixed_reference_auto_gpu(api, opts)
    test_fixed_step_linear_only_exact_cpu_vs_auto_gpu(api, opts)
    test_fixed_step_nonlinear_only_order_auto_gpu(api, opts)
    test_expr_mode_ignores_raman_parameters(api, opts)
    test_fixed_step_fundamental_soliton_order(api, opts)
    test_fixed_step_fundamental_soliton_order_auto_gpu(api, opts)
    print("test_python_operator_regression: runtime-operator propagation regressions validated.")


if __name__ == "__main__":
    main()
