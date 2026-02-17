import math

from nlolib_ctypes import (
    NLO_VECTOR_BACKEND_CPU,
    NLolib,
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


def _build_config(n, dt, omega, beta2, z_final):
    return prepare_sim_config(
        n,
        gamma=0.0,
        betas=[0.0, 0.0, beta2],
        alpha=0.0,
        propagation_distance=z_final,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(om, 0.0) for om in omega],
    )


def main():
    api = NLolib()

    n = 1024
    dt = 0.01
    sigma = 0.20
    beta2 = 0.05
    d = 12.0
    z_final = 1.0

    t = _centered_time_grid(n, dt)
    omega = _omega_grid_unshifted(n, dt)
    opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)

    cfg_pos = _build_config(n, dt, omega, beta2, z_final)
    cfg_neg = _build_config(n, dt, omega, beta2, z_final)

    pulse_pos = _gaussian_with_phase(t, sigma, d)
    pulse_neg = _gaussian_with_phase(t, sigma, -d)

    final_pos = api.propagate(cfg_pos, pulse_pos, 2, opts)[1]
    final_neg = api.propagate(cfg_neg, pulse_neg, 2, opts)[1]

    shift_pos = _intensity_centroid(t, final_pos) - _intensity_centroid(t, pulse_pos)
    shift_neg = _intensity_centroid(t, final_neg) - _intensity_centroid(t, pulse_neg)

    expected_abs = abs(beta2 * d * z_final)
    assert abs(shift_pos) > 0.05
    assert abs(shift_neg) > 0.05
    assert shift_pos * shift_neg < 0.0

    rel_err_pos = abs(abs(shift_pos) - expected_abs) / expected_abs
    rel_err_neg = abs(abs(shift_neg) - expected_abs) / expected_abs
    assert rel_err_pos <= 0.30
    assert rel_err_neg <= 0.30

    mag_sym = abs(abs(shift_pos) - abs(shift_neg)) / max(abs(shift_pos), abs(shift_neg))
    assert mag_sym <= 0.20

    print(
        "test_linear_drift: validated opposite temporal drift directions and expected drift magnitude "
        f"(shift_pos={shift_pos:.6f}, shift_neg={shift_neg:.6f}, expected_abs={expected_abs:.6f})."
    )


if __name__ == "__main__":
    main()
