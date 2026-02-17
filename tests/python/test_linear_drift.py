import math

try:
    from nlolib_cffi import load, ffi
except ModuleNotFoundError:
    print("test_linear_drift: cffi bindings unavailable; skipping.")
    raise SystemExit(0)


def _write_complex_buffer(dst, values):
    for i, val in enumerate(values):
        dst[i].re = float(val.real)
        dst[i].im = float(val.imag)


def _read_complex_buffer(src, n):
    out = [0j] * n
    for i in range(n):
        out[i] = complex(src[i].re, src[i].im)
    return out


def _omega_grid_unshifted(n, dt):
    two_pi = 2.0 * math.pi
    return [two_pi * (float(i) / (float(n) * dt) if i <= (n - 1) // 2 else -float(n - i) / (float(n) * dt))
            for i in range(n)]


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


def _propagate_final_field(lib, cfg, n, inp_values, opts):
    inp = ffi.new("nlo_complex[]", n)
    out = ffi.new("nlo_complex[]", n * 2)
    _write_complex_buffer(inp, inp_values)

    status = int(lib.nlolib_propagate(cfg, n, inp, 2, out, opts))
    assert status == 0

    records = _read_complex_buffer(out, n * 2)
    return records[n:]


def _build_config(n, dt, omega, beta2, z_final):
    cfg = ffi.new("sim_config*")
    cfg.nonlinear.gamma = 0.0
    cfg.dispersion.num_dispersion_terms = 3
    cfg.dispersion.betas[0] = 0.0
    cfg.dispersion.betas[1] = 0.0
    cfg.dispersion.betas[2] = beta2
    cfg.dispersion.alpha = 0.0

    cfg.propagation.propagation_distance = z_final
    cfg.propagation.starting_step_size = 1e-3
    cfg.propagation.max_step_size = 5e-3
    cfg.propagation.min_step_size = 1e-5
    cfg.propagation.error_tolerance = 1e-7

    cfg.time.pulse_period = float(n) * dt
    cfg.time.delta_time = dt

    freq = ffi.new("nlo_complex[]", n)
    for i, om in enumerate(omega):
        freq[i].re = float(om)
        freq[i].im = 0.0
    cfg.frequency.frequency_grid = freq

    cfg.spatial.nx = n
    cfg.spatial.ny = 1
    cfg.spatial.delta_x = 1.0
    cfg.spatial.delta_y = 1.0
    cfg.spatial.grin_gx = 0.0
    cfg.spatial.grin_gy = 0.0
    cfg.spatial.spatial_frequency_grid = ffi.NULL
    cfg.spatial.grin_potential_phase_grid = ffi.NULL

    return cfg, freq


def _cpu_exec_options():
    opts = ffi.new("nlo_execution_options*")
    opts.backend_type = 0  # NLO_VECTOR_BACKEND_CPU
    opts.fft_backend = 0  # NLO_FFT_BACKEND_AUTO
    opts.device_heap_fraction = 0.70
    opts.record_ring_target = 0
    opts.forced_device_budget_bytes = 0
    opts.vulkan.physical_device = ffi.NULL
    opts.vulkan.device = ffi.NULL
    opts.vulkan.queue = ffi.NULL
    opts.vulkan.queue_family_index = 0
    opts.vulkan.command_pool = ffi.NULL
    opts.vulkan.descriptor_set_budget_bytes = 0
    opts.vulkan.descriptor_set_count_override = 0
    return opts


def main():
    lib = load()

    n = 1024
    dt = 0.01
    sigma = 0.20
    beta2 = 0.05
    d = 12.0
    z_final = 1.0

    t = _centered_time_grid(n, dt)
    omega = _omega_grid_unshifted(n, dt)
    opts = _cpu_exec_options()

    cfg_pos, _freq_pos = _build_config(n, dt, omega, beta2, z_final)
    cfg_neg, _freq_neg = _build_config(n, dt, omega, beta2, z_final)

    pulse_pos = _gaussian_with_phase(t, sigma, d)
    pulse_neg = _gaussian_with_phase(t, sigma, -d)

    final_pos = _propagate_final_field(lib, cfg_pos, n, pulse_pos, opts)
    final_neg = _propagate_final_field(lib, cfg_neg, n, pulse_neg, opts)

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
