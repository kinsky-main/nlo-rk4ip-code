"""
Temporal Raman scattering validation with an analytical self-frequency-shift check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.plotting import (
    plot_intensity_colormap_vs_propagation,
    plot_three_curve_drift,
    plot_total_error_over_propagation,
    plot_two_curve_comparison,
)
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from backend.storage import ExampleRunDB
from nlolib_ctypes import (
    NLO_NONLINEAR_MODEL_EXPR,
    NLO_NONLINEAR_MODEL_KERR_RAMAN,
    RuntimeOperators,
)

_C_NM_PER_PS = 299792.458


def _spectral_centroid(freq_axis: np.ndarray, spectral_intensity: np.ndarray) -> np.ndarray:
    weights = np.asarray(spectral_intensity, dtype=np.float64)
    numer = np.sum(weights * freq_axis[None, :], axis=1)
    denom = np.maximum(np.sum(weights, axis=1), 1e-15)
    return numer / denom


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    peaks = np.maximum(np.max(arr, axis=1, keepdims=True), 1e-15)
    return arr / peaks


def _default_raman_response(n: int, dt: float, tau1: float, tau2: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) * float(dt)
    coef = (tau1 * tau1 + tau2 * tau2) / (tau1 * tau2 * tau2)
    response = coef * np.exp(-t / tau2) * np.sin(t / tau1)
    area = float(np.sum(response) * dt)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("invalid Raman response normalization area")
    return (response / area).astype(np.float64)


def _raman_first_moment(response: np.ndarray, dt: float) -> float:
    t = np.arange(response.size, dtype=np.float64) * float(dt)
    return float(np.sum(t * response) * dt)


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    axis = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size != axis.size:
        raise ValueError("y and x must have identical length for cumulative trapezoid integration.")
    out = np.zeros_like(values)
    for i in range(1, values.size):
        out[i] = out[i - 1] + 0.5 * (values[i] + values[i - 1]) * (axis[i] - axis[i - 1])
    return out


def _spectral_map_to_wavelength_map(
    omega_axis: np.ndarray,
    spectral_map: np.ndarray,
    lambda0_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (lambda0_nm > 0.0):
        raise ValueError("lambda0_nm must be > 0.")
    freq0 = _C_NM_PER_PS / float(lambda0_nm)
    freq_detuning = np.asarray(omega_axis, dtype=np.float64) / (2.0 * np.pi)
    freq_total = freq0 + freq_detuning
    valid = freq_total > 0.0
    if not np.any(valid):
        raise ValueError("no positive total frequency samples for wavelength map.")
    lambda_axis = _C_NM_PER_PS / freq_total[valid]
    map_valid = np.asarray(spectral_map, dtype=np.float64)[:, valid]
    order = np.argsort(lambda_axis)
    return lambda_axis[order], map_valid[:, order]


def _omega_centroid_to_wavelength_nm(omega_centroid: np.ndarray, lambda0_nm: float) -> np.ndarray:
    if not (lambda0_nm > 0.0):
        raise ValueError("lambda0_nm must be > 0.")
    freq0 = _C_NM_PER_PS / float(lambda0_nm)
    freq_total = freq0 + (np.asarray(omega_centroid, dtype=np.float64) / (2.0 * np.pi))
    if np.any(freq_total <= 0.0):
        raise ValueError("centroid mapping to wavelength requires positive total frequency.")
    return _C_NM_PER_PS / freq_total


def _row_centroid(axis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    coords = np.asarray(axis, dtype=np.float64).reshape(-1)
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != coords.size:
        raise ValueError("rows must have shape [record, axis].")
    numer = np.sum(data * coords[None, :], axis=1)
    denom = np.maximum(np.sum(data, axis=1), 1e-15)
    return numer / denom


def _crop_wavelength_support(
    lambda_axis_nm: np.ndarray,
    spectral_map: np.ndarray,
    *,
    keep_mass: float = 0.999,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < keep_mass <= 1.0):
        raise ValueError("keep_mass must be in (0, 1].")
    axis = np.asarray(lambda_axis_nm, dtype=np.float64).reshape(-1)
    data = np.asarray(spectral_map, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != axis.size:
        raise ValueError("spectral_map must have shape [record, wavelength].")
    weights = np.sum(np.clip(data, 0.0, None), axis=0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return axis, data
    cdf = np.cumsum(weights) / total
    tail = 0.5 * (1.0 - keep_mass)
    lo = int(np.searchsorted(cdf, tail, side="left"))
    hi = int(np.searchsorted(cdf, 1.0 - tail, side="right"))
    lo = max(lo, 0)
    hi = min(max(hi, lo + 1), axis.size)
    return axis[lo:hi], data[:, lo:hi]


def _raman_centroid_rhs_moment(
    records: np.ndarray,
    dt: float,
    gamma: float,
    f_r: float,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    fields = np.asarray(records, dtype=np.complex128)
    if fields.ndim != 2:
        raise ValueError("records must have shape [record, time].")
    n = fields.shape[1]
    omega_unshifted = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    h_r = _default_raman_response(n, dt, tau1, tau2)
    h_r_fft = np.fft.fft(h_r)

    rhs = np.empty(fields.shape[0], dtype=np.float64)
    for i in range(fields.shape[0]):
        intensity = np.abs(fields[i]) ** 2
        delayed = np.fft.ifft(np.fft.fft(intensity) * h_r_fft).real
        delayed_dt = np.fft.ifft((1.0j * omega_unshifted) * np.fft.fft(delayed)).real
        energy = max(float(np.sum(intensity) * dt), 1e-15)
        rhs[i] = float((gamma * f_r / energy) * np.sum(intensity * delayed_dt) * dt)
    return rhs


def _run_case(
    runner: NloExampleRunner,
    field0: np.ndarray,
    sim_cfg: TemporalSimulationConfig,
    *,
    num_records: int,
    exec_options: SimulationOptions,
    storage_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = storage_kwargs if storage_kwargs is not None else {}
    z_axis, records = runner.propagate_temporal_records(
        field0,
        sim_cfg,
        num_records,
        exec_options,
        **kwargs,
    )
    return np.asarray(z_axis, dtype=np.float64), np.asarray(records, dtype=np.complex128)


def main() -> float:
    parser = build_example_parser(
        example_slug="raman_scattering",
        description="Raman self-frequency shift validation (temporal only) with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "raman_scattering_rk4ip"
    kerr_case_key = "kerr_only"
    raman_case_key = "kerr_raman_shock"

    n = 2**11
    dt = 0.002
    beta2 = -0.01
    gamma = 1.40
    z_final = 0.40
    pulse_width = 0.08
    num_records = 120

    f_r = 0.18
    tau1 = 0.0122
    tau2 = 0.0320
    shock_omega0 = 0.0
    lambda0_nm = 1550.0

    t = centered_time_grid(n, dt)
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    p0 = abs(beta2) / (gamma * pulse_width * pulse_width)
    field0 = (np.sqrt(p0) / np.cosh(t / pulse_width)).astype(np.complex128)
    exec_options = SimulationOptions(backend="auto", fft_backend="auto")
    runner = NloExampleRunner()

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded_kerr = db.load_case(example_name=example_name, run_group=run_group, case_key=kerr_case_key)
        loaded_raman = db.load_case(example_name=example_name, run_group=run_group, case_key=raman_case_key)
        meta = loaded_raman.meta
        n = int(meta["n"])
        dt = float(meta["dt"])
        beta2 = float(meta["beta2"])
        gamma = float(meta["gamma"])
        z_final = float(meta["z_final"])
        pulse_width = float(meta["pulse_width"])
        f_r = float(meta["f_r"])
        tau1 = float(meta["tau1"])
        tau2 = float(meta["tau2"])
        shock_omega0 = float(meta["shock_omega0"])
        lambda0_nm = float(meta.get("lambda0_nm", 1550.0))
        t = centered_time_grid(n, dt)
        p0 = abs(beta2) / (gamma * pulse_width * pulse_width)
        field0 = (np.sqrt(p0) / np.cosh(t / pulse_width)).astype(np.complex128)
        z_kerr = np.asarray(loaded_kerr.z_axis, dtype=np.float64)
        z_raman = np.asarray(loaded_raman.z_axis, dtype=np.float64)
        if z_kerr.shape != z_raman.shape or not np.allclose(z_kerr, z_raman):
            raise RuntimeError("replot data mismatch: Kerr and Raman z-axes differ.")
        z_axis = z_kerr
        kerr_records = np.asarray(loaded_kerr.records, dtype=np.complex128)
        raman_records = np.asarray(loaded_raman.records, dtype=np.complex128)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        runtime_kerr = RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w-c1",
            nonlinear_expr="i*c2*A*I",
            constants=[0.5 * beta2, 0.0, gamma],
            nonlinear_model=NLO_NONLINEAR_MODEL_EXPR,
            nonlinear_gamma=gamma,
            raman_fraction=0.0,
            shock_omega0=0.0,
        )
        runtime_raman = RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w-c1",
            nonlinear_expr="0",
            constants=[0.5 * beta2, 0.0, gamma],
            nonlinear_model=NLO_NONLINEAR_MODEL_KERR_RAMAN,
            nonlinear_gamma=gamma,
            raman_fraction=f_r,
            raman_tau1=tau1,
            raman_tau2=tau2,
            shock_omega0=shock_omega0,
        )
        base_cfg = dict(
            gamma=gamma,
            beta2=beta2,
            alpha=0.0,
            dt=dt,
            z_final=z_final,
            num_time_samples=n,
            pulse_period=n * dt,
            omega=omega,
            starting_step_size=z_final / 500.0,
            max_step_size=z_final / 80.0,
            min_step_size=z_final / 20000.0,
            error_tolerance=2e-6,
            honor_solver_controls=True,
        )
        cfg_kerr = TemporalSimulationConfig(runtime=runtime_kerr, **base_cfg)
        cfg_raman = TemporalSimulationConfig(runtime=runtime_raman, **base_cfg)

        storage_kerr = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=kerr_case_key,
            chunk_records=8,
        )
        storage_raman = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=raman_case_key,
            chunk_records=8,
        )
        z_axis, kerr_records = _run_case(
            runner,
            field0,
            cfg_kerr,
            num_records=num_records,
            exec_options=exec_options,
            storage_kwargs=storage_kerr,
        )
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=kerr_case_key,
            solver_meta=runner.last_meta,
            meta={
                "n": int(n),
                "dt": float(dt),
                "beta2": float(beta2),
                "gamma": float(gamma),
                "z_final": float(z_final),
                "pulse_width": float(pulse_width),
                "f_r": float(f_r),
                "tau1": float(tau1),
                "tau2": float(tau2),
                "shock_omega0": float(shock_omega0),
                "lambda0_nm": float(lambda0_nm),
            },
        )

        z_axis_raman, raman_records = _run_case(
            runner,
            field0,
            cfg_raman,
            num_records=num_records,
            exec_options=exec_options,
            storage_kwargs=storage_raman,
        )
        if z_axis.shape != z_axis_raman.shape or not np.allclose(z_axis, z_axis_raman):
            raise RuntimeError("solver returned inconsistent z-axis between Kerr and Raman runs.")
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=raman_case_key,
            solver_meta=runner.last_meta,
            meta={
                "n": int(n),
                "dt": float(dt),
                "beta2": float(beta2),
                "gamma": float(gamma),
                "z_final": float(z_final),
                "pulse_width": float(pulse_width),
                "f_r": float(f_r),
                "tau1": float(tau1),
                "tau2": float(tau2),
                "shock_omega0": float(shock_omega0),
                "lambda0_nm": float(lambda0_nm),
            },
        )

    omega_axis = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(n, d=dt))
    kerr_spectra = np.fft.fftshift(np.fft.fft(kerr_records, axis=1), axes=1)
    raman_spectra = np.fft.fftshift(np.fft.fft(raman_records, axis=1), axes=1)
    kerr_spec_map = np.abs(kerr_spectra) ** 2
    raman_spec_map = np.abs(raman_spectra) ** 2
    kerr_spec_centroid = _spectral_centroid(omega_axis, kerr_spec_map)
    raman_spec_centroid = _spectral_centroid(omega_axis, raman_spec_map)

    response = _default_raman_response(n, dt, tau1, tau2)
    t_r = _raman_first_moment(response, dt)
    centroid_rhs = _raman_centroid_rhs_moment(raman_records, dt, gamma, f_r, tau1, tau2)
    centroid_derivative_num = np.gradient(raman_spec_centroid, z_axis, edge_order=2)
    predicted_centroid = raman_spec_centroid[0] + _cumulative_trapezoid(centroid_rhs, z_axis)
    centered_num = raman_spec_centroid - raman_spec_centroid[0]
    centered_pred = predicted_centroid - predicted_centroid[0]
    centered_abs_scale = max(float(np.max(np.abs(centered_num))), 1e-12)
    centroid_pointwise_rel_error = np.abs(centered_num - centered_pred) / centered_abs_scale
    centroid_curve_rel_error = float(
        np.linalg.norm(centered_num - centered_pred) / max(np.linalg.norm(centered_num), 1e-15)
    )
    centroid_derivative_rel_error = float(
        np.linalg.norm(centroid_derivative_num - centroid_rhs) / max(np.linalg.norm(centroid_derivative_num), 1e-15)
    )

    final_kerr = _normalized_rows(kerr_spec_map)[-1]
    final_raman = _normalized_rows(raman_spec_map)[-1]
    centroid_delta_final = float(raman_spec_centroid[-1] - kerr_spec_centroid[-1])
    predicted_delta_final = float(predicted_centroid[-1] - predicted_centroid[0])

    lambda_axis_full, wavelength_map_full = _spectral_map_to_wavelength_map(omega_axis, raman_spec_map, lambda0_nm)
    lambda_axis, wavelength_map = _crop_wavelength_support(lambda_axis_full, wavelength_map_full, keep_mass=0.999)
    lambda_axis_kerr, wavelength_map_kerr = _spectral_map_to_wavelength_map(omega_axis, kerr_spec_map, lambda0_nm)
    lambda_kerr = _row_centroid(lambda_axis_kerr, wavelength_map_kerr)
    lambda_raman = _row_centroid(lambda_axis_full, wavelength_map_full)
    lambda_pred = _omega_centroid_to_wavelength_nm(predicted_centroid, lambda0_nm)
    lambda_kerr_shift = lambda_kerr - lambda_kerr[0]
    lambda_raman_shift = lambda_raman - lambda_raman[0]
    lambda_pred_shift = lambda_pred - lambda_pred[0]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    p1 = plot_intensity_colormap_vs_propagation(
        omega_axis,
        z_axis,
        raman_spec_map,
        output_dir / "raman_spectral_intensity_propagation.png",
        x_label="Angular-frequency detuning (rad/time)",
        title="Raman: Spectral Intensity Over Propagation",
        colorbar_label="Normalized spectral intensity",
    )
    if p1 is not None:
        saved_paths.append(p1)

    p2 = plot_two_curve_comparison(
        z_axis,
        centered_num,
        centered_pred,
        output_dir / "raman_spectral_centroid_over_propagation_with_analytic.png",
        label_a="Numerical centroid shift",
        label_b="Analytical centroid shift",
        y_label="Delta spectral centroid (rad/time)",
        title="Raman Analytical Validation: Centroid Shift",
    )
    if p2 is not None:
        saved_paths.append(p2)

    p3 = plot_intensity_colormap_vs_propagation(
        lambda_axis,
        z_axis,
        wavelength_map,
        output_dir / "raman_wavelength_intensity_propagation.png",
        x_label="Wavelength (nm)",
        title="Raman: Wavelength Intensity Over Propagation (around lambda0)",
        colorbar_label="Normalized spectral intensity",
    )
    if p3 is not None:
        saved_paths.append(p3)

    p4 = plot_three_curve_drift(
        z_axis,
        lambda_kerr_shift,
        lambda_raman_shift,
        lambda_pred_shift,
        output_dir / "raman_wavelength_centroid_over_propagation_with_analytic.png",
        label_a="Kerr-only",
        label_b="Kerr+Raman",
        label_c="Moment-theorem prediction",
        y_label="Delta centroid wavelength (nm)",
        title="Raman Analytical Validation: Wavelength Centroid Shift",
    )
    if p4 is not None:
        saved_paths.append(p4)

    p5 = plot_two_curve_comparison(
        omega_axis,
        final_kerr,
        final_raman,
        output_dir / "raman_final_spectrum_comparison.png",
        label_a="Kerr-only final",
        label_b="Kerr+Raman final",
        x_label="Angular-frequency detuning (rad/time)",
        y_label="Normalized spectral intensity",
        title="Final Spectrum: Kerr-only vs Raman",
    )
    if p5 is not None:
        saved_paths.append(p5)

    p6 = plot_two_curve_comparison(
        z_axis,
        centroid_derivative_num,
        centroid_rhs,
        output_dir / "raman_spectral_centroid_derivative_validation.png",
        label_a="Numerical d(centroid)/dz",
        label_b="Analytical moment RHS",
        y_label="Centroid derivative (rad/time/m)",
        title="Raman Analytical Validation: Centroid Derivative",
    )
    if p6 is not None:
        saved_paths.append(p6)

    p8 = plot_total_error_over_propagation(
        z_axis,
        centroid_pointwise_rel_error,
        output_dir / "raman_spectral_centroid_shift_relative_error.png",
        title="Raman Analytical Validation: Pointwise Relative Error",
        y_label="Relative error of centroid shift",
    )
    if p8 is not None:
        saved_paths.append(p8)

    print(f"raman scattering summary (run_group={run_group}):")
    print(f"  n={n}, dt={dt:.6e}, z_final={z_final:.6e}")
    print(f"  gamma={gamma:.6e}, f_r={f_r:.6e}, tau1={tau1:.6e}, tau2={tau2:.6e}, shock_omega0={shock_omega0:.6e}")
    print(f"  wavelength mapping reference      = {lambda0_nm:.3f} nm (time unit assumed: ps)")
    print(f"  Raman first moment T_R           = {t_r:.6e}")
    print(f"  centroid derivative rel. error   = {centroid_derivative_rel_error:.6e}")
    print(f"  centroid curve rel. error        = {centroid_curve_rel_error:.6e}")
    print(f"  max pointwise rel. error         = {float(np.max(centroid_pointwise_rel_error)):.6e}")
    print(f"  final centroid shift (raman-kerr)= {centroid_delta_final:.6e}")
    print(f"  predicted centroid shift         = {predicted_delta_final:.6e}")
    if saved_paths:
        print("saved plots:")
        for path in saved_paths:
            print(f"  {path}")
    return centroid_curve_rel_error


if __name__ == "__main__":
    main()
