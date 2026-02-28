"""
Temporal Raman scattering validation with delayed Raman response + shock term.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.plotting import (
    plot_intensity_colormap_vs_propagation,
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
        description="Delayed Raman + shock validation (temporal only) with DB-backed run/replot.",
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
    pulse_width = 0.05
    num_records = 120

    f_r = 0.18
    tau1 = 0.0122
    tau2 = 0.0320
    shock_omega0 = 20.0

    t = centered_time_grid(n, dt)
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    field0 = np.exp(-((t / pulse_width) ** 2)).astype(np.complex128)
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
        t = centered_time_grid(n, dt)
        field0 = np.exp(-((t / pulse_width) ** 2)).astype(np.complex128)
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
            },
        )

    freq_axis = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
    kerr_spectra = np.fft.fftshift(np.fft.fft(kerr_records, axis=1), axes=1)
    raman_spectra = np.fft.fftshift(np.fft.fft(raman_records, axis=1), axes=1)
    kerr_spec_map = np.abs(kerr_spectra) ** 2
    raman_spec_map = np.abs(raman_spectra) ** 2
    kerr_spec_centroid = _spectral_centroid(freq_axis, kerr_spec_map)
    raman_spec_centroid = _spectral_centroid(freq_axis, raman_spec_map)

    final_kerr = _normalized_rows(kerr_spec_map)[-1]
    final_raman = _normalized_rows(raman_spec_map)[-1]
    centroid_delta_final = float(raman_spec_centroid[-1] - kerr_spec_centroid[-1])

    output_dir = Path(__file__).resolve().parent / "output" / "raman_scattering"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    p1 = plot_intensity_colormap_vs_propagation(
        freq_axis,
        z_axis,
        raman_spec_map,
        output_dir / "raman_spectral_intensity_propagation.png",
        x_label="Frequency detuning (1/time)",
        title="Raman+Shock: Spectral Intensity Over Propagation",
        colorbar_label="Normalized spectral intensity",
    )
    if p1 is not None:
        saved_paths.append(p1)

    p2 = plot_two_curve_comparison(
        z_axis,
        kerr_spec_centroid,
        raman_spec_centroid,
        output_dir / "spectral_centroid_over_propagation.png",
        label_a="Kerr-only",
        label_b="Kerr+Raman+Shock",
        y_label="Spectral centroid (1/time)",
        title="Spectral Centroid Shift Over Propagation",
    )
    if p2 is not None:
        saved_paths.append(p2)

    p3 = plot_two_curve_comparison(
        freq_axis,
        final_kerr,
        final_raman,
        output_dir / "final_spectrum_comparison.png",
        label_a="Kerr-only final",
        label_b="Kerr+Raman+Shock final",
        x_label="Frequency detuning (1/time)",
        y_label="Normalized spectral intensity",
        title="Final Spectrum: Kerr-only vs Raman+Shock",
    )
    if p3 is not None:
        saved_paths.append(p3)

    print(f"raman scattering summary (run_group={run_group}):")
    print(f"  n={n}, dt={dt:.6e}, z_final={z_final:.6e}")
    print(f"  gamma={gamma:.6e}, f_r={f_r:.6e}, tau1={tau1:.6e}, tau2={tau2:.6e}, shock_omega0={shock_omega0:.6e}")
    print(f"  final centroid shift (raman-kerr) = {centroid_delta_final:.6e}")
    if saved_paths:
        print("saved plots:")
        for path in saved_paths:
            print(f"  {path}")
    return abs(centroid_delta_final)


if __name__ == "__main__":
    main()
