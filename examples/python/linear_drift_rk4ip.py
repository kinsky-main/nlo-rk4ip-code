"""
Linear dispersive drift example using the OOP example runner.

The simulation disables nonlinearity (gamma=0) and applies an initial phase
ramp exp(-i*d*t) to a Gaussian pulse, which induces temporal drift as the
pulse propagates along z.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.plotting import (
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from backend.storage import ExampleRunDB


def gaussian_with_phase_ramp(t: np.ndarray, sigma: float, d: float) -> np.ndarray:
    envelope = np.exp(-((t / sigma) ** 2))
    return envelope * np.exp((-1.0j) * d * t)


def centroid_curve(t: np.ndarray, records: np.ndarray) -> np.ndarray:
    intensity = np.abs(records) ** 2
    weighted = intensity @ t
    norm = np.sum(intensity, axis=1)
    safe_norm = np.where(norm > 0.0, norm, 1.0)
    return weighted / safe_norm


def linear_reference_records(
    field0: np.ndarray,
    z_records: np.ndarray,
    beta2: float,
    dt: float,
) -> np.ndarray:
    field = np.asarray(field0, dtype=np.complex128).reshape(-1)
    z = np.asarray(z_records, dtype=np.float64).reshape(-1)
    n = int(field.size)
    if n == 0:
        raise ValueError("field0 must be non-empty.")

    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=float(dt))
    phase_coeff = 0.5 * float(beta2) * (omega**2)
    spectrum0 = np.fft.fft(field)

    references = np.empty((z.size, n), dtype=np.complex128)
    phase_arg = np.empty(n, dtype=np.float64)
    phase = np.empty(n, dtype=np.complex128)
    spectrum_z = np.empty(n, dtype=np.complex128)
    for i, z_i in enumerate(z):
        np.multiply(phase_coeff, z_i, out=phase_arg)
        np.multiply(phase_arg, 1.0j, out=phase)
        np.exp(phase, out=phase)
        np.multiply(spectrum0, phase, out=spectrum_z)
        references[i] = np.fft.ifft(spectrum_z)
    return references


def relative_l2_error_curve(records: np.ndarray, reference_records: np.ndarray) -> np.ndarray:
    if records.shape != reference_records.shape:
        raise ValueError("records and reference_records must have the same shape.")

    ref_norms = np.linalg.norm(np.asarray(reference_records, dtype=np.complex128), axis=1)
    norm_floor = max(float(np.max(ref_norms)) * 1e-12, 1e-15)
    out = np.empty(records.shape[0], dtype=np.float64)
    for i in range(records.shape[0]):
        ref = np.asarray(reference_records[i], dtype=np.complex128)
        num = np.asarray(records[i], dtype=np.complex128)
        ref_norm = float(np.linalg.norm(ref))
        safe_ref_norm = max(ref_norm, norm_floor)
        out[i] = float(np.linalg.norm(num - ref) / safe_ref_norm)
    return out


def main() -> float:
    parser = build_example_parser(
        example_slug="linear_drift",
        description="Linear dispersive drift with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "linear_drift_rk4ip"
    case_key = "default"

    num_samples = 1024
    dt = 0.01
    sigma = 0.20
    beta2 = 0.05
    gamma = 0.0
    chirp = 12.0
    z_final = 1.0
    num_records = 180
    t = centered_time_grid(num_samples, dt)

    exec_options = SimulationOptions(backend="auto", fft_backend="auto")
    runner = NloExampleRunner()
    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        meta = loaded.meta
        num_samples = int(meta["num_samples"])
        dt = float(meta["dt"])
        sigma = float(meta["sigma"])
        beta2 = float(meta["beta2"])
        chirp = float(meta["chirp"])
        t = centered_time_grid(num_samples, dt)
        z_records = np.asarray(loaded.z_axis, dtype=np.float64)
        records = np.asarray(loaded.records, dtype=np.complex128)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        sim_cfg = TemporalSimulationConfig(
            gamma=gamma,
            beta2=beta2,
            alpha=0.0,
            dt=dt,
            z_final=z_final,
            num_time_samples=num_samples,
            pulse_period=float(num_samples) * dt,
            omega=None,
            error_tolerance=1e-5,
        )
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=8,
        )
        z_records, records = runner.propagate_temporal_records(
            field0,
            sim_cfg,
            num_records,
            exec_options,
            **storage_kwargs,
        )
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            solver_meta=runner.last_meta,
            meta={
                "num_samples": int(num_samples),
                "dt": float(dt),
                "sigma": float(sigma),
                "beta2": float(beta2),
                "chirp": float(chirp),
            },
        )

    field0 = gaussian_with_phase_ramp(t, sigma, chirp).astype(np.complex128)
    record_norms = np.asarray([float(np.linalg.norm(record)) for record in records], dtype=np.float64)

    centroid = centroid_curve(t, records)
    centroid_shift = centroid - centroid[0]
    measured_slope = float(np.polyfit(z_records, centroid_shift, 1)[0])
    predicted_slope = beta2 * chirp
    theory_shift = predicted_slope * z_records
    slope_rel_error = abs(measured_slope - predicted_slope) / max(abs(predicted_slope), 1e-12)
    reference_records = linear_reference_records(field0, z_records, beta2, dt)
    error_curve = relative_l2_error_curve(records, reference_records)

    output_dir = Path(__file__).resolve().parent / "output" / "linear_drift"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    saved = plot_intensity_colormap_vs_propagation(
        t,
        z_records,
        np.abs(records) ** 2,
        output_dir / "intensity_propagation_map.png",
        x_label="Time t",
        title="Linear Drift: Temporal Intensity Propagation",
        colorbar_label="Normalized intensity",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_final_re_im_comparison(
        t,
        records[0],
        records[-1],
        output_dir / "final_re_im_comparison.png",
        x_label="Time t",
        title="Linear Drift: Final Re/Im Comparison",
        reference_label="Initial",
        final_label="Final",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_final_intensity_comparison(
        t,
        records[0],
        records[-1],
        output_dir / "final_intensity_comparison.png",
        x_label="Time t",
        title="Linear Drift: Final Intensity Comparison",
        reference_label="Initial",
        final_label="Final",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_total_error_over_propagation(
        z_records,
        error_curve,
        output_dir / "total_error_over_propagation.png",
        title="Linear Drift: Full-Window Relative L2 Error Over Propagation",
        y_label="Relative L2 error (numerical vs analytical)",
    )
    if saved is not None:
        saved_paths.append(saved)

    print(f"linear drift example completed (run_group={run_group}).")
    print(f"record norms: first={record_norms[0]:.6e}, last={record_norms[-1]:.6e}, min={np.min(record_norms):.6e}")
    print(f"centroid shift: z0={centroid_shift[0]:.6e}, z_end={centroid_shift[-1]:.6e}")
    print(f"slope (signed): measured={measured_slope:.6e}, predicted={predicted_slope:.6e}")
    print(f"slope relative error: {slope_rel_error:.6e}")
    print(f"centroid theory final shift: {theory_shift[-1]:.6e}")
    if saved_paths:
        print("saved plots:")
        for path in saved_paths:
            print(f"  {path}")

    return slope_rel_error


if __name__ == "__main__":
    main()
