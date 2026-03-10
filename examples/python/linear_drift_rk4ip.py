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
from backend.app_base import ExampleAppBase
from backend.metrics import relative_l2_error_curve
from backend.plotting import (
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from backend.reference import exact_linear_temporal_records
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

def _run(args: argparse.Namespace) -> float:
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
    field0 = gaussian_with_phase_ramp(t, sigma, chirp).astype(np.complex128)

    exec_options = SimulationOptions(backend="cpu", fft_backend="fftw")
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
        field0 = gaussian_with_phase_ramp(t, sigma, chirp).astype(np.complex128)
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

    record_norms = np.asarray([float(np.linalg.norm(record)) for record in records], dtype=np.float64)

    centroid = centroid_curve(t, records)
    centroid_shift = centroid - centroid[0]
    measured_slope = float(np.polyfit(z_records, centroid_shift, 1)[0])
    predicted_slope = beta2 * chirp
    theory_shift = predicted_slope * z_records
    slope_rel_error = abs(measured_slope - predicted_slope) / max(abs(predicted_slope), 1e-12)
    reference_records = exact_linear_temporal_records(
        field0,
        z_records,
        2.0 * np.pi * np.fft.fftfreq(num_samples, d=dt),
        0.5 * beta2,
        0.0,
    )
    error_curve = 100.0 * relative_l2_error_curve(records, reference_records)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    saved = plot_intensity_colormap_vs_propagation(
        t,
        z_records,
        np.abs(records) ** 2,
        output_dir / "linear_drift_intensity_propagation_map.png",
        x_label="Time t",
        
        colorbar_label="Normalized intensity",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_final_re_im_comparison(
        t,
        records[0],
        records[-1],
        output_dir / "linear_drift_final_re_im_comparison.png",
        x_label="Time t",
        
        reference_label="Initial",
        final_label="Final",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_final_intensity_comparison(
        t,
        records[0],
        records[-1],
        output_dir / "linear_drift_final_intensity_comparison.png",
        x_label="Time t",
        
        reference_label="Initial",
        final_label="Final",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_total_error_over_propagation(
        z_records,
        error_curve,
        output_dir / "linear_drift_total_error_over_propagation.png",
        y_label="Relative L2 field error (%)",
    )
    if saved is not None:
        saved_paths.append(saved)

    print(f"linear drift example completed (run_group={run_group}).")
    print(f"record norms: first={record_norms[0]:.6e}, last={record_norms[-1]:.6e}, min={np.min(record_norms):.6e}")
    print(f"centroid shift: z0={centroid_shift[0]:.6e}, z_end={centroid_shift[-1]:.6e}")
    print(f"slope (signed): measured={measured_slope:.6e}, predicted={predicted_slope:.6e}")
    print(f"slope relative error: {slope_rel_error:.6e}")
    print(f"centroid theory final shift: {theory_shift[-1]:.6e}")

    return slope_rel_error


class LinearDriftApp(ExampleAppBase):
    example_slug = "linear_drift"
    description = "Linear dispersive drift with DB-backed run/replot."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return LinearDriftApp.from_cli().run()


if __name__ == "__main__":
    main()
