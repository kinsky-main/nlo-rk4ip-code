"""
Linear dispersive drift example using the OOP example runner.

The simulation disables nonlinearity (gamma=0) and applies an initial phase
ramp exp(-i*d*t) to a Gaussian pulse, which induces temporal drift as the
pulse propagates along z.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from backend.plotting import (
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from backend.runner import NloExampleRunner, SimulationOptions, TemporalSimulationConfig


def centered_time_grid(num_samples: int, delta_time: float) -> np.ndarray:
    return (np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)) * delta_time


def gaussian_with_phase_ramp(t: np.ndarray, sigma: float, d: float) -> np.ndarray:
    envelope = np.exp(-((t / sigma) ** 2))
    return envelope * np.exp((-1.0j) * d * t)


def centroid_curve(t: np.ndarray, records: np.ndarray) -> np.ndarray:
    intensity = np.abs(records) ** 2
    weighted = intensity @ t
    norm = np.sum(intensity, axis=1)
    safe_norm = np.where(norm > 0.0, norm, 1.0)
    return weighted / safe_norm


def centroid_model_error_curve(
    measured_shift: np.ndarray,
    theory_shift: np.ndarray,
    scale: float,
) -> np.ndarray:
    safe_scale = scale if scale > 0.0 else 1.0
    return np.abs(np.asarray(measured_shift) - np.asarray(theory_shift)) / safe_scale


def main() -> float:
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

    sim_cfg = TemporalSimulationConfig(
        gamma=gamma,
        beta2=beta2,
        alpha=0.0,
        dt=dt,
        z_final=z_final,
        num_time_samples=num_samples,
        pulse_period=float(num_samples) * dt,
        omega=None,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
    )
    exec_opts = SimulationOptions(backend="auto", fft_backend="auto")

    runner = NloExampleRunner()
    z_records, records = runner.propagate_temporal_records(field0, sim_cfg, num_records, exec_opts)
    record_norms = np.asarray([float(np.linalg.norm(record)) for record in records], dtype=np.float64)

    centroid = centroid_curve(t, records)
    centroid_shift = centroid - centroid[0]
    measured_slope = float(np.polyfit(z_records, centroid_shift, 1)[0])
    predicted_slope = beta2 * chirp
    theory_shift = predicted_slope * z_records
    slope_rel_error = abs(measured_slope - predicted_slope) / max(abs(predicted_slope), 1e-12)
    end_shift_scale = abs(predicted_slope * z_final)
    error_curve = centroid_model_error_curve(centroid_shift, theory_shift, end_shift_scale)

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
        cmap="viridis",
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
        title="Linear Drift: Centroid Model Error Over Propagation",
        y_label="Normalized |measured - theory|",
    )
    if saved is not None:
        saved_paths.append(saved)

    print("linear drift example completed.")
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
