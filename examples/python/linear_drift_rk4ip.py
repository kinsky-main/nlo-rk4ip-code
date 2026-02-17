"""
Linear dispersive drift example using the OOP example runner.

The simulation disables nonlinearity (gamma=0) and applies an initial phase
ramp exp(-i*d*t) to a Gaussian pulse, which induces temporal drift as the
pulse propagates along z.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from backend.runner import NloExampleRunner, SimulationOptions, TemporalSimulationConfig

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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


def save_plots(
    t: np.ndarray,
    z_records: np.ndarray,
    records: np.ndarray,
    centroid_shift: np.ndarray,
    theory_shift: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    if plt is None:
        print("matplotlib not available; skipping plot generation.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    intensity = np.abs(records) ** 2
    norm = np.max(intensity)
    intensity_map = intensity / norm if norm > 0.0 else intensity

    fig_1, ax_1 = plt.subplots(figsize=(9.0, 5.0))
    ax_1.plot(t, intensity[0], lw=2.0, label="Initial intensity")
    ax_1.plot(t, intensity[-1], lw=1.8, ls="--", label="Final intensity")
    ax_1.set_xlabel("Time t")
    ax_1.set_ylabel("Intensity |A|^2")
    ax_1.set_title("Linear Drift: Initial vs Final Pulse Intensity")
    ax_1.grid(True, alpha=0.3)
    ax_1.legend()
    p1 = output_dir / "pulse_overlay.png"
    fig_1.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig_1)
    saved_paths.append(p1)

    fig_2, ax_2 = plt.subplots(figsize=(9.0, 5.2))
    mesh = ax_2.pcolormesh(t, z_records, intensity_map, shading="auto", cmap="viridis")
    ax_2.set_xlabel("Time t")
    ax_2.set_ylabel("Propagation distance z")
    ax_2.set_title("Linear Drift: Temporal Intensity Propagation")
    cbar = fig_2.colorbar(mesh, ax=ax_2)
    cbar.set_label("Normalized intensity")
    p2 = output_dir / "intensity_propagation_map.png"
    fig_2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig_2)
    saved_paths.append(p2)

    fig_3, ax_3 = plt.subplots(figsize=(9.0, 4.8))
    ax_3.plot(z_records, centroid_shift, lw=2.0, label="Measured centroid shift")
    ax_3.plot(z_records, theory_shift, lw=1.5, ls="--", label="Linear |beta2*d| prediction")
    ax_3.set_xlabel("Propagation distance z")
    ax_3.set_ylabel("Centroid shift")
    ax_3.set_title("Linear Drift: Centroid Shift vs Propagation")
    ax_3.grid(True, alpha=0.3)
    ax_3.legend()
    p3 = output_dir / "centroid_shift_vs_z.png"
    fig_3.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig_3)
    saved_paths.append(p3)

    return saved_paths


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

    centroid = centroid_curve(t, records)
    centroid_shift = centroid - centroid[0]
    measured_slope = float(np.polyfit(z_records, centroid_shift, 1)[0])
    predicted_abs_slope = abs(beta2 * chirp)
    theory_slope = math.copysign(predicted_abs_slope, measured_slope if measured_slope != 0.0 else 1.0)
    theory_shift = theory_slope * z_records
    slope_rel_error = abs(abs(measured_slope) - predicted_abs_slope) / max(predicted_abs_slope, 1e-12)

    output_dir = Path(__file__).resolve().parent / "output" / "linear_drift"
    saved_paths = save_plots(
        t=t,
        z_records=z_records,
        records=records,
        centroid_shift=centroid_shift,
        theory_shift=theory_shift,
        output_dir=output_dir,
    )

    print("linear drift example completed.")
    print(f"centroid shift: z0={centroid_shift[0]:.6e}, z_end={centroid_shift[-1]:.6e}")
    print(f"slope magnitude: measured={abs(measured_slope):.6e}, predicted={predicted_abs_slope:.6e}")
    print(f"slope relative error: {slope_rel_error:.6e}")
    if saved_paths:
        print("saved plots:")
        for path in saved_paths:
            print(f"  {path}")

    return slope_rel_error


if __name__ == "__main__":
    main()
