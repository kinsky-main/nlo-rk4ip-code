"""
Minimal flattened-XY GRIN propagation example using the ctypes API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from backend.plotting import (
    plot_3d_intensity_scatter_propagation,
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from backend.runner import NloExampleRunner, SimulationOptions


def relative_l2_error_curve(records: np.ndarray, reference_records: np.ndarray) -> np.ndarray:
    if records.shape != reference_records.shape:
        raise ValueError("records and reference_records must have the same shape.")

    out = np.empty(records.shape[0], dtype=np.float64)
    for i in range(records.shape[0]):
        ref = np.asarray(reference_records[i], dtype=np.complex128)
        ref_norm = float(np.linalg.norm(ref))
        safe_ref_norm = ref_norm if ref_norm > 0.0 else 1.0
        out[i] = float(np.linalg.norm(np.asarray(records[i], dtype=np.complex128) - ref) / safe_ref_norm)
    return out


def main() -> None:
    runner = NloExampleRunner()

    nx = 1024
    ny = 1024
    nxy = nx * ny
    dx = 0.5
    dy = 0.5

    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    w0 = 8.0
    grin_gx = 2.0e-4
    grin_gy = 2.0e-4
    field0 = np.exp(-((xx * xx + yy * yy) / (w0 * w0))).astype(np.complex128)
    field0_flat = field0.reshape(-1)

    num_records = 8
    exec_opts = SimulationOptions(
        backend="auto",
        fft_backend="auto",
        device_heap_fraction=0.70,
    )
    z_records, records = runner.propagate_flattened_xy_records(
        field0_flat=field0_flat,
        nx=nx,
        ny=ny,
        num_records=num_records,
        propagation_distance=0.25,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=5e-5,
        error_tolerance=1e-7,
        delta_x=dx,
        delta_y=dy,
        grin_gx=grin_gx,
        grin_gy=grin_gy,
        gamma=0.0,
        alpha=0.0,
        exec_options=exec_opts,
    )

    in_power = float(np.sum(np.abs(records[0]) ** 2))
    out_power = float(np.sum(np.abs(records[-1]) ** 2))
    print(f"GRIN XY propagation completed: records={num_records}, shape=({ny}, {nx})")
    print(f"Power trend: z0={in_power:.6e}, z_end={out_power:.6e}")

    phase_unit = (grin_gx * (xx * xx)) + (grin_gy * (yy * yy))
    analytical_records = np.empty_like(records, dtype=np.complex128)
    for i, z in enumerate(z_records):
        analytical_records[i] = field0 * np.exp((1.0j) * phase_unit * float(z))

    full_error = relative_l2_error_curve(
        np.asarray(records, dtype=np.complex128).reshape(num_records, nxy),
        np.asarray(analytical_records, dtype=np.complex128).reshape(num_records, nxy),
    )

    center_row = ny // 2
    profile_records = np.asarray(records[:, center_row, :], dtype=np.complex128)
    analytical_profile_records = np.asarray(analytical_records[:, center_row, :], dtype=np.complex128)
    analytical_final_profile = analytical_profile_records[-1]
    final_profile = profile_records[-1]
    profile_error = relative_l2_error_curve(profile_records, analytical_profile_records)

    output_dir = Path(__file__).resolve().parent / "output" / "grin_fiber_xy"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    p1 = plot_intensity_colormap_vs_propagation(
        x,
        z_records,
        np.abs(profile_records) ** 2,
        output_dir / "profile_intensity_colormap.png",
        x_label="Transverse coordinate x",
        y_label="Propagation distance z",
        title="GRIN Fiber: Center-Line Intensity Profile vs Propagation",
        colorbar_label="Normalized center-line intensity",
        cmap="viridis",
    )
    if p1 is not None:
        saved_paths.append(p1)

    p2 = plot_final_re_im_comparison(
        x,
        analytical_final_profile,
        final_profile,
        output_dir / "final_re_im_profile_comparison.png",
        x_label="Transverse coordinate x",
        title="GRIN Fiber: Final Re/Im Profile (Analytical vs Numerical)",
        reference_label="Analytical final",
        final_label="Numerical final",
    )
    if p2 is not None:
        saved_paths.append(p2)

    p3 = plot_final_intensity_comparison(
        x,
        analytical_final_profile,
        final_profile,
        output_dir / "final_intensity_profile_comparison.png",
        x_label="Transverse coordinate x",
        title="GRIN Fiber: Final Intensity Profile (Analytical vs Numerical)",
        reference_label="Analytical final",
        final_label="Numerical final",
    )
    if p3 is not None:
        saved_paths.append(p3)

    p4 = plot_total_error_over_propagation(
        z_records,
        full_error,
        output_dir / "total_profile_error_over_propagation.png",
        title="GRIN Fiber: Total Error Over Propagation (Analytical vs Numerical)",
        y_label="Relative L2 error (full transverse field)",
    )
    if p4 is not None:
        saved_paths.append(p4)

    p5 = plot_3d_intensity_scatter_propagation(
        x,
        y,
        z_records,
        np.asarray(records, dtype=np.complex128),
        output_dir / "propagation_3d_intensity_scatter.png",
        intensity_cutoff=0.08,
        xy_stride=24,
        min_marker_size=2.0,
        max_marker_size=40.0,
        title="GRIN Fiber: 3D Propagation Scatter (Intensity-Weighted)",
    )
    if p5 is not None:
        saved_paths.append(p5)

    print(f"analytical comparison: final full-field relative L2 error={full_error[-1]:.6e}")
    print(f"analytical comparison: final center-line relative L2 error={profile_error[-1]:.6e}")
    if saved_paths:
        print("saved profile plots:")
        for path in saved_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
