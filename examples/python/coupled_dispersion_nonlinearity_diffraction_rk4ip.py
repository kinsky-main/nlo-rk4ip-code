"""
Coupled (3+1)D example: dispersion + nonlinearity + diffraction.

This demo compares:
1) full coupled propagation (dispersion + diffraction + Kerr + GRIN potential)
2) linear baseline (dispersion + diffraction + GRIN potential, gamma=0)

Plots include static 3D propagation views and diagnostic cross-sections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.plotting import (
    plot_3d_intensity_scatter_propagation,
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from backend.runner import NloExampleRunner, SimulationOptions
from backend.storage import ExampleRunDB


def _relative_l2_error_curve(records_a: np.ndarray, records_b: np.ndarray) -> np.ndarray:
    if records_a.shape != records_b.shape:
        raise ValueError("records_a and records_b must have the same shape.")

    out = np.empty(records_a.shape[0], dtype=np.float64)
    for i in range(records_a.shape[0]):
        a = np.asarray(records_a[i], dtype=np.complex128).reshape(-1)
        b = np.asarray(records_b[i], dtype=np.complex128).reshape(-1)
        denom = max(float(np.linalg.norm(b)), 1e-12)
        out[i] = float(np.linalg.norm(a - b) / denom)
    return out


def _k2_grid(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    return (kkx * kkx + kky * kky).astype(np.complex128)


def _omega_grid(nt: int, dt: float) -> np.ndarray:
    return (2.0 * np.pi * np.fft.fftfreq(nt, d=dt)).astype(np.float64)


def _save_two_curve_plot(
    output_path: Path,
    z_axis: np.ndarray,
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    *,
    label_a: str,
    label_b: str,
    y_label: str,
    title: str,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping two-curve summary plot.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(z_axis, curve_a, lw=1.9, label=label_a)
    ax.plot(z_axis, curve_b, lw=1.8, ls="--", label=label_b)
    ax.set_xlabel("Propagation distance z")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _run_case(
    runner: NloExampleRunner,
    *,
    gamma: float,
    nt: int,
    nx: int,
    ny: int,
    dt: float,
    dx: float,
    dy: float,
    z_final: float,
    num_records: int,
    beta2: float,
    diffraction_coeff: float,
    grin_strength: float,
    temporal_width: float,
    spatial_width: float,
    chirp: float,
    exec_options: SimulationOptions,
    storage_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    nlo = runner.nlo
    api = runner.api

    t = (np.arange(nt, dtype=np.float64) - 0.5 * (nt - 1)) * dt
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    temporal = np.exp(-((t / temporal_width) ** 2)) * np.exp((-1.0j) * chirp * t)
    spatial = np.exp(-((xx * xx + yy * yy) / (spatial_width * spatial_width)))
    field0 = (temporal[:, None, None] * spatial[None, :, :]).astype(np.complex128)

    omega = _omega_grid(nt, dt)
    k2 = _k2_grid(nx, ny, dx, dy)
    potential = (grin_strength * (xx * xx + yy * yy)).astype(np.complex128)

    pulse = nlo.PulseSpec(
        samples=field0.reshape(-1).tolist(),
        delta_time=dt,
        pulse_period=float(nt) * dt,
        time_nt=nt,
        frequency_grid=[complex(float(w), 0.0) for w in omega],
        spatial_nx=nx,
        spatial_ny=ny,
        delta_x=dx,
        delta_y=dy,
        spatial_frequency_grid=k2.reshape(-1).tolist(),
        potential_grid=potential.reshape(-1).tolist(),
    )
    linear_operator = nlo.OperatorSpec(
        expr="i*beta2*w*w-loss",
        params={"beta2": 0.5 * beta2, "loss": 0.0},
    )
    transverse_operator = nlo.OperatorSpec(
        expr="i*beta_t*w",
        params={"beta_t": diffraction_coeff},
    )
    nonlinear_operator = nlo.OperatorSpec(
        expr="i*A*(gamma*I + V)",
        params={"gamma": gamma},
    )

    exec_options_ctypes = exec_options.to_ctypes(nlo)
    propagate_kwargs: dict[str, object] = {}
    if storage_kwargs is not None:
        propagate_kwargs = {
            "sqlite_path": storage_kwargs["sqlite_path"],
            "run_id": storage_kwargs["run_id"],
            "chunk_records": storage_kwargs["chunk_records"],
            "sqlite_max_bytes": storage_kwargs["sqlite_max_bytes"],
            "log_final_output_field_to_db": storage_kwargs["log_final_output_field_to_db"],
        }
    result = api.propagate(
        pulse,
        linear_operator,
        nonlinear_operator,
        transverse_operator=transverse_operator,
        propagation_distance=float(z_final),
        output="dense",
        preset="accuracy",
        records=num_records,
        exec_options=exec_options_ctypes,
        **propagate_kwargs,
    )
    records = np.asarray(result.records, dtype=np.complex128).reshape(num_records, nt, ny, nx)
    z_records = np.asarray(result.z_axis, dtype=np.float64)
    return t, x, y, z_records, field0, records, dict(result.meta)


def main() -> None:
    parser = build_example_parser(
        example_slug="coupled_dispersion_nonlinearity_diffraction",
        description="Coupled dispersion/nonlinearity/diffraction with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "coupled_dispersion_nonlinearity_diffraction_rk4ip"
    full_case_key = "full"
    linear_case_key = "linear"

    runner = NloExampleRunner()
    exec_options = SimulationOptions(backend="auto", fft_backend="auto", device_heap_fraction=0.70)

    nt = 256
    nx = 64
    ny = 64
    dt = 0.02
    dx = 0.8
    dy = 0.8
    z_final = 0.20
    num_records = 50

    beta2 = 0.06
    gamma_full = 0.45
    diffraction_coeff = -0.020
    grin_strength = 1.6e-4
    temporal_width = 0.22
    spatial_width = 8.0
    chirp = 8.0

    total_samples = nt * nx * ny
    runtime_limits = runner.api.query_runtime_limits(exec_options=exec_options.to_ctypes(runner.nlo))
    if total_samples > int(runtime_limits.max_num_time_samples_runtime):
        raise ValueError(
            "Requested nt*nx*ny="
            f"{total_samples} exceeds runtime max_num_time_samples="
            f"{runtime_limits.max_num_time_samples_runtime}. "
            "Reduce nt/nx/ny or select a backend with a higher runtime limit."
        )

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded_full = db.load_case(example_name=example_name, run_group=run_group, case_key=full_case_key)
        loaded_linear = db.load_case(example_name=example_name, run_group=run_group, case_key=linear_case_key)
        meta = loaded_full.meta
        nt = int(meta["nt"])
        nx = int(meta["nx"])
        ny = int(meta["ny"])
        dt = float(meta["dt"])
        dx = float(meta["dx"])
        dy = float(meta["dy"])
        t = (np.arange(nt, dtype=np.float64) - 0.5 * (nt - 1)) * dt
        x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
        y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
        z_records = np.asarray(loaded_full.z_axis, dtype=np.float64)
        full_records = np.asarray(loaded_full.records, dtype=np.complex128).reshape(-1, nt, ny, nx)
        linear_records = np.asarray(loaded_linear.records, dtype=np.complex128).reshape(-1, nt, ny, nx)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        full_storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=full_case_key,
            chunk_records=2,
        )
        linear_storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=linear_case_key,
            chunk_records=2,
        )
        t, x, y, z_records, _, full_records, full_meta = _run_case(
            runner,
            gamma=gamma_full,
            nt=nt,
            nx=nx,
            ny=ny,
            dt=dt,
            dx=dx,
            dy=dy,
            z_final=z_final,
            num_records=num_records,
            beta2=beta2,
            diffraction_coeff=diffraction_coeff,
            grin_strength=grin_strength,
            temporal_width=temporal_width,
            spatial_width=spatial_width,
            chirp=chirp,
            exec_options=exec_options,
            storage_kwargs=full_storage_kwargs,
        )
        _, _, _, _, _, linear_records, linear_meta = _run_case(
            runner,
            gamma=0.0,
            nt=nt,
            nx=nx,
            ny=ny,
            dt=dt,
            dx=dx,
            dy=dy,
            z_final=z_final,
            num_records=num_records,
            beta2=beta2,
            diffraction_coeff=diffraction_coeff,
            grin_strength=grin_strength,
            temporal_width=temporal_width,
            spatial_width=spatial_width,
            chirp=chirp,
            exec_options=exec_options,
            storage_kwargs=linear_storage_kwargs,
        )
        common_meta = {
            "nt": int(nt),
            "nx": int(nx),
            "ny": int(ny),
            "dt": float(dt),
            "dx": float(dx),
            "dy": float(dy),
        }
        db.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=full_case_key,
            run_id=str(full_meta["storage_result"]["run_id"]),
            meta={
                **common_meta,
                "gamma": float(gamma_full),
            },
        )
        db.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=linear_case_key,
            run_id=str(linear_meta["storage_result"]["run_id"]),
            meta={
                **common_meta,
                "gamma": 0.0,
            },
        )

    full_intensity = np.abs(full_records) ** 2
    linear_intensity = np.abs(linear_records) ** 2
    full_spatial_records = np.sum(full_intensity, axis=1)
    linear_spatial_records = np.sum(linear_intensity, axis=1)
    temporal_center_full = full_intensity[:, :, ny // 2, nx // 2]
    x_center_tmid_full = full_intensity[:, nt // 2, ny // 2, :]
    error_curve = _relative_l2_error_curve(full_records, linear_records)

    power_full = np.sum(full_intensity, axis=(1, 2, 3)).astype(np.float64)
    power_linear = np.sum(linear_intensity, axis=(1, 2, 3)).astype(np.float64)
    power_drift_full = float(abs(power_full[-1] - power_full[0]) / max(power_full[0], 1e-12))
    power_drift_linear = float(abs(power_linear[-1] - power_linear[0]) / max(power_linear[0], 1e-12))
    print(
        "coupled propagation completed: "
        f"grid=(t={nt}, y={ny}, x={nx}), records={num_records}, "
        f"final_full_vs_linear_error={error_curve[-1]:.6e}"
    )
    print(f"run_group={run_group}")
    print(
        "power drift: "
        f"full={power_drift_full:.6e}, linear={power_drift_linear:.6e}"
    )

    output_dir = Path(__file__).resolve().parent / "output" / "coupled_dispersion_nonlinearity_diffraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    p1 = plot_3d_intensity_scatter_propagation(
        x,
        y,
        z_records,
        full_spatial_records,
        output_dir / "full_spatial_integrated_3d_scatter.png",
        intensity_cutoff=0.08,
        xy_stride=4,
        min_marker_size=2.0,
        max_marker_size=36.0,
        title="Full coupled case: spatial intensity integrated over time",
    )
    if p1 is not None:
        saved_paths.append(p1)

    p2 = plot_3d_intensity_scatter_propagation(
        x,
        y,
        z_records,
        linear_spatial_records,
        output_dir / "linear_baseline_spatial_integrated_3d_scatter.png",
        intensity_cutoff=0.08,
        xy_stride=4,
        min_marker_size=2.0,
        max_marker_size=36.0,
        title="Linear baseline: spatial intensity integrated over time",
    )
    if p2 is not None:
        saved_paths.append(p2)

    p3 = plot_intensity_colormap_vs_propagation(
        t,
        z_records,
        temporal_center_full,
        output_dir / "temporal_center_colormap_full.png",
        x_label="Time t",
        y_label="Propagation distance z",
        title="Full coupled case: center-point temporal intensity vs z",
        colorbar_label="Normalized intensity",
        cmap="magma",
    )
    if p3 is not None:
        saved_paths.append(p3)

    p4 = plot_intensity_colormap_vs_propagation(
        x,
        z_records,
        x_center_tmid_full,
        output_dir / "transverse_centerline_colormap_full.png",
        x_label="Transverse x (t = t_mid, y = y_mid)",
        y_label="Propagation distance z",
        title="Full coupled case: transverse center-line intensity vs z",
        colorbar_label="Normalized intensity",
        cmap="viridis",
    )
    if p4 is not None:
        saved_paths.append(p4)

    p5 = plot_final_re_im_comparison(
        t,
        linear_records[-1, :, ny // 2, nx // 2],
        full_records[-1, :, ny // 2, nx // 2],
        output_dir / "final_temporal_center_re_im_comparison.png",
        x_label="Time t",
        title="Final center-point temporal field (linear baseline vs full)",
        reference_label="Linear baseline",
        final_label="Full coupled",
    )
    if p5 is not None:
        saved_paths.append(p5)

    p6 = plot_final_intensity_comparison(
        x,
        linear_records[-1, nt // 2, ny // 2, :],
        full_records[-1, nt // 2, ny // 2, :],
        output_dir / "final_transverse_centerline_intensity_comparison.png",
        x_label="Transverse coordinate x",
        title="Final transverse center-line intensity (linear baseline vs full)",
        reference_label="Linear baseline",
        final_label="Full coupled",
    )
    if p6 is not None:
        saved_paths.append(p6)

    p7 = plot_total_error_over_propagation(
        z_records,
        error_curve,
        output_dir / "full_vs_linear_relative_error_over_propagation.png",
        title="Full coupled vs linear baseline: relative L2 error over z",
        y_label="Relative L2 error",
    )
    if p7 is not None:
        saved_paths.append(p7)

    p8 = _save_two_curve_plot(
        output_dir / "power_over_propagation_full_vs_linear.png",
        z_records,
        power_full,
        power_linear,
        label_a="Full coupled",
        label_b="Linear baseline",
        y_label="Total power sum(|A|^2)",
        title="Power trend over propagation",
    )
    if p8 is not None:
        saved_paths.append(p8)

    if saved_paths:
        print("saved plots:")
        for path in saved_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
