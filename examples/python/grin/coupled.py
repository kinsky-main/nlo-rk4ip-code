"""Full coupled GRIN example: dispersion + diffraction + Kerr nonlinearity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from backend.metrics import mean_pointwise_abs_relative_error_curve
from backend.plotting import (
    plot_3d_intensity_scatter_propagation,
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
    plot_two_curve_comparison,
)
from backend.runner import NloExampleRunner, SimulationOptions
from backend.storage import ExampleRunDB


def _relative_l2_error_curve(records_a: np.ndarray, records_b: np.ndarray) -> np.ndarray:
    return mean_pointwise_abs_relative_error_curve(
        records_a,
        records_b,
        context="grin_coupled:full_vs_linear",
    )


def _omega_grid(nt: int, dt: float) -> np.ndarray:
    return (2.0 * np.pi * np.fft.fftfreq(nt, d=dt)).astype(np.float64)


def _flatten_tfast(volume_tyx: np.ndarray) -> np.ndarray:
    return np.asarray(volume_tyx, dtype=np.complex128).transpose(2, 1, 0).reshape(-1)


def _unflatten_records_tfast(records_flat: np.ndarray, num_records: int, nt: int, ny: int, nx: int) -> np.ndarray:
    return np.asarray(records_flat, dtype=np.complex128).reshape(num_records, nx, ny, nt).transpose(0, 3, 2, 1)


@dataclass(frozen=True)
class CoupledGrinConfig:
    nt: int = 256
    nx: int = 64
    ny: int = 64
    dt: float = 0.02
    dx: float = 0.8
    dy: float = 0.8
    z_final: float = 0.50
    num_records: int = 50
    beta2: float = 0.6
    gamma_full: float = 0.35
    diffraction_coeff: float = -0.15
    grin_strength: float = 1.6e-2
    temporal_width: float = 0.26
    spatial_width: float = 8.0
    chirp: float = 4.0


def _run_case(
    runner: NloExampleRunner,
    *,
    gamma: float,
    cfg: CoupledGrinConfig,
    exec_options: SimulationOptions,
    storage_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    nlo = runner.nlo

    t = (np.arange(cfg.nt, dtype=np.float64) - 0.5 * (cfg.nt - 1)) * cfg.dt
    x = (np.arange(cfg.nx, dtype=np.float64) - 0.5 * (cfg.nx - 1)) * cfg.dx
    y = (np.arange(cfg.ny, dtype=np.float64) - 0.5 * (cfg.ny - 1)) * cfg.dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    temporal = np.exp(-((t / cfg.temporal_width) ** 2)) * np.exp((-1.0j) * cfg.chirp * t)
    spatial = np.exp(-((xx * xx + yy * yy) / (cfg.spatial_width * cfg.spatial_width)))
    field0 = (temporal[:, None, None] * spatial[None, :, :]).astype(np.complex128)

    omega = _omega_grid(cfg.nt, cfg.dt)
    potential = (cfg.grin_strength * (xx * xx + yy * yy)).astype(np.complex128)
    potential_tfast = np.tile(potential.T.reshape(-1), cfg.nt)

    runtime = nlo.RuntimeOperators(
        linear_factor_fn=lambda A, wt, kx, ky: (1.0j)
        * ((0.5 * cfg.beta2) * (wt * wt) + cfg.diffraction_coeff * ((kx * kx) + (ky * ky))),
        linear_fn=lambda A, D, h: np.exp(h * D),
        nonlinear_fn=lambda A, I, V: (1.0j * A) * (gamma * I + V),
    )
    propagate_kwargs: dict[str, object] = {}
    if storage_kwargs is not None:
        propagate_kwargs = {
            "sqlite_path": storage_kwargs["sqlite_path"],
            "run_id": storage_kwargs["run_id"],
            "chunk_records": storage_kwargs["chunk_records"],
            "sqlite_max_bytes": storage_kwargs["sqlite_max_bytes"],
            "log_final_output_field_to_db": storage_kwargs["log_final_output_field_to_db"],
        }
    z_records, records_flat = runner.propagate_tensor3d_records(
        _flatten_tfast(field0),
        nt=int(cfg.nt),
        nx=int(cfg.nx),
        ny=int(cfg.ny),
        num_records=int(cfg.num_records),
        propagation_distance=float(cfg.z_final),
        starting_step_size=8.0e-4,
        max_step_size=2.0e-3,
        min_step_size=1.0e-8,
        error_tolerance=2.0e-6,
        delta_x=float(cfg.dx),
        delta_y=float(cfg.dy),
        delta_time=float(cfg.dt),
        pulse_period=float(cfg.nt) * cfg.dt,
        frequency_grid=omega,
        potential_grid=potential_tfast,
        runtime=runtime,
        exec_options=exec_options,
        **propagate_kwargs,
    )
    records = _unflatten_records_tfast(records_flat, int(cfg.num_records), int(cfg.nt), int(cfg.ny), int(cfg.nx))
    return t, x, y, z_records, field0, records, dict(runner.last_meta)


class FullCoupledGrinApp:
    """Runs a full coupled GRIN simulation and a linear baseline comparison."""

    def __init__(self, args: Any):
        self.args = args
        self.db = ExampleRunDB(args.db_path)
        self.runner = NloExampleRunner()
        self.nlo = self.runner.nlo
        self.exec_options = SimulationOptions(backend="auto", fft_backend="auto", device_heap_fraction=0.70)
        self.example_name = "grin_full_coupled_rk4ip"
        self.full_case_key = "full"
        self.linear_case_key = "linear"

    def run(self) -> tuple[str, list[Path]]:
        cfg = CoupledGrinConfig()
        total_samples = cfg.nt * cfg.nx * cfg.ny
        runtime_limits = self.runner.api.query_runtime_limits(exec_options=self.exec_options.to_ctypes(self.nlo))
        if total_samples > int(runtime_limits.max_num_time_samples_runtime):
            raise ValueError(
                "Requested nt*nx*ny="
                f"{total_samples} exceeds runtime max_num_time_samples="
                f"{runtime_limits.max_num_time_samples_runtime}. "
                "Reduce nt/nx/ny or select a backend with a higher runtime limit."
            )

        if self.args.replot:
            run_group = self.db.resolve_replot_group(self.example_name, self.args.run_group)
            loaded_full = self.db.load_case(
                example_name=self.example_name,
                run_group=run_group,
                case_key=self.full_case_key,
            )
            loaded_linear = self.db.load_case(
                example_name=self.example_name,
                run_group=run_group,
                case_key=self.linear_case_key,
            )
            meta = loaded_full.meta
            cfg = CoupledGrinConfig(
                nt=int(meta["nt"]),
                nx=int(meta["nx"]),
                ny=int(meta["ny"]),
                dt=float(meta["dt"]),
                dx=float(meta["dx"]),
                dy=float(meta["dy"]),
                z_final=float(meta["z_final"]),
                num_records=int(meta["num_records"]),
                beta2=float(meta["beta2"]),
                gamma_full=float(meta["gamma_full"]),
                diffraction_coeff=float(meta["diffraction_coeff"]),
                grin_strength=float(meta["grin_strength"]),
                temporal_width=float(meta["temporal_width"]),
                spatial_width=float(meta["spatial_width"]),
                chirp=float(meta["chirp"]),
            )
            t = (np.arange(cfg.nt, dtype=np.float64) - 0.5 * (cfg.nt - 1)) * cfg.dt
            x = (np.arange(cfg.nx, dtype=np.float64) - 0.5 * (cfg.nx - 1)) * cfg.dx
            y = (np.arange(cfg.ny, dtype=np.float64) - 0.5 * (cfg.ny - 1)) * cfg.dy
            z_records = np.asarray(loaded_full.z_axis, dtype=np.float64)
            full_records = np.asarray(loaded_full.records, dtype=np.complex128).reshape(-1, cfg.nt, cfg.ny, cfg.nx)
            linear_records = np.asarray(loaded_linear.records, dtype=np.complex128).reshape(-1, cfg.nt, cfg.ny, cfg.nx)
        else:
            run_group = self.db.begin_group(self.example_name, self.args.run_group)
            full_storage_kwargs = self.db.storage_kwargs(
                example_name=self.example_name,
                run_group=run_group,
                case_key=self.full_case_key,
                chunk_records=2,
            )
            linear_storage_kwargs = self.db.storage_kwargs(
                example_name=self.example_name,
                run_group=run_group,
                case_key=self.linear_case_key,
                chunk_records=2,
            )
            t, x, y, z_records, _, full_records, full_meta = _run_case(
                self.runner,
                gamma=cfg.gamma_full,
                cfg=cfg,
                exec_options=self.exec_options,
                storage_kwargs=full_storage_kwargs,
            )
            _, _, _, _, _, linear_records, linear_meta = _run_case(
                self.runner,
                gamma=0.0,
                cfg=cfg,
                exec_options=self.exec_options,
                storage_kwargs=linear_storage_kwargs,
            )
            common_meta = {
                "nt": int(cfg.nt),
                "nx": int(cfg.nx),
                "ny": int(cfg.ny),
                "dt": float(cfg.dt),
                "dx": float(cfg.dx),
                "dy": float(cfg.dy),
                "z_final": float(cfg.z_final),
                "num_records": int(cfg.num_records),
                "beta2": float(cfg.beta2),
                "gamma_full": float(cfg.gamma_full),
                "diffraction_coeff": float(cfg.diffraction_coeff),
                "grin_strength": float(cfg.grin_strength),
                "temporal_width": float(cfg.temporal_width),
                "spatial_width": float(cfg.spatial_width),
                "chirp": float(cfg.chirp),
            }
            self.db.save_case(
                example_name=self.example_name,
                run_group=run_group,
                case_key=self.full_case_key,
                run_id=str(full_meta["storage_result"]["run_id"]),
                meta=common_meta,
            )
            self.db.save_case(
                example_name=self.example_name,
                run_group=run_group,
                case_key=self.linear_case_key,
                run_id=str(linear_meta["storage_result"]["run_id"]),
                meta=common_meta,
            )

        full_intensity = np.abs(full_records) ** 2
        linear_intensity = np.abs(linear_records) ** 2
        full_spatial_records = np.sum(full_intensity, axis=1)
        linear_spatial_records = np.sum(linear_intensity, axis=1)
        temporal_center_full = full_intensity[:, :, cfg.ny // 2, cfg.nx // 2]
        temporal_center_linear = linear_intensity[:, :, cfg.ny // 2, cfg.nx // 2]
        x_center_tmid_full = full_intensity[:, cfg.nt // 2, cfg.ny // 2, :]
        x_center_tmid_linear = linear_intensity[:, cfg.nt // 2, cfg.ny // 2, :]
        error_curve = _relative_l2_error_curve(full_records, linear_records)

        power_full = np.sum(full_intensity, axis=(1, 2, 3)).astype(np.float64)
        power_linear = np.sum(linear_intensity, axis=(1, 2, 3)).astype(np.float64)
        power_drift_full = float(abs(power_full[-1] - power_full[0]) / max(power_full[0], 1e-12))
        power_drift_linear = float(abs(power_linear[-1] - power_linear[0]) / max(power_linear[0], 1e-12))

        print(
            "full coupled GRIN propagation completed: "
            f"grid=(t={cfg.nt}, y={cfg.ny}, x={cfg.nx}), records={cfg.num_records}, "
            f"final_full_vs_linear_error={error_curve[-1]:.6e}"
        )
        print(f"run_group={run_group}")
        print(f"power drift: full={power_drift_full:.6e}, linear={power_drift_linear:.6e}")

        output_dir = self.args.output_dir / run_group
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        spatial_peak = float(max(np.max(full_spatial_records), np.max(linear_spatial_records)))
        temporal_peak = float(max(np.max(temporal_center_full), np.max(temporal_center_linear)))
        transverse_peak = float(max(np.max(x_center_tmid_full), np.max(x_center_tmid_linear)))

        p1 = plot_3d_intensity_scatter_propagation(
            x,
            y,
            z_records,
            full_spatial_records,
            output_dir / "grin_full_spatial_integrated_3d_scatter.png",
            intensity_cutoff=0.08,
            xy_stride=4,
            min_marker_size=2.0,
            max_marker_size=36.0,
            normalization_peak=spatial_peak,
            
            input_is_intensity=True,
        )
        if p1 is not None:
            saved_paths.append(p1)

        p2 = plot_3d_intensity_scatter_propagation(
            x,
            y,
            z_records,
            linear_spatial_records,
            output_dir / "grin_linear_baseline_spatial_integrated_3d_scatter.png",
            intensity_cutoff=0.08,
            xy_stride=4,
            min_marker_size=2.0,
            max_marker_size=36.0,
            normalization_peak=spatial_peak,
            
            input_is_intensity=True,
        )
        if p2 is not None:
            saved_paths.append(p2)

        p3 = plot_intensity_colormap_vs_propagation(
            t,
            z_records,
            temporal_center_full,
            output_dir / "grin_temporal_center_colormap_full.png",
            x_label="Time t",
            y_label="Propagation distance z",
            
            colorbar_label="Normalized intensity",
            normalization_peak=temporal_peak,
        )
        if p3 is not None:
            saved_paths.append(p3)

        p4 = plot_intensity_colormap_vs_propagation(
            x,
            z_records,
            x_center_tmid_full,
            output_dir / "grin_transverse_centerline_colormap_full.png",
            x_label="Transverse x (t=t_mid, y=y_mid)",
            y_label="Propagation distance z",
            
            colorbar_label="Normalized intensity",
            normalization_peak=transverse_peak,
        )
        if p4 is not None:
            saved_paths.append(p4)

        p5 = plot_final_re_im_comparison(
            t,
            linear_records[-1, :, cfg.ny // 2, cfg.nx // 2],
            full_records[-1, :, cfg.ny // 2, cfg.nx // 2],
            output_dir / "grin_final_temporal_center_re_im_comparison.png",
            x_label="Time t",
            
            reference_label="Linear baseline",
            final_label="Full coupled GRIN",
        )
        if p5 is not None:
            saved_paths.append(p5)

        p6 = plot_final_intensity_comparison(
            x,
            linear_records[-1, cfg.nt // 2, cfg.ny // 2, :],
            full_records[-1, cfg.nt // 2, cfg.ny // 2, :],
            output_dir / "grin_final_transverse_centerline_intensity_comparison.png",
            x_label="Transverse coordinate x",
            
            reference_label="Linear baseline",
            final_label="Full coupled GRIN",
        )
        if p6 is not None:
            saved_paths.append(p6)

        p7 = plot_total_error_over_propagation(
            z_records,
            error_curve,
            output_dir / "grin_full_vs_linear_relative_error_over_propagation.png",
            
            y_label="Mean pointwise abs-relative error",
        )
        if p7 is not None:
            saved_paths.append(p7)

        p8 = plot_two_curve_comparison(
            z_records,
            power_full,
            power_linear,
            output_dir / "grin_power_over_propagation_full_vs_linear.png",
            label_a="Full coupled GRIN",
            label_b="Linear baseline",
            y_label="Total power sum(|A|^2)",
            
        )
        if p8 is not None:
            saved_paths.append(p8)

        return run_group, saved_paths
