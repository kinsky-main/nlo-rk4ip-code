"""
High-order temporal soliton propagation in a GRIN-guided tensor geometry.

The example launches a single-lobed guided spatial mode with a higher-order
temporal soliton envelope and compares the nonlinear propagation against a
linear baseline using the C-backed Python API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from backend.app_base import ExampleAppBase
from backend.plotting import (
    plot_3d_intensity_contours_propagation,
    plot_intensity_colormap_vs_propagation,
    save_3d_intensity_time_sweep_video,
    plot_three_curve_drift,
    plot_two_curve_comparison,
)
from backend.runner import centered_time_grid
from backend.storage import ExampleRunDB


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

import nlolib as nlo


def centered_spatial_grid(num_samples: int, delta: float) -> np.ndarray:
    return (np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)) * float(delta)


def guided_spatial_mode(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    mode_width: float,
    chirp: float,
) -> np.ndarray:
    xx, yy = np.meshgrid(np.asarray(x_axis, dtype=np.float64), np.asarray(y_axis, dtype=np.float64))
    r2 = (xx * xx) + (yy * yy)
    envelope = np.exp(-r2 / (2.0 * float(mode_width) * float(mode_width)))
    return envelope * np.exp(1.0j * float(chirp) * r2)


def sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(np.asarray(x, dtype=np.float64))


def dispersion_length(beta2: float, temporal_width: float) -> float:
    return (float(temporal_width) * float(temporal_width)) / abs(float(beta2))


def soliton_period(beta2: float, temporal_width: float) -> float:
    return 0.5 * np.pi * dispersion_length(beta2, temporal_width)


def fundamental_soliton_power(beta2: float, gamma: float, temporal_width: float) -> float:
    return abs(float(beta2)) / (float(gamma) * float(temporal_width) * float(temporal_width))


def diffraction_length(beta_t: float, mode_width: float) -> float:
    return (float(mode_width) * float(mode_width)) / abs(float(beta_t))


def temporal_envelope(
    t_axis: np.ndarray,
    *,
    temporal_width: float,
    beta2: float,
    gamma: float,
    soliton_order: float,
) -> np.ndarray:
    t = np.asarray(t_axis, dtype=np.float64)
    p1 = fundamental_soliton_power(beta2, gamma, temporal_width)
    return float(soliton_order) * np.sqrt(p1) * sech(t / float(temporal_width))


def grin_launch_field(
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    temporal_width: float,
    beta2: float,
    gamma: float,
    soliton_order: float,
    mode_width: float,
    spatial_chirp: float,
) -> np.ndarray:
    temporal = temporal_envelope(
        t_axis,
        temporal_width=temporal_width,
        beta2=beta2,
        gamma=gamma,
        soliton_order=soliton_order,
    ).astype(np.complex128)
    transverse = guided_spatial_mode(
        x_axis,
        y_axis,
        mode_width=mode_width,
        chirp=spatial_chirp,
    ).astype(np.complex128)
    return temporal[:, None, None] * transverse[None, :, :]


def flatten_tfast(field_tyx: np.ndarray) -> np.ndarray:
    field = np.asarray(field_tyx, dtype=np.complex128)
    if field.ndim != 3:
        raise ValueError("field_tyx must have shape [nt, ny, nx].")
    return np.transpose(field, (2, 1, 0)).reshape(-1)


def unflatten_tfast_records(
    records_flat: np.ndarray,
    *,
    num_records: int,
    nt: int,
    ny: int,
    nx: int,
) -> np.ndarray:
    flat = np.asarray(records_flat, dtype=np.complex128).reshape(int(num_records), int(nx), int(ny), int(nt))
    return np.transpose(flat, (0, 3, 2, 1))


def overlap_fidelity_curve(records_tyx: np.ndarray, launch_tyx: np.ndarray) -> np.ndarray:
    reference = np.asarray(launch_tyx, dtype=np.complex128).reshape(-1)
    reference_norm = max(float(np.linalg.norm(reference)), 1.0e-30)
    out = np.empty(int(records_tyx.shape[0]), dtype=np.float64)
    for idx in range(int(records_tyx.shape[0])):
        record = np.asarray(records_tyx[idx], dtype=np.complex128).reshape(-1)
        denom = max(reference_norm * float(np.linalg.norm(record)), 1.0e-30)
        out[idx] = float(abs(np.vdot(reference, record)) / denom)
    return out


def rms_radius_curve(records_tyx: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(np.asarray(x_axis, dtype=np.float64), np.asarray(y_axis, dtype=np.float64))
    r2 = (xx * xx) + (yy * yy)
    out = np.empty(int(records_tyx.shape[0]), dtype=np.float64)
    for idx in range(int(records_tyx.shape[0])):
        intensity = np.sum(np.abs(np.asarray(records_tyx[idx], dtype=np.complex128)) ** 2, axis=0)
        total = max(float(np.sum(intensity)), 1.0e-30)
        out[idx] = float(np.sqrt(np.sum(r2 * intensity) / total))
    return out


def rms_temporal_width_curve(records_tyx: np.ndarray, t_axis: np.ndarray) -> np.ndarray:
    t = np.asarray(t_axis, dtype=np.float64).reshape(-1)
    out = np.empty(int(records_tyx.shape[0]), dtype=np.float64)
    for idx in range(int(records_tyx.shape[0])):
        weights = np.sum(np.abs(np.asarray(records_tyx[idx], dtype=np.complex128)) ** 2, axis=(1, 2))
        total = max(float(np.sum(weights)), 1.0e-30)
        center = float(np.sum(t * weights) / total)
        out[idx] = float(np.sqrt(np.sum(((t - center) ** 2) * weights) / total))
    return out


def peak_intensity_curve(records_tyx: np.ndarray) -> np.ndarray:
    return np.asarray([float(np.max(np.abs(record) ** 2)) for record in records_tyx], dtype=np.float64)


def total_power_curve(records_tyx: np.ndarray) -> np.ndarray:
    return np.asarray([float(np.sum(np.abs(record) ** 2)) for record in records_tyx], dtype=np.float64)


def relative_power_drift_curve(power_curve: np.ndarray) -> np.ndarray:
    power = np.asarray(power_curve, dtype=np.float64).reshape(-1)
    baseline = max(float(power[0]), 1.0e-30)
    return np.abs(power - float(power[0])) / baseline


def centerline_intensity_map(records_tyx: np.ndarray) -> np.ndarray:
    records = np.asarray(records_tyx, dtype=np.complex128)
    center_row = int(records.shape[2] // 2)
    return np.sum(np.abs(records[:, :, center_row, :]) ** 2, axis=1)


def time_integrated_xy_records(records_tyx: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.asarray(records_tyx, dtype=np.complex128)) ** 2, axis=1)


def temporal_marginal_curve(records_tyx: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.asarray(records_tyx, dtype=np.complex128)) ** 2, axis=(2, 3))


def grin_potential_grid(x_axis: np.ndarray, y_axis: np.ndarray, grin_strength: float) -> np.ndarray:
    xx, yy = np.meshgrid(np.asarray(x_axis, dtype=np.float64), np.asarray(y_axis, dtype=np.float64))
    return (float(grin_strength) * ((xx * xx) + (yy * yy))).astype(np.complex128)


def _run_case(
    api: nlo.NLolib,
    field0_tyx: np.ndarray,
    *,
    nt: int,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    beta2: float,
    beta_t: float,
    gamma: float,
    grin_strength: float,
    z_final: float,
    num_records: int,
    exec_options,
    storage_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    potential = grin_potential_grid(
        centered_spatial_grid(nx, dx),
        centered_spatial_grid(ny, dy),
        grin_strength,
    ).reshape(-1)
    cfg = nlo.prepare_sim_config(
        nt * nx * ny,
        propagation_distance=float(z_final),
        starting_step_size=5.0e-3,
        max_step_size=2.0e-1,
        min_step_size=1.0e-5,
        error_tolerance=1.0e-6,
        pulse_period=float(nt) * float(dt),
        delta_time=float(dt),
        tensor_nt=int(nt),
        tensor_nx=int(nx),
        tensor_ny=int(ny),
        tensor_layout=int(nlo.TENSOR_LAYOUT_XYT_T_FAST),
        delta_x=float(dx),
        delta_y=float(dy),
        frequency_grid=(2.0 * np.pi * np.fft.fftfreq(int(nt), d=float(dt))).astype(np.complex128).tolist(),
        potential_grid=potential.tolist(),
        runtime=nlo.RuntimeOperators(
            linear_factor_expr="i*(c0*(wt*wt) + c1*(kx*kx + ky*ky))",
            linear_expr="exp(h*D)",
            nonlinear_expr="i*A*(c2*I + V)",
            constants=[0.5 * float(beta2), float(beta_t), float(gamma), 0.0],
        ),
    )
    kwargs = dict(storage_kwargs or {})
    result = api.propagate(
        cfg,
        flatten_tfast(field0_tyx).tolist(),
        int(num_records),
        exec_options,
        t_eval=np.linspace(0.0, float(z_final), int(num_records)).tolist(),
        **kwargs,
    )
    z_axis = np.asarray(result.z_axis, dtype=np.float64)
    records_tyx = unflatten_tfast_records(
        np.asarray(result.records, dtype=np.complex128),
        num_records=len(z_axis),
        nt=nt,
        ny=ny,
        nx=nx,
    )
    return z_axis, records_tyx, dict(result.meta)


def _run(args: argparse.Namespace) -> float:
    nlo.set_progress_options(enabled=False)
    try:
        nlo.set_log_level(nlo.NLOLIB_LOG_LEVEL_ERROR)
    except RuntimeError:
        pass

    db = ExampleRunDB(args.db_path)
    api = nlo.NLolib()
    example_name = "high_order_grin_soliton_rk4ip"
    nonlinear_case_key = "nonlinear"
    linear_case_key = "linear_baseline"

    nt = 1024
    nx = 64
    ny = 64
    dt = 0.06
    dx = 0.24
    dy = 0.24
    temporal_width = 0.30
    soliton_order = 1.0
    mode_width = 2.4
    spatial_chirp = 0.0
    beta2 = -0.08
    beta_t = -0.08
    grin_strength = 1.5e-3
    gamma_nonlinear = 1.0
    z_period = soliton_period(beta2, temporal_width)
    z_final = 2.0 * z_period
    num_records = 72

    t_axis = centered_time_grid(nt, dt)
    x_axis = centered_spatial_grid(nx, dx)
    y_axis = centered_spatial_grid(ny, dy)
    field0_tyx = grin_launch_field(
        t_axis,
        x_axis,
        y_axis,
        temporal_width=temporal_width,
        beta2=beta2,
        gamma=gamma_nonlinear,
        soliton_order=soliton_order,
        mode_width=mode_width,
        spatial_chirp=spatial_chirp,
    ).astype(np.complex128)

    exec_options = nlo.default_execution_options(
        backend_type=nlo.VECTOR_BACKEND_CPU,
        fft_backend=nlo.FFT_BACKEND_FFTW,
    )

    if args.replot:
        run_group = db.resolve_replot_group(
            example_name,
            args.run_group,
            required_case_keys=[nonlinear_case_key, linear_case_key],
        )
        loaded_nonlinear = db.load_case(example_name=example_name, run_group=run_group, case_key=nonlinear_case_key)
        loaded_linear = db.load_case(example_name=example_name, run_group=run_group, case_key=linear_case_key)
        meta = loaded_nonlinear.meta
        nt = int(meta["nt"])
        nx = int(meta["nx"])
        ny = int(meta["ny"])
        dt = float(meta["dt"])
        dx = float(meta["dx"])
        dy = float(meta["dy"])
        temporal_width = float(meta["temporal_width"])
        soliton_order = float(meta["soliton_order"])
        mode_width = float(meta["mode_width"])
        spatial_chirp = float(meta["spatial_chirp"])
        beta2 = float(meta["beta2"])
        beta_t = float(meta["beta_t"])
        grin_strength = float(meta["grin_strength"])
        gamma_nonlinear = float(meta["gamma_nonlinear"])
        t_axis = centered_time_grid(nt, dt)
        x_axis = centered_spatial_grid(nx, dx)
        y_axis = centered_spatial_grid(ny, dy)
        field0_tyx = grin_launch_field(
            t_axis,
            x_axis,
            y_axis,
            temporal_width=temporal_width,
            beta2=beta2,
            gamma=gamma_nonlinear,
            soliton_order=soliton_order,
            mode_width=mode_width,
            spatial_chirp=spatial_chirp,
        ).astype(np.complex128)
        z_axis = np.asarray(loaded_nonlinear.z_axis, dtype=np.float64)
        nonlinear_records = unflatten_tfast_records(
            loaded_nonlinear.records,
            num_records=len(z_axis),
            nt=nt,
            ny=ny,
            nx=nx,
        )
        linear_records = unflatten_tfast_records(
            loaded_linear.records,
            num_records=len(loaded_linear.z_axis),
            nt=nt,
            ny=ny,
            nx=nx,
        )
    else:
        run_group = db.begin_group(example_name, args.run_group)
        storage_nonlinear = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=nonlinear_case_key,
            chunk_records=4,
        )
        storage_linear = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=linear_case_key,
            chunk_records=4,
        )
        z_axis, nonlinear_records, nonlinear_meta = _run_case(
            api,
            field0_tyx,
            nt=nt,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            dt=dt,
            beta2=beta2,
            beta_t=beta_t,
            gamma=gamma_nonlinear,
            grin_strength=grin_strength,
            z_final=z_final,
            num_records=num_records,
            exec_options=exec_options,
            storage_kwargs=storage_nonlinear,
        )
        z_linear, linear_records, linear_meta = _run_case(
            api,
            field0_tyx,
            nt=nt,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            dt=dt,
            beta2=beta2,
            beta_t=beta_t,
            gamma=0.0,
            grin_strength=grin_strength,
            z_final=z_final,
            num_records=num_records,
            exec_options=exec_options,
            storage_kwargs=storage_linear,
        )
        if z_axis.shape != z_linear.shape or not np.allclose(z_axis, z_linear):
            raise RuntimeError("linear and nonlinear z axes did not match.")
        meta = {
            "nt": int(nt),
            "nx": int(nx),
            "ny": int(ny),
            "dt": float(dt),
            "dx": float(dx),
            "dy": float(dy),
            "temporal_width": float(temporal_width),
            "soliton_order": float(soliton_order),
            "mode_width": float(mode_width),
            "spatial_chirp": float(spatial_chirp),
            "beta2": float(beta2),
            "beta_t": float(beta_t),
            "grin_strength": float(grin_strength),
            "gamma_nonlinear": float(gamma_nonlinear),
        }
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=nonlinear_case_key,
            solver_meta=nonlinear_meta,
            meta=meta,
        )
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=linear_case_key,
            solver_meta=linear_meta,
            meta=meta,
        )

    launch_centerline = np.sum(np.abs(field0_tyx[:, ny // 2, :]) ** 2, axis=0)
    launch_temporal = np.sum(np.abs(field0_tyx) ** 2, axis=(1, 2))
    nonlinear_xy = time_integrated_xy_records(nonlinear_records)
    linear_xy = time_integrated_xy_records(linear_records)

    nonlinear_radius = rms_radius_curve(nonlinear_records, x_axis, y_axis)
    linear_radius = rms_radius_curve(linear_records, x_axis, y_axis)
    nonlinear_temporal_width = rms_temporal_width_curve(nonlinear_records, t_axis)
    linear_temporal_width = rms_temporal_width_curve(linear_records, t_axis)
    nonlinear_peak = peak_intensity_curve(nonlinear_records)
    linear_peak = peak_intensity_curve(linear_records)
    nonlinear_overlap = overlap_fidelity_curve(nonlinear_records, field0_tyx)
    linear_overlap = overlap_fidelity_curve(linear_records, field0_tyx)
    nonlinear_power = total_power_curve(nonlinear_records)
    linear_power = total_power_curve(linear_records)
    nonlinear_power_drift_curve = relative_power_drift_curve(nonlinear_power)
    linear_power_drift_curve = relative_power_drift_curve(linear_power)

    nonlinear_centerline = centerline_intensity_map(nonlinear_records)
    linear_centerline = centerline_intensity_map(linear_records)
    nonlinear_temporal = temporal_marginal_curve(nonlinear_records)
    linear_temporal = temporal_marginal_curve(linear_records)
    ld = dispersion_length(beta2, temporal_width)
    lnl = 1.0 / (gamma_nonlinear * fundamental_soliton_power(beta2, gamma_nonlinear, temporal_width))
    ldiff = diffraction_length(beta_t, mode_width)
    t_scaled = t_axis / float(temporal_width)
    x_scaled = x_axis / float(mode_width)
    y_scaled = y_axis / float(mode_width)
    z_period = soliton_period(beta2, temporal_width)
    z_final = float(z_axis[-1]) if z_axis.size > 0 else float(z_final)
    z_scaled = z_axis / ld

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_intensity_colormap_vs_propagation(
        x_scaled,
        z_scaled,
        linear_centerline,
        output_dir / "high_order_grin_soliton_linear_centerline_map.png",
        x_label="x / w0",
        y_label="z / L_D",
        colorbar_label="Normalized center-line intensity",
    )
    plot_intensity_colormap_vs_propagation(
        x_scaled,
        z_scaled,
        nonlinear_centerline,
        output_dir / "high_order_grin_soliton_nonlinear_centerline_map.png",
        x_label="x / w0",
        y_label="z / L_D",
        colorbar_label="Normalized center-line intensity",
    )
    plot_two_curve_comparison(
        z_scaled,
        nonlinear_radius,
        linear_radius,
        output_dir / "high_order_grin_soliton_rms_radius.png",
        label_a="Nonlinear",
        label_b="Linear baseline",
        x_label="z / L_D",
        y_label="RMS transverse radius",
    )
    plot_two_curve_comparison(
        z_scaled,
        nonlinear_peak,
        linear_peak,
        output_dir / "high_order_grin_soliton_peak_intensity.png",
        label_a="Nonlinear",
        label_b="Linear baseline",
        x_label="z / L_D",
        y_label="Peak intensity",
    )
    plot_two_curve_comparison(
        z_scaled,
        nonlinear_temporal_width,
        linear_temporal_width,
        output_dir / "high_order_grin_soliton_rms_temporal_width.png",
        label_a="Nonlinear",
        label_b="Linear baseline",
        x_label="z / L_D",
        y_label="RMS temporal width",
    )
    plot_two_curve_comparison(
        z_scaled,
        nonlinear_overlap,
        linear_overlap,
        output_dir / "high_order_grin_soliton_overlap_fidelity.png",
        label_a="Nonlinear",
        label_b="Linear baseline",
        x_label="z / L_D",
        y_label="Overlap fidelity to launch mode",
    )
    plot_two_curve_comparison(
        z_scaled,
        nonlinear_power_drift_curve,
        linear_power_drift_curve,
        output_dir / "high_order_grin_soliton_power_drift.png",
        label_a="Nonlinear",
        label_b="Linear baseline",
        x_label="z / L_D",
        y_label="Relative power drift",
    )
    plot_three_curve_drift(
        x_scaled,
        launch_centerline,
        linear_centerline[-1],
        nonlinear_centerline[-1],
        output_dir / "high_order_grin_soliton_final_centerline_comparison.png",
        label_a="Launch",
        label_b="Linear final",
        label_c="Nonlinear final",
        x_label="x / w0",
        y_label="Center-line intensity",
    )
    plot_three_curve_drift(
        t_scaled,
        launch_temporal,
        linear_temporal[-1],
        nonlinear_temporal[-1],
        output_dir / "high_order_grin_soliton_final_temporal_comparison.png",
        label_a="Launch",
        label_b="Linear final",
        label_c="Nonlinear final",
        x_label="t / T0",
        y_label="Temporal marginal intensity",
    )
    plot_intensity_colormap_vs_propagation(
        t_scaled,
        z_scaled,
        nonlinear_temporal,
        output_dir / "high_order_grin_soliton_nonlinear_temporal_map.png",
        x_label="t / T0",
        y_label="z / L_D",
        colorbar_label="Normalized temporal marginal intensity",
    )
    plot_intensity_colormap_vs_propagation(
        x_scaled,
        y_scaled,
        nonlinear_xy[-1],
        output_dir / "high_order_grin_soliton_nonlinear_final_xy_map.png",
        x_label="x / w0",
        y_label="y / w0",
        colorbar_label="Normalized final intensity",
    )
    plot_3d_intensity_contours_propagation(
        x_scaled,
        y_scaled,
        z_scaled,
        linear_xy,
        output_dir / "high_order_grin_soliton_linear_3d_intensity_contour_surfaces.png",
        input_is_intensity=True,
        z_label="z / L_D",
    )
    plot_3d_intensity_contours_propagation(
        x_scaled,
        y_scaled,
        z_scaled,
        nonlinear_xy,
        output_dir / "high_order_grin_soliton_nonlinear_3d_intensity_contour_surfaces.png",
        input_is_intensity=True,
        z_label="z / L_D",
    )
    # save_3d_intensity_time_sweep_video(
    #     t_scaled,
    #     x_scaled,
    #     y_scaled,
    #     z_scaled,
    #     nonlinear_records,
    #     output_dir / "high_order_grin_soliton_nonlinear_time_sweep.mp4",
    # )

    nonlinear_power_drift = float(nonlinear_power_drift_curve[-1])
    linear_power_drift = float(linear_power_drift_curve[-1])

    print("high-order GRIN soliton summary")
    print(f"  nonlinear final radius / launch radius = {float(nonlinear_radius[-1] / nonlinear_radius[0]):.6f}")
    print(f"  linear final radius / launch radius = {float(linear_radius[-1] / linear_radius[0]):.6f}")
    print(f"  nonlinear radius excursion = {float(np.max(nonlinear_radius) / nonlinear_radius[0]):.6f}")
    print(f"  linear radius excursion = {float(np.max(linear_radius) / linear_radius[0]):.6f}")
    print(f"  nonlinear final temporal width / launch width = {float(nonlinear_temporal_width[-1] / nonlinear_temporal_width[0]):.6f}")
    print(f"  linear final temporal width / launch width = {float(linear_temporal_width[-1] / linear_temporal_width[0]):.6f}")
    print(f"  nonlinear min overlap fidelity = {float(np.min(nonlinear_overlap)):.6f}")
    print(f"  linear min overlap fidelity = {float(np.min(linear_overlap)):.6f}")
    print(f"  nonlinear power drift = {nonlinear_power_drift:.6e}")
    print(f"  linear power drift = {linear_power_drift:.6e}")
    print(f"  nonlinear max power drift = {float(np.max(nonlinear_power_drift_curve)):.6e}")
    print(f"  linear max power drift = {float(np.max(linear_power_drift_curve)):.6e}")
    print(f"  L_D = {ld:.6f}")
    print(f"  L_NL(fundamental) = {lnl:.6f}")
    print(f"  L_diff = {ldiff:.6f}")
    print(f"  soliton period / L_D = {float(z_period / ld):.6f}")
    print(f"  periods covered = {float(z_final / z_period):.2f}")
    return float(max(np.max(nonlinear_power_drift_curve), np.max(linear_power_drift_curve)))


class HighOrderGrinSolitonApp(ExampleAppBase):
    example_slug = "high_order_grin_soliton"
    description = "High-order GRIN-guided nonlinear tensor propagation using the C-backed runtime operators."

    def run(self) -> float:
        return _run(self.args)


def main(argv: list[str] | None = None) -> float:
    return HighOrderGrinSolitonApp.from_cli(argv).run()


if __name__ == "__main__":
    main()
