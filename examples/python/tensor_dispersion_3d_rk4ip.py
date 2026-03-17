"""
Tensor 3D linear dispersion/diffraction example with analytical validation.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
from backend.app_base import ExampleAppBase
from backend.metrics import relative_l2_intensity_error, relative_l2_intensity_error_curve
from backend.plotting import (
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
    plot_two_curve_comparison,
    plot_3d_intensity_scatter_propagation
)
from backend.reference import exact_linear_tensor3d_records
from backend.runner import NloExampleRunner, SimulationOptions, centered_time_grid
from backend.storage import ExampleRunDB


def centered_spatial_grid(num_samples: int, delta: float) -> np.ndarray:
    """Return a centered spatial axis with spacing ``delta``."""
    return (np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)) * float(delta)


def gaussian_tensor_field(
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    temporal_width: float,
    x_width: float,
    y_width: float,
) -> np.ndarray:
    """Return a separable Gaussian field with shape ``[nt, ny, nx]``."""
    temporal = np.exp(-((t_axis / float(temporal_width)) ** 2))
    xx, yy = np.meshgrid(x_axis, y_axis)
    transverse = np.exp(-((xx / float(x_width)) ** 2) - ((yy / float(y_width)) ** 2))
    return temporal[:, None, None] * transverse[None, :, :]


def flatten_tyx_row_major(field_tyx: np.ndarray) -> np.ndarray:
    """Flatten ``[nt, ny, nx]`` into the solver's t-fast layout."""
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
    """Reshape flat solver output into ``[record, nt, ny, nx]``."""
    flat = np.asarray(records_flat, dtype=np.complex128).reshape(int(num_records), int(nx), int(ny), int(nt))
    return np.transpose(flat, (0, 3, 2, 1))


def relative_l2_curve(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return per-record relative L2 field error along axis 0."""
    pred = np.asarray(prediction, dtype=np.complex128)
    ref = np.asarray(reference, dtype=np.complex128)
    if pred.shape != ref.shape:
        raise ValueError(f"prediction and reference must have identical shape: {pred.shape} != {ref.shape}")

    values = []
    for idx in range(int(pred.shape[0])):
        diff_norm = float(np.linalg.norm(pred[idx] - ref[idx]))
        ref_norm = float(np.linalg.norm(ref[idx]))
        values.append(diff_norm / max(ref_norm, 1.0e-30))
    return np.asarray(values, dtype=np.float64)


def relative_l2_real(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return ``||pred-ref||_2 / ||ref||_2`` for real-valued curves."""
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if pred.shape != ref.shape:
        raise ValueError(f"prediction and reference must have identical shape: {pred.shape} != {ref.shape}")
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def marginal_intensity_profiles(field_tyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return temporal, x, and y intensity marginals for ``[nt, ny, nx]``."""
    intensity = np.abs(np.asarray(field_tyx, dtype=np.complex128)) ** 2
    temporal = np.sum(intensity, axis=(1, 2))
    x_profile = np.sum(intensity, axis=(0, 1))
    y_profile = np.sum(intensity, axis=(0, 2))
    return temporal, x_profile, y_profile


def rms_width(axis: np.ndarray, weights: np.ndarray) -> float:
    """Return the RMS width of ``weights`` distributed on ``axis``."""
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(mass))
    if total <= 0.0:
        return 0.0
    center = float(np.sum(values * mass) / total)
    variance = float(np.sum(((values - center) ** 2) * mass) / total)
    return float(np.sqrt(max(variance, 0.0)))


def _run(args: argparse.Namespace) -> float:
    db = ExampleRunDB(args.db_path)
    example_name = "tensor_dispersion_3d_rk4ip"
    case_key = "default"

    nt = 512
    nx = 256
    ny = 256
    dt = 0.04
    dx = 0.15
    dy = 0.15
    temporal_width = 0.24
    x_width = 0.60
    y_width = 0.70
    beta2 = 0.08
    beta_t = -0.20
    z_final = 0.80
    num_records = 100
    starting_step_size = 2.0e-2
    max_step_size = 8.0e-2
    min_step_size = 1.0e-5
    error_tolerance = 1.0e-7

    t_axis = centered_time_grid(nt, dt)
    x_axis = centered_spatial_grid(nx, dx)
    y_axis = centered_spatial_grid(ny, dy)
    omega = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    field0_tyx = gaussian_tensor_field(
        t_axis,
        x_axis,
        y_axis,
        temporal_width=temporal_width,
        x_width=x_width,
        y_width=y_width,
    ).astype(np.complex128)

    runner = NloExampleRunner()
    exec_options = SimulationOptions(backend="auto", fft_backend="auto")

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        meta = loaded.meta
        nt = int(meta["nt"])
        nx = int(meta["nx"])
        ny = int(meta["ny"])
        dt = float(meta["dt"])
        dx = float(meta["dx"])
        dy = float(meta["dy"])
        temporal_width = float(meta["temporal_width"])
        x_width = float(meta["x_width"])
        y_width = float(meta["y_width"])
        beta2 = float(meta["beta2"])
        beta_t = float(meta["beta_t"])
        t_axis = centered_time_grid(nt, dt)
        x_axis = centered_spatial_grid(nx, dx)
        y_axis = centered_spatial_grid(ny, dy)
        omega = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
        field0_tyx = gaussian_tensor_field(
            t_axis,
            x_axis,
            y_axis,
            temporal_width=temporal_width,
            x_width=x_width,
            y_width=y_width,
        ).astype(np.complex128)
        z_records = np.asarray(loaded.z_axis, dtype=np.float64)
        records_flat = np.asarray(loaded.records, dtype=np.complex128)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        runtime = SimpleNamespace(
            linear_factor_fn=lambda A, wt, kx, ky: (1.0j * ((0.5 * beta2) * (wt**2) + beta_t * ((kx**2) + (ky**2)))),  # noqa: E731
            nonlinear_fn=lambda A, I, V: 0.0,  # noqa: E731
        )
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=2,
        )
        z_records, records_flat = runner.propagate_tensor3d_records(
            flatten_tyx_row_major(field0_tyx),
            nt=nt,
            nx=nx,
            ny=ny,
            num_records=num_records,
            propagation_distance=z_final,
            starting_step_size=starting_step_size,
            max_step_size=max_step_size,
            min_step_size=min_step_size,
            error_tolerance=error_tolerance,
            delta_x=dx,
            delta_y=dy,
            delta_time=dt,
            pulse_period=float(nt) * dt,
            frequency_grid=omega,
            runtime=runtime,
            exec_options=exec_options,
            **storage_kwargs,
        )
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            solver_meta=runner.last_meta,
            meta={
                "nt": int(nt),
                "nx": int(nx),
                "ny": int(ny),
                "dt": float(dt),
                "dx": float(dx),
                "dy": float(dy),
                "temporal_width": float(temporal_width),
                "x_width": float(x_width),
                "y_width": float(y_width),
                "beta2": float(beta2),
                "beta_t": float(beta_t),
            },
        )

    records_tyx = unflatten_tfast_records(records_flat, num_records=len(z_records), nt=nt, ny=ny, nx=nx)
    reference_records = exact_linear_tensor3d_records(
        field0_tyx,
        z_records,
        omega,
        kx,
        ky,
        0.5 * beta2,
        beta_t,
    )

    field_error_curve = 100.0 * relative_l2_curve(records_tyx, reference_records)
    intensity_error_curve = 100.0 * relative_l2_intensity_error_curve(records_tyx, reference_records)
    final_field_error = float(field_error_curve[-1] / 100.0)
    final_intensity_error = float(relative_l2_intensity_error(records_tyx[-1], reference_records[-1]))

    final_temporal_ref, final_x_ref, final_y_ref = marginal_intensity_profiles(reference_records[-1])
    final_temporal_num, final_x_num, final_y_num = marginal_intensity_profiles(records_tyx[-1])
    temporal_profile_error = relative_l2_real(final_temporal_num, final_temporal_ref)
    x_profile_error = relative_l2_real(final_x_num, final_x_ref)
    y_profile_error = relative_l2_real(final_y_num, final_y_ref)

    temporal_width_ref = rms_width(t_axis, final_temporal_ref)
    temporal_width_num = rms_width(t_axis, final_temporal_num)
    x_width_ref = rms_width(x_axis, final_x_ref)
    x_width_num = rms_width(x_axis, final_x_num)
    y_width_ref = rms_width(y_axis, final_y_ref)
    y_width_num = rms_width(y_axis, final_y_num)
    initial_temporal_num, initial_x_num, initial_y_num = marginal_intensity_profiles(records_tyx[0])
    initial_temporal_width = rms_width(t_axis, initial_temporal_num)
    initial_x_width = rms_width(x_axis, initial_x_num)
    initial_y_width = rms_width(y_axis, initial_y_num)

    record_norms = np.asarray([float(np.linalg.norm(record)) for record in records_flat], dtype=np.float64)
    initial_power = float(np.sum(np.abs(records_tyx[0]) ** 2))
    final_power = float(np.sum(np.abs(records_tyx[-1]) ** 2))
    power_drift = abs(final_power - initial_power) / max(initial_power, 1.0e-30)

    x_marginal_map = np.sum(np.abs(records_tyx) ** 2, axis=(1, 2))
    t_marginal_map = np.sum(np.abs(records_tyx) ** 2, axis=(2, 3))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    saved = plot_total_error_over_propagation(
        z_records,
        intensity_error_curve,
        output_dir / "tensor_dispersion_3d_intensity_error_over_propagation.png",
        y_label="Relative L2 intensity error (%)",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_total_error_over_propagation(
        z_records,
        field_error_curve,
        output_dir / "tensor_dispersion_3d_field_error_over_propagation.png",
        y_label="Relative L2 field error (%)",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_two_curve_comparison(
        t_axis,
        final_temporal_ref,
        final_temporal_num,
        output_dir / "tensor_dispersion_3d_final_temporal_intensity_marginal.png",
        label_a="Reference final",
        label_b="Numerical final",
        x_label="Time t",
        y_label="Marginal intensity",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_two_curve_comparison(
        x_axis,
        final_x_ref,
        final_x_num,
        output_dir / "tensor_dispersion_3d_final_x_intensity_marginal.png",
        label_a="Reference final",
        label_b="Numerical final",
        x_label="x",
        y_label="Marginal intensity",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_two_curve_comparison(
        y_axis,
        final_y_ref,
        final_y_num,
        output_dir / "tensor_dispersion_3d_final_y_intensity_marginal.png",
        label_a="Reference final",
        label_b="Numerical final",
        x_label="y",
        y_label="Marginal intensity",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_intensity_colormap_vs_propagation(
        x_axis,
        z_records,
        x_marginal_map,
        output_dir / "tensor_dispersion_3d_x_marginal_map.png",
        x_label="x",
        colorbar_label="Normalized x-marginal intensity",
    )
    if saved is not None:
        saved_paths.append(saved)

    saved = plot_intensity_colormap_vs_propagation(
        t_axis,
        z_records,
        t_marginal_map,
        output_dir / "tensor_dispersion_3d_temporal_marginal_map.png",
        x_label="Time t",
        colorbar_label="Normalized temporal-marginal intensity",
    )
    if saved is not None:
        saved_paths.append(saved)
        
    saved = plot_3d_intensity_scatter_propagation(
        x_axis,
        y_axis,
        z_records,
        np.sum(np.abs(records_tyx) ** 2, axis=1),
        output_dir / "tensor_dispersion_3d_3d_intensity_scatter.png"
    )
    if saved is not None:
        saved_paths.append(saved)

    print(f"tensor 3D dispersion example completed (run_group={run_group}).")
    print(f"records shape: {records_flat.shape}")
    print(f"record norms: first={record_norms[0]:.6e}, last={record_norms[-1]:.6e}, min={np.min(record_norms):.6e}")
    print(f"power drift: {power_drift:.6e}")
    print(f"final field relative error: {final_field_error:.6e}")
    print(f"final intensity relative error: {final_intensity_error:.6e}")
    print(
        "final marginal intensity errors: "
        f"temporal={temporal_profile_error:.6e}, "
        f"x={x_profile_error:.6e}, "
        f"y={y_profile_error:.6e}"
    )
    print(
        "rms widths (initial -> final num/ref): "
        f"t={initial_temporal_width:.6e}->{temporal_width_num:.6e}/{temporal_width_ref:.6e}, "
        f"x={initial_x_width:.6e}->{x_width_num:.6e}/{x_width_ref:.6e}, "
        f"y={initial_y_width:.6e}->{y_width_num:.6e}/{y_width_ref:.6e}"
    )
    for path in saved_paths:
        print(f"saved plot: {path}")

    return max(final_intensity_error, temporal_profile_error, x_profile_error, y_profile_error)


class TensorDispersion3DApp(ExampleAppBase):
    example_slug = "tensor_dispersion_3d"
    description = "Tensor 3D linear dispersion/diffraction with DB-backed run/replot."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return TensorDispersion3DApp.from_cli().run()


if __name__ == "__main__":
    main()
