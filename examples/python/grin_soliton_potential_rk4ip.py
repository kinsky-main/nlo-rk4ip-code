"""
GRIN potential + temporal soliton analytical validation (tensor 3D API).

This example validates a separable analytical solution for:
  A_z = i*(beta2/2)*A_tt + i*gamma*|A|^2*A + i*V(x,y)*A
with no diffraction term. The exact solution is:
  A(t,x,y,z) = A_soliton(t,z) * exp(i*V(x,y)*z)
where A_soliton is the fundamental temporal soliton.
"""

from __future__ import annotations

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


def _flatten_xy_tfast(field_yx: np.ndarray) -> np.ndarray:
    return np.asarray(field_yx, dtype=np.complex128).T.reshape(-1)


def _flatten_tfast(field_tyx: np.ndarray) -> np.ndarray:
    return np.asarray(field_tyx, dtype=np.complex128).transpose(2, 1, 0).reshape(-1)


def _unflatten_record_tfast(record_flat: np.ndarray, nt: int, ny: int, nx: int) -> np.ndarray:
    return np.asarray(record_flat, dtype=np.complex128).reshape(nx, ny, nt).transpose(2, 1, 0)


def _relative_l2_error_curve(records_num: np.ndarray, records_ref: np.ndarray) -> np.ndarray:
    if records_num.shape != records_ref.shape:
        raise ValueError("records_num and records_ref must have the same shape")

    out = np.empty(records_num.shape[0], dtype=np.float64)
    for i in range(records_num.shape[0]):
        ref = np.asarray(records_ref[i], dtype=np.complex128).reshape(-1)
        num = np.asarray(records_num[i], dtype=np.complex128).reshape(-1)
        denom = max(float(np.linalg.norm(ref)), 1e-15)
        out[i] = float(np.linalg.norm(num - ref) / denom)
    return out


def _build_case(
    nt: int,
    nx: int,
    ny: int,
    dt: float,
    dx: float,
    dy: float,
    beta2: float,
    gamma: float,
    t0: float,
    grin_g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = (np.arange(nt, dtype=np.float64) - 0.5 * float(nt - 1)) * dt
    x = (np.arange(nx, dtype=np.float64) - 0.5 * float(nx - 1)) * dx
    y = (np.arange(ny, dtype=np.float64) - 0.5 * float(ny - 1)) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    p0 = abs(beta2) / (gamma * t0 * t0)
    a0_t = (np.sqrt(p0) / np.cosh(t / t0)).astype(np.complex128)
    field0 = np.tile(a0_t[:, None, None], (1, ny, nx))
    potential_xy = (grin_g * (xx * xx + yy * yy)).astype(np.complex128)
    omega = (2.0 * np.pi * np.fft.fftfreq(nt, d=dt)).astype(np.float64)
    return t, x, y, field0, potential_xy, omega


def _analytical_records(
    field0_t: np.ndarray,
    potential_xy: np.ndarray,
    z_axis: np.ndarray,
    beta2: float,
    t0: float,
) -> np.ndarray:
    ld = (t0 * t0) / abs(beta2)
    out = np.empty((z_axis.size, field0_t.size, potential_xy.shape[0], potential_xy.shape[1]), dtype=np.complex128)
    for i, z in enumerate(z_axis):
        soliton_phase = np.exp((1.0j) * 0.5 * (float(z) / ld))
        temporal = field0_t * soliton_phase
        grin_phase = np.exp((1.0j) * potential_xy * float(z))
        out[i] = temporal[:, None, None] * grin_phase[None, :, :]
    return out


def main() -> None:
    parser = build_example_parser(
        example_slug="grin_soliton_potential",
        description="GRIN potential + temporal soliton analytical validation (tensor 3D API).",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "grin_soliton_potential_rk4ip"
    case_key = "default"

    nt = 512
    nx = 172
    ny = 172
    beta2 = -0.01
    gamma = 0.01
    t0 = 0.1 / (2.0 * np.log(1.0 + np.sqrt(2.0)))
    z_final = 0.5 * ((t0 * t0) / abs(beta2))
    num_records = 20
    dt = (16.0 * t0) / float(nt)
    dx = 0.6
    dy = 0.6
    grin_g = 0.020
    step_size = z_final / 320.0
    lambda0_nm = 1550.0

    runner = NloExampleRunner()
    nlo = runner.nlo
    exec_options = SimulationOptions(backend="auto", fft_backend="auto", device_heap_fraction=0.70)

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
        beta2 = float(meta["beta2"])
        gamma = float(meta["gamma"])
        t0 = float(meta["t0"])
        z_final = float(meta["z_final"])
        grin_g = float(meta["grin_g"])
        num_records = int(meta["num_records"])
        step_size = float(meta.get("step_size", z_final / 320.0))
        lambda0_nm = float(meta.get("lambda0_nm", 1550.0))
        t, x, y, field0, potential_xy, _ = _build_case(nt, nx, ny, dt, dx, dy, beta2, gamma, t0, grin_g)
        z_axis = np.asarray(loaded.z_axis, dtype=np.float64)
        records_flat = np.asarray(loaded.records, dtype=np.complex128).reshape(-1, nt * ny * nx)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        t, x, y, field0, potential_xy, omega = _build_case(nt, nx, ny, dt, dx, dy, beta2, gamma, t0, grin_g)
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=2,
        )

        runtime = nlo.RuntimeOperators(
            linear_factor_expr="i*c0*wt*wt",
            linear_expr="exp(h*D)",
            nonlinear_expr="i*A*(c1*I + V)",
            constants=[0.5 * beta2, gamma, 0.0, 0.0],
        )
        cfg = nlo.prepare_sim_config(
            nt * nx * ny,
            propagation_distance=float(z_final),
            starting_step_size=float(step_size),
            max_step_size=float(step_size),
            min_step_size=float(step_size),
            error_tolerance=1e-6,
            pulse_period=float(nt) * dt,
            delta_time=dt,
            tensor_nt=nt,
            tensor_nx=nx,
            tensor_ny=ny,
            tensor_layout=int(nlo.NLO_TENSOR_LAYOUT_XYT_T_FAST),
            frequency_grid=[complex(float(w), 0.0) for w in omega],
            delta_x=dx,
            delta_y=dy,
            potential_grid=_flatten_xy_tfast(potential_xy).tolist(),
            runtime=runtime,
        )
        result = runner.api.propagate(
            cfg,
            _flatten_tfast(field0).tolist(),
            int(num_records),
            exec_options=exec_options.to_ctypes(nlo),
            **storage_kwargs,
        )
        runner.last_meta = dict(result.meta)
        z_axis = np.asarray(result.z_axis, dtype=np.float64)
        records_flat = np.asarray(result.records, dtype=np.complex128).reshape(num_records, nt * ny * nx)

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
                "beta2": float(beta2),
                "gamma": float(gamma),
                "t0": float(t0),
                "z_final": float(z_final),
                "grin_g": float(grin_g),
                "num_records": int(num_records),
                "step_size": float(step_size),
                "lambda0_nm": float(lambda0_nm),
            },
        )

    records = np.asarray(
        [_unflatten_record_tfast(row, nt, ny, nx) for row in records_flat],
        dtype=np.complex128,
    )
    records_ref = _analytical_records(
        field0[:, 0, 0],
        potential_xy,
        z_axis,
        beta2,
        t0,
    )

    error_curve = _relative_l2_error_curve(records, records_ref)
    final_error = float(error_curve[-1])

    power_num = np.sum(np.abs(records) ** 2, axis=(1, 2, 3))
    power_drift = float(abs(power_num[-1] - power_num[0]) / max(float(power_num[0]), 1e-15))

    center_y = ny // 2
    center_x = nx // 2
    temporal_num = np.asarray(records[:, :, center_y, center_x], dtype=np.complex128)
    temporal_ref = np.asarray(records_ref[:, :, center_y, center_x], dtype=np.complex128)
    xline_num = np.asarray(records[-1, nt // 2, center_y, :], dtype=np.complex128)
    xline_ref = np.asarray(records_ref[-1, nt // 2, center_y, :], dtype=np.complex128)

    output_dir = Path(__file__).resolve().parent / "output" / "grin_soliton_potential"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    p1 = plot_intensity_colormap_vs_propagation(
        t,
        z_axis,
        np.abs(temporal_num) ** 2,
        output_dir / "center_temporal_intensity_colormap.png",
        x_label="Time t",
        y_label="Propagation distance z",
        title="Center-point temporal intensity (numerical)",
        colorbar_label="Normalized intensity",
    )
    if p1 is not None:
        saved.append(p1)

    p2 = plot_final_intensity_comparison(
        t,
        temporal_ref[-1],
        temporal_num[-1],
        output_dir / "final_center_temporal_intensity_comparison.png",
        x_label="Time t",
        title="Final temporal intensity at GRIN center (analytical vs numerical)",
        reference_label="Analytical",
        final_label="Numerical",
    )
    if p2 is not None:
        saved.append(p2)

    p3 = plot_final_re_im_comparison(
        x,
        xline_ref,
        xline_num,
        output_dir / "final_xline_re_im_comparison_tmid_ycenter.png",
        x_label="Transverse coordinate x",
        title="Final transverse line field (t=t_mid, y=y_mid)",
        reference_label="Analytical",
        final_label="Numerical",
    )
    if p3 is not None:
        saved.append(p3)

    p4 = plot_total_error_over_propagation(
        z_axis,
        error_curve,
        output_dir / "relative_error_over_propagation.png",
        title="GRIN potential + soliton: relative L2 error over z",
        y_label="Relative L2 error",
    )
    if p4 is not None:
        saved.append(p4)

    spatial_num = np.sum(np.abs(records) ** 2, axis=1)
    spatial_ref = np.sum(np.abs(records_ref) ** 2, axis=1)
    p5 = plot_3d_intensity_scatter_propagation(
        x,
        y,
        z_axis,
        spatial_num,
        output_dir / "spatial_integrated_3d_numerical_scatter.png",
        intensity_cutoff=0.02,
        xy_stride=2,
        z_stride=1,
        min_marker_size=1.0,
        max_marker_size=24.0,
        alpha_min=0.04,
        alpha_max=0.95,
        dpi=400,
        title="GRIN+Soliton: spatial intensity integrated over time (numerical)",
    )
    if p5 is not None:
        saved.append(p5)

    p6 = plot_3d_intensity_scatter_propagation(
        x,
        y,
        z_axis,
        spatial_ref,
        output_dir / "spatial_integrated_3d_expected_scatter.png",
        intensity_cutoff=0.02,
        xy_stride=2,
        z_stride=1,
        min_marker_size=1.0,
        max_marker_size=24.0,
        alpha_min=0.04,
        alpha_max=0.95,
        dpi=400,
        title="GRIN+Soliton: spatial intensity integrated over time (expected)",
    )
    if p6 is not None:
        saved.append(p6)

    spec_num = np.fft.fftshift(np.fft.fft(temporal_num, axis=1), axes=1)
    spec_ref = np.fft.fftshift(np.fft.fft(temporal_ref, axis=1), axes=1)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))

    p7 = plot_intensity_colormap_vs_propagation(
        freq_axis,
        z_axis,
        np.abs(spec_num) ** 2,
        output_dir / "frequency_profile_numerical.png",
        x_label="Frequency detuning (1/time)",
        y_label="Propagation distance z",
        title="Frequency propagation profile (numerical)",
        colorbar_label="Normalized spectral intensity",
    )
    if p7 is not None:
        saved.append(p7)

    p8 = plot_intensity_colormap_vs_propagation(
        freq_axis,
        z_axis,
        np.abs(spec_ref) ** 2,
        output_dir / "frequency_profile_expected.png",
        x_label="Frequency detuning (1/time)",
        y_label="Propagation distance z",
        title="Frequency propagation profile (expected)",
        colorbar_label="Normalized spectral intensity",
    )
    if p8 is not None:
        saved.append(p8)

    c_m_per_s = 299792458.0
    f0_hz = c_m_per_s / (lambda0_nm * 1e-9)
    freq_abs_hz = f0_hz + (freq_axis * 1e12)
    valid = freq_abs_hz > 0.0
    wavelength_nm = (c_m_per_s / freq_abs_hz[valid]) * 1e9
    order = np.argsort(wavelength_nm)
    wl_axis = wavelength_nm[order]
    wl_num = np.abs(spec_num[:, valid][:, order]) ** 2
    wl_ref = np.abs(spec_ref[:, valid][:, order]) ** 2

    p9 = plot_intensity_colormap_vs_propagation(
        wl_axis,
        z_axis,
        wl_num,
        output_dir / "wavelength_profile_numerical.png",
        x_label="Wavelength (nm)",
        y_label="Propagation distance z",
        title="Wavelength propagation profile (numerical)",
        colorbar_label="Normalized spectral intensity",
    )
    if p9 is not None:
        saved.append(p9)

    p10 = plot_intensity_colormap_vs_propagation(
        wl_axis,
        z_axis,
        wl_ref,
        output_dir / "wavelength_profile_expected.png",
        x_label="Wavelength (nm)",
        y_label="Propagation distance z",
        title="Wavelength propagation profile (expected)",
        colorbar_label="Normalized spectral intensity",
    )
    if p10 is not None:
        saved.append(p10)

    print(f"grin-soliton potential validation summary (run_group={run_group}):")
    print(f"  grid=(t={nt}, y={ny}, x={nx}), records={len(z_axis)}")
    print(f"  final relative L2 error = {final_error:.6e}")
    print(f"  relative power drift    = {power_drift:.6e}")
    if saved:
        print("saved plots:")
        for path in saved:
            print(f"  {path}")


if __name__ == "__main__":
    main()
