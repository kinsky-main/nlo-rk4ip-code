"""
Unsaturable Kerr propagation in a GRIN-guided tensor geometry.

This example mirrors ``high_order_grin_soliton_rk4ip.py`` but replaces the
saturable nonlinear response with a pure Kerr term. The launch condition uses a
high-order temporal soliton with transverse azimuthal and noise perturbations so
that the run highlights temporal soliton fission and transverse filamentation.
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
    plot_summary_curve,
    plot_two_curve_comparison,
)
from backend.runner import centered_time_grid
from backend.storage import ExampleRunDB


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

import nlolib as nlo

from high_order_grin_soliton_rk4ip import (
    centerline_intensity_map,
    centered_spatial_grid,
    diffraction_length,
    dispersion_length,
    flatten_tfast,
    fundamental_soliton_power,
    grin_launch_field,
    grin_potential_grid,
    overlap_fidelity_curve,
    peak_intensity_curve,
    relative_power_drift_curve,
    rms_radius_curve,
    rms_temporal_width_curve,
    soliton_period,
    spectral_marginal_curve,
    temporal_frequency_axis,
    temporal_marginal_curve,
    temporal_peak_count_curve,
    time_integrated_xy_records,
    total_power_curve,
    unflatten_tfast_records,
)


DEFAULT_SOLITON_ORDER = 3.5
DEFAULT_PROPAGATION_PERIODS = 0.5
DEFAULT_AZIMUTHAL_AMPLITUDE = 0.35
DEFAULT_AZIMUTHAL_ORDER = 6
DEFAULT_NOISE_AMPLITUDE = 0.025
DEFAULT_NOISE_SEED = 271828
DEFAULT_GAMMA = 1.2


def transverse_peak_count_curve(xy_records: np.ndarray, *, relative_threshold: float = 0.20) -> np.ndarray:
    records = np.asarray(xy_records, dtype=np.float64)
    out = np.zeros(int(records.shape[0]), dtype=np.int64)
    for idx, image in enumerate(records):
        if image.shape[0] < 3 or image.shape[1] < 3:
            continue
        threshold = float(relative_threshold) * float(np.max(image))
        peaks = 0
        for yi in range(1, int(image.shape[0]) - 1):
            for xi in range(1, int(image.shape[1]) - 1):
                value = float(image[yi, xi])
                if value <= threshold:
                    continue
                neighbourhood = image[yi - 1 : yi + 2, xi - 1 : xi + 2]
                if value >= float(np.max(neighbourhood)):
                    peaks += 1
        out[idx] = peaks
    return out


def filamentation_index_curve(records_tyx: np.ndarray, radius_curve: np.ndarray) -> np.ndarray:
    peak = peak_intensity_curve(records_tyx)
    radius = np.maximum(np.asarray(radius_curve, dtype=np.float64), 1.0e-30)
    reference = max(float(peak[0] / (radius[0] * radius[0])), 1.0e-30)
    return (peak / (radius * radius)) / reference


def _run_unsaturable_case(
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
        starting_step_size=2.5e-3,
        max_step_size=7.5e-2,
        min_step_size=1.0e-9,
        error_tolerance=7.5e-7,
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
            constants=[0.5 * float(beta2), float(beta_t), float(gamma)],
        ),
    )
    kwargs = dict(storage_kwargs or {})
    if storage_kwargs is not None:
        kwargs.setdefault("return_records", False)
    result = api.propagate(
        cfg,
        flatten_tfast(field0_tyx).tolist(),
        int(num_records),
        exec_options,
        t_eval=np.linspace(0.0, float(z_final), int(num_records)).tolist(),
        **kwargs,
    )
    z_axis = np.asarray(result.z_axis, dtype=np.float64)
    if result.records:
        records_tyx = unflatten_tfast_records(
            np.asarray(result.records, dtype=np.complex128),
            num_records=len(z_axis),
            nt=nt,
            ny=ny,
            nx=nx,
        )
    else:
        records_tyx = np.empty((0, int(nt), int(ny), int(nx)), dtype=np.complex128)
    return z_axis, records_tyx, dict(result.meta)


def _run(args: argparse.Namespace) -> float:
    try:
        nlo.set_log_level(nlo.NLOLIB_LOG_LEVEL_ERROR)
    except RuntimeError:
        pass

    db = ExampleRunDB(args.db_path)
    api = nlo.NLolib()
    api.set_progress_options(enabled=True, milestone_percent=2, emit_on_step_adjust=True)
    example_name = "unsaturable_grin_filamentation_fission_rk4ip"
    case_key = "unsaturable_kerr"

    nt = 512
    nx = 96
    ny = 96
    dt = 0.02
    dx = 0.03
    dy = 0.03
    temporal_width = 0.30
    soliton_order = float(args.soliton_order)
    mode_width_x = 0.60
    mode_width_y = 0.72
    spatial_chirp = 0.65
    azimuthal_amplitude = float(args.azimuthal_amplitude)
    azimuthal_order = int(args.azimuthal_order)
    noise_amplitude = float(args.noise_amplitude)
    noise_seed = int(args.noise_seed)
    beta2 = -0.08
    beta_t = -0.08
    grin_strength = 1.5e-3
    gamma = float(args.gamma)
    propagation_periods = float(args.propagation_periods)
    if not np.isfinite(soliton_order) or soliton_order <= 1.0:
        raise ValueError("soliton_order must be finite and greater than 1 to drive fission.")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite value.")
    if not np.isfinite(propagation_periods) or propagation_periods <= 0.0:
        raise ValueError("propagation_periods must be a positive finite value.")
    z_period = soliton_period(beta2, temporal_width)
    z_final = propagation_periods * z_period
    num_records = 84

    t_axis = centered_time_grid(nt, dt)
    x_axis = centered_spatial_grid(nx, dx)
    y_axis = centered_spatial_grid(ny, dy)
    field0_tyx = grin_launch_field(
        t_axis,
        x_axis,
        y_axis,
        temporal_width=temporal_width,
        beta2=beta2,
        gamma=gamma,
        soliton_order=soliton_order,
        mode_width_x=mode_width_x,
        mode_width_y=mode_width_y,
        spatial_chirp=spatial_chirp,
        azimuthal_amplitude=azimuthal_amplitude,
        azimuthal_order=azimuthal_order,
        noise_amplitude=noise_amplitude,
        noise_seed=noise_seed,
    ).astype(np.complex128)

    exec_options = nlo.default_execution_options(
        backend_type=nlo.VECTOR_BACKEND_AUTO,
        fft_backend=nlo.FFT_BACKEND_AUTO,
    )

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group, required_case_keys=[case_key])
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        meta = loaded.meta
        nt = int(meta["nt"])
        nx = int(meta["nx"])
        ny = int(meta["ny"])
        dt = float(meta["dt"])
        dx = float(meta["dx"])
        dy = float(meta["dy"])
        temporal_width = float(meta["temporal_width"])
        soliton_order = float(meta["soliton_order"])
        mode_width_x = float(meta["mode_width_x"])
        mode_width_y = float(meta["mode_width_y"])
        spatial_chirp = float(meta["spatial_chirp"])
        azimuthal_amplitude = float(meta.get("azimuthal_amplitude", 0.0))
        azimuthal_order = int(meta.get("azimuthal_order", 6))
        noise_amplitude = float(meta.get("noise_amplitude", 0.0))
        noise_seed = int(meta.get("noise_seed", 0))
        beta2 = float(meta["beta2"])
        beta_t = float(meta["beta_t"])
        grin_strength = float(meta["grin_strength"])
        gamma = float(meta["gamma"])
        propagation_periods = float(meta.get("propagation_periods", 0.0))
        t_axis = centered_time_grid(nt, dt)
        x_axis = centered_spatial_grid(nx, dx)
        y_axis = centered_spatial_grid(ny, dy)
        field0_tyx = grin_launch_field(
            t_axis,
            x_axis,
            y_axis,
            temporal_width=temporal_width,
            beta2=beta2,
            gamma=gamma,
            soliton_order=soliton_order,
            mode_width_x=mode_width_x,
            mode_width_y=mode_width_y,
            spatial_chirp=spatial_chirp,
            azimuthal_amplitude=azimuthal_amplitude,
            azimuthal_order=azimuthal_order,
            noise_amplitude=noise_amplitude,
            noise_seed=noise_seed,
        ).astype(np.complex128)
        z_axis = np.asarray(loaded.z_axis, dtype=np.float64)
        records = unflatten_tfast_records(loaded.records, num_records=len(z_axis), nt=nt, ny=ny, nx=nx)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        storage = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=4,
        )
        z_axis, records, solver_meta = _run_unsaturable_case(
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
            gamma=gamma,
            grin_strength=grin_strength,
            z_final=z_final,
            num_records=num_records,
            exec_options=exec_options,
            storage_kwargs=storage,
        )
        meta = {
            "nt": int(nt),
            "nx": int(nx),
            "ny": int(ny),
            "dt": float(dt),
            "dx": float(dx),
            "dy": float(dy),
            "temporal_width": float(temporal_width),
            "soliton_order": float(soliton_order),
            "mode_width_x": float(mode_width_x),
            "mode_width_y": float(mode_width_y),
            "spatial_chirp": float(spatial_chirp),
            "azimuthal_amplitude": float(azimuthal_amplitude),
            "azimuthal_order": int(azimuthal_order),
            "noise_amplitude": float(noise_amplitude),
            "noise_seed": int(noise_seed),
            "beta2": float(beta2),
            "beta_t": float(beta_t),
            "grin_strength": float(grin_strength),
            "gamma": float(gamma),
            "propagation_periods": float(propagation_periods),
            "nonlinear_response": "unsaturable_kerr",
        }
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            solver_meta=solver_meta,
            meta=meta,
        )
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        z_axis = np.asarray(loaded.z_axis, dtype=np.float64)
        records = unflatten_tfast_records(loaded.records, num_records=len(z_axis), nt=nt, ny=ny, nx=nx)

    launch_centerline = np.sum(np.abs(field0_tyx[:, ny // 2, :]) ** 2, axis=0)
    launch_temporal = np.sum(np.abs(field0_tyx) ** 2, axis=(1, 2))
    xy_records = time_integrated_xy_records(records)
    centerline = centerline_intensity_map(records)
    temporal = temporal_marginal_curve(records)
    spectral = spectral_marginal_curve(records)

    radius = rms_radius_curve(records, x_axis, y_axis)
    temporal_width_curve = rms_temporal_width_curve(records, t_axis)
    peak = peak_intensity_curve(records)
    overlap = overlap_fidelity_curve(records, field0_tyx)
    power_drift = relative_power_drift_curve(total_power_curve(records))
    temporal_peaks = temporal_peak_count_curve(temporal, relative_threshold=0.18)
    transverse_peaks = transverse_peak_count_curve(xy_records, relative_threshold=0.18)
    filamentation_index = filamentation_index_curve(records, radius)

    ld = dispersion_length(beta2, temporal_width)
    lnl = 1.0 / (gamma * fundamental_soliton_power(beta2, gamma, temporal_width))
    ldiff = diffraction_length(beta_t, 0.5 * (mode_width_x + mode_width_y))
    t_scaled = t_axis / float(temporal_width)
    omega_scaled = temporal_frequency_axis(nt, dt, temporal_width)
    x_scaled = x_axis / float(mode_width_x)
    y_scaled = y_axis / float(mode_width_y)
    z_period = soliton_period(beta2, temporal_width)
    z_final = float(z_axis[-1]) if z_axis.size > 0 else float(z_final)
    if propagation_periods <= 0.0:
        propagation_periods = z_final / z_period
    z_scaled = z_axis / ld

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "unsaturable_grin_filamentation_fission"
    plot_intensity_colormap_vs_propagation(
        x_scaled,
        z_scaled,
        centerline,
        output_dir / f"{prefix}_centerline_map.png",
        x_label=r"$x / w0$",
        y_label=r"$z / L_D$",
        colorbar_label="Normalisedcenter-line intensity",
    )
    plot_intensity_colormap_vs_propagation(
        t_scaled,
        z_scaled,
        temporal,
        output_dir / f"{prefix}_temporal_fission_map.png",
        x_label=r"$t / T_0$",
        y_label=r"$z / L_D$",
        colorbar_label="Normalisedtemporal intensity",
        xlimit=(-5, 5),
    )
    plot_intensity_colormap_vs_propagation(
        omega_scaled,
        z_scaled,
        spectral,
        output_dir / f"{prefix}_spectral_broadening_map.png",
        x_label=r"$\omega / \omega_0$",
        y_label=r"$z / L_D$",
        colorbar_label="Normalisedspectral intensity",
        xlimit=(-7, 7),
    )
    plot_intensity_colormap_vs_propagation(
        x_scaled,
        y_scaled,
        xy_records[-1],
        output_dir / f"{prefix}_final_xy_filaments.png",
        x_label=r"$x / w0$",
        y_label=r"$y / w0$",
        colorbar_label="Normalisedfinal intensity",
    )
    plot_summary_curve(
        z_scaled,
        peak,
        output_dir / f"{prefix}_peak_intensity.png",
        x_label=r"$z / L_D$",
        y_label="Peak intensity",
    )
    plot_summary_curve(
        z_scaled,
        filamentation_index,
        output_dir / f"{prefix}_filamentation_index.png",
        x_label=r"$z / L_D$",
        y_label="Filamentation index",
    )
    plot_summary_curve(
        z_scaled,
        transverse_peaks,
        output_dir / f"{prefix}_transverse_peak_count.png",
        x_label=r"$z / L_D$",
        y_label="Transverse local maxima",
    )
    plot_summary_curve(
        z_scaled,
        temporal_peaks,
        output_dir / f"{prefix}_temporal_peak_count.png",
        x_label=r"$z / L_D$",
        y_label="Temporal local maxima",
    )
    plot_summary_curve(
        z_scaled,
        radius,
        output_dir / f"{prefix}_rms_radius.png",
        x_label=r"$z / L_D$",
        y_label="RMS transverse radius",
    )
    plot_summary_curve(
        z_scaled,
        temporal_width_curve,
        output_dir / f"{prefix}_rms_temporal_width.png",
        x_label=r"$z / L_D$",
        y_label="RMS temporal width",
    )
    plot_summary_curve(
        z_scaled,
        overlap,
        output_dir / f"{prefix}_overlap_fidelity.png",
        x_label=r"$z / L_D$",
        y_label="Overlap fidelity to launch mode",
    )
    plot_summary_curve(
        z_scaled,
        power_drift,
        output_dir / f"{prefix}_power_drift.png",
        x_label=r"$z / L_D$",
        y_label="Relative power drift",
    )
    plot_two_curve_comparison(
        x_scaled,
        launch_centerline,
        centerline[-1],
        output_dir / f"{prefix}_final_centerline_comparison.png",
        label_a="Launch",
        label_b="Unsaturable Kerr final",
        x_label=r"$x / w0$",
        y_label="Center-line intensity",
    )
    plot_two_curve_comparison(
        t_scaled,
        launch_temporal,
        temporal[-1],
        output_dir / f"{prefix}_final_temporal_comparison.png",
        label_a="Launch",
        label_b="Unsaturable Kerr final",
        x_label=r"$t / T_0$",
        y_label="Temporal marginal intensity",
    )
    plot_3d_intensity_contours_propagation(
        x_scaled,
        y_scaled,
        z_scaled,
        xy_records,
        output_dir / f"{prefix}_3d_filament_contours.png",
        input_is_intensity=True,
        z_label=r"$z / L_D$",
    )

    final_power_drift = float(power_drift[-1])
    print("unsaturable GRIN filamentation and fission summary")
    print(f"  soliton order = {soliton_order:.6f}")
    print(f"  gamma = {gamma:.6f}")
    print("  nonlinear response = gamma*I (unsaturable Kerr)")
    print(f"  final radius / launch radius = {float(radius[-1] / radius[0]):.6f}")
    print(f"  radius excursion = {float(np.max(radius) / radius[0]):.6f}")
    print(f"  max filamentation index = {float(np.max(filamentation_index)):.6f}")
    print(f"  max transverse peak count = {int(np.max(transverse_peaks))}")
    print(f"  max temporal peaks = {int(np.max(temporal_peaks))}")
    print(f"  final temporal peaks = {int(temporal_peaks[-1])}")
    print(f"  max spectral intensity = {float(np.max(spectral)):.6e}")
    print(f"  final temporal width / launch width = {float(temporal_width_curve[-1] / temporal_width_curve[0]):.6f}")
    print(f"  min overlap fidelity = {float(np.min(overlap)):.6f}")
    print(f"  power drift = {final_power_drift:.6e}")
    print(f"  max power drift = {float(np.max(power_drift)):.6e}")
    print(f"  L_D = {ld:.6f}")
    print(f"  L_NL(fundamental) = {lnl:.6f}")
    print(f"  L_diff = {ldiff:.6f}")
    print(f"  soliton period / L_D = {float(z_period / ld):.6f}")
    print(f"  periods covered = {float(propagation_periods):.2f}")
    return float(np.max(power_drift))


class UnsaturableGrinFilamentationFissionApp(ExampleAppBase):
    example_slug = "unsaturable_grin_filamentation_fission"
    description = "Unsaturable GRIN Kerr tensor propagation showing filamentation and soliton fission."

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--soliton-order",
            type=float,
            default=DEFAULT_SOLITON_ORDER,
            help="Launch soliton order used for the temporal sech envelope.",
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=DEFAULT_GAMMA,
            help="Dimensionless unsaturable Kerr nonlinearity coefficient.",
        )
        parser.add_argument(
            "--propagation-periods",
            type=float,
            default=DEFAULT_PROPAGATION_PERIODS,
            help="Propagation distance in units of the high-order soliton period.",
        )
        parser.add_argument(
            "--azimuthal-amplitude",
            type=float,
            default=DEFAULT_AZIMUTHAL_AMPLITUDE,
            help="Amplitude epsilon of the cos(m*theta) azimuthal modulation seeding filamentation.",
        )
        parser.add_argument(
            "--azimuthal-order",
            type=int,
            default=DEFAULT_AZIMUTHAL_ORDER,
            help="Azimuthal mode number m for the cos(m*theta) filamentation seed.",
        )
        parser.add_argument(
            "--noise-amplitude",
            type=float,
            default=DEFAULT_NOISE_AMPLITUDE,
            help="Amplitude eta of the random transverse perturbation xi(x,y).",
        )
        parser.add_argument(
            "--noise-seed",
            type=int,
            default=DEFAULT_NOISE_SEED,
            help="RNG seed used to draw the noise field xi(x,y).",
        )

    def run(self) -> float:
        return _run(self.args)


def main(argv: list[str] | None = None) -> float:
    return UnsaturableGrinFilamentationFissionApp.from_cli(argv).run()


if __name__ == "__main__":
    main()
