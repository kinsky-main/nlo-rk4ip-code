"""OOP implementation for GRIN soliton validation examples."""

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
    plot_two_curve_comparison,
    plot_total_error_over_propagation,
)
from backend.runner import NloExampleRunner, SimulationOptions
from backend.spectral import omega_detuning_to_wavelength_nm
from backend.storage import ExampleRunDB

from .models import PlotArtifact, ValidationCheck, ValidationReport
from .validation import (
    PlotImageValidator,
    WavelengthWindowSelector,
    profile_correlation,
    write_report,
)


def _flatten_xy_tfast(field_yx: np.ndarray) -> np.ndarray:
    return np.asarray(field_yx, dtype=np.complex128).T.reshape(-1)


def _flatten_tfast(field_tyx: np.ndarray) -> np.ndarray:
    return np.asarray(field_tyx, dtype=np.complex128).transpose(2, 1, 0).reshape(-1)


def _unflatten_record_tfast(record_flat: np.ndarray, nt: int, ny: int, nx: int) -> np.ndarray:
    return np.asarray(record_flat, dtype=np.complex128).reshape(nx, ny, nt).transpose(2, 1, 0)


def _relative_l2_error_curve(records_num: np.ndarray, records_ref: np.ndarray) -> np.ndarray:
    return mean_pointwise_abs_relative_error_curve(
        records_num,
        records_ref,
        context="grin_soliton:record_error",
    )


def _dispersion_length(beta2: float, t0: float) -> float:
    if beta2 == 0.0:
        raise ValueError("beta2 must be non-zero to compute dispersion-length normalization.")
    ld = (float(t0) * float(t0)) / abs(float(beta2))
    if not np.isfinite(ld) or ld <= 0.0:
        raise ValueError("computed dispersion length is not finite and positive.")
    return float(ld)


def _normalized_propagation_axis(z_axis: np.ndarray, beta2: float, t0: float) -> np.ndarray:
    return np.asarray(z_axis, dtype=np.float64) / _dispersion_length(beta2, t0)


@dataclass(frozen=True)
class PhaseOnlyConfig:
    nt: int = 512
    nx: int = 64
    ny: int = 64
    beta2: float = -0.01
    gamma: float = 0.01
    t0: float = 0.1 / (2.0 * np.log(1.0 + np.sqrt(2.0)))
    dx: float = 0.6
    dy: float = 0.6
    grin_g: float = 0.020
    num_records: int = 40
    step_size: float = 0.001
    lambda0_nm: float = 1550.0

    @property
    def z_final(self) -> float:
        return 0.5 * ((self.t0 * self.t0) / abs(self.beta2))

    @property
    def dt(self) -> float:
        return (16.0 * self.t0) / float(self.nt)


@dataclass(frozen=True)
class DiffractionConfig:
    nt: int = 64
    nx: int = 64
    ny: int = 48
    beta2: float = -0.002
    diffraction_coeff: float = -0.004
    dt: float = 0.0035
    dx: float = 0.45
    dy: float = 0.62
    t0: float = 0.090
    wx: float = 9.0
    wy: float = 2.8
    z_final: float = 0.05
    num_records: int = 12
    step_size: float = 1.0e-3


def _build_phase_case(
    cfg: PhaseOnlyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = (np.arange(cfg.nt, dtype=np.float64) - 0.5 * float(cfg.nt - 1)) * cfg.dt
    x = (np.arange(cfg.nx, dtype=np.float64) - 0.5 * float(cfg.nx - 1)) * cfg.dx
    y = (np.arange(cfg.ny, dtype=np.float64) - 0.5 * float(cfg.ny - 1)) * cfg.dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    p0 = abs(cfg.beta2) / (cfg.gamma * cfg.t0 * cfg.t0)
    a0_t = (np.sqrt(p0) / np.cosh(t / cfg.t0)).astype(np.complex128)
    field0 = np.tile(a0_t[:, None, None], (1, cfg.ny, cfg.nx))
    potential_xy = (cfg.grin_g * (xx * xx + yy * yy)).astype(np.complex128)
    omega = (2.0 * np.pi * np.fft.fftfreq(cfg.nt, d=cfg.dt)).astype(np.float64)
    return t, x, y, field0, potential_xy, omega


def _analytical_phase_records(
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


def _build_diffraction_case(
    cfg: DiffractionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = (np.arange(cfg.nt, dtype=np.float64) - 0.5 * float(cfg.nt - 1)) * cfg.dt
    x = (np.arange(cfg.nx, dtype=np.float64) - 0.5 * float(cfg.nx - 1)) * cfg.dx
    y = (np.arange(cfg.ny, dtype=np.float64) - 0.5 * float(cfg.ny - 1)) * cfg.dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    sech_t = (1.0 / np.cosh(t / cfg.t0)).astype(np.float64)
    spatial = np.exp(-(((xx / cfg.wx) ** 2) + ((yy / cfg.wy) ** 2))).astype(np.float64)
    field0 = (sech_t[:, None, None] * spatial[None, :, :]).astype(np.complex128)
    omega = (2.0 * np.pi * np.fft.fftfreq(cfg.nt, d=cfg.dt)).astype(np.float64)
    return t, x, y, field0, omega


def _fft3_linear_reference(
    field0: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z_axis: np.ndarray,
    beta2: float,
    diffraction_coeff: float,
) -> np.ndarray:
    dt = float(t[1] - t[0])
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    wt = 2.0 * np.pi * np.fft.fftfreq(t.size, d=dt)
    kx = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(y.size, d=dy)
    wt_grid, ky_grid, kx_grid = np.meshgrid(wt, ky, kx, indexing="ij")
    linear = (0.5 * beta2) * (wt_grid * wt_grid) + diffraction_coeff * ((kx_grid * kx_grid) + (ky_grid * ky_grid))

    spectrum0 = np.fft.fftn(field0, axes=(0, 1, 2))
    out = np.empty((z_axis.size, t.size, y.size, x.size), dtype=np.complex128)
    for i, z in enumerate(z_axis):
        out[i] = np.fft.ifftn(spectrum0 * np.exp((1.0j) * linear * float(z)), axes=(0, 1, 2))
    return out


class GrinSolitonApp:
    """Runs GRIN soliton validation workflows with plot/data gating."""

    def __init__(self, args: Any):
        self.args = args
        self.db = ExampleRunDB(args.db_path)
        self.runner = NloExampleRunner()
        self.nlo = self.runner.nlo
        self.exec_options = SimulationOptions(backend="auto", fft_backend="auto", device_heap_fraction=0.70)
        self.example_name = "grin_soliton_potential_rk4ip"
        self.report_path: Path | None = None

    def _selected_cases(self) -> list[str]:
        value = str(getattr(self.args, "validation_case", "both")).strip().lower()
        if value == "both":
            return ["phase_only", "diffraction"]
        if value in {"phase_only", "diffraction"}:
            return [value]
        raise ValueError("validation_case must be one of: phase_only, diffraction, both.")

    def _run_phase_only(
        self,
        *,
        run_group: str,
        output_root: Path,
        replot: bool,
        report: ValidationReport,
    ) -> list[PlotArtifact]:
        cfg = PhaseOnlyConfig()
        case_key = "phase_only"
        alt_case_key = "default"

        if replot:
            cases = {case.case_key for case in self.db.list_cases(example_name=self.example_name, run_group=run_group)}
            selected_key = case_key if case_key in cases else (alt_case_key if alt_case_key in cases else case_key)
            loaded = self.db.load_case(example_name=self.example_name, run_group=run_group, case_key=selected_key)
            meta = loaded.meta
            cfg = PhaseOnlyConfig(
                nt=int(meta["nt"]),
                nx=int(meta["nx"]),
                ny=int(meta["ny"]),
                beta2=float(meta["beta2"]),
                gamma=float(meta["gamma"]),
                t0=float(meta["t0"]),
                dx=float(meta["dx"]),
                dy=float(meta["dy"]),
                grin_g=float(meta["grin_g"]),
                num_records=int(meta["num_records"]),
                step_size=float(meta.get("step_size", cfg.step_size)),
                lambda0_nm=float(meta.get("lambda0_nm", cfg.lambda0_nm)),
            )
            t, x, y, field0, potential_xy, _ = _build_phase_case(cfg)
            z_axis = np.asarray(loaded.z_axis, dtype=np.float64)
            records_flat = np.asarray(loaded.records, dtype=np.complex128).reshape(-1, cfg.nt * cfg.ny * cfg.nx)
        else:
            t, x, y, field0, potential_xy, omega = _build_phase_case(cfg)
            storage_kwargs = self.db.storage_kwargs(
                example_name=self.example_name,
                run_group=run_group,
                case_key=case_key,
                chunk_records=2,
            )
            runtime = self.nlo.RuntimeOperators(
                linear_factor_fn=lambda A, wt: (1.0j * (0.5 * cfg.beta2)) * (wt * wt),
                linear_fn=lambda A, D, h: np.exp(h * D),
                nonlinear_fn=lambda A, I, V: (1.0j * A) * (cfg.gamma * I + V),
            )
            sim_cfg = self.nlo.prepare_sim_config(
                cfg.nt * cfg.nx * cfg.ny,
                propagation_distance=float(cfg.z_final),
                starting_step_size=float(cfg.step_size),
                max_step_size=0.01,
                min_step_size=0.00001,
                error_tolerance=1e-6,
                pulse_period=float(cfg.nt) * cfg.dt,
                delta_time=cfg.dt,
                tensor_nt=cfg.nt,
                tensor_nx=cfg.nx,
                tensor_ny=cfg.ny,
                tensor_layout=int(self.nlo.NLO_TENSOR_LAYOUT_XYT_T_FAST),
                frequency_grid=[complex(float(w), 0.0) for w in omega],
                delta_x=cfg.dx,
                delta_y=cfg.dy,
                potential_grid=_flatten_xy_tfast(potential_xy).tolist(),
                runtime=runtime,
            )
            result = self.runner.api.propagate(
                sim_cfg,
                _flatten_tfast(field0).tolist(),
                int(cfg.num_records),
                exec_options=self.exec_options.to_ctypes(self.nlo),
                **storage_kwargs,
            )
            self.runner.last_meta = dict(result.meta)
            z_axis = np.asarray(result.z_axis, dtype=np.float64)
            records_flat = np.asarray(result.records, dtype=np.complex128).reshape(cfg.num_records, cfg.nt * cfg.ny * cfg.nx)
            self.db.save_case_from_solver_meta(
                example_name=self.example_name,
                run_group=run_group,
                case_key=case_key,
                solver_meta=self.runner.last_meta,
                meta={
                    "nt": int(cfg.nt),
                    "nx": int(cfg.nx),
                    "ny": int(cfg.ny),
                    "dt": float(cfg.dt),
                    "dx": float(cfg.dx),
                    "dy": float(cfg.dy),
                    "beta2": float(cfg.beta2),
                    "gamma": float(cfg.gamma),
                    "t0": float(cfg.t0),
                    "z_final": float(cfg.z_final),
                    "grin_g": float(cfg.grin_g),
                    "num_records": int(cfg.num_records),
                    "step_size": float(cfg.step_size),
                    "lambda0_nm": float(cfg.lambda0_nm),
                },
            )

        records = np.asarray([_unflatten_record_tfast(row, cfg.nt, cfg.ny, cfg.nx) for row in records_flat], dtype=np.complex128)
        records_ref = _analytical_phase_records(field0[:, 0, 0], potential_xy, z_axis, cfg.beta2, cfg.t0)
        z_axis_norm = _normalized_propagation_axis(z_axis, cfg.beta2, cfg.t0)
        z_axis_label = "Normalized propagation z / L_D"
        z_axis_label_3d = "z / L_D"

        error_curve = _relative_l2_error_curve(records, records_ref)
        final_error = float(error_curve[-1])
        power_num = np.sum(np.abs(records) ** 2, axis=(1, 2, 3))
        power_drift = float(abs(power_num[-1] - power_num[0]) / max(float(power_num[0]), 1e-15))

        center_y = cfg.ny // 2
        center_x = cfg.nx // 2
        center_t = cfg.nt // 2
        temporal_num = np.asarray(records[:, :, center_y, center_x], dtype=np.complex128)
        temporal_ref = np.asarray(records_ref[:, :, center_y, center_x], dtype=np.complex128)
        xline_num = np.asarray(records[-1, center_t, center_y, :], dtype=np.complex128)
        xline_ref = np.asarray(records_ref[-1, center_t, center_y, :], dtype=np.complex128)

        xz_profile_num = np.abs(records[:, center_t, center_y, :]) ** 2
        xz_profile_ref = np.abs(records_ref[:, center_t, center_y, :]) ** 2
        yz_profile_num = np.abs(records[:, center_t, :, center_x]) ** 2
        yz_profile_ref = np.abs(records_ref[:, center_t, :, center_x]) ** 2
        xz_error_curve = _relative_l2_error_curve(xz_profile_num, xz_profile_ref)
        yz_error_curve = _relative_l2_error_curve(yz_profile_num, yz_profile_ref)
        final_xz_error = float(xz_error_curve[-1])
        final_yz_error = float(yz_error_curve[-1])

        temporal_center_num = np.abs(temporal_num) ** 2
        temporal_weighted_sum = np.sum(temporal_center_num * t[None, :], axis=1)
        temporal_norm = np.sum(temporal_center_num, axis=1)
        temporal_centroid = temporal_weighted_sum / np.maximum(temporal_norm, 1e-30)
        temporal_centroid_drift = float(abs(temporal_centroid[-1] - temporal_centroid[0]))

        out_dir = output_root / "phase_only"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[PlotArtifact] = []

        def _record(path: Path | None, *, allow_uniform: bool = False) -> None:
            if path is not None:
                artifacts.append(PlotArtifact(key=path.stem, path=path, allow_uniform=allow_uniform))

        _record(
            plot_intensity_colormap_vs_propagation(
                t,
                z_axis_norm,
                np.abs(temporal_num) ** 2,
                out_dir / "center_temporal_intensity_colormap.png",
                x_label="Time t",
                y_label=z_axis_label,
                title="Center-point temporal intensity (numerical)",
                colorbar_label="Normalized intensity",
            )
        )
        _record(
            plot_final_intensity_comparison(
                t,
                temporal_ref[-1],
                temporal_num[-1],
                out_dir / "final_center_temporal_intensity_comparison.png",
                x_label="Time t",
                title="Final temporal intensity at GRIN center (analytical vs numerical)",
                reference_label="Analytical",
                final_label="Numerical",
            )
        )
        _record(
            plot_final_re_im_comparison(
                x,
                xline_ref,
                xline_num,
                out_dir / "final_xline_re_im_comparison_tmid_ycenter.png",
                x_label="Transverse coordinate x",
                title="Final transverse line field (t=t_mid, y=y_mid)",
                reference_label="Analytical",
                final_label="Numerical",
            )
        )
        _record(
            plot_total_error_over_propagation(
                z_axis_norm,
                error_curve,
                out_dir / "relative_error_over_propagation.png",
                title="GRIN potential + soliton: mean pointwise abs-relative error over z",
                y_label="Mean pointwise abs-relative error",
                x_label=z_axis_label,
            )
        )

        spatial_num = np.sum(np.abs(records) ** 2, axis=1)
        spatial_ref = np.sum(np.abs(records_ref) ** 2, axis=1)
        spatial_peak = float(max(np.max(spatial_num), np.max(spatial_ref)))
        _record(
            plot_3d_intensity_scatter_propagation(
                x,
                y,
                z_axis_norm,
                spatial_num,
                out_dir / "spatial_integrated_3d_numerical_scatter.png",
                intensity_cutoff=0.02,
                xy_stride=2,
                z_stride=1,
                min_marker_size=1.0,
                max_marker_size=24.0,
                alpha_min=0.04,
                alpha_max=0.95,
                dpi=400,
                input_is_intensity=True,
                normalization_peak=spatial_peak,
                title="GRIN+Soliton: spatial intensity integrated over time (numerical, uniform expected)",
                z_label=z_axis_label_3d,
            ),
            allow_uniform=True,
        )
        _record(
            plot_3d_intensity_scatter_propagation(
                x,
                y,
                z_axis_norm,
                spatial_ref,
                out_dir / "spatial_integrated_3d_expected_scatter.png",
                intensity_cutoff=0.02,
                xy_stride=2,
                z_stride=1,
                min_marker_size=1.0,
                max_marker_size=24.0,
                alpha_min=0.04,
                alpha_max=0.95,
                dpi=400,
                input_is_intensity=True,
                normalization_peak=spatial_peak,
                title="GRIN+Soliton: spatial intensity integrated over time (expected, uniform)",
                z_label=z_axis_label_3d,
            ),
            allow_uniform=True,
        )
        spec_num = np.fft.fftshift(np.fft.fft(temporal_num, axis=1), axes=1)
        spec_ref = np.fft.fftshift(np.fft.fft(temporal_ref, axis=1), axes=1)
        freq_axis = np.fft.fftshift(np.fft.fftfreq(cfg.nt, d=cfg.dt))
        spec_peak = float(max(np.max(np.abs(spec_num) ** 2), np.max(np.abs(spec_ref) ** 2)))
        _record(
            plot_intensity_colormap_vs_propagation(
                freq_axis,
                z_axis_norm,
                np.abs(spec_num) ** 2,
                out_dir / "frequency_profile_numerical.png",
                x_label="Frequency detuning (1/time)",
                y_label=z_axis_label,
                title="Frequency propagation profile (numerical)",
                colorbar_label="Normalized spectral intensity",
                normalization_peak=spec_peak,
            )
        )
        _record(
            plot_intensity_colormap_vs_propagation(
                freq_axis,
                z_axis_norm,
                np.abs(spec_ref) ** 2,
                out_dir / "frequency_profile_expected.png",
                x_label="Frequency detuning (1/time)",
                y_label=z_axis_label,
                title="Frequency propagation profile (expected)",
                colorbar_label="Normalized spectral intensity",
                normalization_peak=spec_peak,
            )
        )

        wavelength_nm, valid = omega_detuning_to_wavelength_nm(
            2.0 * np.pi * freq_axis,
            cfg.lambda0_nm,
            time_unit_seconds=1.0e-12,
        )
        order = np.argsort(wavelength_nm)
        wl_axis = wavelength_nm[order]
        wl_num = np.abs(spec_num[:, valid][:, order]) ** 2
        wl_ref = np.abs(spec_ref[:, valid][:, order]) ** 2
        window_selector = WavelengthWindowSelector(float(getattr(self.args, "wavelength_mass", 0.999)))
        wl_axis, wl_num, wl_ref, window = window_selector.select(wl_axis, wl_num, wl_ref)
        wl_peak = float(max(np.max(wl_num), np.max(wl_ref)))
        _record(
            plot_intensity_colormap_vs_propagation(
                wl_axis,
                z_axis_norm,
                wl_num,
                out_dir / "wavelength_profile_numerical.png",
                x_label="Wavelength (nm)",
                y_label=z_axis_label,
                title="Wavelength propagation profile (numerical)",
                colorbar_label="Normalized spectral intensity",
                normalization_peak=wl_peak,
            )
        )
        _record(
            plot_intensity_colormap_vs_propagation(
                wl_axis,
                z_axis_norm,
                wl_ref,
                out_dir / "wavelength_profile_expected.png",
                x_label="Wavelength (nm)",
                y_label=z_axis_label,
                title="Wavelength propagation profile (expected)",
                colorbar_label="Normalized spectral intensity",
                normalization_peak=wl_peak,
            )
        )

        xz_peak = float(max(np.max(xz_profile_num), np.max(xz_profile_ref)))
        _record(
            plot_intensity_colormap_vs_propagation(
                x,
                z_axis_norm,
                xz_profile_num,
                out_dir / "x_profile_vs_z_numerical_tmid_ycenter.png",
                x_label="Transverse coordinate x",
                y_label=z_axis_label,
                title="X-profile intensity propagation (numerical, t=t_mid, y=y_mid)",
                colorbar_label="Normalized intensity",
                normalization_peak=xz_peak,
            ),
            allow_uniform=True,
        )
        _record(
            plot_intensity_colormap_vs_propagation(
                x,
                z_axis_norm,
                xz_profile_ref,
                out_dir / "x_profile_vs_z_expected_tmid_ycenter.png",
                x_label="Transverse coordinate x",
                y_label=z_axis_label,
                title="X-profile intensity propagation (expected, t=t_mid, y=y_mid)",
                colorbar_label="Normalized intensity",
                normalization_peak=xz_peak,
            ),
            allow_uniform=True,
        )
        yz_peak = float(max(np.max(yz_profile_num), np.max(yz_profile_ref)))
        _record(
            plot_intensity_colormap_vs_propagation(
                y,
                z_axis_norm,
                yz_profile_num,
                out_dir / "y_profile_vs_z_numerical_tmid_xcenter.png",
                x_label="Transverse coordinate y",
                y_label=z_axis_label,
                title="Y-profile intensity propagation (numerical, t=t_mid, x=x_mid)",
                colorbar_label="Normalized intensity",
                normalization_peak=yz_peak,
            ),
            allow_uniform=True,
        )
        _record(
            plot_intensity_colormap_vs_propagation(
                y,
                z_axis_norm,
                yz_profile_ref,
                out_dir / "y_profile_vs_z_expected_tmid_xcenter.png",
                x_label="Transverse coordinate y",
                y_label=z_axis_label,
                title="Y-profile intensity propagation (expected, t=t_mid, x=x_mid)",
                colorbar_label="Normalized intensity",
                normalization_peak=yz_peak,
            ),
            allow_uniform=True,
        )
        _record(
            plot_two_curve_comparison(
                z_axis_norm,
                xz_error_curve,
                yz_error_curve,
                out_dir / "xy_profile_error_over_propagation.png",
                label_a="X-profile mean pointwise abs-relative error",
                label_b="Y-profile mean pointwise abs-relative error",
                x_label=z_axis_label,
                y_label="Mean pointwise abs-relative error",
                title="X/Y profile intensity error over propagation",
            ),
            allow_uniform=True,
        )

        report.add_threshold(name="phase_only:final_relative_l2", value=final_error, threshold=1e-3)
        report.add_threshold(name="phase_only:final_x_profile_error", value=final_xz_error, threshold=5e-4)
        report.add_threshold(name="phase_only:final_y_profile_error", value=final_yz_error, threshold=5e-4)
        report.add_threshold(name="phase_only:power_drift", value=power_drift, threshold=1e-3)
        report.add_threshold(name="phase_only:center_temporal_centroid_drift", value=temporal_centroid_drift, threshold=1e-4)
        report.add_threshold(
            name="phase_only:wavelength_window_mass",
            value=float(window.mass_fraction),
            threshold=float(getattr(self.args, "wavelength_mass", 0.999)),
            comparator=">=",
            level="warn",
        )
        report.metadata["phase_only"] = {
            "final_error": final_error,
            "final_x_profile_error": final_xz_error,
            "final_y_profile_error": final_yz_error,
            "power_drift": power_drift,
            "temporal_centroid_drift": temporal_centroid_drift,
            "wavelength_window": [float(wl_axis[0]), float(wl_axis[-1])] if wl_axis.size > 0 else [],
            "wavelength_window_mass": float(window.mass_fraction),
        }
        return artifacts

    def _run_diffraction(
        self,
        *,
        run_group: str,
        output_root: Path,
        replot: bool,
        report: ValidationReport,
    ) -> list[PlotArtifact]:
        cfg = DiffractionConfig()
        case_key = "diffraction"

        if replot:
            loaded = self.db.load_case(example_name=self.example_name, run_group=run_group, case_key=case_key)
            meta = loaded.meta
            cfg = DiffractionConfig(
                nt=int(meta["nt"]),
                nx=int(meta["nx"]),
                ny=int(meta["ny"]),
                beta2=float(meta["beta2"]),
                diffraction_coeff=float(meta["diffraction_coeff"]),
                dt=float(meta["dt"]),
                dx=float(meta["dx"]),
                dy=float(meta["dy"]),
                t0=float(meta["t0"]),
                wx=float(meta["wx"]),
                wy=float(meta["wy"]),
                z_final=float(meta["z_final"]),
                num_records=int(meta["num_records"]),
                step_size=float(meta.get("step_size", cfg.step_size)),
            )
            t, x, y, field0, _ = _build_diffraction_case(cfg)
            z_axis = np.asarray(loaded.z_axis, dtype=np.float64)
            records_flat = np.asarray(loaded.records, dtype=np.complex128).reshape(-1, cfg.nt * cfg.ny * cfg.nx)
            records = np.asarray([_unflatten_record_tfast(row, cfg.nt, cfg.ny, cfg.nx) for row in records_flat], dtype=np.complex128)
        else:
            t, x, y, field0, omega = _build_diffraction_case(cfg)
            runtime = self.nlo.RuntimeOperators(
                linear_factor_fn=lambda A, wt, kx, ky: (1.0j)
                * ((0.5 * cfg.beta2) * (wt * wt) + cfg.diffraction_coeff * ((kx * kx) + (ky * ky))),
                linear_fn=lambda A, D, h: np.exp(h * D),
                nonlinear_fn=lambda A, I: 0.0,
            )
            storage_kwargs = self.db.storage_kwargs(
                example_name=self.example_name,
                run_group=run_group,
                case_key=case_key,
                chunk_records=2,
            )
            sim_cfg = self.nlo.prepare_sim_config(
                cfg.nt * cfg.nx * cfg.ny,
                propagation_distance=float(cfg.z_final),
                starting_step_size=float(cfg.step_size),
                max_step_size=0.005,
                min_step_size=1.0e-5,
                error_tolerance=1e-7,
                pulse_period=float(cfg.nt) * cfg.dt,
                delta_time=cfg.dt,
                tensor_nt=cfg.nt,
                tensor_nx=cfg.nx,
                tensor_ny=cfg.ny,
                tensor_layout=int(self.nlo.NLO_TENSOR_LAYOUT_XYT_T_FAST),
                frequency_grid=[complex(float(w), 0.0) for w in omega],
                delta_x=cfg.dx,
                delta_y=cfg.dy,
                potential_grid=[0.0 + 0.0j] * (cfg.nt * cfg.nx * cfg.ny),
                runtime=runtime,
            )
            result = self.runner.api.propagate(
                sim_cfg,
                _flatten_tfast(field0).tolist(),
                int(cfg.num_records),
                exec_options=self.exec_options.to_ctypes(self.nlo),
                **storage_kwargs,
            )
            self.runner.last_meta = dict(result.meta)
            z_axis = np.asarray(result.z_axis, dtype=np.float64)
            records_flat = np.asarray(result.records, dtype=np.complex128).reshape(cfg.num_records, cfg.nt * cfg.ny * cfg.nx)
            records = np.asarray([_unflatten_record_tfast(row, cfg.nt, cfg.ny, cfg.nx) for row in records_flat], dtype=np.complex128)
            self.db.save_case_from_solver_meta(
                example_name=self.example_name,
                run_group=run_group,
                case_key=case_key,
                solver_meta=self.runner.last_meta,
                meta={
                    "nt": int(cfg.nt),
                    "nx": int(cfg.nx),
                    "ny": int(cfg.ny),
                    "beta2": float(cfg.beta2),
                    "diffraction_coeff": float(cfg.diffraction_coeff),
                    "dt": float(cfg.dt),
                    "dx": float(cfg.dx),
                    "dy": float(cfg.dy),
                    "t0": float(cfg.t0),
                    "wx": float(cfg.wx),
                    "wy": float(cfg.wy),
                    "z_final": float(cfg.z_final),
                    "num_records": int(cfg.num_records),
                    "step_size": float(cfg.step_size),
                },
            )
        records_ref = _fft3_linear_reference(field0, t, x, y, z_axis, cfg.beta2, cfg.diffraction_coeff)
        z_axis_norm = _normalized_propagation_axis(z_axis, cfg.beta2, cfg.t0)
        z_axis_label = "Normalized propagation z / L_D"
        z_axis_label_3d = "z / L_D"
        error_curve = _relative_l2_error_curve(records, records_ref)
        final_error = float(error_curve[-1])

        center_t = cfg.nt // 2
        center_x = cfg.nx // 2
        center_y = cfg.ny // 2
        xz_num = np.abs(records[:, center_t, center_y, :]) ** 2
        yz_num = np.abs(records[:, center_t, :, center_x]) ** 2
        xz_ref = np.abs(records_ref[:, center_t, center_y, :]) ** 2
        yz_ref = np.abs(records_ref[:, center_t, :, center_x]) ** 2

        final_x_num = xz_num[-1]
        final_y_num = yz_num[-1]
        final_x_ref = xz_ref[-1]
        final_y_ref = yz_ref[-1]
        corr_x = profile_correlation(final_x_num, final_x_ref)
        corr_y = profile_correlation(final_y_num, final_y_ref)
        cross_xy = profile_correlation(final_x_num, np.interp(np.linspace(0.0, 1.0, cfg.nx), np.linspace(0.0, 1.0, cfg.ny), final_y_ref))
        cross_yx = profile_correlation(final_y_num, np.interp(np.linspace(0.0, 1.0, cfg.ny), np.linspace(0.0, 1.0, cfg.nx), final_x_ref))
        swap_indicator = bool((corr_x < cross_xy) or (corr_y < cross_yx))

        out_dir = output_root / "diffraction"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[PlotArtifact] = []

        def _record(path: Path | None) -> None:
            if path is not None:
                artifacts.append(PlotArtifact(key=path.stem, path=path, allow_uniform=False))

        _record(
            plot_total_error_over_propagation(
                z_axis_norm,
                error_curve,
                out_dir / "relative_error_over_propagation.png",
                title="GRIN diffraction: mean pointwise abs-relative error over z",
                y_label="Mean pointwise abs-relative error",
                x_label=z_axis_label,
            )
        )
        xz_peak = float(max(np.max(xz_num), np.max(xz_ref)))
        yz_peak = float(max(np.max(yz_num), np.max(yz_ref)))
        _record(
            plot_intensity_colormap_vs_propagation(
                x,
                z_axis_norm,
                xz_num,
                out_dir / "x_profile_vs_z_numerical_tmid_ycenter.png",
                x_label="Transverse coordinate x",
                y_label=z_axis_label,
                title="X-profile intensity propagation (numerical, diffraction)",
                colorbar_label="Normalized intensity",
                normalization_peak=xz_peak,
            )
        )
        _record(
            plot_intensity_colormap_vs_propagation(
                x,
                z_axis_norm,
                xz_ref,
                out_dir / "x_profile_vs_z_expected_tmid_ycenter.png",
                x_label="Transverse coordinate x",
                y_label=z_axis_label,
                title="X-profile intensity propagation (expected, diffraction)",
                colorbar_label="Normalized intensity",
                normalization_peak=xz_peak,
            )
        )
        _record(
            plot_intensity_colormap_vs_propagation(
                y,
                z_axis_norm,
                yz_num,
                out_dir / "y_profile_vs_z_numerical_tmid_xcenter.png",
                x_label="Transverse coordinate y",
                y_label=z_axis_label,
                title="Y-profile intensity propagation (numerical, diffraction)",
                colorbar_label="Normalized intensity",
                normalization_peak=yz_peak,
            )
        )
        _record(
            plot_intensity_colormap_vs_propagation(
                y,
                z_axis_norm,
                yz_ref,
                out_dir / "y_profile_vs_z_expected_tmid_xcenter.png",
                x_label="Transverse coordinate y",
                y_label=z_axis_label,
                title="Y-profile intensity propagation (expected, diffraction)",
                colorbar_label="Normalized intensity",
                normalization_peak=yz_peak,
            )
        )

        spatial_num = np.sum(np.abs(records) ** 2, axis=1)
        spatial_ref = np.sum(np.abs(records_ref) ** 2, axis=1)
        spatial_peak = float(max(np.max(spatial_num), np.max(spatial_ref)))
        _record(
            plot_3d_intensity_scatter_propagation(
                x,
                y,
                z_axis_norm,
                spatial_num,
                out_dir / "spatial_integrated_3d_numerical_scatter.png",
                intensity_cutoff=0.01,
                xy_stride=2,
                z_stride=1,
                min_marker_size=1.0,
                max_marker_size=22.0,
                alpha_min=0.04,
                alpha_max=0.95,
                dpi=320,
                input_is_intensity=True,
                normalization_peak=spatial_peak,
                title="GRIN diffraction: spatial integrated intensity (numerical)",
                z_label=z_axis_label_3d,
            )
        )
        _record(
            plot_3d_intensity_scatter_propagation(
                x,
                y,
                z_axis_norm,
                spatial_ref,
                out_dir / "spatial_integrated_3d_expected_scatter.png",
                intensity_cutoff=0.01,
                xy_stride=2,
                z_stride=1,
                min_marker_size=1.0,
                max_marker_size=22.0,
                alpha_min=0.04,
                alpha_max=0.95,
                dpi=320,
                input_is_intensity=True,
                normalization_peak=spatial_peak,
                title="GRIN diffraction: spatial integrated intensity (expected)",
                z_label=z_axis_label_3d,
            )
        )

        report.add_threshold(name="diffraction:final_relative_l2", value=final_error, threshold=6e-2)
        report.add_threshold(name="diffraction:corr_x_num_ref", value=corr_x, threshold=0.98, comparator=">=")
        report.add_threshold(name="diffraction:corr_y_num_ref", value=corr_y, threshold=0.98, comparator=">=")
        report.add_threshold(name="diffraction:corr_x_num_y_ref_cross", value=cross_xy, threshold=0.92, comparator="<=")
        report.add_threshold(name="diffraction:corr_y_num_x_ref_cross", value=cross_yx, threshold=0.92, comparator="<=")
        report.add_threshold(
            name="diffraction:x_y_swap_suspected",
            value=1.0 if swap_indicator else 0.0,
            threshold=0.0,
            comparator="<=",
            detail="1 indicates possible x/y swap.",
        )
        report.metadata["diffraction"] = {
            "final_error": final_error,
            "corr_x": corr_x,
            "corr_y": corr_y,
            "cross_xy": cross_xy,
            "cross_yx": cross_yx,
            "swap_suspected": swap_indicator,
        }
        return artifacts

    def run(self) -> tuple[str, list[Path], ValidationReport]:
        if self.args.replot:
            run_group = self.db.resolve_replot_group(self.example_name, self.args.run_group)
        else:
            run_group = self.db.begin_group(self.example_name, self.args.run_group)
        output_root = self.args.output_dir / run_group
        output_root.mkdir(parents=True, exist_ok=True)
        report = ValidationReport(example_name=self.example_name, run_group=run_group)
        report.metadata["mode"] = "replot" if self.args.replot else "run"
        report.metadata["output_root"] = str(output_root)

        artifacts: list[PlotArtifact] = []
        for case_name in self._selected_cases():
            try:
                if case_name == "phase_only":
                    artifacts.extend(
                        self._run_phase_only(run_group=run_group, output_root=output_root, replot=self.args.replot, report=report)
                    )
                elif case_name == "diffraction":
                    artifacts.extend(
                        self._run_diffraction(run_group=run_group, output_root=output_root, replot=self.args.replot, report=report)
                    )
            except RuntimeError as exc:
                if self.args.replot and "missing case" in str(exc).lower():
                    report.add(
                        ValidationCheck(
                            name=f"replot:{case_name}:missing_case",
                            level="warn",
                            passed=False,
                            detail=str(exc),
                        )
                    )
                    continue
                raise

        if bool(getattr(self.args, "plot_validate", True)):
            PlotImageValidator().validate_artifacts(report, artifacts)

        report_path = (
            Path(self.args.plot_validation_report)
            if getattr(self.args, "plot_validation_report", None) is not None
            else (output_root / "plot_validation_report.json")
        )
        self.report_path = write_report(report, report_path)

        print(f"grin-soliton potential validation summary (run_group={run_group}):")
        for check in report.checks:
            if check.value is None:
                continue
            threshold_txt = f" <= {check.threshold:.6e}" if check.threshold is not None else ""
            print(f"  {check.name}: value={check.value:.6e}{threshold_txt} {'PASS' if check.passed else 'FAIL'}")
        print("saved plots:")
        for artifact in artifacts:
            print(f"  {artifact.path}")
        print(f"plot validation report: {self.report_path}")

        has_failures = report.fail_count() > 0
        has_warnings = report.warn_count() > 0
        if has_failures or (has_warnings and bool(getattr(self.args, "fail_on_plot_warning", False))):
            raise RuntimeError(
                "plot validation failed"
                + (f": fails={report.fail_count()}" if has_failures else "")
                + (f", warns={report.warn_count()}" if has_warnings else "")
            )
        return run_group, [a.path for a in artifacts], report
