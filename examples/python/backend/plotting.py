"""Shared plotting helpers for nlolib Python examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_CMAP_NAME = "nlolib_white_cyan_yellow_hdr"
_DEFAULT_CMAP = None
_STYLE_CMAP_NAME = "nlolib_hdr"


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _default_colormap():
    global _DEFAULT_CMAP
    if _DEFAULT_CMAP is not None:
        return _DEFAULT_CMAP

    from matplotlib.colors import LinearSegmentedColormap

    _DEFAULT_CMAP = LinearSegmentedColormap.from_list(
        _DEFAULT_CMAP_NAME,
        [
            (0.00, "#ffffff"),
            (0.05, "#7ee3ed"),
            (0.25, "#4e8ec3"),
            (0.50, "#002b56"),
            (0.75, "#4d2d99"),
            (0.95, "#fd5ddd"),
            (1.00, "#ff207d"),
        ],
    )
    return _DEFAULT_CMAP


def _ensure_default_colormap_registered():
    if _STYLE_CMAP_NAME not in matplotlib.colormaps:
        matplotlib.colormaps.register(_default_colormap(), name=_STYLE_CMAP_NAME)


_ensure_default_colormap_registered()
_STYLE_PATH = Path(__file__).with_name("figures.mplstyle")
if _STYLE_PATH.is_file():
    plt.style.use(_STYLE_PATH)


def _resolve_cmap(plt, cmap):
    if cmap is None or cmap == _STYLE_CMAP_NAME:
        return _default_colormap()
    return cmap

def plot_intensity_colormap_vs_propagation(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    intensity_map: np.ndarray,
    output_path: Path,
    *,
    x_label: str,
    y_label: str = "Propagation distance z",
    title: str = "Intensity vs Propagation",
    colorbar_label: str = "Normalized intensity",
    cmap="nlolib_hdr",
) -> Path | None:


    data = np.asarray(intensity_map, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    peak = float(np.max(data))
    if peak > 0.0:
        data = data / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(x_axis, z_axis, data, shading="auto", cmap=_resolve_cmap(plt, cmap))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_re_im_comparison(
    x_axis: np.ndarray,
    reference_field: np.ndarray,
    final_field: np.ndarray,
    output_path: Path,
    *,
    x_label: str,
    title: str = "Final Re/Im Comparison",
    reference_label: str = "Reference",
    final_label: str = "Final",
) -> Path | None:


    ref = np.asarray(reference_field, dtype=np.complex128)
    out = np.asarray(final_field, dtype=np.complex128)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(x_axis, np.real(ref), lw=2.0, color="C0", label=f"{reference_label} Re")
    ax.plot(x_axis, np.imag(ref), lw=2.0, color="C1", label=f"{reference_label} Im")
    ax.plot(x_axis, np.real(out), lw=1.5, color="C0", ls="--", label=f"{final_label} Re")
    ax.plot(x_axis, np.imag(out), lw=1.5, color="C1", ls="--", label=f"{final_label} Im")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Field amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_two_curve_comparison(
    x_axis: np.ndarray,
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    output_path: Path,
    *,
    label_a: str,
    label_b: str,
    x_label: str = "Propagation distance z",
    y_label: str,
    title: str,
) -> Path | None:


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(x_axis, curve_a, lw=2.0, label=label_a)
    ax.plot(x_axis, curve_b, lw=1.5, ls="--", label=label_b)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_three_curve_drift(
    x_axis: np.ndarray,
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    curve_c: np.ndarray,
    output_path: Path,
    *,
    label_a: str,
    label_b: str,
    label_c: str,
    x_label: str = "Propagation distance z",
    y_label: str = "Relative drift",
    title: str = "Conservation Checks: Relative Drift Over Propagation",
) -> Path | None:


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(x_axis, curve_a, lw=1.8, label=label_a)
    ax.plot(x_axis, curve_b, lw=1.8, label=label_b)
    ax.plot(x_axis, curve_c, lw=1.8, label=label_c)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_mode_power_exchange(
    z_axis: np.ndarray,
    mode1_num: np.ndarray,
    mode2_num: np.ndarray,
    mode1_ref: np.ndarray,
    mode2_ref: np.ndarray,
    output_path: Path,
) -> Path | None:


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(z_axis, mode1_ref, lw=2.0, color="C0", label="|A1|^2 analytical")
    ax.plot(z_axis, mode2_ref, lw=2.0, color="C1", label="|A2|^2 analytical")
    ax.plot(z_axis, mode1_num, "--", lw=1.5, color="C0", label="|A1|^2 numerical")
    ax.plot(z_axis, mode2_num, "--", lw=1.5, color="C1", label="|A2|^2 numerical")
    ax.set_xlabel("Propagation distance z")
    ax.set_ylabel("Mode power")
    ax.set_title("Two-Mode Linear Beating: Power Exchange")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_phase_shift_comparison(
    t_axis: np.ndarray,
    phase_num: np.ndarray,
    phase_ref: np.ndarray,
    phase_mask: np.ndarray,
    output_path: Path,
    *,
    title: str = "SPM Final Phase Shift (Masked by Pulse Support)",
) -> Path | None:


    phase_num_plot = np.where(phase_mask, phase_num, np.nan)
    phase_ref_plot = np.where(phase_mask, phase_ref, np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(t_axis, phase_ref_plot, lw=2.0, label="Analytical phase shift")
    ax.plot(t_axis, phase_num_plot, "--", lw=1.5, label="Numerical phase shift")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Phase shift (rad)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_convergence_loglog(
    step_sizes: np.ndarray,
    errors: np.ndarray,
    fit_mask: np.ndarray,
    fitted_order: float,
    fitted_intercept: float,
    output_path: Path,
) -> Path | None:


    order = np.argsort(step_sizes)
    step_sizes_plot = step_sizes[order]
    errors_plot = errors[order]
    fit_mask_plot = fit_mask[order]

    fit_indices = np.flatnonzero(fit_mask_plot)
    anchor = int(fit_indices[0]) if fit_indices.size > 0 else 0
    ref = errors_plot[anchor] * (step_sizes_plot / step_sizes_plot[anchor]) ** 4
    fit_line = np.exp(fitted_intercept) * (step_sizes_plot**fitted_order)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.loglog(step_sizes_plot, errors_plot, "o", lw=1.8, ms=3.0, label="Numerical error")
    ax.loglog(step_sizes_plot, fit_line, "--", lw=1.6, color="C4", label="Fitted power law")
    ax.loglog(step_sizes_plot, ref, "--", lw=1.5, label=r"Reference $O(\Delta z^4)$")
    ax.set_xlabel("Step size Delta z (m)")
    ax.set_ylabel("Total relative L2 error")
    ax.set_title(f"Fixed-Step Soliton Convergence (fitted order p = {fitted_order:.3f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_summary_curve(
    x_values: np.ndarray | list[float],
    y_values: np.ndarray | list[float],
    output_path: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> Path | None:


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, marker="o", lw=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_wavelength_step_history(
    z_samples: np.ndarray,
    lambda_nm: np.ndarray,
    spectral_map: np.ndarray,
    output_path: Path,
    *,
    accepted_z: np.ndarray | None = None,
    accepted_step_sizes: np.ndarray | None = None,
) -> Path | None:

    z_axis = np.asarray(z_samples, dtype=np.float64)
    lambda_axis = np.asarray(lambda_nm, dtype=np.float64)
    data = np.asarray(spectral_map, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    if data.shape != (z_axis.size, lambda_axis.size):
        raise ValueError("spectral_map shape must be [record, wavelength].")

    peak = float(np.max(data))
    if peak > 0.0:
        data = data / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4.0, 6.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.4], hspace=0.24)

    ax_map = fig.add_subplot(grid[0, 0])
    mesh = ax_map.pcolormesh(z_axis, lambda_axis, data.T, shading="auto", cmap=_resolve_cmap(plt, None))
    ax_map.set_xlabel("Propagation distance z (m)")
    ax_map.set_ylabel("Wavelength (nm)")
    ax_map.set_title("")
    ax_map.set_box_aspect(1.0)
    cbar = fig.colorbar(mesh, ax=ax_map, pad=0.02)
    cbar.set_label("Normalized spectral intensity")

    ax_step = fig.add_subplot(grid[1, 0])
    has_series = False
    if accepted_z is not None and accepted_step_sizes is not None:
        z_plot = np.asarray(accepted_z, dtype=np.float64).reshape(-1)
        step_plot = np.asarray(accepted_step_sizes, dtype=np.float64).reshape(-1)
        n = min(z_plot.size, step_plot.size)
        if n > 0:
            order = np.argsort(z_plot[:n])
            ax_step.plot(
                z_plot[:n][order],
                step_plot[:n][order],
                lw=1.2,
                color="C1",
                label="Accepted step_size",
            )
            has_series = True

    if has_series:
        ax_step.set_xlabel("Propagation distance z (m)")
        ax_step.set_ylabel("Step size (m)")
        ax_step.set_title("Adaptive RK4IP Step Sizes")
        ax_step.grid(True, alpha=0.3)
        ax_step.legend()
    else:
        ax_step.text(
            0.5,
            0.5,
            "No adaptive step-adjustment events captured",
            transform=ax_step.transAxes,
            ha="center",
            va="center",
        )
        ax_step.set_xticks([])
        ax_step.set_yticks([])
        ax_step.set_title("Adaptive RK4IP Step Sizes")

    map_pos = ax_map.get_position()
    step_pos = ax_step.get_position()
    ax_step.set_position([map_pos.x0, step_pos.y0, map_pos.width, step_pos.height])

    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_intensity_comparison(
    x_axis: np.ndarray,
    reference_field: np.ndarray,
    final_field: np.ndarray,
    output_path: Path,
    *,
    x_label: str,
    title: str = "Final Intensity Comparison",
    reference_label: str = "Reference",
    final_label: str = "Final",
) -> Path | None:


    ref_intensity = np.abs(np.asarray(reference_field, dtype=np.complex128)) ** 2
    out_intensity = np.abs(np.asarray(final_field, dtype=np.complex128)) ** 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(x_axis, ref_intensity, lw=2.0, color="C0", label=f"{reference_label} |A|^2")
    ax.plot(x_axis, out_intensity, lw=1.5, ls="--", color="C1", label=f"{final_label} |A|^2")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Intensity |A|^2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_total_error_over_propagation(
    z_axis: np.ndarray,
    error_curve: np.ndarray,
    output_path: Path,
    *,
    title: str = "Total Error Over Propagation",
    y_label: str = "Relative L2 error",
) -> Path | None:


    errors = np.asarray(error_curve, dtype=np.float64)
    errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
    errors = np.clip(errors, 0.0, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(z_axis, errors, lw=1.8, color="C3")
    ax.set_xlabel("Propagation distance z")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_3d_intensity_scatter_propagation(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    field_records: np.ndarray,
    output_path: Path,
    *,
    intensity_cutoff: float = 0.05,
    xy_stride: int = 16,
    z_stride: int = 1,
    min_marker_size: float = 2.0,
    max_marker_size: float = 36.0,
    alpha_min: float = 0.08,
    alpha_max: float = 0.90,
    dpi: int = 320,
    title: str = "3D Propagation Intensity Scatter",
) -> Path | None:


    records = np.asarray(field_records, dtype=np.complex128)
    if records.ndim != 3:
        raise ValueError("field_records must be [record, y, x].")

    x = np.asarray(x_axis, dtype=np.float64)
    y = np.asarray(y_axis, dtype=np.float64)
    z = np.asarray(z_axis, dtype=np.float64)
    if records.shape[0] != z.size or records.shape[1] != y.size or records.shape[2] != x.size:
        raise ValueError("Axes lengths must match field_records shape.")

    if xy_stride <= 0:
        raise ValueError("xy_stride must be positive.")
    if z_stride <= 0:
        raise ValueError("z_stride must be positive.")
    if intensity_cutoff < 0.0 or intensity_cutoff >= 1.0:
        raise ValueError("intensity_cutoff must be in [0, 1).")
    if alpha_min < 0.0 or alpha_min > 1.0 or alpha_max < 0.0 or alpha_max > 1.0 or alpha_min > alpha_max:
        raise ValueError("alpha_min/alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    intensity = np.abs(records) ** 2
    max_intensity = float(np.max(intensity))
    if max_intensity <= 0.0:
        print("intensity is zero everywhere; skipping 3D propagation scatter plot.")
        return None

    x_points: list[float] = []
    y_points: list[float] = []
    z_points: list[float] = []
    c_points: list[float] = []
    s_points: list[float] = []

    for zi in range(0, z.size, z_stride):
        for yi in range(0, y.size, xy_stride):
            for xi in range(0, x.size, xy_stride):
                norm_intensity = float(intensity[zi, yi, xi] / max_intensity)
                if norm_intensity < intensity_cutoff:
                    continue

                x_points.append(float(x[xi]))
                y_points.append(float(y[yi]))
                z_points.append(float(z[zi]))
                c_points.append(norm_intensity)
                s_points.append(min_marker_size + (max_marker_size - min_marker_size) * norm_intensity)

    if not x_points:
        print("no points passed intensity cutoff; skipping 3D propagation scatter plot.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    norm_values = np.asarray(c_points, dtype=np.float64)
    cmap = _resolve_cmap(plt, None)
    rgba = cmap(np.clip(norm_values, 0.0, 1.0))
    rgba[:, 3] = alpha_min + (alpha_max - alpha_min) * np.clip(norm_values, 0.0, 1.0)
    scatter = ax.scatter(
        np.asarray(x_points),
        np.asarray(y_points),
        np.asarray(z_points),
        c=rgba,
        s=np.asarray(s_points),
        marker="o",
        depthshade=False,
        linewidths=0.0,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap),
        ax=ax,
        pad=0.10,
    )
    cbar.set_label("Normalized intensity")
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path
