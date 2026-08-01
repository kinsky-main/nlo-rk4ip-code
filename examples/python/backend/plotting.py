"""Shared plotting helpers for nlolib Python examples."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, List

import numpy as np

_DEFAULT_CMAP_NAME = "nlolib_white_cyan_yellow_hdr"
_DEFAULT_CMAP = None
_STYLE_CMAP_NAME = "nlolib_hdr"
_PRIMARY_OUTPUT_DIR: Path | None = None
_REPORT_OUTPUT_DIR: Path | None = None
_SELECTED_PLOT_KEYS: set[str] | None = None


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_debug_enabled() -> bool:
    raw = os.environ.get("NLOLIB_PLOT_DEBUG_SAVE", "")
    return raw.strip().lower() in {"1", "true", "yes", "on", "debug"}


def _plot_debug(message: str) -> None:
    if _plot_debug_enabled():
        print(f"[plotting-debug] {message}")


def _default_colormap():
    global _DEFAULT_CMAP
    if _DEFAULT_CMAP is not None:
        return _DEFAULT_CMAP

    from matplotlib.colors import LinearSegmentedColormap

    _DEFAULT_CMAP = LinearSegmentedColormap.from_list(
        _DEFAULT_CMAP_NAME,
        [
            (0.00, "#ffffff"),
            (0.03, "#7ee3ed"),
            (0.30, "#4e8ec3"),
            (0.70, "#4d2d99"),
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


def _rc_color(name: str, fallback: str) -> str:
    value = plt.rcParams.get(name, fallback)
    try:
        from matplotlib.colors import to_hex

        return str(to_hex(value, keep_alpha=False))
    except Exception:
        return str(fallback)


def _rc_float(name: str, fallback: float) -> float:
    value = plt.rcParams.get(name, fallback)
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _crop_rendered_image_to_content(image: np.ndarray, *, padding_fraction: float = 0.04) -> np.ndarray:
    cropped, _ = _crop_rendered_image_to_content_with_bounds(image, padding_fraction=padding_fraction)
    return cropped


def _crop_rendered_image_to_content_with_bounds(
    image: np.ndarray,
    *,
    padding_fraction: float = 0.04,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    img = np.asarray(image)
    if img.ndim < 3 or img.shape[0] <= 0 or img.shape[1] <= 0:
        width = int(img.shape[1] if img.ndim >= 2 else 0)
        height = int(img.shape[0] if img.ndim >= 1 else 0)
        return img, (0, 0, width, height)
    rgb = img[:, :, :3].astype(np.int16, copy=False)
    background = rgb[0, 0, :]
    diff = np.max(np.abs(rgb - background), axis=2)
    mask = diff > 4
    if not np.any(mask):
        return img, (0, 0, int(img.shape[1]), int(img.shape[0]))

    rows, cols = np.nonzero(mask)
    y0 = int(np.min(rows))
    y1 = int(np.max(rows)) + 1
    x0 = int(np.min(cols))
    x1 = int(np.max(cols)) + 1
    pad_y = max(2, int(round(float(padding_fraction) * float(y1 - y0))))
    pad_x = max(2, int(round(float(padding_fraction) * float(x1 - x0))))
    y0 = max(0, y0 - pad_y)
    y1 = min(img.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(img.shape[1], x1 + pad_x)
    return img[y0:y1, x0:x1, :], (x0, y0, x1, y1)


def _image_with_mpl_colorbar(
    image: np.ndarray,
    *,
    colorbar_label: str = "Normalised intensity",
    axis_label_specs: list[tuple[str, float, float, float]] | None = None,
) -> np.ndarray:
    """Compose a rendered scene image with a matplotlib colorbar on the right.

    `axis_label_specs` is an optional list of ``(label, x_axes, y_axes, rot_deg)``
    tuples drawn over the image via matplotlib so that LaTeX (``$z / L_D$``)
    renders correctly. Coordinates are in the *displayed* image's axes frame.
    When specs are provided the image is assumed to already be cropped to
    content (so the projected coords match the display); when specs are
    ``None`` the function crops internally for backwards compatibility.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    img = np.asarray(image)
    if axis_label_specs is None:
        img = _crop_rendered_image_to_content(img)
    fig = plt.figure(figsize=(4.0, 2.64), dpi=450, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[22.0, 1.0])
    ax_img = fig.add_subplot(grid[0, 0])
    ax_cbar = fig.add_subplot(grid[0, 1])
    ax_img.imshow(img)
    ax_img.set_aspect("equal", adjustable="box", anchor="C")
    ax_img.set_axis_off()
    if axis_label_specs:
        label_color = _rc_color("axes.labelcolor", _rc_color("text.color", "#000000"))
        label_size = _rc_float("axes.labelsize", _rc_float("font.size", 10.0))
        for label, x_pos, y_pos, rotation in axis_label_specs:
            if not label:
                continue
            ax_img.text(
                float(x_pos),
                float(y_pos),
                str(label),
                transform=ax_img.transAxes,
                ha="center",
                va="center",
                rotation=float(rotation),
                color=label_color,
                fontsize=label_size,
                clip_on=False,
            )
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=_resolve_cmap(plt, None))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label(colorbar_label, labelpad=4.0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    out = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    out = np.asarray(out[:, :, :3]).copy()
    plt.close(fig)
    return out


def _figure_to_rgb_array(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    out = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return np.asarray(out[:, :, :3]).copy()


def _apply_nice_ticks(ax: Any) -> None:
    try:
        from matplotlib.ticker import MaxNLocator
        locator_x = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        locator_y = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        ax.xaxis.set_major_locator(locator_x)
        ax.yaxis.set_major_locator(locator_y)
        ax.set_axisbelow(True)
    except Exception:
        pass


def _normalized_nonnegative_data(values: np.ndarray, *, normalization_peak: float | None) -> tuple[np.ndarray, float]:
    data = np.asarray(values, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    peak = float(np.max(data))
    norm_peak = peak if normalization_peak is None else float(normalization_peak)
    if normalization_peak is not None and norm_peak <= 0.0:
        raise ValueError("normalization_peak must be positive when provided.")
    if norm_peak > 0.0:
        data = np.clip(data / norm_peak, 0.0, 1.0)
    return data, norm_peak


def _evenly_spaced_indices(count: int, *, max_count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_count <= 0:
        raise ValueError("max_count must be positive.")
    if count <= max_count:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0.0, float(count - 1), int(max_count))).astype(np.int64))


def _evenly_spaced_indices_with_required(
    count: int,
    *,
    max_count: int,
    required_indices: np.ndarray,
) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    if count <= max_count:
        return np.arange(count, dtype=np.int64)

    required = np.asarray(required_indices, dtype=np.int64).reshape(-1)
    required = np.unique(required[(required >= 0) & (required < int(count))])
    if required.size == 0:
        return _evenly_spaced_indices(count, max_count=max_count)

    base_count = max(int(max_count) - int(required.size), 1)
    indices = np.unique(
        np.concatenate(
            [
                _evenly_spaced_indices(count, max_count=base_count),
                required,
            ]
        )
    )
    if indices.size > max_count:
        required_set = {int(value) for value in required}
        keep = np.ones(indices.size, dtype=bool)
        while int(np.count_nonzero(keep)) > int(max_count):
            candidates = np.flatnonzero(keep & np.asarray([int(value) not in required_set for value in indices]))
            if candidates.size == 0:
                break
            distances = np.min(np.abs(indices[candidates, None] - required[None, :]), axis=1)
            remove_at = int(candidates[int(np.argmin(distances))])
            keep[remove_at] = False
        indices = indices[keep]
    return np.asarray(indices, dtype=np.int64)


def _axis_feature_indices(profile: np.ndarray, *, threshold_fraction: float = 0.35, max_count: int = 9) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    peak = float(np.max(values))
    if peak <= 0.0:
        return np.asarray([int(np.argmax(values))], dtype=np.int64)
    support = np.flatnonzero(values >= float(threshold_fraction) * peak)
    if support.size == 0:
        support = np.asarray([int(np.argmax(values))], dtype=np.int64)
    if support.size > int(max_count):
        support = support[_evenly_spaced_indices(support.size, max_count=int(max_count))]
    return np.unique(np.concatenate([support, np.asarray([int(np.argmax(values))], dtype=np.int64)]))


def _pulse_supported_time_indices(
    field_records_tyx: np.ndarray,
    *,
    max_count: int,
    support_threshold: float,
    padding_fraction: float = 0.04,
) -> np.ndarray:
    records = np.asarray(field_records_tyx)
    if records.ndim != 4:
        raise ValueError("field_records_tyx must have shape [record, time, y, x].")

    time_count = int(records.shape[1])
    profile = np.zeros(time_count, dtype=np.float64)
    for time_index in range(time_count):
        frame = records[:, time_index, :, :]
        if np.iscomplexobj(frame):
            profile[time_index] = float(np.sum(np.abs(frame) ** 2))
        else:
            profile[time_index] = float(np.sum(np.clip(frame, 0.0, None)))
    print(f"Number of time samples: {time_count}, peak profile value: {float(np.max(profile)):.3g}")
    print(f"Range of time values: min={float(np.min(time_count)):.3g}, mean={float(np.mean(time_count)):.3g}, max={float(np.max(profile)):.3g}")
    peak = float(np.max(profile))
    if peak <= 0.0:
        return _evenly_spaced_indices(time_count, max_count=max_count)

    normalized = profile / peak
    occupied = np.flatnonzero(normalized >= max(float(support_threshold), 1.0e-4))
    if occupied.size == 0:
        occupied = np.flatnonzero(normalized >= 1.0e-4)
    if occupied.size == 0:
        return _evenly_spaced_indices(time_count, max_count=max_count)

    start = int(occupied[0])
    stop = int(occupied[-1])
    padding = max(1, int(np.ceil((stop - start + 1) * max(float(padding_fraction), 0.0))))
    start = max(0, start - padding)
    stop = min(time_count - 1, stop + padding)

    supported_count = stop - start + 1
    if supported_count <= max_count:
        return np.arange(start, stop + 1, dtype=np.int64)
    return np.unique(np.rint(np.linspace(float(start), float(stop), int(max_count))).astype(np.int64))


def configure_plot_saving(
    *,
    primary_output_dir: Path | None = None,
    report_dir: Path | None = None,
    selected_plot_keys: set[str] | None = None,
) -> None:
    global _PRIMARY_OUTPUT_DIR
    global _REPORT_OUTPUT_DIR
    global _SELECTED_PLOT_KEYS

    _PRIMARY_OUTPUT_DIR = (
        Path(primary_output_dir).resolve()
        if primary_output_dir is not None
        else None
    )
    if _PRIMARY_OUTPUT_DIR is not None:
        print(f"Configured primary plot output directory: {_PRIMARY_OUTPUT_DIR}")
    _REPORT_OUTPUT_DIR = (
        Path(report_dir).resolve()
        if report_dir is not None
        else None
    )
    if _REPORT_OUTPUT_DIR is not None:
        print(f"Configured report plot output directory: {_REPORT_OUTPUT_DIR}")
    _SELECTED_PLOT_KEYS = (
        {str(key).strip().lower() for key in selected_plot_keys}
        if selected_plot_keys is not None
        else None
    )


def _plot_key_from_path(output_path: Path) -> str:
    return output_path.stem.strip().lower()


def _plot_is_selected(output_path: Path) -> bool:
    if _SELECTED_PLOT_KEYS is None:
        return True
    return _plot_key_from_path(output_path) in _SELECTED_PLOT_KEYS


def _resolve_report_output_path(output_path: Path) -> Path | None:
    if _REPORT_OUTPUT_DIR is None:
        return None

    relative_path = Path(output_path.name)
    if _PRIMARY_OUTPUT_DIR is not None:
        try:
            relative_path = output_path.resolve().relative_to(_PRIMARY_OUTPUT_DIR)
        except Exception:
            relative_path = Path(output_path.name)
    return _REPORT_OUTPUT_DIR / relative_path


def _save_rendered_image(source_path: Path, output_path: Path) -> Path | None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = _plot_is_selected(output_path)
    report_output_path = _resolve_report_output_path(output_path)
    shutil.copyfile(source_path, output_path)
    if report_output_path is None or not selected:
        return output_path
    report_output_path = Path(report_output_path)
    if report_output_path.resolve() == output_path.resolve():
        return output_path
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, report_output_path)
    return output_path

def _save_figure(fig: Any, output_path: Path, **kwargs: Any) -> Path | None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = _plot_is_selected(output_path)
    report_output_path = _resolve_report_output_path(output_path)
    _plot_debug(
        "save start "
        f"primary='{output_path}' selected={selected} report='{report_output_path}' kwargs={kwargs}"
    )

    try:
        fig.savefig(output_path, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "primary figure save failed: "
            f"path='{output_path}', cwd='{Path.cwd()}', kwargs={kwargs}"
        ) from exc

    _plot_debug(
        "primary save done "
        f"exists={output_path.exists()} size={output_path.stat().st_size if output_path.exists() else -1}"
    )

    if report_output_path is None:
        _plot_debug("report save skipped: report_dir is not configured")
        return output_path
    if not selected:
        _plot_debug("report save skipped: plot key filtered by --save-plots")
        return output_path

    report_output_path = Path(report_output_path)
    if report_output_path.resolve() == output_path.resolve():
        _plot_debug("report save skipped: report path resolves to same file as primary output")
        return output_path

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(report_output_path, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "report figure save failed: "
            f"path='{report_output_path}', cwd='{Path.cwd()}', kwargs={kwargs}"
        ) from exc

    _plot_debug(
        "report save done "
        f"path='{report_output_path}' exists={report_output_path.exists()} "
        f"size={report_output_path.stat().st_size if report_output_path.exists() else -1}"
    )
    return output_path


def plot_intensity_colormap_vs_propagation(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    intensity_map: np.ndarray,
    output_path: Path,
    *,
    x_label: str,
    y_label: str = "Propagation distance z",
    colorbar_label: str = "Normalised intensity",
    normalization_peak: float | None = None,
    xlimit: tuple[float, float] | None = None,
    ylimit: tuple[float, float] | None = None,
    cmap="nlolib_hdr",
) -> Path | None:


    data = np.asarray(intensity_map, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    peak = float(np.max(data))
    norm_peak = peak if normalization_peak is None else float(normalization_peak)
    if norm_peak < 0.0:
        raise ValueError("normalization_peak must be non-negative.")
    if norm_peak > 0.0:
        data = np.clip(data / norm_peak, 0.0, 1.0)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        x_axis,
        z_axis,
        data,
        shading="auto",
        cmap=_resolve_cmap(plt, cmap),
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if xlimit is not None:
        ax.set_xlim(xlimit)
    if ylimit is not None:
        ax.set_ylim(ylimit)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)
    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved


def plot_final_re_im_comparison(
    x_axis: np.ndarray,
    reference_field: np.ndarray,
    final_field: np.ndarray,
    output_path: Path,
    *,
    x_label: str,
    reference_label: str = "Reference",
    final_label: str = "Final",
) -> Path | None:


    ref = np.asarray(reference_field, dtype=np.complex128)
    out = np.asarray(final_field, dtype=np.complex128)

    fig, ax = plt.subplots()
    ax.plot(x_axis, np.real(ref), lw=2.0, color="C0", label=f"{reference_label} Re")
    ax.plot(x_axis, np.imag(ref), lw=2.0, color="C1", label=f"{reference_label} Im")
    ax.plot(x_axis, np.real(out), lw=1.5, color="C0", ls="--", label=f"{final_label} Re")
    ax.plot(x_axis, np.imag(out), lw=1.5, color="C1", ls="--", label=f"{final_label} Im")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Field amplitude")
    _apply_nice_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved


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
) -> Path | None:


    fig, ax = plt.subplots()
    ax.plot(x_axis, curve_a, lw=2.0, label=label_a)
    ax.plot(x_axis, curve_b, lw=1.5, ls="--", label=label_b)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_nice_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved


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
) -> Path | None:


    fig, ax = plt.subplots()
    ax.plot(x_axis, curve_a, lw=1.8, label=label_a)
    ax.plot(x_axis, curve_b, lw=1.8, label=label_b)
    ax.plot(x_axis, curve_c, lw=1.8, label=label_c)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_nice_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved


def plot_mode_power_exchange(
    z_axis: np.ndarray,
    mode1_num: np.ndarray,
    mode2_num: np.ndarray,
    mode1_ref: np.ndarray,
    mode2_ref: np.ndarray,
    output_path: Path,
) -> Path | None:


    fig, ax = plt.subplots()
    ax.plot(z_axis, mode1_ref, lw=2.0, color="C0", label="|A1|^2 analytical")
    ax.plot(z_axis, mode2_ref, lw=2.0, color="C1", label="|A2|^2 analytical")
    ax.plot(z_axis, mode1_num, "--", lw=1.5, color="C0", label="|A1|^2 numerical")
    ax.plot(z_axis, mode2_num, "--", lw=1.5, color="C1", label="|A2|^2 numerical")
    ax.set_xlabel("Propagation distance z")
    ax.set_ylabel("Mode power")
    _apply_nice_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved


def plot_phase_shift_comparison(
    t_axis: np.ndarray,
    phase_num: np.ndarray,
    phase_ref: np.ndarray,
    phase_mask: np.ndarray,
    output_path: Path,
) -> Path | None:


    phase_num_plot = np.where(phase_mask, phase_num, np.nan)
    phase_ref_plot = np.where(phase_mask, phase_ref, np.nan)

    fig, ax = plt.subplots()
    ax.plot(t_axis, phase_ref_plot, lw=2.0, label="Analytical phase shift")
    ax.plot(t_axis, phase_num_plot, "--", lw=1.5, label="Numerical phase shift")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Phase shift (rad)")
    _apply_nice_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved


def plot_convergence_loglog(
    step_sizes: np.ndarray,
    errors: np.ndarray,
    fit_mask: np.ndarray,
    fitted_order: float,
    fitted_intercept: float,
    output_path: Path,
    *,
    x_label: str = "Step size Delta z (m)",
    y_label: str = "Mean pointwise abs-relative error",
    reference_order: float = 4.0,
    legend_label_parts: List[str] = [r"Fitted power law $\propto h^{", r"}$"],
) -> Path | None:


    order = np.argsort(step_sizes)
    step_sizes_plot = step_sizes[order]
    errors_plot = errors[order]
    fit_mask_plot = fit_mask[order]

    fit_indices = np.flatnonzero(fit_mask_plot)
    anchor = int(fit_indices[0]) if fit_indices.size > 0 else 0
    ref = errors_plot[anchor] * (step_sizes_plot / step_sizes_plot[anchor]) ** reference_order
    fit_line = np.exp(fitted_intercept) * (step_sizes_plot**fitted_order)

    fig, ax = plt.subplots()
    ax.loglog(
        step_sizes_plot[fit_mask_plot],
        fit_line[fit_mask_plot],
        "--",
        lw=1.6,
        color="C3",
        label=legend_label_parts[0] + f"{fitted_order:.0f}" + legend_label_parts[1],
    )
    ax.loglog(step_sizes_plot, errors_plot, "o", color="C1", lw=1.8, ms=3.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # _apply_nice_ticks(ax)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    saved = _save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)
    return saved


def plot_summary_curve(
    x_values: np.ndarray | list[float],
    y_values: np.ndarray | list[float],
    output_path: Path,
    *,
    x_label: str,
    y_label: str,
) -> Path | None:


    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, marker="o", lw=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_nice_ticks(ax)
    ax.grid(True, alpha=0.3)
    saved = _save_figure(fig, output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return saved


def plot_wavelength_step_history(
    z_samples: np.ndarray,
    lambda_nm: np.ndarray,
    spectral_map: np.ndarray,
    output_path: Path,
    *,
    accepted_z: np.ndarray | None = None,
    accepted_step_sizes: np.ndarray | None = None,
    map_x_label: str = r"Propagation distance $z$ (m)",
    map_y_label: str = "Wavelength (nm)",
    step_x_label: str = r"Propagation distance $z$ (m)",
    step_y_label: str = "Step size (m)",
    normalization_peak: float | None = None,
) -> Path | None:

    z_axis = np.asarray(z_samples, dtype=np.float64)
    lambda_axis = np.asarray(lambda_nm, dtype=np.float64)
    data = np.asarray(spectral_map, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    if data.shape != (z_axis.size, lambda_axis.size):
        raise ValueError("spectral_map shape must be [record, wavelength].")

    peak = float(np.max(data)) if normalization_peak is None else float(normalization_peak)
    if peak < 0.0:
        raise ValueError("normalization_peak must be non-negative.")
    if peak > 0.0:
        data = np.clip(data / peak, 0.0, 1.0)

    fig = plt.figure(figsize=(4.0, 6.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.4], hspace=0.24)

    ax_map = fig.add_subplot(grid[0, 0])
    mesh = ax_map.pcolormesh(
        z_axis,
        lambda_axis,
        data.T,
        shading="auto",
        cmap=_resolve_cmap(plt, None),
        vmin=0.0,
        vmax=1.0,
    )
    ax_map.set_xlabel(map_x_label)
    ax_map.set_ylabel(map_y_label)
    ax_map.set_box_aspect(1.0)
    ax_map.tick_params(labelbottom=False, bottom=False)
    cbar = fig.colorbar(mesh, ax=ax_map, pad=0.05, shrink=0.79)
    cbar.set_label("Normalised intensity")

    ax_step = fig.add_subplot(grid[1, 0])
    ax_step.tick_params(labeltop=False, top=False)
    has_series = False
    if accepted_z is not None and accepted_step_sizes is not None:
        z_plot = np.asarray(accepted_z, dtype=np.float64).reshape(-1)
        step_plot = np.asarray(accepted_step_sizes, dtype=np.float64).reshape(-1)
        n = min(z_plot.size, step_plot.size)
        if n > 0:
            order = np.argsort(z_plot[:n])
            z_sorted = z_plot[:n][order]
            step_sorted = step_plot[:n][order]
            ax_step.plot(
                z_sorted,
                step_sorted * 1000,
                lw=1.2,
                color="C1",
            )
            has_series = True

    if has_series:
        ax_step.set_xlabel(step_x_label)
        ax_step.set_ylabel(step_y_label)
        ax_step.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        _apply_nice_ticks(ax_step)
        ax_step.grid(True, alpha=0.3)
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

    map_pos = ax_map.get_position()
    step_pos = ax_step.get_position()
    ax_step.set_position([map_pos.x0, step_pos.y0 + 0.07, map_pos.width, step_pos.height])

    saved = _save_figure(fig, output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return saved


def plot_final_intensity_comparison(
    x_axis: np.ndarray,
    reference_field: np.ndarray,
    final_field: np.ndarray,
    output_path: Path,
    *,
    x_label: str,
    reference_label: str = "Reference",
    final_label: str = "Final",
) -> Path | None:


    ref_intensity = np.abs(np.asarray(reference_field, dtype=np.complex128)) ** 2
    out_intensity = np.abs(np.asarray(final_field, dtype=np.complex128)) ** 2

    fig, ax = plt.subplots()
    ax.plot(x_axis, ref_intensity, lw=2.0, color="C0", label=f"{reference_label} $|A|^2$")
    ax.plot(x_axis, out_intensity, lw=1.5, ls="--", color="C1", label=f"{final_label} $|A|^2$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Intensity $|A|^2$")
    _apply_nice_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    saved = _save_figure(fig, output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return saved


def plot_total_error_over_propagation(
    z_axis: np.ndarray,
    error_curve: np.ndarray,
    output_path: Path,
    *,
    y_label: str = "Mean pointwise abs-relative error",
    x_label: str = "Propagation distance z",
) -> Path | None:


    z_values = np.asarray(z_axis, dtype=np.float64).reshape(-1)
    errors = np.asarray(error_curve, dtype=np.float64)
    if errors.ndim == 0:
        errors = np.full(z_values.shape, float(errors), dtype=np.float64)
    else:
        errors = errors.reshape(-1)
        if errors.size == 1 and z_values.size > 1:
            errors = np.full(z_values.shape, float(errors[0]), dtype=np.float64)
        elif errors.size != z_values.size:
            raise ValueError(
                "error_curve must be a scalar or have the same length as z_axis."
            )
    errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
    errors = np.clip(errors, 0.0, None)

    fig, ax = plt.subplots()
    ax.plot(z_values, errors, lw=1.8, color="C3")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_nice_ticks(ax)
    ax.grid(True, alpha=0.3)
    saved = _save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)
    return saved


def plot_frequency_time_propagation_grid(
    frequency_axis: np.ndarray,
    time_axis: np.ndarray,
    z_axis: np.ndarray,
    upper_frequency_map: np.ndarray,
    upper_time_map: np.ndarray,
    lower_frequency_map: np.ndarray,
    lower_time_map: np.ndarray,
    output_path: Path,
    *,
    upper_row_label: str,
    lower_row_label: str,
    left_title: str = "Frequency-domain intensity",
    right_title: str = "Time-domain intensity",
    colorbar_label: str = "Normalised intensity",
    upper_left_annotation: str | None = None,
    lower_left_annotation: str | None = None,
    cmap="nlolib_hdr",
) -> Path | None:
    frequency_values = np.asarray(frequency_axis, dtype=np.float64).reshape(-1)
    time_values = np.asarray(time_axis, dtype=np.float64).reshape(-1)
    z_values = np.asarray(z_axis, dtype=np.float64).reshape(-1)
    maps = [
        np.asarray(upper_frequency_map, dtype=np.float64),
        np.asarray(upper_time_map, dtype=np.float64),
        np.asarray(lower_frequency_map, dtype=np.float64),
        np.asarray(lower_time_map, dtype=np.float64),
    ]
    expected_frequency_shape = (z_values.size, frequency_values.size)
    expected_time_shape = (z_values.size, time_values.size)
    if maps[0].shape != expected_frequency_shape or maps[2].shape != expected_frequency_shape:
        raise ValueError("frequency intensity maps must have shape [record, frequency].")
    if maps[1].shape != expected_time_shape or maps[3].shape != expected_time_shape:
        raise ValueError("time intensity maps must have shape [record, time].")

    normalized_maps = []
    for data in maps:
        normalized, _ = _normalized_nonnegative_data(data, normalization_peak=None)
        normalized_maps.append(normalized)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(4.0, 2.64),
        sharey=True,
        constrained_layout=True,
    )
    plot_specs = (
        (
            axes[0, 0],
            frequency_values,
            normalized_maps[0],
            f"",
            r"",
            r"$z / L_D$",
        ),
        (
            axes[0, 1],
            time_values,
            normalized_maps[1],
            f"",
            r"",
            "",
        ),
        (
            axes[1, 0],
            frequency_values,
            normalized_maps[2],
            f"",
            r"Frequency detuning $\omega - \omega_0$",
            r"$z / L_D$",
        ),
        (
            axes[1, 1],
            time_values,
            normalized_maps[3],
            f"",
            r"Time $t$ (ps)",
            "",
        ),
    )

    last_mesh = None
    resolved_cmap = _resolve_cmap(plt, cmap)
    for num, (ax, x_values, data, title, x_label, y_label) in enumerate(plot_specs):
        last_mesh = ax.pcolormesh(
            x_values,
            z_values,
            data,
            shading="auto",
            cmap=resolved_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)

        
        if num == 0 :
            ax.set_xlim(-20, 20)
        else:
            ax.set_xlim(-2, 2)
        if y_label:
            ax.set_ylabel(y_label)
        if num == 2:
            ax.set_xlim(-4, 0)

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes, shrink=0.96, pad=0.02)
        cbar.set_label(colorbar_label)

    if upper_left_annotation:
        axes[0, 0].text(
            0.04,
            0.94,
            upper_left_annotation,
            transform=axes[0, 0].transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": (1.0, 1.0, 1.0, 0.72),
                "edgecolor": "none",
            },
        )
    if lower_left_annotation:
        axes[1, 0].text(
            0.04,
            0.94,
            lower_left_annotation,
            transform=axes[1, 0].transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": (1.0, 1.0, 1.0, 0.72),
                "edgecolor": "none",
            },
        )

    saved = _save_figure(fig, output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return saved


def plot_3d_intensity_contours_propagation(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    field_records: np.ndarray,
    output_path: Path,
    *,
    intensity_cutoff: float = 0.01,
    num_levels: int = 100,
    max_x_samples: int = 128,
    max_y_samples: int = 128,
    max_z_samples: int = 128,
    alpha_min: float = 0.05,
    alpha_max: float = 0.3,
    input_is_intensity: bool = False,
    normalization_peak: float | None = None,
    x_label: str = r"$x$",
    y_label: str = r"$y$",
    z_label: str = r"$z$",
    annotation_text: str | None = None,
    xy_crop_inset: float = 0.0,
    xy_crop_padding: float = 0.16,
    z_axis_ratio: float = 2,
) -> Path | None:

    image = _render_3d_intensity_contours_frame(
        x_axis,
        y_axis,
        z_axis,
        field_records,
        intensity_cutoff=intensity_cutoff,
        num_levels=num_levels,
        max_x_samples=max_x_samples,
        max_y_samples=max_y_samples,
        max_z_samples=max_z_samples,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        input_is_intensity=input_is_intensity,
        normalization_peak=normalization_peak,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        annotation_text=annotation_text,
        xy_crop_inset=xy_crop_inset,
        xy_crop_padding=xy_crop_padding,
        z_axis_ratio=z_axis_ratio,
    )
    if image is None:
        return None
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 220.0, height / 220.0), dpi=220)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(image)
    ax.set_axis_off()
    saved = _save_figure(fig, output_path, dpi=220)
    plt.close(fig)
    return saved


def _render_3d_intensity_contours_frame(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    field_records: np.ndarray,
    *,
    intensity_cutoff: float,
    num_levels: int,
    max_x_samples: int,
    max_y_samples: int,
    max_z_samples: int,
    alpha_min: float,
    alpha_max: float,
    input_is_intensity: bool,
    normalization_peak: float | None,
    x_label: str,
    y_label: str,
    z_label: str,
    annotation_text: str | None,
    xy_crop_inset: float = 0.0,
    xy_crop_padding: float = 0.16,
    z_axis_ratio: float = 2.0,
) -> np.ndarray | None:
    if input_is_intensity:
        records = np.asarray(field_records, dtype=np.float64)
    else:
        records = np.asarray(field_records, dtype=np.complex128)
    if records.ndim != 3:
        raise ValueError("field_records must be [record, y, x].")

    x = np.asarray(x_axis, dtype=np.float64)
    y = np.asarray(y_axis, dtype=np.float64)
    z = np.asarray(z_axis, dtype=np.float64)
    if records.shape[0] != z.size or records.shape[1] != y.size or records.shape[2] != x.size:
        raise ValueError("Axes lengths must match field_records shape.")

    if intensity_cutoff < 0.0 or intensity_cutoff >= 1.0:
        raise ValueError("intensity_cutoff must be in [0, 1).")
    if num_levels <= 0:
        raise ValueError("num_levels must be positive.")
    if max_x_samples <= 0 or max_y_samples <= 0 or max_z_samples <= 0:
        raise ValueError("max_x_samples/max_y_samples/max_z_samples must be positive.")
    if alpha_min < 0.0 or alpha_min > 1.0 or alpha_max < 0.0 or alpha_max > 1.0 or alpha_min > alpha_max:
        raise ValueError("alpha_min/alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1.")
    if xy_crop_inset < 0.0 or xy_crop_inset >= 1.0:
        raise ValueError("xy_crop_inset must be in [0, 1).")
    if xy_crop_padding < 0.0 or xy_crop_padding >= 1.0:
        raise ValueError("xy_crop_padding must be in [0, 1).")
    if not np.isfinite(z_axis_ratio) or z_axis_ratio <= 0.0:
        raise ValueError("z_axis_ratio must be a positive finite value.")
    if input_is_intensity:
        intensity = np.asarray(records, dtype=np.float64)
    else:
        intensity = np.abs(records) ** 2
    intensity, _ = _normalized_nonnegative_data(intensity, normalization_peak=normalization_peak)
    if float(np.max(intensity)) <= 0.0:
        print("intensity is zero everywhere; skipping 3D propagation contour-surface plot.")
        return None

    x_indices = _evenly_spaced_indices_with_required(
        x.size,
        max_count=int(max_x_samples),
        required_indices=_axis_feature_indices(np.max(intensity, axis=(0, 1))),
    )
    y_indices = _evenly_spaced_indices_with_required(
        y.size,
        max_count=int(max_y_samples),
        required_indices=_axis_feature_indices(np.max(intensity, axis=(0, 2))),
    )
    z_indices = _evenly_spaced_indices_with_required(
        z.size,
        max_count=int(max_z_samples),
        required_indices=_axis_feature_indices(np.max(intensity, axis=(1, 2))),
    )
    intensity_small = intensity[np.ix_(z_indices, y_indices, x_indices)]
    x_small = x[x_indices]
    y_small = y[y_indices]
    z_small = z[z_indices]

    max_intensity = float(np.max(intensity_small))
    _plot_debug(
        "3d contour sampled intensity "
        f"full_max={float(np.max(intensity)):.4g} sampled_max={max_intensity:.4g} "
        f"samples=({x_small.size}, {y_small.size}, {z_small.size})"
    )
    level_upper = min(0.92, max_intensity)
    if level_upper <= float(intensity_cutoff):
        print("no contours passed intensity cutoff; skipping 3D propagation contour-surface plot.")
        return None
    levels = np.linspace(float(intensity_cutoff), level_upper, int(num_levels), dtype=np.float64)
    x_small, y_small, intensity_small = _crop_xy_within_low_contour(
        x_small,
        y_small,
        intensity_small,
        intensity_cutoff=float(intensity_cutoff),
        level_upper=float(level_upper),
        xy_crop_inset=float(xy_crop_inset),
        xy_crop_padding=float(xy_crop_padding),
    )

    # Reorder axes so the isometric view projects:
    #   data x -> screen lower-left (world X)
    #   data z -> screen lower-right, propagation goes right (world Y)
    #   data y -> screen up (world Z)
    #
    # Normalize world coordinates so the bounding box has a controlled aspect
    # ratio: the transverse axes (data x, data y) share the same world span,
    # and the propagation axis (data z) is `z_axis_ratio` times that span. The
    # default makes the propagation direction visibly longer than the
    # transverse axes so the direction of propagation reads at a glance.
    x_min, x_max = float(np.min(x_small)), float(np.max(x_small))
    y_min, y_max = float(np.min(y_small)), float(np.max(y_small))
    z_min, z_max = float(np.min(z_small)), float(np.max(z_small))
    xy_span = max(max(x_max - x_min, y_max - y_min), 1.0e-9)
    z_span = float(z_axis_ratio) * xy_span
    half_xy = 0.5 * xy_span
    half_z = 0.5 * z_span
    x_world = np.linspace(-half_xy, half_xy, x_small.size)
    z_world = np.linspace(-half_z, half_z, z_small.size)
    y_world = np.linspace(-half_xy, half_xy, y_small.size)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FixedLocator, FixedFormatter

    facecolor = _rc_color("figure.facecolor", "#ffffff")
    foreground = _rc_color("text.color", _rc_color("axes.labelcolor", "#000000"))
    edgecolor = _rc_color("axes.edgecolor", "#000000")
    gridcolor = _rc_color("grid.color", "#cccccc")
    fig = plt.figure(figsize=(5.2, 3.6), dpi=360, constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.view_init(elev=24.0, azim=45.0)
    ax.set_box_aspect((xy_span, z_span, xy_span))

    cmap = _resolve_cmap(plt, None)
    render_indices = _evenly_spaced_indices(levels.size, max_count=min(levels.size, 18))
    render_levels = levels[render_indices]
    level_span = max(float(level_upper) - float(intensity_cutoff), 1.0e-12)
    any_surface = False
    for level in render_levels:
        level_fraction = float(np.clip((float(level) - float(intensity_cutoff)) / level_span, 0.0, 1.0))
        opacity = float(np.clip(alpha_min + (alpha_max - alpha_min) * level_fraction, 0.0, 1.0))
        # opacity *= level_fraction ** 0.2
        any_surface = _plot_intensity_level_envelope_shells(
            ax,
            intensity_small,
            x_world=x_world,
            y_world=y_world,
            z_world=z_world,
            level=float(level),
            color=tuple(float(value) for value in cmap(float(level))[:3]),
            opacity=opacity,
            zorder=10.0 + 100.0 * level_fraction,
        ) or any_surface
    if not any_surface:
        print("no contours passed intensity cutoff; skipping 3D propagation contour-surface plot.")
        plt.close(fig)
        return None

    ax.set_xlim(float(x_world[0]), float(x_world[-1]))
    ax.set_ylim(float(z_world[0]), float(z_world[-1]))
    ax.set_zlim(float(y_world[0]), float(y_world[-1]))

    def apply_physical_ticks(axis: Any, world_axis: np.ndarray, physical_axis: np.ndarray, *, count: int) -> None:
        tick_count = max(int(count), 2)
        world_values = np.asarray(world_axis, dtype=np.float64)
        physical_values = np.asarray(physical_axis, dtype=np.float64)
        values = np.linspace(float(physical_values[0]), float(physical_values[-1]), tick_count)
        if abs(float(physical_values[-1] - physical_values[0])) <= 1.0e-12:
            ticks = np.full(tick_count, float(world_values[0]), dtype=np.float64)
        else:
            fraction = (values - float(physical_values[0])) / float(physical_values[-1] - physical_values[0])
            ticks = float(world_values[0]) + fraction * float(world_values[-1] - world_values[0])
        labels = [f"{float(value):.2g}" for value in values]
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FixedFormatter(labels))

    apply_physical_ticks(ax.xaxis, x_world, x_small, count=3)
    apply_physical_ticks(ax.yaxis, z_world, z_small, count=4)
    apply_physical_ticks(ax.zaxis, y_world, y_small, count=3)
    ax.set_xlabel(x_label, labelpad=4.0)
    ax.set_ylabel(z_label, labelpad=4.0)
    ax.set_zlabel(y_label, labelpad=4.0)
    ax.tick_params(colors=foreground, labelsize=0.72 * _rc_float("font.size", 10.0), pad=1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(foreground)
        try:
            axis.line.set_color(edgecolor)
            axis._axinfo["grid"]["color"] = gridcolor
            axis._axinfo["tick"]["color"] = edgecolor
            axis._axinfo["axisline"]["color"] = edgecolor
        except Exception:
            pass
    try:
        ax.xaxis.pane.set_facecolor(facecolor)
        ax.yaxis.pane.set_facecolor(facecolor)
        ax.zaxis.pane.set_facecolor(facecolor)
        ax.xaxis.pane.set_edgecolor(edgecolor)
        ax.yaxis.pane.set_edgecolor(edgecolor)
        ax.zaxis.pane.set_edgecolor(edgecolor)
    except Exception:
        pass
    ax.grid(True)

    if annotation_text:
        ax.text2D(
            0.02,
            0.96,
            str(annotation_text),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=foreground,
            fontsize=1.15 * _rc_float("font.size", 10.0),
        )

    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.74, pad=0.04)
    cbar.set_label("Normalised intensity", labelpad=6.0, fontsize=11.0)

    image = _figure_to_rgb_array(fig)
    plt.close(fig)
    return image


def _crop_xy_within_low_contour(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    intensity_zyx: np.ndarray,
    *,
    intensity_cutoff: float,
    level_upper: float,
    xy_crop_inset: float,
    xy_crop_padding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_axis, dtype=np.float64).reshape(-1)
    y = np.asarray(y_axis, dtype=np.float64).reshape(-1)
    intensity = np.asarray(intensity_zyx, dtype=np.float64)
    if intensity.ndim != 3 or x.size < 6 or y.size < 6:
        return x, y, intensity
    if intensity.shape[1] != y.size or intensity.shape[2] != x.size:
        return x, y, intensity

    crop_threshold = float(intensity_cutoff) + float(xy_crop_inset) * max(float(level_upper) - float(intensity_cutoff), 0.0)
    crop_threshold = min(float(level_upper), max(float(intensity_cutoff), crop_threshold))
    support_xy = np.max(intensity, axis=0) >= crop_threshold
    if not np.any(support_xy):
        return x, y, intensity

    y_idx, x_idx = np.nonzero(support_xy)
    x0 = int(np.min(x_idx))
    x1 = int(np.max(x_idx))
    y0 = int(np.min(y_idx))
    y1 = int(np.max(y_idx))

    x_pad_value = max(float(xy_crop_padding) * max(float(x[-1] - x[0]), 0.0), _median_axis_step(x))
    y_pad_value = max(float(xy_crop_padding) * max(float(y[-1] - y[0]), 0.0), _median_axis_step(y))
    x0, x1 = _expand_crop_window_by_value(x, x0, x1, pad_value=x_pad_value)
    y0, y1 = _expand_crop_window_by_value(y, y0, y1, pad_value=y_pad_value)

    if x0 == 0 and x1 == (x.size - 1) and y0 == 0 and y1 == (y.size - 1):
        return x, y, intensity

    return x[x0:x1 + 1], y[y0:y1 + 1], intensity[:, y0:y1 + 1, x0:x1 + 1]


def _median_axis_step(axis: np.ndarray) -> float:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return 0.0
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _expand_crop_window_by_value(axis: np.ndarray, start: int, stop: int, *, pad_value: float) -> tuple[int, int]:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return int(start), int(stop)
    target_min = max(float(values[0]), float(values[int(start)]) - float(pad_value))
    target_max = min(float(values[-1]), float(values[int(stop)]) + float(pad_value))
    start_idx = int(np.searchsorted(values, target_min, side="left"))
    stop_idx = int(np.searchsorted(values, target_max, side="right") - 1)
    start_idx = max(0, min(start_idx, values.size - 1))
    stop_idx = max(start_idx, min(stop_idx, values.size - 1))
    return start_idx, stop_idx


def _threshold_envelope_coordinates(
    axis: np.ndarray,
    values: np.ndarray,
    *,
    level: float,
    axis_index: int,
    upper: bool,
) -> np.ndarray:
    axis_values = np.asarray(axis, dtype=np.float64).reshape(-1)
    data = np.moveaxis(np.asarray(values, dtype=np.float64), int(axis_index), 0)
    if data.shape[0] != axis_values.size:
        raise ValueError("axis length must match the selected values dimension.")

    passed = data >= float(level)
    has_passed = np.any(passed, axis=0)
    if not np.any(has_passed):
        return np.full(data.shape[1:], np.nan, dtype=np.float64)

    if upper:
        inside_idx = data.shape[0] - 1 - np.argmax(passed[::-1, ...], axis=0)
        outside_idx = np.minimum(inside_idx + 1, data.shape[0] - 1)
    else:
        inside_idx = np.argmax(passed, axis=0)
        outside_idx = np.maximum(inside_idx - 1, 0)

    inside = np.take_along_axis(data, np.expand_dims(inside_idx, axis=0), axis=0)[0]
    outside = np.take_along_axis(data, np.expand_dims(outside_idx, axis=0), axis=0)[0]
    inside_coord = axis_values[inside_idx]
    outside_coord = axis_values[outside_idx]

    denom = inside - outside
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (float(level) - outside) / denom
    fraction = np.clip(fraction, 0.0, 1.0)
    coords = outside_coord + fraction * (inside_coord - outside_coord)
    coords = np.where((inside_idx == outside_idx) | ~np.isfinite(coords), inside_coord, coords)
    return np.where(has_passed, coords, np.nan)


def _plot_intensity_level_envelope_shells(
    ax: Any,
    intensity_zyx: np.ndarray,
    *,
    x_world: np.ndarray,
    y_world: np.ndarray,
    z_world: np.ndarray,
    level: float,
    color: tuple[float, float, float, float],
    opacity: float,
    zorder: float,
) -> bool:
    surface_kwargs = {
        "color": color,
        "alpha": float(opacity),
        "linewidth": 0.0,
        "antialiased": True,
        "shade": False,
        "rstride": 1,
        "cstride": 1,
        "zsort": "min",
        "zorder": float(zorder),
    }
    any_surface = False

    x_grid_zx, z_grid_zx = np.meshgrid(x_world, z_world)
    for upper in (False, True):
        y_coords = _threshold_envelope_coordinates(
            y_world,
            intensity_zyx,
            level=float(level),
            axis_index=1,
            upper=upper,
        )
        if np.any(np.isfinite(y_coords)):
            ax.plot_surface(x_grid_zx, z_grid_zx, y_coords, **surface_kwargs)
            any_surface = True

    z_grid_zy, y_grid_zy = np.meshgrid(z_world, y_world, indexing="ij")
    for upper in (False, True):
        x_coords = _threshold_envelope_coordinates(
            x_world,
            intensity_zyx,
            level=float(level),
            axis_index=2,
            upper=upper,
        )
        if np.any(np.isfinite(x_coords)):
            ax.plot_surface(x_coords, z_grid_zy, y_grid_zy, **surface_kwargs)
            any_surface = True

    x_grid_yx, y_grid_yx = np.meshgrid(x_world, y_world)
    for upper in (False, True):
        z_coords = _threshold_envelope_coordinates(
            z_world,
            intensity_zyx,
            level=float(level),
            axis_index=0,
            upper=upper,
        )
        if np.any(np.isfinite(z_coords)):
            ax.plot_surface(x_grid_yx, z_coords, y_grid_yx, **surface_kwargs)
            any_surface = True

    return any_surface


def save_3d_intensity_time_sweep_video(
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    field_records_tyx: np.ndarray,
    output_path: Path,
    *,
    max_time_frames: int = 128,
    fps: int = 24,
    intensity_cutoff: float = 0.05,
    num_levels: int = 20,
    max_x_samples: int = 64,
    max_y_samples: int = 64,
    max_z_samples: int = 100,
    alpha_min: float = 0.05,
    alpha_max: float = 0.90,
) -> Path | None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "save_3d_intensity_time_sweep_video requires imageio. "
            "Install the Python example dependencies first."
        ) from exc

    t = np.asarray(t_axis, dtype=np.float64).reshape(-1)
    records = np.asarray(field_records_tyx)
    if records.ndim != 4:
        raise ValueError("field_records_tyx must have shape [record, time, y, x].")
    if records.shape[1] != t.size:
        raise ValueError("t axis length must match field_records_tyx time dimension.")

    if np.iscomplexobj(records):
        peak = float(np.max(np.abs(records) ** 2))
    else:
        peak = float(np.max(records))
    if peak <= 0.0:
        print("time-sweep video skipped because intensity is zero everywhere.")
        return None

    frame_indices = _pulse_supported_time_indices(
        records,
        max_count=int(max_time_frames),
        support_threshold=max(0.01, float(intensity_cutoff)),
    )
    if frame_indices.size == 0:
        print("time-sweep video skipped because there are no valid time samples.")
        return None
    print(
        "Rendering "
        f"{len(frame_indices)} frames for time-sweep video over pulse support "
        f"[{float(t[int(frame_indices[0])]):+.3f}, {float(t[int(frame_indices[-1])]):+.3f}]..."
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    for time_index in frame_indices:
        frame = _render_3d_intensity_contours_frame(
            x_axis,
            y_axis,
            z_axis,
            records[:, int(time_index), :, :],
            intensity_cutoff=float(intensity_cutoff),
            num_levels=int(num_levels),
            max_x_samples=int(max_x_samples),
            max_y_samples=int(max_y_samples),
            max_z_samples=int(max_z_samples),
            alpha_min=float(alpha_min),
            alpha_max=float(alpha_max),
            input_is_intensity=not np.iscomplexobj(records),
            normalization_peak=None,
            x_label=r"$x$",
            y_label=r"$y$",
            z_label="z",
            annotation_text=f"t = {float(t[int(time_index)]):+.3f}",
        )
        if frame is not None:
            frames.append(frame)
    if not frames:
        output_path.unlink(missing_ok=True)
        return None
    try:
        writer = imageio.get_writer(
            str(output_path),
            format="FFMPEG",
            fps=int(fps),
            codec="libx264",
            quality=7,
            macro_block_size=None,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        gif_path = output_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), frames, duration=(1.0 / max(int(fps), 1)))
        return gif_path

    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return output_path


def plot_3d_intensity_volume_propagation(*args: Any, **kwargs: Any) -> Path | None:
    """Backward-compatible alias for the contour-surface tensor propagation view."""
    return plot_3d_intensity_contours_propagation(*args, **kwargs)


def plot_3d_intensity_scatter_propagation(*args: Any, **kwargs: Any) -> Path | None:
    """Backward-compatible alias for the contour-surface tensor propagation view."""
    return plot_3d_intensity_contours_propagation(*args, **kwargs)
