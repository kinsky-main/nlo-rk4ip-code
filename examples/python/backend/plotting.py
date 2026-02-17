"""Shared plotting helpers for nlolib Python examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


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
    cmap: str = "magma",
) -> Path | None:
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping intensity colormap plot.")
        return None

    data = np.asarray(intensity_map, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    peak = float(np.max(data))
    if peak > 0.0:
        data = data / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    mesh = ax.pcolormesh(x_axis, z_axis, data, shading="auto", cmap=cmap)
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
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping Re/Im comparison plot.")
        return None

    ref = np.asarray(reference_field, dtype=np.complex128)
    out = np.asarray(final_field, dtype=np.complex128)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(x_axis, np.real(ref), lw=1.8, label=f"{reference_label} Re")
    ax.plot(x_axis, np.imag(ref), lw=1.8, ls="--", label=f"{reference_label} Im")
    ax.plot(x_axis, np.real(out), lw=1.6, label=f"{final_label} Re")
    ax.plot(x_axis, np.imag(out), lw=1.6, ls="--", label=f"{final_label} Im")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Field amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping intensity comparison plot.")
        return None

    ref_intensity = np.abs(np.asarray(reference_field, dtype=np.complex128)) ** 2
    out_intensity = np.abs(np.asarray(final_field, dtype=np.complex128)) ** 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(x_axis, ref_intensity, lw=2.0, label=f"{reference_label} |A|^2")
    ax.plot(x_axis, out_intensity, lw=1.8, ls="--", label=f"{final_label} |A|^2")
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
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping propagation error plot.")
        return None

    errors = np.asarray(error_curve, dtype=np.float64)
    errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
    errors = np.clip(errors, 0.0, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(z_axis, errors, lw=1.8, color="tab:red")
    ax.set_xlabel("Propagation distance z")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
    min_marker_size: float = 2.0,
    max_marker_size: float = 36.0,
    title: str = "3D Propagation Intensity Scatter",
) -> Path | None:
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping 3D propagation scatter plot.")
        return None

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
    if intensity_cutoff < 0.0 or intensity_cutoff >= 1.0:
        raise ValueError("intensity_cutoff must be in [0, 1).")

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

    for zi in range(z.size):
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
    fig = plt.figure(figsize=(10.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        np.asarray(x_points),
        np.asarray(y_points),
        np.asarray(z_points),
        c=np.asarray(c_points),
        s=np.asarray(s_points),
        cmap="plasma",
        alpha=0.70,
        linewidths=0.0,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.10)
    cbar.set_label("Normalized intensity")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
