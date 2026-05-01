from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PYTHON = REPO_ROOT / "examples" / "python"
if str(EXAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PYTHON))

from grin.models import PlotArtifact, ValidationReport
from grin.validation import PlotImageValidator, WavelengthWindowSelector
from backend.plotting import _image_with_mpl_colorbar


def _check_condition(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def test_wavelength_window_selector_focuses_signal() -> None:
    axis = np.linspace(600.0, 500000.0, 4096, dtype=np.float64)
    center = 1550.0
    sigma = 25.0
    signal = np.exp(-0.5 * ((axis - center) / sigma) ** 2)
    signal += 1.0e-9 * np.exp(-0.5 * ((axis - 300000.0) / 8000.0) ** 2)

    map_a = np.tile(signal[None, :], (6, 1))
    map_b = np.tile((0.8 * signal)[None, :], (6, 1))
    selector = WavelengthWindowSelector(0.999)
    axis_win, _, _, window = selector.select(axis, map_a, map_b)

    _check_condition(axis_win.size < axis.size, "selector should shrink oversized axis support")
    _check_condition(float(axis_win.min()) > 1300.0, "window lower bound should stay near carrier support")
    _check_condition(float(axis_win.max()) < 1800.0, "window upper bound should stay near carrier support")
    _check_condition(window.mass_fraction >= 0.999, "selected support must satisfy requested cumulative mass")


def test_plot_image_validator_flags_blank_plot() -> None:
    with tempfile.TemporaryDirectory(prefix="nlolib_grin_plot_validation_") as tmp:
        root = Path(tmp)
        good_path = root / "good_plot.png"
        blank_path = root / "blank_plot.png"

        x = np.linspace(-1.0, 1.0, 64)
        z = np.linspace(0.0, 1.0, 32)
        xx, zz = np.meshgrid(x, z, indexing="xy")
        intensity = np.exp(-6.0 * (xx * xx)) * (1.0 + 0.2 * np.sin(8.0 * zz))

        fig, ax = plt.subplots()
        ax.pcolormesh(x, z, intensity, shading="auto")
        fig.savefig(good_path)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.imshow(np.ones((64, 64), dtype=np.float64), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        fig.savefig(blank_path)
        plt.close(fig)

        report = ValidationReport(example_name="grin_plot_validation_test", run_group="unit")
        PlotImageValidator().validate_artifacts(
            report,
            [
                PlotArtifact(key="good_plot", path=good_path, allow_uniform=False),
                PlotArtifact(key="blank_plot", path=blank_path, allow_uniform=False),
            ],
        )
        checks = {check.name: check for check in report.checks}
        _check_condition(checks["plot_exists:good_plot"].passed, "good plot existence check failed")
        _check_condition(checks["plot_exists:blank_plot"].passed, "blank plot existence check failed")
        _check_condition(checks["plot_nonwhite_ratio:good_plot"].passed, "good plot should pass nonwhite ratio")
        _check_condition(
            not checks["plot_nonwhite_ratio:blank_plot"].passed or not checks["plot_gray_std:blank_plot"].passed,
            "blank plot should fail at least one image sanity criterion",
        )


def test_3d_contour_compositor_draws_projected_grid_lines() -> None:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    axis_line_specs = [
        (0.10, 0.15, 0.88, 0.18, 1.0, 1.8, "axis"),
        (0.12, 0.16, 0.25, 0.82, 1.0, 1.8, "axis"),
        (0.25, 0.82, 0.88, 0.78, 0.55, 1.1, "grid"),
    ]

    rendered = _image_with_mpl_colorbar(
        image,
        axis_line_specs=axis_line_specs,
        colorbar_label="Normalized intensity",
    )
    rgb = np.asarray(rendered[:, :, :3], dtype=np.int16)
    dark_pixels = np.any(rgb < 245, axis=2)

    _check_condition(rendered.ndim == 3 and rendered.shape[2] == 3, "composited image must be RGB")
    _check_condition(
        int(np.count_nonzero(dark_pixels)) > 100,
        "projected grid lines should alter the blank render",
    )


def main() -> None:
    test_wavelength_window_selector_focuses_signal()
    test_plot_image_validator_flags_blank_plot()
    test_3d_contour_compositor_draws_projected_grid_lines()
    print("test_python_grin_plot_validation: all checks passed.")


if __name__ == "__main__":
    main()
