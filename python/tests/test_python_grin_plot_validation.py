"""
Plot-compositor checks that outlived the GRIN example validation harness.

This file used to cover ``grin.validation.WavelengthWindowSelector`` and
``grin.validation.PlotImageValidator`` as well. Commit 8ffc4b8 ("Removed old
grin examples") deleted the whole ``examples/python/grin`` package, so those two
cases have been importing a module that no longer exists ever since; they are
retired rather than revived, because reviving them would mean restoring example
code that was removed on purpose. What remains is the part that still exercises
live code: ``backend.plotting._image_with_mpl_colorbar``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PYTHON = REPO_ROOT / "examples"
if str(EXAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PYTHON))

from backend.plotting import _image_with_mpl_colorbar  # noqa: E402


def _check_condition(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def test_image_with_mpl_colorbar_composes_image_and_colorbar() -> None:
    # Render a non-trivial scene image (gradient) and verify the compositor
    # preserves the rendered content and adds a colorbar strip on the right.
    height, width = 120, 180
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float64)
    scene = np.tile(gradient[None, :, None], (height, 1, 3))
    scene = (scene * 255.0).astype(np.uint8)

    rendered = _image_with_mpl_colorbar(scene, colorbar_label="Normalized intensity")

    _check_condition(rendered.ndim == 3 and rendered.shape[2] == 3, "composited image must be RGB")
    _check_condition(rendered.shape[0] > 0 and rendered.shape[1] > 0, "composited image must be non-empty")

    # A colorbar adds a strip of saturated colors on the right side of the
    # figure that does not appear in the input gradient (which is greyscale).
    rgb = np.asarray(rendered, dtype=np.int16)
    right_half = rgb[:, rendered.shape[1] // 2:, :]
    chroma = np.max(right_half, axis=2) - np.min(right_half, axis=2)
    _check_condition(int(np.max(chroma)) > 30, "colorbar should introduce non-grey pixels on the right")


def main() -> None:
    test_image_with_mpl_colorbar_composes_image_and_colorbar()
    print("test_python_grin_plot_validation: all checks passed.")


if __name__ == "__main__":
    main()
