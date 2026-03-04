from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PYTHON = REPO_ROOT / "examples" / "python"
if str(EXAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PYTHON))

from backend.metrics import (  # noqa: E402
    mean_pointwise_abs_relative_error,
    mean_pointwise_abs_relative_error_curve,
    pointwise_abs_relative_error,
)


def _check_close(actual: float, expected: float, tol: float = 1e-12) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tol:
        raise AssertionError(f"value mismatch: actual={actual}, expected={expected}, tol={tol}")


def test_pointwise_abs_relative_error_formula() -> None:
    pred = np.asarray([2.0, 5.0, 1.0], dtype=np.float64)
    ref = np.asarray([1.0, 4.0, 2.0], dtype=np.float64)
    got = pointwise_abs_relative_error(pred, ref, eps=1e-12, warn_on_clamp=False)
    expected = np.asarray([1.0, 0.25, 0.5], dtype=np.float64)
    if not np.allclose(got, expected):
        raise AssertionError(f"pointwise formula mismatch: {got} vs {expected}")


def test_mean_pointwise_abs_relative_error_curve_reduction() -> None:
    pred = np.asarray(
        [
            [2.0, 5.0, 1.0],
            [3.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )
    ref = np.asarray(
        [
            [1.0, 4.0, 2.0],
            [2.0, 2.0, 4.0],
        ],
        dtype=np.float64,
    )
    curve = mean_pointwise_abs_relative_error_curve(pred, ref, eps=1e-12, warn_on_clamp=False)
    expected_curve = np.asarray(
        [
            (1.0 + 0.25 + 0.5) / 3.0,
            (0.5 + 0.5 + 0.25) / 3.0,
        ],
        dtype=np.float64,
    )
    if not np.allclose(curve, expected_curve):
        raise AssertionError(f"curve mismatch: {curve} vs {expected_curve}")

    total = mean_pointwise_abs_relative_error(pred, ref, eps=1e-12, warn_on_clamp=False)
    _check_close(total, float(np.mean(expected_curve)))


def test_near_zero_reference_clamp_warning() -> None:
    pred = np.asarray([0.0, 1.0], dtype=np.float64)
    ref = np.asarray([0.0, 0.0], dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = pointwise_abs_relative_error(pred, ref, eps=1e-6, warn_on_clamp=True, context="unit_test")
    if len(caught) <= 0:
        raise AssertionError("expected clamp warning to be emitted")
    if not np.all(np.isfinite(got)):
        raise AssertionError("clamped pointwise error should remain finite")


def main() -> None:
    test_pointwise_abs_relative_error_formula()
    test_mean_pointwise_abs_relative_error_curve_reduction()
    test_near_zero_reference_clamp_warning()
    print("test_python_example_metrics: all checks passed.")


if __name__ == "__main__":
    main()
