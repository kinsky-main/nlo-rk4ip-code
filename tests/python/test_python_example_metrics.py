"""
Checks for the ``backend.metrics`` relative-error family used by the examples.

Commit 8ffc4b8 replaced the old clamped pointwise metrics
(``pointwise_abs_relative_error`` and friends, with ``eps`` / ``warn_on_clamp``
/ ``context`` keywords) with intensity L2-norm ratios, and rewrote only the
import names here. The cases below were written against the removed contract,
so they are re-derived against the current one: a true relative L2 norm of the
intensity difference, with no clamping keywords and NaN for a zero reference.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PYTHON = REPO_ROOT / "examples" / "python"
if str(EXAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PYTHON))

from backend.metrics import (  # noqa: E402
    filtered_relative_l2_intensity_error,
    relative_l2_intensity_error,
    relative_l2_intensity_error_curve,
)


def _check_close(actual: float, expected: float, tol: float = 1e-12) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tol:
        raise AssertionError(f"value mismatch: actual={actual}, expected={expected}, tol={tol}")


def test_relative_l2_intensity_error_formula() -> None:
    # |pred|^2 = [4, 25, 1], |ref|^2 = [1, 16, 4]; difference [3, 9, -3].
    pred = np.asarray([2.0, 5.0, 1.0], dtype=np.float64)
    ref = np.asarray([1.0, 4.0, 2.0], dtype=np.float64)
    got = relative_l2_intensity_error(pred, ref)
    _check_close(got, math.sqrt(99.0 / 273.0))


def test_relative_l2_intensity_error_ignores_pure_phase() -> None:
    # The metric compares intensities, so a global phase rotation is invisible.
    # Callers that need a phase-sensitive comparison must not use this family.
    field = np.asarray([1.0, 2.0, 3.0], dtype=np.complex128)
    rotated = field * np.exp(0.7j)
    got = relative_l2_intensity_error(rotated, field)
    _check_close(got, 0.0, tol=1e-14)


def test_relative_l2_intensity_error_curve_reduction() -> None:
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
    curve = relative_l2_intensity_error_curve(pred, ref)
    expected_curve = np.asarray(
        [math.sqrt(99.0 / 273.0), math.sqrt(83.0 / 288.0)],
        dtype=np.float64,
    )
    if not np.allclose(curve, expected_curve, rtol=0.0, atol=1e-12):
        raise AssertionError(f"curve mismatch: {curve} vs {expected_curve}")

    # The whole-array value is a single norm ratio over every point, which is
    # not the mean of the per-record curve; pin the distinction so a future
    # refactor cannot quietly swap one for the other.
    total = relative_l2_intensity_error(pred, ref)
    _check_close(total, math.sqrt(182.0 / 561.0))
    if abs(total - float(np.mean(expected_curve))) <= 1e-12:
        raise AssertionError("aggregate error must not coincide with the curve mean on this fixture")


def test_relative_l2_intensity_error_curve_treats_1d_as_one_record() -> None:
    pred = np.asarray([2.0, 5.0, 1.0], dtype=np.float64)
    ref = np.asarray([1.0, 4.0, 2.0], dtype=np.float64)
    curve = relative_l2_intensity_error_curve(pred, ref)
    if curve.shape != (1,):
        raise AssertionError(f"1-D input should reduce to a single record: {curve.shape}")
    _check_close(float(curve[0]), math.sqrt(99.0 / 273.0))


def test_zero_reference_is_not_finite() -> None:
    # A zero reference has no norm to divide by; the metric reports NaN rather
    # than clamping the denominator, so callers see the degenerate case.
    pred = np.asarray([0.0, 1.0], dtype=np.float64)
    ref = np.asarray([0.0, 0.0], dtype=np.float64)
    got = relative_l2_intensity_error(pred, ref)
    if not math.isnan(got):
        raise AssertionError(f"zero reference should give NaN, got {got}")


def test_filtered_variant_suppresses_low_intensity_tail() -> None:
    # Disagreement confined to points far below the reference peak is dropped
    # by the intensity floor but counted by the unfiltered metric.
    ref = np.asarray([1.0, 1.0e-6, 1.0e-6], dtype=np.float64)
    pred = np.asarray([1.0, 2.0e-6, 0.0], dtype=np.float64)
    unfiltered = relative_l2_intensity_error(pred, ref)
    filtered = filtered_relative_l2_intensity_error(pred, ref, min_relative_intensity=1.0e-3)
    if not (unfiltered > 0.0):
        raise AssertionError("unfiltered metric should see the tail disagreement")
    _check_close(filtered, 0.0, tol=1e-14)


def test_shape_mismatch_is_rejected() -> None:
    pred = np.asarray([1.0, 2.0], dtype=np.float64)
    ref = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    try:
        relative_l2_intensity_error(pred, ref)
    except ValueError:
        return
    raise AssertionError("mismatched shapes should raise ValueError")


def main() -> None:
    test_relative_l2_intensity_error_formula()
    test_relative_l2_intensity_error_ignores_pure_phase()
    test_relative_l2_intensity_error_curve_reduction()
    test_relative_l2_intensity_error_curve_treats_1d_as_one_record()
    test_zero_reference_is_not_finite()
    test_filtered_variant_suppresses_low_intensity_tail()
    test_shape_mismatch_is_rejected()
    print("test_python_example_metrics: all checks passed.")


if __name__ == "__main__":
    main()
