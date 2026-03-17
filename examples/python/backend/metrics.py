"""Shared numerical error metrics for Python examples."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


DEFAULT_RELATIVE_ERROR_EPS = 1.0e-12


def _validate_shapes(prediction: np.ndarray, reference: np.ndarray) -> None:
    if prediction.shape != reference.shape:
        raise ValueError(
            "prediction and reference must have identical shape: "
            f"{prediction.shape} != {reference.shape}"
        )


def relative_l2_intensity_error(
    prediction: np.ndarray | Any,
    reference: np.ndarray | Any,
) -> float:
    """Return ``|| |pred|^2 - |ref|^2 ||_2 / || |ref|^2 ||_2``."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    _validate_shapes(pred, ref)

    pred_intensity = np.abs(pred) ** 2
    ref_intensity = np.abs(ref) ** 2
    diff_norm = float(np.linalg.norm(pred_intensity - ref_intensity))
    ref_norm = float(np.linalg.norm(ref_intensity))
    if not np.isfinite(diff_norm) or not np.isfinite(ref_norm) or ref_norm <= 0.0:
        return float("nan")
    return diff_norm / ref_norm


def relative_l2_intensity_error_curve(
    prediction_records: np.ndarray | Any,
    reference_records: np.ndarray | Any,
) -> np.ndarray:
    """Return per-record relative L2 intensity error along axis 0."""
    pred = np.asarray(prediction_records)
    ref = np.asarray(reference_records)
    _validate_shapes(pred, ref)

    if pred.ndim == 0:
        return np.asarray([relative_l2_intensity_error(pred, ref)], dtype=np.float64)
    if pred.ndim == 1:
        return np.asarray([relative_l2_intensity_error(pred, ref)], dtype=np.float64)

    values = [
        relative_l2_intensity_error(pred[idx], ref[idx])
        for idx in range(int(pred.shape[0]))
    ]
    return np.asarray(values, dtype=np.float64)


def filtered_relative_l2_intensity_error(
    prediction: np.ndarray | Any,
    reference: np.ndarray | Any,
    *,
    min_relative_intensity: float = 1.0e-8,
) -> float:
    """
    Return ``||pred-ref||_2 / ||ref||_2`` on points above a reference-intensity floor.

    The mask is derived only from ``|ref|^2`` so low-intensity tails are suppressed
    without assuming a Gaussian pulse shape.
    """
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    _validate_shapes(pred, ref)
    if min_relative_intensity < 0.0:
        raise ValueError("min_relative_intensity must be >= 0.")

    ref_intensity = np.abs(ref) ** 2
    peak_intensity = float(np.max(ref_intensity)) if ref_intensity.size > 0 else 0.0
    if not np.isfinite(peak_intensity) or peak_intensity <= 0.0:
        return relative_l2_intensity_error(pred, ref)

    mask = ref_intensity >= (peak_intensity * float(min_relative_intensity))
    if not bool(np.any(mask)):
        return relative_l2_intensity_error(pred, ref)

    return relative_l2_intensity_error(pred[mask], ref[mask])


def filtered_relative_l2_intensity_error_curve(
    prediction_records: np.ndarray | Any,
    reference_records: np.ndarray | Any,
    *,
    min_relative_intensity: float = 1.0e-8,
) -> np.ndarray:
    """Return per-record filtered relative L2 intensity error along axis 0."""
    pred = np.asarray(prediction_records)
    ref = np.asarray(reference_records)
    _validate_shapes(pred, ref)

    if pred.ndim == 0 or pred.ndim == 1:
        return np.asarray(
            [filtered_relative_l2_intensity_error(pred, ref, min_relative_intensity=min_relative_intensity)],
            dtype=np.float64,
        )

    values = [
        filtered_relative_l2_intensity_error(
            pred[idx],
            ref[idx],
            min_relative_intensity=min_relative_intensity,
        )
        for idx in range(int(pred.shape[0]))
    ]
    print(values)
    return np.asarray(values, dtype=np.float64)
