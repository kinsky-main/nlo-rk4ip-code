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


def pointwise_abs_relative_error(
    prediction: np.ndarray | Any,
    reference: np.ndarray | Any,
    *,
    eps: float = DEFAULT_RELATIVE_ERROR_EPS,
    warn_on_clamp: bool = True,
    context: str | None = None,
) -> np.ndarray:
    """Return pointwise ``abs(pred-ref) / max(abs(ref), eps)``."""
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    _validate_shapes(pred, ref)
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")

    abs_ref = np.abs(ref)
    clamped_mask = abs_ref < float(eps)
    if warn_on_clamp and bool(np.any(clamped_mask)):
        clamped_count = int(np.count_nonzero(clamped_mask))
        total_count = int(clamped_mask.size)
        prefix = f"{context}: " if context else ""
        warnings.warn(
            (
                f"{prefix}relative-error denominator clamped at {eps:.3e} for "
                f"{clamped_count}/{total_count} points; near-zero references may "
                "amplify floating-point noise."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    denominator = np.maximum(abs_ref, float(eps))
    return np.abs(pred - ref) / denominator


def mean_pointwise_abs_relative_error(
    prediction: np.ndarray | Any,
    reference: np.ndarray | Any,
    *,
    eps: float = DEFAULT_RELATIVE_ERROR_EPS,
    warn_on_clamp: bool = True,
    context: str | None = None,
) -> float:
    """Return mean of pointwise absolute relative error over all points."""
    rel = pointwise_abs_relative_error(
        prediction,
        reference,
        eps=eps,
        warn_on_clamp=warn_on_clamp,
        context=context,
    )
    return float(np.mean(rel, dtype=np.float64))


def mean_pointwise_abs_relative_error_curve(
    prediction_records: np.ndarray | Any,
    reference_records: np.ndarray | Any,
    *,
    eps: float = DEFAULT_RELATIVE_ERROR_EPS,
    warn_on_clamp: bool = True,
    context: str | None = None,
) -> np.ndarray:
    """
    Return per-record mean pointwise absolute relative error.

    The first dimension is treated as the record axis; remaining dimensions are averaged.
    """
    rel = pointwise_abs_relative_error(
        prediction_records,
        reference_records,
        eps=eps,
        warn_on_clamp=warn_on_clamp,
        context=context,
    )
    rel = np.asarray(rel, dtype=np.float64)
    if rel.ndim == 0:
        return np.asarray([float(rel)], dtype=np.float64)
    if rel.ndim == 1:
        return rel.astype(np.float64, copy=False)
    reduce_axes = tuple(range(1, rel.ndim))
    return np.mean(rel, axis=reduce_axes, dtype=np.float64)


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
