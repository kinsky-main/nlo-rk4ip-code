"""Reference propagators for Python examples and diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def exact_linear_temporal_records(
    field0: np.ndarray | Any,
    z_records: np.ndarray | Any,
    omega: np.ndarray | Any,
    c0: float,
    c1: float = 0.0,
) -> np.ndarray:
    """
    Return the exact linear temporal propagation records for ``D(w)=i*c0*w^2-c1``.

    This follows the same internal convention as the solver: forward FFT to
    frequency, multiply by ``exp(D*z)``, then inverse FFT back to time.
    """
    field = np.asarray(field0, dtype=np.complex128).reshape(-1)
    z_axis = np.asarray(z_records, dtype=np.float64).reshape(-1)
    omega_axis = np.asarray(omega, dtype=np.float64).reshape(-1)
    if field.size <= 0:
        raise ValueError("field0 must be non-empty.")
    if omega_axis.size != field.size:
        raise ValueError("omega size must match field0 size.")

    spectrum0 = np.fft.fft(field)
    dispersion = ((1.0j * float(c0)) * (omega_axis**2)) - float(c1)
    out = np.empty((z_axis.size, field.size), dtype=np.complex128)
    for idx, z_value in enumerate(z_axis):
        phase = np.exp(dispersion * float(z_value))
        out[idx] = np.fft.ifft(spectrum0 * phase)
    return out


def exact_linear_temporal_final(
    field0: np.ndarray | Any,
    z_final: float,
    omega: np.ndarray | Any,
    c0: float,
    c1: float = 0.0,
) -> np.ndarray:
    """Return the final exact linear temporal field for ``z_final``."""
    return exact_linear_temporal_records(
        field0,
        np.asarray([float(z_final)], dtype=np.float64),
        omega,
        c0,
        c1,
    )[0]
