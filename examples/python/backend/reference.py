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


def second_order_soliton_period(
    beta2: float,
    T0: float,
) -> float:
    """Return the second-order soliton recurrence period in meters."""
    ld = (float(T0) * float(T0)) / abs(float(beta2))
    return 0.5 * np.pi * ld

def second_order_soliton_normalized_envelope(
    t: np.ndarray,
    z: float,
    beta2: float,
    T0: float,
) -> np.ndarray:
    """Analytical normalized N=2 soliton for ``iU_z + 0.5 U_tt + |U|^2 U = 0`` with ``U(0,t)=2 sech(t)``."""
    ld = (T0 * T0) / abs(beta2)
    xi = z / ld
    numerator = 4.0 * (
        np.cosh(3.0 * t) + 3.0 * np.exp(4.0j * xi) * np.cosh(t)
    ) * np.exp(0.5j * xi)
    denominator = np.cosh(4.0 * t) + 4.0 * np.cosh(2.0 * t) + 3.0 * np.cos(4.0 * xi)
    return numerator / denominator


def second_order_soliton_normalized_records(
    t: np.ndarray,
    z_records: np.ndarray,
    beta2: float,
    T0: float,
) -> np.ndarray:
    """Return analytical normalized second-order soliton records on ``z_records``."""
    tau = np.asarray(t, dtype=np.float64).reshape(-1)
    z_axis = np.asarray(z_records, dtype=np.float64).reshape(-1)
    out = np.empty((z_axis.size, tau.size), dtype=np.complex128)
    for idx, z_value in enumerate(z_axis):
        out[idx] = second_order_soliton_normalized_envelope(tau, float(z_value), beta2, T0)
    return out


def peak_intensity_over_records(records: np.ndarray) -> np.ndarray:
    """Return the maximum intensity at each propagation record."""
    fields = np.asarray(records, dtype=np.complex128)
    if fields.ndim != 2:
        raise ValueError("records must have shape [record, sample].")
    return np.max(np.abs(fields) ** 2, axis=1).astype(np.float64, copy=False)


def analytical_initial_condition_error(
    t: np.ndarray,
    beta2: float,
    T0: float,
) -> float:
    u_ref = 2.0 * (1.0 / np.cosh(t))
    u_analytic = second_order_soliton_normalized_envelope(t, 0.0, beta2, T0)
    return float(np.max(np.abs(u_ref - u_analytic)))
