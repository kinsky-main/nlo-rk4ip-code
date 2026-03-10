"""Reference propagators for Python examples and diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp as _solve_ivp
except Exception:  # pragma: no cover - optional dependency
    _solve_ivp = None


def scipy_available() -> bool:
    """Return ``True`` when SciPy's ODE integrator is available."""
    return _solve_ivp is not None


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


def solve_temporal_nlse_scipy_records(
    field0: np.ndarray | Any,
    z_records: np.ndarray | Any,
    omega: np.ndarray | Any,
    c0: float,
    c2: float,
    *,
    c1: float = 0.0,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    method: str = "DOP853",
    max_step: float | None = None,
) -> np.ndarray:
    """
    Solve the temporal NLSE reference ODE with SciPy on the host.

    The semi-discrete RHS matches the solver's temporal convention:

    ``dA/dz = ifft(D(w) * fft(A)) + i*c2*A*|A|^2``

    where ``D(w)=i*c0*w^2-c1``.
    """
    if _solve_ivp is None:
        raise ImportError("SciPy is not available in the active Python environment.")

    field = np.asarray(field0, dtype=np.complex128).reshape(-1)
    z_axis = np.asarray(z_records, dtype=np.float64).reshape(-1)
    omega_axis = np.asarray(omega, dtype=np.float64).reshape(-1)
    if field.size <= 0:
        raise ValueError("field0 must be non-empty.")
    if z_axis.size <= 0:
        raise ValueError("z_records must be non-empty.")
    if omega_axis.size != field.size:
        raise ValueError("omega size must match field0 size.")
    if np.any(~np.isfinite(z_axis)):
        raise ValueError("z_records must be finite.")
    if np.any(np.diff(z_axis) < 0.0):
        raise ValueError("z_records must be sorted in nondecreasing order.")

    dispersion = ((1.0j * float(c0)) * (omega_axis**2)) - float(c1)
    z_final = float(z_axis[-1])
    if max_step is None:
        positive_spacings = np.diff(z_axis)
        positive_spacings = positive_spacings[positive_spacings > 0.0]
        spacing_cap = float(np.min(positive_spacings)) if positive_spacings.size > 0 else z_final
        max_step_value = min(spacing_cap, z_final / 256.0) if z_final > 0.0 else np.inf
    else:
        max_step_value = float(max_step)
    if not np.isfinite(max_step_value) or max_step_value <= 0.0:
        max_step_value = np.inf
    first_step_value = min(max_step_value, z_final / 1024.0) if np.isfinite(max_step_value) and z_final > 0.0 else None
    if first_step_value is not None and first_step_value <= 0.0:
        first_step_value = None

    def rhs(_z: float, field_z: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fft(field_z)
        linear = np.fft.ifft(dispersion * spectrum)
        if c2 == 0.0:
            return linear
        intensity = np.abs(field_z) ** 2
        nonlinear = (1.0j * float(c2)) * field_z * intensity
        return linear + nonlinear

    solution = _solve_ivp(
        rhs,
        (0.0, z_final),
        field,
        method=method,
        t_eval=z_axis,
        rtol=float(rtol),
        atol=float(atol),
        first_step=first_step_value,
        max_step=max_step_value,
    )
    if not solution.success:
        raise RuntimeError(f"SciPy reference integration failed: {solution.message}")
    return np.asarray(solution.y.T, dtype=np.complex128)


def solve_temporal_nlse_scipy_final(
    field0: np.ndarray | Any,
    z_final: float,
    omega: np.ndarray | Any,
    c0: float,
    c2: float,
    *,
    c1: float = 0.0,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    method: str = "DOP853",
    max_step: float | None = None,
) -> np.ndarray:
    """Return the final SciPy reference field at ``z_final``."""
    return solve_temporal_nlse_scipy_records(
        field0,
        np.asarray([float(z_final)], dtype=np.float64),
        omega,
        c0,
        c2,
        c1=c1,
        rtol=rtol,
        atol=atol,
        method=method,
        max_step=max_step,
    )[0]
