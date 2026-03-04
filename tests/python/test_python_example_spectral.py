from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PYTHON = REPO_ROOT / "examples" / "python"
if str(EXAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PYTHON))

from backend.spectral import (  # noqa: E402
    carrier_wavelength_nm_to_frequency_hz,
    frequency_hz_to_wavelength_nm,
    omega_centroid_to_wavelength_nm,
    omega_detuning_to_wavelength_nm,
)


def _check_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tol:
        raise AssertionError(f"value mismatch: actual={actual}, expected={expected}, tol={tol}")


def test_wavelength_frequency_roundtrip() -> None:
    lambda0 = 1550.0
    freq0 = carrier_wavelength_nm_to_frequency_hz(lambda0)
    lambda_back = float(frequency_hz_to_wavelength_nm(np.asarray([freq0]))[0])
    _check_close(lambda_back, lambda0, tol=1e-9)


def test_omega_axis_to_wavelength_axis() -> None:
    # Large negative detuning should produce invalid total frequencies.
    omega = np.asarray([-2.0e15, -1.0e14, 0.0, 1.0e14], dtype=np.float64)
    wavelengths_nm, valid_mask = omega_detuning_to_wavelength_nm(omega, 1550.0, time_unit_seconds=1.0e-12)
    if wavelengths_nm.size <= 0:
        raise AssertionError("expected at least one valid wavelength sample")
    if valid_mask.dtype != bool:
        raise AssertionError("valid mask must be boolean")
    if np.any(wavelengths_nm <= 0.0):
        raise AssertionError("wavelength samples must be positive")
    if int(np.count_nonzero(valid_mask)) != int(wavelengths_nm.size):
        raise AssertionError("valid-mask count should match returned wavelength count")


def test_omega_centroid_to_wavelength_nm() -> None:
    omega = np.asarray([0.0, 2.0e12], dtype=np.float64)
    wavelength = omega_centroid_to_wavelength_nm(omega, 1550.0, time_unit_seconds=1.0e-12)
    if wavelength.shape != omega.shape:
        raise AssertionError("centroid wavelength conversion should preserve shape")
    if not (float(wavelength[1]) < float(wavelength[0])):
        raise AssertionError("positive frequency detuning should reduce wavelength")


def main() -> None:
    test_wavelength_frequency_roundtrip()
    test_omega_axis_to_wavelength_axis()
    test_omega_centroid_to_wavelength_nm()
    print("test_python_example_spectral: all checks passed.")


if __name__ == "__main__":
    main()
