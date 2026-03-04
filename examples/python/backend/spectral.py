"""Shared spectral conversion helpers for Python examples."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def carrier_wavelength_nm_to_frequency_hz(lambda0_nm: float) -> float:
    """Convert carrier wavelength in nm to absolute frequency in Hz."""
    wavelength_nm = float(lambda0_nm)
    if wavelength_nm <= 0.0:
        raise ValueError("lambda0_nm must be > 0.")
    return SPEED_OF_LIGHT_M_PER_S / (wavelength_nm * 1.0e-9)


def frequency_hz_to_wavelength_nm(frequency_hz: np.ndarray | Any) -> np.ndarray:
    """Convert absolute frequency in Hz to wavelength in nm."""
    freq = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(freq <= 0.0):
        raise ValueError("frequency_hz must be strictly positive.")
    return (SPEED_OF_LIGHT_M_PER_S / freq) * 1.0e9


def omega_detuning_to_frequency_hz(
    omega_detuning: np.ndarray | Any,
    *,
    time_unit_seconds: float = 1.0e-12,
) -> np.ndarray:
    """Convert angular-frequency detuning to frequency detuning in Hz."""
    if time_unit_seconds <= 0.0:
        raise ValueError("time_unit_seconds must be > 0.")
    omega = np.asarray(omega_detuning, dtype=np.float64)
    return omega / (2.0 * math.pi * float(time_unit_seconds))


def omega_detuning_to_wavelength_nm(
    omega_detuning: np.ndarray | Any,
    lambda0_nm: float,
    *,
    time_unit_seconds: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map angular-frequency detuning to wavelength axis.

    Returns ``(wavelength_nm, valid_mask)`` where ``valid_mask`` marks positive
    absolute-frequency samples from the input detuning axis.
    """
    detuning_hz = omega_detuning_to_frequency_hz(
        omega_detuning,
        time_unit_seconds=time_unit_seconds,
    )
    carrier_hz = carrier_wavelength_nm_to_frequency_hz(lambda0_nm)
    absolute_hz = carrier_hz + detuning_hz
    valid_mask = absolute_hz > 0.0
    if not np.any(valid_mask):
        raise ValueError("no positive total frequency samples for wavelength map.")
    wavelengths_nm = frequency_hz_to_wavelength_nm(absolute_hz[valid_mask])
    return wavelengths_nm, valid_mask


def omega_centroid_to_wavelength_nm(
    omega_centroid: np.ndarray | Any,
    lambda0_nm: float,
    *,
    time_unit_seconds: float = 1.0e-12,
) -> np.ndarray:
    """Convert spectral-centroid detuning values to centroid wavelength in nm."""
    detuning_hz = omega_detuning_to_frequency_hz(
        omega_centroid,
        time_unit_seconds=time_unit_seconds,
    )
    absolute_hz = carrier_wavelength_nm_to_frequency_hz(lambda0_nm) + detuning_hz
    return frequency_hz_to_wavelength_nm(absolute_hz)
