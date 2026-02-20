"""
Second-order soliton propagation check using the Python ctypes API.

This example compares the numerical solver output against the known analytical
breather solution at one soliton period and reports the average relative
intensity error epsilon. The comparison is performed in normalized variables
using t = T / T0 and A(z, t) = sqrt(P0) * exp(-alpha * z / 2) * U(z, t).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from backend.plotting import (
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

C_NM_PER_PS = 299792.458


def sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x)


def to_dimensionless_time(T: np.ndarray, T0: float) -> np.ndarray:
    return T / T0


def to_normalized_envelope(A: np.ndarray, z: float, P0: float, alpha: float) -> np.ndarray:
    return np.asarray(A, dtype=np.complex128) * np.exp(0.5 * alpha * z) / math.sqrt(P0)


def to_physical_envelope(U: np.ndarray, z: float, P0: float, alpha: float) -> np.ndarray:
    return np.asarray(U, dtype=np.complex128) * (math.sqrt(P0) * np.exp(-0.5 * alpha * z))


def normalized_nlse_coefficients(beta2: float, gamma: float, T0: float, P0: float) -> tuple[float, float, float]:
    if beta2 == 0.0:
        raise ValueError("beta2 must be non-zero for NLSE normalization.")
    ld = (T0 * T0) / abs(beta2)
    lnl = 1.0 / (gamma * P0)
    sgn_beta2 = 1.0 if beta2 > 0.0 else -1.0
    return sgn_beta2, ld, lnl


def second_order_soliton_normalized_envelope(
    t: np.ndarray,
    z: float,
    beta2: float,
    T0: float,
) -> np.ndarray:
    """Analytical normalized envelope U for an N=2 soliton of the NLSE."""
    ld = (T0 * T0) / abs(beta2)
    xi = z / ld
    numerator = 4.0 * (
        np.cosh(3.0 * t) + 3.0 * np.exp(4.0j * xi) * np.cosh(t)
    ) * np.exp(0.5j * xi)
    denominator = np.cosh(4.0 * t) + 4.0 * np.cosh(2.0 * t) + 3.0 * np.cos(4.0 * xi)
    return numerator / denominator


def average_relative_intensity_error(A_num: np.ndarray, A_true: np.ndarray) -> float:
    intensity_num = np.abs(A_num) ** 2
    intensity_true = np.abs(A_true) ** 2
    finite = np.isfinite(intensity_num) & np.isfinite(intensity_true)
    if not np.any(finite):
        return float("nan")
    numerator = np.mean(np.abs(intensity_num[finite] - intensity_true[finite]))
    denominator = float(np.max(intensity_true[finite]))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def analytical_initial_condition_error(
    t: np.ndarray,
    beta2: float,
    T0: float,
) -> float:
    u_ref = 2.0 * sech(t)
    u_analytic = second_order_soliton_normalized_envelope(t, 0.0, beta2, T0)
    return float(np.max(np.abs(u_ref - u_analytic)))


def compute_wavelength_spectral_map_from_records(
    A_records: np.ndarray,
    z_samples: np.ndarray,
    dt: float,
    lambda0_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if A_records.ndim != 2:
        raise ValueError("A_records must be a 2D array [record, time].")
    if A_records.shape[0] != z_samples.size:
        raise ValueError("A_records row count must match z_samples length.")

    n = int(A_records.shape[1])
    freq_shifted = np.fft.fftshift(np.fft.fftfreq(n, d=dt))

    nu0 = C_NM_PER_PS / lambda0_nm
    nu = nu0 + freq_shifted
    valid = nu > 0.0
    lambda_nm = C_NM_PER_PS / nu[valid]

    field_records = np.asarray(A_records, dtype=np.complex128)
    spectra = np.fft.fftshift(np.fft.fft(field_records, axis=1), axes=1)
    spec_map = np.abs(spectra[:, valid]) ** 2

    order = np.argsort(lambda_nm)
    lambda_nm = lambda_nm[order]
    spec_map = spec_map[:, order]
    spec_map = np.nan_to_num(spec_map, nan=0.0, posinf=0.0, neginf=0.0)
    spec_map = np.clip(spec_map, 0.0, None)

    max_value = float(np.max(spec_map))
    if max_value > 0.0:
        spec_map /= max_value

    # Keep only the visibly occupied spectral band; this avoids an effectively
    # blank map when tiny near-zero-frequency bins stretch lambda to huge values.
    spectral_profile = np.max(spec_map, axis=0)
    if spectral_profile.size > 0:
        support_threshold = max(float(np.max(spectral_profile)) * 1e-3, 1e-12)
        support_idx = np.flatnonzero(spectral_profile >= support_threshold)
        if support_idx.size >= 8:
            left = max(int(support_idx[0]) - 2, 0)
            right = min(int(support_idx[-1]) + 3, lambda_nm.size)
            lambda_nm = lambda_nm[left:right]
            spec_map = spec_map[:, left:right]

    return z_samples, lambda_nm, spec_map


def relative_l2_error_curve(records: np.ndarray, reference_records: np.ndarray) -> np.ndarray:
    if records.shape != reference_records.shape:
        raise ValueError("records and reference_records must have the same shape.")

    ref_norms = np.linalg.norm(np.asarray(reference_records, dtype=np.complex128), axis=1)
    norm_floor = max(float(np.max(ref_norms)) * 1e-12, 1e-15)
    out = np.empty(records.shape[0], dtype=np.float64)
    for i in range(records.shape[0]):
        ref = np.asarray(reference_records[i], dtype=np.complex128)
        num = np.asarray(records[i], dtype=np.complex128)
        ref_norm = float(np.linalg.norm(ref))
        safe_ref_norm = max(ref_norm, norm_floor)
        out[i] = float(np.linalg.norm(num - ref) / safe_ref_norm)
    return out


def save_plots(
    t: np.ndarray,
    U_num: np.ndarray,
    U_true: np.ndarray,
    error_curve: np.ndarray,
    z_final: float,
    z_samples: np.ndarray,
    lambda_nm: np.ndarray,
    spectral_map: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    p1 = plot_intensity_colormap_vs_propagation(
        lambda_nm,
        z_samples,
        spectral_map,
        output_dir / "wavelength_intensity_colormap.png",
        x_label="Wavelength (nm)",
        y_label="Propagation distance z (m)",
        title="Spectral Intensity Envelope vs Propagation Distance",
        colorbar_label="Normalized spectral intensity",
        cmap="magma",
    )
    if p1 is not None:
        saved_paths.append(p1)

    p2 = plot_final_re_im_comparison(
        t,
        U_true,
        U_num,
        output_dir / "final_re_im_comparison.png",
        x_label="Dimensionless time t = T/T0",
        title=f"Second-Order Soliton at z = {z_final:.3f} m: Re/Im Comparison",
        reference_label="Analytical",
        final_label="Numerical",
    )
    if p2 is not None:
        saved_paths.append(p2)

    p3 = plot_final_intensity_comparison(
        t,
        U_true,
        U_num,
        output_dir / "final_intensity_comparison.png",
        x_label="Dimensionless time t = T/T0",
        title=f"Second-Order Soliton at z = {z_final:.3f} m: Intensity Comparison",
        reference_label="Analytical",
        final_label="Numerical",
    )
    if p3 is not None:
        saved_paths.append(p3)

    p4 = plot_total_error_over_propagation(
        z_samples,
        error_curve,
        output_dir / "total_error_over_propagation.png",
        title="Second-Order Soliton: Total Error Over Propagation",
        y_label="Relative L2 error (numerical vs analytical)",
    )
    if p4 is not None:
        saved_paths.append(p4)

    return saved_paths


def diagnose_first_nonfinite_record(
    A_records: np.ndarray,
    z_samples: np.ndarray,
) -> tuple[float | None, float]:
    max_finite_amplitude = 0.0
    for i, z in enumerate(z_samples):
        field_z = np.asarray(A_records[i], dtype=np.complex128)
        finite_mask = np.isfinite(field_z.real) & np.isfinite(field_z.imag)
        if not finite_mask.all():
            return float(z), max_finite_amplitude
        max_finite_amplitude = max(max_finite_amplitude, float(np.max(np.abs(field_z))))
    return None, max_finite_amplitude


def ensure_finite_records_or_raise(
    A_records: np.ndarray,
    z_samples: np.ndarray,
) -> None:
    first_bad_z, max_finite_amplitude = diagnose_first_nonfinite_record(A_records, z_samples)
    if first_bad_z is None:
        return
    raise RuntimeError(
        "numerical propagation diverged with non-finite field values near "
        f"z = {first_bad_z:.6e} m; max finite |A| before divergence = {max_finite_amplitude:.6e}. "
        "Plotting was aborted to avoid blank/invalid figures."
    )


def main() -> float:
    beta2 = -0.01
    gamma = 0.01
    alpha = 0.0
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    p0 = (2**2) * abs(beta2) / (gamma * t0 * t0)
    sgn_beta2, ld, lnl = normalized_nlse_coefficients(beta2, gamma, t0, p0)
    z_final = 0.5 * math.pi * ld

    n = 2**10
    dt = (16.0 * t0) / n
    T = centered_time_grid(n, dt)
    t = to_dimensionless_time(T, t0)
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)

    U0 = 2.0 * sech(t)
    A0 = to_physical_envelope(U0, 0.0, p0, alpha)

    num_recorded_samples = 160
    sim_cfg = TemporalSimulationConfig(
        gamma=gamma,
        beta2=beta2,
        alpha=alpha,
        dt=dt,
        z_final=z_final,
        num_time_samples=n,
        pulse_period=n * dt,
        omega=omega,
        starting_step_size=2e-4,
        max_step_size=1e-2,
        min_step_size=1e-7,
        error_tolerance=5e-6,
    )
    exec_opts = SimulationOptions(backend="auto", fft_backend="auto")

    runner = NloExampleRunner()
    z_records, A_records = runner.propagate_temporal_records(
        np.asarray(A0, dtype=np.complex128),
        sim_cfg,
        num_recorded_samples,
        exec_opts,
    )
    ensure_finite_records_or_raise(A_records, z_records)
    U_num_records = np.empty_like(A_records, dtype=np.complex128)
    U_true_records = np.empty_like(A_records, dtype=np.complex128)
    for i, z in enumerate(z_records):
        U_num_records[i] = to_normalized_envelope(A_records[i], float(z), p0, alpha)
        U_true_records[i] = second_order_soliton_normalized_envelope(t, float(z), beta2, t0)

    U_num = U_num_records[-1]
    U_true = U_true_records[-1]
    epsilon = average_relative_intensity_error(U_num, U_true)
    error_curve = relative_l2_error_curve(U_num_records, U_true_records)
    z0_analytic_error = analytical_initial_condition_error(t, beta2, t0)
    if not np.isfinite(epsilon):
        raise RuntimeError("final epsilon is non-finite; numerical output is invalid.")

    lambda0_nm = 1550.0
    z_map, lambda_nm, spectral_map = compute_wavelength_spectral_map_from_records(
        A_records,
        z_records,
        dt,
        lambda0_nm,
    )
    if not np.all(np.isfinite(spectral_map)):
        raise RuntimeError("spectral map contains non-finite values; refusing to render blank output.")

    output_dir = Path(__file__).resolve().parent / "output" / "second_order_soliton"
    saved_paths = save_plots(
        t,
        U_num,
        U_true,
        error_curve,
        z_final,
        z_map,
        lambda_nm,
        spectral_map,
        output_dir,
    )

    print(
        "normalized NLSE coefficients: "
        f"sgn(beta2)={int(sgn_beta2):+d}, "
        f"1/(2*LD)={0.5 / ld:.6e} 1/m, "
        f"exp(-alpha*z_final)/LNL={math.exp(-alpha * z_final) / lnl:.6e} 1/m."
    )
    print(f"analytical z=0 envelope max error = {z0_analytic_error:.6e}")
    print(f"epsilon = {epsilon:.6e}")
    if saved_paths:
        print("saved plots:")
        for path in saved_paths:
            print(f"  {path}")

    return epsilon


if __name__ == "__main__":
    main()
