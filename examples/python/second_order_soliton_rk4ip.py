"""
Second-order soliton propagation check using the Python CFFI API.

This example compares the numerical solver output against the known analytical
breather solution at one soliton period and reports the average relative
intensity error epsilon.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

C_NM_PER_PS = 299792.458


def sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x)


def second_order_soliton_field(
    T: np.ndarray,
    z: float,
    beta2: float,
    gamma: float,
    T0: float,
) -> np.ndarray:
    """Analytical field for an N=2 soliton of the normalized NLSE."""
    ld = (T0 * T0) / abs(beta2)
    xi = z / ld
    tau = T / T0

    numerator = 4.0 * (
        np.cosh(3.0 * tau) + 3.0 * np.exp(4.0j * xi) * np.cosh(tau)
    ) * np.exp(0.5j * xi)
    denominator = np.cosh(4.0 * tau) + 4.0 * np.cosh(2.0 * tau) + 3.0 * np.cos(4.0 * xi)
    u2 = numerator / denominator

    amplitude_scale = math.sqrt(abs(beta2) / (gamma * T0 * T0))
    return amplitude_scale * u2


def _write_complex_buffer(dst, values: np.ndarray) -> None:
    for i, val in enumerate(values):
        dst[i].re = float(val.real)
        dst[i].im = float(val.imag)


def _read_complex_buffer(src, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        out[i] = complex(src[i].re, src[i].im)
    return out


def rk4ip_solver(A0: np.ndarray, z_final: float, params: dict) -> np.ndarray:
    """Thin Python wrapper around nlolib_propagate."""
    try:
        from nlolib_cffi import ffi, load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nlolib_cffi/cffi is not available. Install cffi and ensure "
            "PYTHONPATH includes the repo's python/ directory."
        ) from exc

    lib = load()

    n = int(A0.size)
    dt = float(params["dt"])
    omega = np.asarray(params["omega"], dtype=np.float64)
    if omega.size != n:
        raise ValueError("omega grid size must match A0 size.")

    cfg = ffi.new("sim_config*")
    cfg.nonlinear.gamma = float(params["gamma"])
    cfg.dispersion.num_dispersion_terms = 3
    cfg.dispersion.betas[0] = 0.0
    cfg.dispersion.betas[1] = 0.0
    cfg.dispersion.betas[2] = float(params["beta2"])
    cfg.dispersion.alpha = 0.0

    cfg.propagation.propagation_distance = float(z_final)
    cfg.propagation.starting_step_size = float(params.get("starting_step_size", 1e-4))
    cfg.propagation.max_step_size = float(params.get("max_step_size", 1e-4))
    cfg.propagation.min_step_size = float(params.get("min_step_size", 1e-4))
    cfg.propagation.error_tolerance = float(params.get("error_tolerance", 1e-8))

    cfg.time.pulse_period = float(params.get("pulse_period", n * dt))
    cfg.time.delta_time = dt

    freq = ffi.new("nlo_complex[]", n)
    for i, om in enumerate(omega):
        freq[i].re = float(om)
        freq[i].im = 0.0
    cfg.frequency.frequency_grid = freq

    inp = ffi.new("nlo_complex[]", n)
    out = ffi.new("nlo_complex[]", n)
    _write_complex_buffer(inp, np.asarray(A0, dtype=np.complex128))

    status = int(lib.nlolib_propagate(cfg, n, inp, out))
    if status != 0:
        raise RuntimeError(f"nlolib_propagate failed with status={status}.")

    return _read_complex_buffer(out, n)


def average_relative_intensity_error(A_num: np.ndarray, A_true: np.ndarray) -> float:
    intensity_num = np.abs(A_num) ** 2
    intensity_true = np.abs(A_true) ** 2
    numerator = np.mean(np.abs(intensity_num - intensity_true))
    denominator = float(np.max(intensity_true))
    return float(numerator / denominator)


def _spectral_intensity(field_t: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(field_t))
    return np.abs(spectrum) ** 2


def compute_wavelength_spectral_map(
    A0: np.ndarray,
    z_samples: np.ndarray,
    params: dict,
    lambda0_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(A0.size)
    dt = float(params["dt"])
    freq_shifted = np.fft.fftshift(np.fft.fftfreq(n, d=dt))

    nu0 = C_NM_PER_PS / lambda0_nm
    nu = nu0 + freq_shifted
    valid = nu > 0.0
    lambda_nm = C_NM_PER_PS / nu[valid]

    spec_map = np.empty((z_samples.size, valid.sum()), dtype=np.float64)
    for i, z in enumerate(z_samples):
        field_z = np.asarray(A0, dtype=np.complex128) if z == 0.0 else rk4ip_solver(A0, float(z), params)
        spec = _spectral_intensity(field_z)
        spec_map[i, :] = spec[valid]

    order = np.argsort(lambda_nm)
    lambda_nm = lambda_nm[order]
    spec_map = spec_map[:, order]

    max_value = float(np.max(spec_map))
    if max_value > 0.0:
        spec_map /= max_value

    return z_samples, lambda_nm, spec_map


def save_plots(
    T: np.ndarray,
    A_num: np.ndarray,
    A_true: np.ndarray,
    epsilon: float,
    z_final: float,
    z_samples: np.ndarray,
    lambda_nm: np.ndarray,
    spectral_map: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    if plt is None:
        print("matplotlib not available; skipping plot generation.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    intensity_num = np.abs(A_num) ** 2
    intensity_true = np.abs(A_true) ** 2
    abs_intensity_error = np.abs(intensity_num - intensity_true)

    fig_1, ax_1 = plt.subplots(figsize=(9.0, 5.0))
    ax_1.plot(T, intensity_true, lw=2.0, label="Analytical |A_true|^2")
    ax_1.plot(T, intensity_num, "--", lw=1.5, label="Numerical |A_num|^2")
    ax_1.set_xlabel("Time (ps)")
    ax_1.set_ylabel("Intensity (W)")
    ax_1.set_title(f"Second-Order Soliton at z = {z_final:.3f} m")
    ax_1.grid(True, alpha=0.3)
    ax_1.legend()
    p1 = output_dir / "intensity_comparison.png"
    fig_1.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig_1)
    saved_paths.append(p1)

    fig_2, ax_2 = plt.subplots(figsize=(9.0, 4.5))
    ax_2.plot(T, abs_intensity_error, lw=1.5, color="tab:red")
    ax_2.set_xlabel("Time (ps)")
    ax_2.set_ylabel(r"$||A_{num}|^2 - |A_{true}|^2|$")
    ax_2.set_title(f"Absolute Intensity Error, epsilon = {epsilon:.3e}")
    ax_2.grid(True, alpha=0.3)
    p2 = output_dir / "intensity_error.png"
    fig_2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig_2)
    saved_paths.append(p2)

    fig_3, ax_3 = plt.subplots(figsize=(9.0, 5.5))
    img = ax_3.pcolormesh(lambda_nm, z_samples, spectral_map, shading="auto", cmap="magma")
    ax_3.set_xlabel("Wavelength (nm)")
    ax_3.set_ylabel("Propagation distance z (m)")
    ax_3.set_title("Spectral Intensity Envelope vs Propagation Distance")
    colorbar = fig_3.colorbar(img, ax=ax_3)
    colorbar.set_label("Normalized spectral intensity")
    p3 = output_dir / "wavelength_intensity_colormap.png"
    fig_3.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig_3)
    saved_paths.append(p3)

    return saved_paths


def diagnose_first_nonfinite_z(A0: np.ndarray, params: dict, z_final: float) -> tuple[float | None, float]:
    z_probe = np.linspace(0.0, z_final, 41)
    max_finite_amplitude = 0.0
    for z in z_probe:
        field_z = np.asarray(A0, dtype=np.complex128) if z == 0.0 else rk4ip_solver(A0, float(z), params)
        finite_mask = np.isfinite(field_z.real) & np.isfinite(field_z.imag)
        if not finite_mask.all():
            return float(z), max_finite_amplitude
        max_finite_amplitude = max(max_finite_amplitude, float(np.max(np.abs(field_z))))
    return None, max_finite_amplitude


def main() -> float:
    beta2 = -0.01
    gamma = 0.01
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    p0 = (2**2) * abs(beta2) / (gamma * t0 * t0)
    z_final = 0.506

    n = 2**12
    tmax = 8.0 * t0
    T = np.linspace(-tmax, tmax, n)
    dt = float(T[1] - T[0])
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)

    A0 = np.sqrt(p0) * sech(T / t0)

    params = {
        "beta2": beta2,
        "gamma": gamma,
        "dt": dt,
        "omega": omega,
        "pulse_period": n * dt,
        "starting_step_size": 1e-4,
        "max_step_size": 1e-4,
        "min_step_size": 1e-4,
        "error_tolerance": 1e-8,
    }

    A_true = second_order_soliton_field(T, z_final, beta2, gamma, t0)
    A_num = rk4ip_solver(A0, z_final, params)
    epsilon = average_relative_intensity_error(A_num, A_true)
    if not np.isfinite(epsilon):
        first_bad_z, max_finite_amplitude = diagnose_first_nonfinite_z(A0, params, z_final)
        print("warning: epsilon is non-finite because numerical field contains non-finite values.")
        if first_bad_z is not None:
            print(
                "diagnostic: first non-finite numerical value detected near "
                f"z = {first_bad_z:.6e} m; max finite |A| before divergence = "
                f"{max_finite_amplitude:.6e}."
            )

    lambda0_nm = 1550.0
    z_samples = np.linspace(0.0, z_final, 40)
    map_params = dict(params)
    map_step = z_final / 400.0
    map_params["starting_step_size"] = map_step
    map_params["max_step_size"] = map_step
    map_params["min_step_size"] = map_step
    z_map, lambda_nm, spectral_map = compute_wavelength_spectral_map(A0, z_samples, map_params, lambda0_nm)

    output_dir = Path(__file__).resolve().parent / "output" / "second_order_soliton"
    saved_paths = save_plots(
        T,
        A_num,
        A_true,
        epsilon,
        z_final,
        z_map,
        lambda_nm,
        spectral_map,
        output_dir,
    )

    print(f"epsilon = {epsilon:.6e}")
    if saved_paths:
        print("saved plots:")
        for path in saved_paths:
            print(f"  {path}")

    return epsilon


if __name__ == "__main__":
    main()
