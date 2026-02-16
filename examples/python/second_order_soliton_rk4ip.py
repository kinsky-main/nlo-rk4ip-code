"""
Second-order soliton propagation check using the Python CFFI API.

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


def _write_complex_buffer(dst, values: np.ndarray) -> None:
    for i, val in enumerate(values):
        dst[i].re = float(val.real)
        dst[i].im = float(val.imag)


def _read_complex_buffer(src, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        out[i] = complex(src[i].re, src[i].im)
    return out


def _parse_pointer_value(value: int | str) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            return int(text, 16)
        return int(text, 10)
    return int(value)


def _to_vk_handle(ffi, value: int | str | None, ctype: str):
    if value is None:
        return ffi.NULL
    parsed = _parse_pointer_value(value)
    if parsed == 0:
        return ffi.NULL
    return ffi.cast(ctype, parsed)


def _build_execution_options(ffi, params: dict):
    backend_cfg = params.get("backend")
    if backend_cfg is None:
        return None

    opts = ffi.new("nlo_execution_options*")
    opts.backend_type = 0  # NLO_VECTOR_BACKEND_CPU
    opts.fft_backend = 0  # NLO_FFT_BACKEND_AUTO
    opts.device_heap_fraction = float(params.get("device_heap_fraction", 0.70))
    opts.record_ring_target = int(params.get("record_ring_target", 0))
    opts.forced_device_budget_bytes = int(params.get("forced_device_budget_bytes", 0))
    opts.vulkan.physical_device = ffi.NULL
    opts.vulkan.device = ffi.NULL
    opts.vulkan.queue = ffi.NULL
    opts.vulkan.queue_family_index = 0
    opts.vulkan.command_pool = ffi.NULL
    opts.vulkan.descriptor_set_budget_bytes = 0
    opts.vulkan.descriptor_set_count_override = 0

    fft_backend = str(params.get("fft_backend", "auto")).strip().lower()
    fft_backend_map = {"auto": 0, "fftw": 1, "vkfft": 2}
    if fft_backend not in fft_backend_map:
        raise ValueError("fft_backend must be one of: auto, fftw, vkfft.")
    opts.fft_backend = fft_backend_map[fft_backend]

    if isinstance(backend_cfg, str):
        backend_type = backend_cfg.strip().lower()
        cfg = {}
    elif isinstance(backend_cfg, dict):
        backend_type = str(backend_cfg.get("type", "cpu")).strip().lower()
        cfg = backend_cfg
    else:
        raise TypeError("backend must be a string or dict when provided.")

    if backend_type == "cpu":
        opts.backend_type = 0
        return opts

    if backend_type != "vulkan":
        raise ValueError("backend type must be either 'cpu' or 'vulkan'.")

    vk_cfg = cfg.get("vulkan", cfg)
    required = ("physical_device", "device", "queue", "queue_family_index")
    missing = [name for name in required if name not in vk_cfg]
    if missing:
        raise ValueError(
            "vulkan backend requires handle fields: "
            + ", ".join(required)
            + ". Missing: "
            + ", ".join(missing)
        )

    opts.backend_type = 1  # NLO_VECTOR_BACKEND_VULKAN
    opts.vulkan.physical_device = _to_vk_handle(ffi, vk_cfg.get("physical_device"), "VkPhysicalDevice")
    opts.vulkan.device = _to_vk_handle(ffi, vk_cfg.get("device"), "VkDevice")
    opts.vulkan.queue = _to_vk_handle(ffi, vk_cfg.get("queue"), "VkQueue")
    opts.vulkan.queue_family_index = int(vk_cfg.get("queue_family_index", 0))
    opts.vulkan.command_pool = _to_vk_handle(ffi, vk_cfg.get("command_pool"), "VkCommandPool")
    opts.vulkan.descriptor_set_budget_bytes = int(vk_cfg.get("descriptor_set_budget_bytes", 0))
    opts.vulkan.descriptor_set_count_override = int(vk_cfg.get("descriptor_set_count_override", 0))
    return opts


def rk4ip_solver_recorded(
    A0: np.ndarray,
    z_final: float,
    params: dict,
    num_recorded_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one propagation and return z records and envelope records."""
    try:
        from nlolib_cffi import ffi, load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nlolib_cffi/cffi is not available. Install cffi and ensure "
            "PYTHONPATH includes the repo's python/ directory."
        ) from exc

    lib = load()

    if num_recorded_samples <= 0:
        raise ValueError("num_recorded_samples must be positive.")

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
    cfg.dispersion.alpha = float(params.get("alpha", 0.0))

    cfg.propagation.propagation_distance = float(z_final)
    cfg.propagation.starting_step_size = float(params.get("starting_step_size", 1e-4))
    cfg.propagation.max_step_size = float(params.get("max_step_size", 1e-2))
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
    out = ffi.new("nlo_complex[]", n * int(num_recorded_samples))
    _write_complex_buffer(inp, np.asarray(A0, dtype=np.complex128))

    exec_opts = _build_execution_options(ffi, params)
    exec_opts_ptr = exec_opts if exec_opts is not None else ffi.NULL
    status = int(lib.nlolib_propagate(cfg, n, inp, int(num_recorded_samples), out, exec_opts_ptr))
    if status != 0:
        raise RuntimeError(f"nlolib_propagate failed with status={status}.")

    flat = _read_complex_buffer(out, n * int(num_recorded_samples))
    records = flat.reshape(int(num_recorded_samples), n)
    if int(num_recorded_samples) == 1:
        z_records = np.asarray([float(z_final)], dtype=np.float64)
    else:
        z_records = np.linspace(0.0, float(z_final), int(num_recorded_samples))
    return z_records, records


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


def _spectral_intensity(field_t: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(field_t))
    return np.abs(spectrum) ** 2


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

    spec_map = np.empty((z_samples.size, valid.sum()), dtype=np.float64)
    for i in range(z_samples.size):
        field_z = np.asarray(A_records[i], dtype=np.complex128)
        spec = np.nan_to_num(_spectral_intensity(field_z), nan=0.0, posinf=0.0, neginf=0.0)
        spec_map[i, :] = spec[valid]

    order = np.argsort(lambda_nm)
    lambda_nm = lambda_nm[order]
    spec_map = spec_map[:, order]
    spec_map = np.nan_to_num(spec_map, nan=0.0, posinf=0.0, neginf=0.0)
    spec_map = np.clip(spec_map, 0.0, None)

    max_value = float(np.max(spec_map))
    if max_value > 0.0:
        spec_map /= max_value

    return z_samples, lambda_nm, spec_map


def save_plots(
    t: np.ndarray,
    U_num: np.ndarray,
    U_true: np.ndarray,
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

    intensity_num = np.nan_to_num(np.abs(U_num) ** 2, nan=0.0, posinf=0.0, neginf=0.0)
    intensity_true = np.nan_to_num(np.abs(U_true) ** 2, nan=0.0, posinf=0.0, neginf=0.0)
    abs_intensity_error = np.abs(intensity_num - intensity_true)

    fig_1, ax_1 = plt.subplots(figsize=(9.0, 5.0))
    ax_1.plot(t, intensity_true, lw=2.0, label="Analytical |U_true|^2")
    ax_1.plot(t, intensity_num, "--", lw=1.5, label="Numerical |U_num|^2")
    ax_1.set_xlabel("Dimensionless time t = T/T0")
    ax_1.set_ylabel("Normalized intensity |U|^2")
    ax_1.set_title(f"Second-Order Soliton at z = {z_final:.3f} m (Normalized)")
    if np.max(intensity_true) > 0.0:
        ax_1.set_ylim(0.0, 1.1 * float(np.max(intensity_true)))
    ax_1.grid(True, alpha=0.3)
    ax_1.legend()
    p1 = output_dir / "intensity_comparison.png"
    fig_1.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig_1)
    saved_paths.append(p1)

    fig_2, ax_2 = plt.subplots(figsize=(9.0, 4.5))
    ax_2.plot(t, abs_intensity_error, lw=1.5, color="tab:red")
    ax_2.set_xlabel("Dimensionless time t = T/T0")
    ax_2.set_ylabel(r"$||U_{num}|^2 - |U_{true}|^2|$")
    ax_2.set_title(f"Absolute Intensity Error, epsilon = {epsilon:.3e}")
    ax_2.grid(True, alpha=0.3)
    p2 = output_dir / "intensity_error.png"
    fig_2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig_2)
    saved_paths.append(p2)

    if spectral_map.size > 0 and lambda_nm.size > 0 and z_samples.size > 0:
        fig_3, ax_3 = plt.subplots(figsize=(9.0, 5.5))
        safe_map = np.nan_to_num(spectral_map, nan=0.0, posinf=0.0, neginf=0.0)
        safe_map = np.clip(safe_map, 0.0, None)
        spectral_db = 10.0 * np.log10(np.maximum(safe_map, 1e-12))
        img = ax_3.pcolormesh(lambda_nm, z_samples, spectral_db, shading="auto", cmap="magma", vmin=-80.0, vmax=0.0)
        ax_3.set_xlabel("Wavelength (nm)")
        ax_3.set_ylabel("Propagation distance z (m)")
        ax_3.set_title("Spectral Intensity Envelope vs Propagation Distance")
        colorbar = fig_3.colorbar(img, ax=ax_3)
        colorbar.set_label("Normalized spectral intensity (dB)")
        p3 = output_dir / "wavelength_intensity_colormap.png"
        fig_3.savefig(p3, dpi=200, bbox_inches="tight")
        plt.close(fig_3)
        saved_paths.append(p3)

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


def main() -> float:
    beta2 = -0.01
    gamma = 0.01
    alpha = 0.0
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    p0 = (2**2) * abs(beta2) / (gamma * t0 * t0)
    z_final = 0.506
    sgn_beta2, ld, lnl = normalized_nlse_coefficients(beta2, gamma, t0, p0)

    n = 2**12
    tmax = 8.0 * t0
    T = np.linspace(-tmax, tmax, n)
    t = to_dimensionless_time(T, t0)
    dt = float(T[1] - T[0])
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)

    U0 = 2.0 * sech(t)
    A0 = to_physical_envelope(U0, 0.0, p0, alpha)

    params = {
        "beta2": beta2,
        "gamma": gamma,
        "alpha": alpha,
        "backend": "cpu",
        "fft_backend": "auto",
        "dt": dt,
        "omega": omega,
        "pulse_period": n * dt,
        "starting_step_size": 1e-4,
        "max_step_size": 1e-2,
        "min_step_size": 1e-6,
        "error_tolerance": 1e-8,
    }

    num_recorded_samples = 40
    z_records, A_records = rk4ip_solver_recorded(A0, z_final, params, num_recorded_samples)
    A_num = np.asarray(A_records[-1], dtype=np.complex128)
    U_true = second_order_soliton_normalized_envelope(t, z_final, beta2, t0)
    U_num = to_normalized_envelope(A_num, z_final, p0, alpha)
    epsilon = average_relative_intensity_error(U_num, U_true)
    z0_analytic_error = analytical_initial_condition_error(t, beta2, t0)
    if not np.isfinite(epsilon):
        first_bad_z, max_finite_amplitude = diagnose_first_nonfinite_record(A_records, z_records)
        print("warning: epsilon is non-finite because numerical field contains non-finite values.")
        if first_bad_z is not None:
            print(
                "diagnostic: first non-finite numerical value detected near "
                f"z = {first_bad_z:.6e} m; max finite |A| before divergence = "
                f"{max_finite_amplitude:.6e}."
            )

    lambda0_nm = 1550.0
    z_map, lambda_nm, spectral_map = compute_wavelength_spectral_map_from_records(
        A_records,
        z_records,
        dt,
        lambda0_nm,
    )

    output_dir = Path(__file__).resolve().parent / "output" / "second_order_soliton"
    saved_paths = save_plots(
        t,
        U_num,
        U_true,
        epsilon,
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
