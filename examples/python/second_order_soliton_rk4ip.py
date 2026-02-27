"""
Second-order soliton propagation check using the Python ctypes API.

This example compares the numerical solver output against the known analytical
breather solution at one soliton period and reports the average relative
intensity error epsilon. The comparison is performed in normalized variables
using t = T / T0 and A(z, t) = sqrt(P0) * exp(-alpha * z / 2) * U(z, t).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.plotting import (
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_total_error_over_propagation,
)
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from backend.storage import ExampleRunDB


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

C_NM_PER_PS = 299792.458


@dataclass(frozen=True)
class StepTelemetry:
    accepted_z: np.ndarray
    accepted_step_sizes: np.ndarray
    next_z: np.ndarray
    next_step_sizes: np.ndarray
    dropped: int

    @staticmethod
    def empty() -> "StepTelemetry":
        return StepTelemetry(
            accepted_z=np.empty(0, dtype=np.float64),
            accepted_step_sizes=np.empty(0, dtype=np.float64),
            next_z=np.empty(0, dtype=np.float64),
            next_step_sizes=np.empty(0, dtype=np.float64),
            dropped=0,
        )


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
    fft_size_visual: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if A_records.ndim != 2:
        raise ValueError("A_records must be a 2D array [record, time].")
    if A_records.shape[0] != z_samples.size:
        raise ValueError("A_records row count must match z_samples length.")

    n = int(A_records.shape[1])
    n_fft = int(n if fft_size_visual is None else max(int(fft_size_visual), n))
    freq_shifted = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt))

    nu0 = C_NM_PER_PS / lambda0_nm
    nu = nu0 + freq_shifted
    valid = nu > 0.0
    lambda_nm = C_NM_PER_PS / nu[valid]

    field_records = np.asarray(A_records, dtype=np.complex128)
    spectra = np.fft.fftshift(np.fft.fft(field_records, n=n_fft, axis=1), axes=1)
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
            support_min = float(lambda_nm[int(support_idx[0])])
            support_max = float(lambda_nm[int(support_idx[-1])])
            half_span = max(abs(support_min - float(lambda0_nm)), abs(support_max - float(lambda0_nm)))
            if half_span > 0.0:
                half_span *= 1.02
                lower = float(lambda0_nm) - half_span
                upper = float(lambda0_nm) + half_span
                band = (lambda_nm >= lower) & (lambda_nm <= upper)
                if int(np.count_nonzero(band)) >= 8:
                    lambda_nm = lambda_nm[band]
                    spec_map = spec_map[:, band]

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


def _load_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def step_telemetry_from_meta(meta: dict[str, object] | None) -> StepTelemetry:
    if not meta:
        return StepTelemetry.empty()
    raw = meta.get("step_history")
    if not isinstance(raw, dict):
        return StepTelemetry.empty()

    z = np.asarray(raw.get("z", []), dtype=np.float64).reshape(-1)
    accepted = np.asarray(raw.get("step_size", []), dtype=np.float64).reshape(-1)
    next_sizes = np.asarray(raw.get("next_step_size", []), dtype=np.float64).reshape(-1)
    n = min(z.size, accepted.size, next_sizes.size)
    if n <= 0:
        dropped = int(raw.get("dropped", 0))
        return StepTelemetry.empty() if dropped <= 0 else StepTelemetry(
            accepted_z=np.empty(0, dtype=np.float64),
            accepted_step_sizes=np.empty(0, dtype=np.float64),
            next_z=np.empty(0, dtype=np.float64),
            next_step_sizes=np.empty(0, dtype=np.float64),
            dropped=dropped,
        )
    return StepTelemetry(
        accepted_z=z[:n],
        accepted_step_sizes=accepted[:n],
        next_z=z[:n],
        next_step_sizes=next_sizes[:n],
        dropped=int(raw.get("dropped", 0)),
    )


def filter_record_clipped_steps(
    telemetry: StepTelemetry,
    z_samples: np.ndarray,
) -> tuple[StepTelemetry, int]:
    z_axis = np.asarray(z_samples, dtype=np.float64).reshape(-1)
    n = int(telemetry.accepted_z.size)
    if n <= 0 or z_axis.size <= 1:
        return telemetry, 0

    z = np.asarray(telemetry.accepted_z, dtype=np.float64)
    accepted = np.asarray(telemetry.accepted_step_sizes, dtype=np.float64)
    proposed = np.asarray(telemetry.next_step_sizes, dtype=np.float64)
    if z.size != accepted.size or z.size != proposed.size:
        return telemetry, 0

    spacing = float((z_axis[-1] - z_axis[0]) / float(z_axis.size - 1))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return telemetry, 0

    z_end = float(z_axis[-1])
    expected_boundaries = np.asarray(z_axis[1:-1], dtype=np.float64)
    if expected_boundaries.size <= 0:
        return telemetry, 0

    proximity_eps = max(64.0 * np.finfo(np.float64).eps * max(1.0, abs(z_end)), spacing * 0.25)
    distance_to_expected = np.min(np.abs(z[:, None] - expected_boundaries[None, :]), axis=1)
    clipped = distance_to_expected <= proximity_eps

    if not np.any(clipped):
        return telemetry, 0

    keep = ~clipped
    filtered = StepTelemetry(
        accepted_z=z[keep],
        accepted_step_sizes=accepted[keep],
        next_z=np.asarray(telemetry.next_z, dtype=np.float64)[keep],
        next_step_sizes=proposed[keep],
        dropped=telemetry.dropped,
    )
    return filtered, int(np.count_nonzero(clipped))


def plot_wavelength_step_history(
    z_samples: np.ndarray,
    lambda_nm: np.ndarray,
    spectral_map: np.ndarray,
    telemetry: StepTelemetry,
    output_path: Path,
) -> Path | None:
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping wavelength + step-size plot.")
        return None

    z_axis = np.asarray(z_samples, dtype=np.float64)
    lambda_axis = np.asarray(lambda_nm, dtype=np.float64)
    data = np.asarray(spectral_map, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    if data.shape != (z_axis.size, lambda_axis.size):
        raise ValueError("spectral_map shape must be [record, wavelength].")

    peak = float(np.max(data))
    if peak > 0.0:
        data = data / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.8, 10.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.4], hspace=0.24)

    ax_map = fig.add_subplot(grid[0, 0])
    mesh = ax_map.pcolormesh(z_axis, lambda_axis, data.T, shading="auto", cmap="magma")
    ax_map.set_xlabel("Propagation distance z (m)")
    ax_map.set_ylabel("Wavelength (nm)")
    ax_map.set_title("Spectral Intensity Envelope vs Propagation Distance")
    ax_map.set_box_aspect(1.0)
    cbar = fig.colorbar(mesh, ax=ax_map, pad=0.02)
    cbar.set_label("Normalized spectral intensity")

    ax_step = fig.add_subplot(grid[1, 0])
    telemetry_plot, _ = filter_record_clipped_steps(telemetry, z_axis)
    has_series = False
    if telemetry_plot.accepted_z.size > 0:
        order = np.argsort(telemetry_plot.accepted_z)
        ax_step.plot(
            telemetry_plot.accepted_z[order],
            telemetry_plot.accepted_step_sizes[order],
            lw=1.2,
            color="tab:blue",
            label="Accepted step_size",
        )
        has_series = True

    if has_series:
        ax_step.set_xlabel("Propagation distance z (m)")
        ax_step.set_ylabel("Step size (m)")
        ax_step.set_title("Adaptive RK4IP Step Sizes")
        ax_step.grid(True, alpha=0.3)
        ax_step.legend()
    else:
        ax_step.text(
            0.5,
            0.5,
            "No adaptive step-adjustment events captured",
            transform=ax_step.transAxes,
            ha="center",
            va="center",
        )
        ax_step.set_xticks([])
        ax_step.set_yticks([])
        ax_step.set_title("Adaptive RK4IP Step Sizes")

    # Keep both panels at identical drawable width after colorbar shrinkage.
    map_pos = ax_map.get_position()
    step_pos = ax_step.get_position()
    ax_step.set_position([map_pos.x0, step_pos.y0, map_pos.width, step_pos.height])

    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_plots(
    t: np.ndarray,
    U_num: np.ndarray,
    U_true: np.ndarray,
    error_curve: np.ndarray,
    z_final: float,
    z_samples: np.ndarray,
    lambda_nm: np.ndarray,
    spectral_map: np.ndarray,
    telemetry: StepTelemetry,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    p1 = plot_wavelength_step_history(
        z_samples,
        lambda_nm,
        spectral_map,
        telemetry,
        output_dir / "wavelength_intensity_colormap.png",
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
    parser = build_example_parser(
        example_slug="second_order_soliton",
        description="Second-order soliton validation with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "second_order_soliton_rk4ip"
    case_key = "default"

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        meta = loaded.meta
        beta2 = float(meta["beta2"])
        gamma = float(meta["gamma"])
        alpha = float(meta["alpha"])
        t0 = float(meta["t0"])
        p0 = float(meta["p0"])
        n = int(meta["n"])
        dt = float(meta["dt"])
        z_final = float(meta["z_final"])
        T = centered_time_grid(n, dt)
        t = to_dimensionless_time(T, t0)
        z_records = np.asarray(loaded.z_axis, dtype=np.float64)
        A_records = np.asarray(loaded.records, dtype=np.complex128)
        step_history = db.load_step_history(run_id=loaded.run_id)
        if step_history is None:
            raise RuntimeError(
                "stored run is missing step telemetry; rerun without --replot to capture full-fidelity data."
            )
        telemetry = step_telemetry_from_meta({"step_history": step_history})
    else:
        run_group = db.begin_group(example_name, args.run_group)
        beta2 = -0.01
        gamma = 0.01
        alpha = 0.0
        tfwhm = 100e-3
        t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
        p0 = (2**2) * abs(beta2) / (gamma * t0 * t0)
        ld_tmp = (t0 * t0) / abs(beta2)
        z_final = 0.5 * math.pi * ld_tmp

        n = 2**10
        dt = (16.0 * t0) / n
        T = centered_time_grid(n, dt)
        t = to_dimensionless_time(T, t0)
        omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)
        U0 = 2.0 * sech(t)
        A0 = to_physical_envelope(U0, 0.0, p0, alpha)

        num_recorded_samples = 100
        sim_cfg = TemporalSimulationConfig(
            gamma=gamma,
            beta2=beta2,
            alpha=alpha,
            dt=dt,
            z_final=z_final,
            num_time_samples=n,
            pulse_period=n * dt,
            omega=omega,
            starting_step_size=z_final / 500.0,
            max_step_size=z_final / 25.0,
            min_step_size=z_final / 1000.0,
            error_tolerance=5e-6,
            honor_solver_controls=True,
        )
        exec_options = SimulationOptions(backend="auto", fft_backend="auto")
        runner = NloExampleRunner()
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=8,
        )
        z_records, A_records = runner.propagate_temporal_records(
            np.asarray(A0, dtype=np.complex128),
            sim_cfg,
            num_recorded_samples,
            exec_options,
            capture_step_history=True,
            step_history_capacity=200000,
            **storage_kwargs,
        )
        telemetry = step_telemetry_from_meta(runner.last_meta)
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            solver_meta=runner.last_meta,
            meta={
                "beta2": float(beta2),
                "gamma": float(gamma),
                "alpha": float(alpha),
                "t0": float(t0),
                "p0": float(p0),
                "n": int(n),
                "dt": float(dt),
                "z_final": float(z_final),
            },
            save_step_history=True,
        )

    sgn_beta2, ld, lnl = normalized_nlse_coefficients(beta2, gamma, t0, p0)
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
        fft_size_visual=4 * n,
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
        telemetry,
        output_dir,
    )

    print(f"second-order soliton summary (run_group={run_group})")
    print(
        "normalized NLSE coefficients: "
        f"sgn(beta2)={int(sgn_beta2):+d}, "
        f"1/(2*LD)={0.5 / ld:.6e} 1/m, "
        f"exp(-alpha*z_final)/LNL={math.exp(-alpha * z_final) / lnl:.6e} 1/m."
    )
    print(f"analytical z=0 envelope max error = {z0_analytic_error:.6e}")
    print(f"epsilon = {epsilon:.6e}")
    print(
        "step telemetry events: "
        f"accepted={telemetry.accepted_z.size}, "
        f"next={telemetry.next_z.size}, "
        f"dropped={telemetry.dropped}"
    )
    if saved_paths:
        print("saved plots:")
        for path in saved_paths:
            print(f"  {path}")

    return epsilon


if __name__ == "__main__":
    main()
