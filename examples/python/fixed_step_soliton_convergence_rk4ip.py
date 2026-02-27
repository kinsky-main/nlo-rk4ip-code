"""
Fixed-step soliton convergence sweep with analytical benchmark.

Runs full solver propagations for a set of fixed step sizes and plots the
final error against step size on log-log axes. The fitted slope is computed
on a coarse-to-mid step window to avoid roundoff-floor contamination.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)


def _load_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _configure_runtime_logging(runner: NloExampleRunner) -> None:
    try:
        runner.api.set_log_level(runner.nlo.NLOLIB_LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        runner.api.set_progress_options(enabled=False, milestone_percent=5, emit_on_step_adjust=False)
    except Exception:
        pass


def _relative_l2_error(num: np.ndarray, ref: np.ndarray) -> float:
    ref_norm = float(np.linalg.norm(ref))
    denom = max(ref_norm, 1e-15)
    return float(np.linalg.norm(num - ref) / denom)


def _fit_loglog_slope(
    step_sizes: np.ndarray,
    errors: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    valid_mask = (
        np.asarray(fit_mask, dtype=bool)
        & np.isfinite(step_sizes)
        & np.isfinite(errors)
        & (step_sizes > 0.0)
        & (errors > 0.0)
    )
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < 2:
        raise RuntimeError(
            "insufficient valid points for convergence slope fit; "
            f"valid_points={valid_count}. "
            "Check for diverged runs (non-finite errors) or zero/negative fit values."
        )

    log_x = np.log(step_sizes[valid_mask])
    log_y = np.log(errors[valid_mask])
    mean_x = float(np.mean(log_x))
    mean_y = float(np.mean(log_y))
    dx = log_x - mean_x
    dy = log_y - mean_y
    var_x = float(np.dot(dx, dx))
    if var_x <= 0.0:
        raise RuntimeError("cannot fit convergence slope: zero variance in log(step_sizes).")
    slope = float(np.dot(dx, dy) / var_x)
    intercept = mean_y - (slope * mean_x)
    return slope, intercept, valid_mask


def _save_convergence_plot(
    output_path: Path,
    step_sizes: np.ndarray,
    errors: np.ndarray,
    fit_mask: np.ndarray,
    fitted_order: float,
    fitted_intercept: float,
) -> Path | None:
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping convergence plot.")
        return None

    order = np.argsort(step_sizes)
    step_sizes_plot = step_sizes[order]
    errors_plot = errors[order]
    fit_mask_plot = fit_mask[order]

    fit_indices = np.flatnonzero(fit_mask_plot)
    anchor = int(fit_indices[0]) if fit_indices.size > 0 else 0
    ref = errors_plot[anchor] * (step_sizes_plot / step_sizes_plot[anchor]) ** 4
    fit_line = np.exp(fitted_intercept) * (step_sizes_plot ** fitted_order)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.loglog(step_sizes_plot, errors_plot, "o", lw=1.8, ms=3.0, label="Numerical error")
    ax.loglog(step_sizes_plot, fit_line, "--", lw=1.6, color="tab:green", label="Fitted power law")
    ax.loglog(step_sizes_plot, ref, "--", lw=1.5, label=r"Reference $O(\Delta z^4)$")
    ax.loglog(
        step_sizes_plot[fit_mask_plot],
        errors_plot[fit_mask_plot],
        "o",
        ms=8.0,
        markerfacecolor="none",
        markeredgewidth=1.2,
        label="Fit window",
    )
    ax.set_xlabel("Step size Delta z (m)")
    ax.set_ylabel("Final relative L2 error")
    ax.set_title(f"Fixed-Step Soliton Convergence (fitted order p = {fitted_order:.3f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> float:
    beta2 = -0.01
    gamma = 0.01
    alpha = 0.0
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    p0 = abs(beta2) / (gamma * t0 * t0)
    ld = (t0 * t0) / abs(beta2)
    z_final = 10.0

    n = 2**9
    dt = (16.0 * t0) / n
    t_axis = centered_time_grid(n, dt)
    tau = t_axis / t0
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)
    a0 = (math.sqrt(p0) / np.cosh(tau)).astype(np.complex128)

    step_counts = np.round(np.geomspace(16, 1024, 32), decimals=0).astype(int)
    step_sizes = np.asarray([z_final / float(k) for k in step_counts], dtype=np.float64)
    fit_mask = step_counts <= 128

    runner = NloExampleRunner()
    _configure_runtime_logging(runner)
    exec_options = SimulationOptions(backend="auto", fft_backend="auto")

    errors = np.empty(step_sizes.size, dtype=np.float64)
    for i, dz in enumerate(step_sizes):
        sim_cfg = TemporalSimulationConfig(
            gamma=gamma,
            beta2=beta2,
            alpha=alpha,
            dt=dt,
            z_final=z_final,
            num_time_samples=n,
            pulse_period=n * dt,
            omega=omega,
            starting_step_size=dz,
            max_step_size=dz,
            min_step_size=dz,
            error_tolerance=1e-6,
            honor_solver_controls=True,
        )
        z_records, a_records = runner.propagate_temporal_records(
            a0,
            sim_cfg,
            num_records=2,
            exec_options=exec_options,
        )
        z_end = float(z_records[-1])
        a_num = np.asarray(a_records[-1], dtype=np.complex128)
        a_true = (math.sqrt(p0) / np.cosh(tau)) * np.exp(0.5j * (z_end / ld))
        errors[i] = _relative_l2_error(a_num, a_true)

    fitted_order, fitted_intercept, fit_mask_valid = _fit_loglog_slope(step_sizes, errors, fit_mask)

    output_dir = Path(__file__).resolve().parent / "output" / "fixed_step_soliton_convergence"
    plot_path = _save_convergence_plot(
        output_dir / "error_vs_step_size.png",
        step_sizes,
        errors,
        fit_mask_valid,
        fitted_order,
        fitted_intercept,
    )

    print("fixed-step soliton convergence summary:")
    for k, dz, err in zip(step_counts.tolist(), step_sizes, errors):
        print(f"  steps={k:4d}  dz={dz:.6e}  error={err:.6e}")
    fit_counts = step_counts[fit_mask_valid].tolist()
    print(f"fit window step counts = {fit_counts}")
    print(f"fitted order p = {fitted_order:.6f}")
    print(f"fitted line: error ~= exp({fitted_intercept:.6f}) * (Delta z)^{fitted_order:.6f}")
    print("expected RK4 scaling reference: O(Delta z^4)")
    if fitted_order < 3.5:
        print("warning: observed order is below 4th-order expectation for this coupled soliton benchmark.")
        print("         this suggests a coupled-case integrator limitation rather than a plotting issue.")
    if plot_path is not None:
        print(f"saved plot: {plot_path}")

    return fitted_order


if __name__ == "__main__":
    main()
