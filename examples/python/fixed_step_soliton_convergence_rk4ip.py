"""
Fixed-step soliton convergence sweep with analytical benchmark.

Runs full solver propagations for a set of fixed step sizes and plots the
total trajectory error against step size on log-log axes. The fitted slope is
computed on a coarse-to-mid step window to avoid roundoff-floor contamination.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from backend.storage import ExampleRunDB


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


def _total_trajectory_error(
    records: np.ndarray,
    z_axis: np.ndarray,
    tau: np.ndarray,
    p0: float,
    ld: float,
    num_records_common: int,
) -> float:
    m = min(int(num_records_common), int(records.shape[0]), int(z_axis.size))
    if m <= 0:
        return float("nan")
    amp0 = (math.sqrt(p0) / np.cosh(tau)).astype(np.complex128)
    z = np.asarray(z_axis[:m], dtype=np.float64)
    ref = amp0[None, :] * np.exp(0.5j * (z[:, None] / float(ld)))
    num = np.asarray(records[:m], dtype=np.complex128)
    return _relative_l2_error(num.reshape(-1), ref.reshape(-1))


def _step_count_from_case(case_key: str, meta: dict[str, object]) -> int:
    meta_steps = int(meta.get("step_count", 0))
    if meta_steps > 0:
        return meta_steps
    if case_key.startswith("steps_"):
        try:
            return int(case_key.split("_", 1)[1])
        except Exception:
            return 0
    return 0


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
    fit_line = np.exp(fitted_intercept) * (step_sizes_plot**fitted_order)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.loglog(step_sizes_plot, errors_plot, "o", lw=1.8, ms=3.0, label="Numerical error")
    ax.loglog(step_sizes_plot, fit_line, "--", lw=1.6, color="tab:green", label="Fitted power law")
    ax.loglog(step_sizes_plot, ref, "--", lw=1.5, label=r"Reference $O(\Delta z^4)$")
    ax.set_xlabel("Step size Delta z (m)")
    ax.set_ylabel("Total relative L2 error")
    ax.set_title(f"Fixed-Step Soliton Convergence (fitted order p = {fitted_order:.3f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.savefig(output_path, bbox_inches="tight")
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

    parser = build_example_parser(
        example_slug="fixed_step_soliton_convergence",
        description="Fixed-step soliton convergence sweep with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "fixed_step_soliton_convergence_rk4ip"

    step_counts_base = np.round(np.geomspace(16, 1024, 32), decimals=0).astype(int)

    runner = NloExampleRunner()
    _configure_runtime_logging(runner)
    exec_options = SimulationOptions(backend="auto", fft_backend="auto")

    run_data: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        cases = db.list_cases(example_name=example_name, run_group=run_group)
        if not cases:
            raise RuntimeError(f"no cases found in run_group '{run_group}'.")
        for case in cases:
            loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case.case_key)
            case_steps = _step_count_from_case(case.case_key, loaded.meta)
            case_dz = float(loaded.meta.get("step_size", 0.0))
            if case_dz <= 0.0 and case_steps > 0:
                case_dz = z_final / float(case_steps)
            if case_steps <= 0 or case_dz <= 0.0:
                continue
            run_data.append(
                (
                    int(case_steps),
                    float(case_dz),
                    np.asarray(loaded.z_axis, dtype=np.float64),
                    np.asarray(loaded.records, dtype=np.complex128),
                )
            )
        if len(run_data) < 2:
            raise RuntimeError("replot mode needs at least two valid stored step-size cases.")
    else:
        run_group = db.begin_group(example_name, args.run_group)
        target_records = int(np.min(step_counts_base)) + 1
        for step_count in step_counts_base.tolist():
            dz = z_final / float(step_count)
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
            case_key = f"steps_{int(step_count)}"
            storage_kwargs = db.storage_kwargs(
                example_name=example_name,
                run_group=run_group,
                case_key=case_key,
                chunk_records=2,
            )
            z_records, a_records = runner.propagate_temporal_records(
                a0,
                sim_cfg,
                num_records=target_records,
                exec_options=exec_options,
                **storage_kwargs,
            )
            db.save_case_from_solver_meta(
                example_name=example_name,
                run_group=run_group,
                case_key=case_key,
                solver_meta=runner.last_meta,
                meta={
                    "step_count": int(step_count),
                    "step_size": float(dz),
                    "target_records": int(target_records),
                },
            )
            run_data.append(
                (
                    int(step_count),
                    float(dz),
                    np.asarray(z_records, dtype=np.float64),
                    np.asarray(a_records, dtype=np.complex128),
                )
            )

    if len(run_data) < 2:
        raise RuntimeError("insufficient runs to compute convergence.")

    min_records_common = min(int(records.shape[0]) for _, _, _, records in run_data)
    step_counts = np.asarray([entry[0] for entry in run_data], dtype=int)
    step_sizes = np.asarray([entry[1] for entry in run_data], dtype=np.float64)
    fit_mask = step_counts <= 128
    errors = np.asarray(
        [
            _total_trajectory_error(records, z_axis, tau, p0, ld, min_records_common)
            for _, _, z_axis, records in run_data
        ],
        dtype=np.float64,
    )

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

    print(f"fixed-step soliton convergence summary (run_group={run_group}):")
    print(f"  total-error samples per run used = {min_records_common}")
    for k, dz, err in zip(step_counts.tolist(), step_sizes, errors):
        print(f"  steps={k:4d}  dz={dz:.6e}  total_error={err:.6e}")
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
