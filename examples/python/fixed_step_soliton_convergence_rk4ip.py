"""
Fixed-step soliton convergence sweep with internal numerical reference.

Runs full solver propagations for a set of fixed step sizes and plots the
total trajectory relative L2 error against step size on log-log axes. The
finest finite step case is used as the internal reference trajectory.
"""

from __future__ import annotations

import math

import numpy as np
from backend.app_base import ExampleAppBase
from backend.plotting import plot_convergence_loglog
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from backend.storage import ExampleRunDB
from backend.metrics import mean_pointwise_abs_relative_error


def _configure_runtime_logging(runner: NloExampleRunner) -> None:
    try:
        runner.api.set_log_level(runner.nlo.NLOLIB_LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        runner.api.set_progress_options(enabled=False, milestone_percent=5, emit_on_step_adjust=False)
    except Exception:
        pass


def _step_count_from_key(case_key: str) -> int:
    if not case_key.startswith("steps_"):
        return 0
    try:
        return int(case_key.split("_", 1)[1])
    except Exception:
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

    log_x = np.asarray(np.log(step_sizes[valid_mask]), dtype=np.float64)
    log_y = np.asarray(np.log(errors[valid_mask]), dtype=np.float64)
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    slope = float(slope)
    intercept = float(intercept)
    return slope, intercept, valid_mask


def _run(args) -> float:
    beta2 = -0.01
    gamma = 0.01
    alpha = 0.0
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    ld = (t0 * t0) / abs(beta2)
    z0 = 0.5 * math.pi * ld
    z_final = z0

    n = 2**9
    dt = (16.0 * t0) / n
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)
    tau = centered_time_grid(n, dt) / t0
    p0 = abs(beta2) / (gamma * t0 * t0)
    a0 = (math.sqrt(p0) / np.cosh(tau)).astype(np.complex128)

    db = ExampleRunDB(args.db_path)
    example_name = "fixed_step_soliton_convergence_rk4ip"

    base_record_intervals = 16
    step_counts_base = np.geomspace(base_record_intervals, 1024 + 1, base_record_intervals, dtype=int)
    target_records = base_record_intervals + 1

    runner = NloExampleRunner()
    _configure_runtime_logging(runner)
    exec_options = SimulationOptions(backend="auto", fft_backend="auto")

    run_data: list[tuple[int, float, np.ndarray]] = []
    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        cases = db.list_cases(example_name=example_name, run_group=run_group)
        if not cases:
            raise RuntimeError(f"no cases found in run_group '{run_group}'.")
        for case in cases:
            loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case.case_key)
            case_steps = int(loaded.meta.get("step_count", 0))
            if case_steps <= 0:
                case_steps = _step_count_from_key(case.case_key)
            case_dz = (float(loaded.z_end) / float(case_steps)) if case_steps > 0 else 0.0
            if case_steps <= 0 or case_dz <= 0.0:
                continue
            run_data.append(
                (
                    int(case_steps),
                    float(case_dz),
                    np.asarray(loaded.records, dtype=np.complex128),
                )
            )
        if len(run_data) < 2:
            raise RuntimeError("replot mode needs at least two valid stored step-size cases.")
    else:
        run_group = db.begin_group(example_name, args.run_group)
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
            _, a_records = runner.propagate_temporal_records(
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
                save_step_history=False,
            )
            run_data.append(
                (
                    int(step_count),
                    float(dz),
                    np.asarray(a_records, dtype=np.complex128),
                )
            )

    if len(run_data) < 2:
        raise RuntimeError("insufficient runs to compute convergence.")

    run_data.sort(key=lambda row: int(row[0]))
    min_records_common = min(int(records.shape[0]) for _, _, records in run_data)
    step_counts = np.asarray([entry[0] for entry in run_data], dtype=int)
    step_sizes = np.asarray([entry[1] for entry in run_data], dtype=np.float64)
    step_sizes_norm = step_sizes / float(z0)
    fit_mask = step_counts <= 128
    records_finite = np.asarray(
        [bool(np.all(np.isfinite(records[: min_records_common]))) for _, _, records in run_data],
        dtype=bool,
    )
    finite_indices = np.flatnonzero(records_finite)
    if finite_indices.size <= 0:
        raise RuntimeError("all runs diverged; cannot compute convergence error.")
    ref_idx = int(finite_indices[-1])
    reference_records = np.asarray(run_data[ref_idx][2][:min_records_common], dtype=np.complex128)
    errors = np.asarray(
        [mean_pointwise_abs_relative_error(records, reference_records, context="fixed_step_soliton_convergence:record_error") for _, _, records in run_data],
        dtype=np.float64,
    )

    fitted_order, fitted_intercept, fit_mask_valid = _fit_loglog_slope(step_sizes_norm, errors, fit_mask)

    output_dir = args.output_dir
    plot_convergence_loglog(
        step_sizes_norm,
        errors,
        fit_mask_valid,
        fitted_order,
        fitted_intercept,
        output_dir / "error_vs_fixed_step_size.png",
        x_label="Normalized step size Delta z / Z0",
    )

    print(f"fixed-step soliton convergence summary (run_group={run_group}):")
    print(f"  total-error samples per run used = {min_records_common}")
    print(f"  reference case (internal) = steps_{step_counts[ref_idx]}")
    for k, dz, dz_norm, finite_records, err in zip(
        step_counts.tolist(),
        step_sizes,
        step_sizes_norm,
        records_finite.tolist(),
        errors.tolist(),
    ):
        status = "finite" if finite_records else "diverged"
        err_text = f"{err:.6e}" if math.isfinite(err) else "nan"
        print(
            f"  steps={k:4d}  dz={dz:.6e}  dz_over_z0={dz_norm:.6e}  "
            f"status={status}  total_error={err_text}"
        )
    fit_counts = step_counts[fit_mask_valid].tolist()
    print(f"fit window step counts = {fit_counts}")
    print(f"fitted order p = {fitted_order:.6f}")
    print(f"fitted line: error ~= exp({fitted_intercept:.6f}) * (Delta z / Z0)^{fitted_order:.6f}")
    print("expected RK4 scaling reference: O((Delta z / Z0)^4)")
    if fitted_order < 3.5:
        print("warning: observed order is below 4th-order expectation for this coupled soliton benchmark.")
        print("         this suggests a coupled-case integrator limitation rather than a plotting issue.")

    return fitted_order


class FixedStepSolitonConvergenceApp(ExampleAppBase):
    example_slug = "fixed_step_soliton_convergence"
    description = "Fixed-step soliton convergence sweep with DB-backed run/replot."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return FixedStepSolitonConvergenceApp.from_cli().run()


if __name__ == "__main__":
    main()
