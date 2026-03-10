"""
Fixed-step soliton convergence sweep with analytical benchmark.

Runs full solver propagations for a set of fixed step sizes and plots the
final-field relative L2 error against step size on log-log axes. Errors are
computed against the analytical fundamental-soliton solution and use a
reference-intensity floor to suppress near-zero tail noise without assuming a
Gaussian pulse shape.
"""

from __future__ import annotations

import math

import numpy as np
from backend.app_base import ExampleAppBase
from backend.metrics import filtered_relative_l2_error
from backend.plotting import plot_convergence_loglog
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from backend.storage import ExampleRunDB


INTENSITY_FILTER_RATIO = 1.0e-8
NUM_TIME_SAMPLES = 2**10
WINDOW_MULTIPLE_T0 = 64.0
FIXED_STEP_COUNTS = np.asarray([16, 32, 64, 128, 256, 512, 1024, 2048], dtype=int)
ADAPTIVE_TOLERANCES = np.asarray([1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13, 1.0e-14, 1.0e-15], dtype=np.float64)


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


def _tolerance_from_key(case_key: str) -> float:
    if not case_key.startswith("adaptive_tol_"):
        return 0.0
    try:
        return float(case_key.split("_", 2)[2])
    except Exception:
        return 0.0


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


def _fundamental_soliton_reference(
    tau: np.ndarray,
    z_axis: np.ndarray,
    p0: float,
    ld: float,
) -> np.ndarray:
    envelope = (math.sqrt(p0) / np.cosh(tau)).astype(np.complex128)
    phase = np.exp(0.5j * (np.asarray(z_axis, dtype=np.float64) / float(ld)))
    return phase[:, None] * envelope[None, :]


def _final_error_against_reference(
    records: np.ndarray,
    reference_final: np.ndarray,
) -> float:
    return filtered_relative_l2_error(
        np.asarray(records, dtype=np.complex128)[-1],
        reference_final,
        min_relative_intensity=INTENSITY_FILTER_RATIO,
    )


def _run(args) -> float:
    beta2 = -0.01
    gamma = 0.01
    alpha = 0.0
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    ld = (t0 * t0) / abs(beta2)
    z0 = 0.5 * math.pi * ld
    z_final = z0

    # A narrow periodic window creates a benchmark-side floor in the fine regime
    # before the RK4IP truncation error is visible. Keep the time window wide
    # enough that the final-field benchmark measures the solver, not wraparound.
    n = NUM_TIME_SAMPLES
    dt = (WINDOW_MULTIPLE_T0 * t0) / n
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)
    tau = centered_time_grid(n, dt) / t0
    p0 = abs(beta2) / (gamma * t0 * t0)
    a0 = (math.sqrt(p0) / np.cosh(tau)).astype(np.complex128)

    db = ExampleRunDB(args.db_path)
    example_name = "fixed_step_soliton_convergence_rk4ip"

    target_records = 2

    runner = NloExampleRunner()
    _configure_runtime_logging(runner)
    exec_options = SimulationOptions(backend="cpu", fft_backend="fftw")

    fixed_run_data: list[tuple[int, float, np.ndarray]] = []
    adaptive_run_data: list[tuple[float, np.ndarray]] = []
    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        cases = db.list_cases(example_name=example_name, run_group=run_group)
        if not cases:
            raise RuntimeError(f"no cases found in run_group '{run_group}'.")
        for case in cases:
            loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case.case_key)
            mode = str(loaded.meta.get("mode", "")).strip().lower()
            if mode == "adaptive":
                tol = float(loaded.meta.get("error_tolerance", 0.0))
                if tol <= 0.0:
                    tol = _tolerance_from_key(case.case_key)
                if tol > 0.0:
                    adaptive_run_data.append((tol, np.asarray(loaded.records, dtype=np.complex128)))
                continue

            case_steps = int(loaded.meta.get("step_count", 0))
            if case_steps <= 0:
                case_steps = _step_count_from_key(case.case_key)
            case_dz = float(loaded.meta.get("step_size", 0.0))
            if case_dz <= 0.0 and case_steps > 0:
                case_dz = float(loaded.z_end) / float(case_steps)
            if case_steps > 0 and case_dz > 0.0:
                fixed_run_data.append(
                    (
                        int(case_steps),
                        float(case_dz),
                        np.asarray(loaded.records, dtype=np.complex128),
                    )
                )
        if len(fixed_run_data) < 2:
            raise RuntimeError("replot mode needs at least two valid fixed-step cases.")
    else:
        run_group = db.begin_group(example_name, args.run_group)
        for step_count in FIXED_STEP_COUNTS.tolist():
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
                    "mode": "fixed",
                    "step_count": int(step_count),
                    "step_size": float(dz),
                    "intensity_filter_ratio": float(INTENSITY_FILTER_RATIO),
                },
                save_step_history=False,
            )
            fixed_run_data.append(
                (
                    int(step_count),
                    float(dz),
                    np.asarray(a_records, dtype=np.complex128),
                )
            )

        adaptive_start_step = z_final / 64.0
        adaptive_max_step = z_final / 4.0
        adaptive_min_step = z_final / 4096.0
        for tolerance in ADAPTIVE_TOLERANCES.tolist():
            sim_cfg = TemporalSimulationConfig(
                gamma=gamma,
                beta2=beta2,
                alpha=alpha,
                dt=dt,
                z_final=z_final,
                num_time_samples=n,
                pulse_period=n * dt,
                omega=omega,
                starting_step_size=adaptive_start_step,
                max_step_size=adaptive_max_step,
                min_step_size=adaptive_min_step,
                error_tolerance=float(tolerance),
                honor_solver_controls=True,
            )
            case_key = f"adaptive_tol_{float(tolerance):.1e}"
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
                    "mode": "adaptive",
                    "error_tolerance": float(tolerance),
                    "starting_step_size": float(adaptive_start_step),
                    "max_step_size": float(adaptive_max_step),
                    "min_step_size": float(adaptive_min_step),
                    "intensity_filter_ratio": float(INTENSITY_FILTER_RATIO),
                },
                save_step_history=False,
            )
            adaptive_run_data.append((float(tolerance), np.asarray(a_records, dtype=np.complex128)))

    if len(fixed_run_data) < 2:
        raise RuntimeError("insufficient fixed-step runs to compute convergence.")
    if len(adaptive_run_data) < 2:
        raise RuntimeError("insufficient adaptive-tolerance runs to compute convergence.")

    fixed_run_data.sort(key=lambda row: int(row[0]))
    adaptive_run_data.sort(key=lambda row: float(row[0]), reverse=True)
    records_by_tolerance = {
        float(tolerance): np.asarray(records, dtype=np.complex128)
        for tolerance, records in adaptive_run_data
    }
    step_counts = np.asarray([entry[0] for entry in fixed_run_data], dtype=int)
    step_sizes = np.asarray([entry[1] for entry in fixed_run_data], dtype=np.float64)
    step_sizes_norm = step_sizes / float(z0)

    records_finite = np.asarray(
        [bool(np.all(np.isfinite(records))) for _, _, records in fixed_run_data],
        dtype=bool,
    )
    if not bool(np.any(records_finite)):
        raise RuntimeError("all runs diverged; cannot compute convergence error.")

    reference_final = _fundamental_soliton_reference(
        tau,
        np.asarray([z_final], dtype=np.float64),
        p0,
        ld,
    )[0]
    fixed_errors_analytic = np.asarray(
        [_final_error_against_reference(records, reference_final) for _, _, records in fixed_run_data],
        dtype=np.float64,
    )
    fixed_fit_mask = np.asarray(
        np.isfinite(step_sizes_norm) & np.isfinite(fixed_errors_analytic) & (step_sizes_norm > 0.0) & (fixed_errors_analytic > 0.0),
        dtype=bool,
    )
    fixed_fitted_order, fixed_fitted_intercept, fixed_fit_mask_valid = _fit_loglog_slope(
        step_sizes_norm,
        fixed_errors_analytic,
        fixed_fit_mask,
    )

    tolerances = np.asarray([entry[0] for entry in adaptive_run_data], dtype=np.float64)
    adaptive_errors_analytic = np.asarray(
        [_final_error_against_reference(records, reference_final) for _, records in adaptive_run_data],
        dtype=np.float64,
    )
    adaptive_fit_mask = np.asarray(
        np.isfinite(tolerances) & np.isfinite(adaptive_errors_analytic) & (tolerances > 0.0) & (adaptive_errors_analytic > 0.0),
        dtype=bool,
    )
    adaptive_fitted_order, adaptive_fitted_intercept, adaptive_fit_mask_valid = _fit_loglog_slope(
        tolerances,
        adaptive_errors_analytic,
        adaptive_fit_mask,
    )

    output_dir = args.output_dir
    plot_convergence_loglog(
        step_sizes_norm,
        fixed_errors_analytic,
        fixed_fit_mask_valid,
        fixed_fitted_order,
        fixed_fitted_intercept,
        output_dir / "error_vs_fixed_step_size.png",
        x_label="Normalized step size Delta z / Z0",
        y_label="Filtered relative L2 final-field error",
    )
    plot_convergence_loglog(
        tolerances,
        adaptive_errors_analytic,
        adaptive_fit_mask_valid,
        adaptive_fitted_order,
        adaptive_fitted_intercept,
        output_dir / "error_vs_adaptive_tolerance.png",
        x_label="Adaptive relative local error tolerance",
        y_label="Filtered relative L2 final-field error",
        reference_order=1.0,
    )

    print(f"fixed-step soliton convergence summary (run_group={run_group}):")
    print("  benchmark = corrected analytical fundamental soliton final field")
    print(f"  records per run = {target_records}")
    print(f"  intensity_filter_ratio = {INTENSITY_FILTER_RATIO:.1e}")
    print(f"  num_time_samples = {n}")
    print(f"  time_window = {WINDOW_MULTIPLE_T0:.1f} * T0")
    for k, dz, dz_norm, finite_records, final_err in zip(
        step_counts.tolist(),
        step_sizes,
        step_sizes_norm,
        records_finite.tolist(),
        fixed_errors_analytic.tolist(),
    ):
        status = "finite" if finite_records else "diverged"
        final_text = f"{final_err:.6e}" if math.isfinite(final_err) else "nan"
        print(
            f"  steps={k:4d}  dz={dz:.6e}  dz_over_z0={dz_norm:.6e}  "
            f"status={status}  solver_vs_analytical_error={final_text}"
        )
    fixed_fit_counts = step_counts[fixed_fit_mask_valid].tolist()
    print(f"fixed-step fit window step counts = {fixed_fit_counts}")
    print(f"fixed-step fitted order vs analytical p = {fixed_fitted_order:.6f}")
    print(
        f"fixed-step fitted line: error ~= exp({fixed_fitted_intercept:.6f}) * "
        f"(Delta z / Z0)^{fixed_fitted_order:.6f}"
    )
    print("expected fixed-step RK4 scaling reference: O((Delta z / Z0)^4)")
    if fixed_fitted_order < 3.0:
        print("warning: observed order is below 4th-order expectation for this fixed-step soliton benchmark.")

    print("adaptive tolerance sweep summary:")
    for tolerance in tolerances.tolist():
        analytical_err = _final_error_against_reference(records_by_tolerance[float(tolerance)], reference_final)
        print(
            f"  tol={tolerance:.1e}  solver_vs_analytical_error={analytical_err:.6e}"
        )
    adaptive_fit_tolerances = tolerances[adaptive_fit_mask_valid].tolist()
    print(f"adaptive fit window tolerances = {[f'{v:.1e}' for v in adaptive_fit_tolerances]}")
    print(f"adaptive fitted order vs analytical p = {adaptive_fitted_order:.6f}")
    print(
        f"adaptive fitted line: error ~= exp({adaptive_fitted_intercept:.6f}) * "
        f"tol^{adaptive_fitted_order:.6f}"
    )
    print("expected adaptive scaling reference: O(tol)")

    return fixed_fitted_order


class FixedStepSolitonConvergenceApp(ExampleAppBase):
    example_slug = "fixed_step_soliton_convergence"
    description = "Fixed-step soliton convergence sweep with DB-backed run/replot."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return FixedStepSolitonConvergenceApp.from_cli().run()


if __name__ == "__main__":
    main()
