"""
Four-panel propagation comparison for pure nonlinearity (SPM) and drifted dispersion.

The generated figure is arranged as a 2x2 grid:
  - left column: frequency-domain intensity propagation
  - right column: time-domain intensity propagation
  - top row: pure Kerr nonlinearity / SPM
  - bottom row: pure dispersion with linear drift
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from backend.app_base import ExampleAppBase
from backend.plotting import plot_frequency_time_propagation_grid
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from backend.storage import ExampleRunDB


@dataclass(frozen=True)
class CaseConfig:
    case_key: str
    gamma: float
    beta2: float
    chirp: float = 0.0


def _configure_runtime_logging(runner: NloExampleRunner) -> None:
    try:
        runner.api.set_log_level(runner.nlo.NLOLIB_LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        runner.api.set_progress_options(enabled=False, milestone_percent=5, emit_on_step_adjust=False)
    except Exception:
        pass


def _row_label(case: CaseConfig, *, chirp: float) -> str:
    if case.case_key == "dispersion" and abs(chirp) > 0.0:
        return "Pure dispersion with linear drift"
    if case.case_key == "spm":
        return "Pure nonlinearity (SPM)"
    return "Pure dispersion"


def _initial_field(time_axis: np.ndarray, pulse_width: float, *, chirp: float = 0.0) -> np.ndarray:
    envelope = np.exp(-((time_axis / pulse_width) ** 2))
    if chirp == 0.0:
        return envelope.astype(np.complex128)
    return (envelope * np.exp((-1.0j) * chirp * time_axis)).astype(np.complex128)


def _row_annotation(case: CaseConfig, *, chirp: float) -> str:
    if case.case_key == "dispersion" and abs(chirp) > 0.0:
        return "Dispersion"
    if case.case_key == "spm":
        return "SPM"
    return "Dispersion"


def _frequency_intensity_map(records: np.ndarray) -> np.ndarray:
    spectra = np.fft.fftshift(np.fft.fft(records, axis=1), axes=1)
    return np.abs(spectra) ** 2


def _run_case(
    *,
    db: ExampleRunDB,
    runner: NloExampleRunner,
    args: argparse.Namespace,
    example_name: str,
    run_group: str,
    case: CaseConfig,
    num_samples: int,
    dt: float,
    pulse_width: float,
    z_final: float,
    num_records: int,
    exec_options: SimulationOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if args.replot:
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case.case_key)
        meta = loaded.meta
        num_samples = int(meta["num_samples"])
        dt = float(meta["dt"])
        pulse_width = float(meta["pulse_width"])
        chirp = float(meta.get("chirp", case.chirp))
        time_axis = centered_time_grid(num_samples, dt)
        return (
            time_axis,
            np.asarray(loaded.z_axis, dtype=np.float64),
            np.asarray(loaded.records, dtype=np.complex128),
            chirp,
        )

    time_axis = centered_time_grid(num_samples, dt)
    field0 = _initial_field(time_axis, pulse_width, chirp=case.chirp)
    sim_cfg = TemporalSimulationConfig(
        gamma=case.gamma,
        beta2=case.beta2,
        alpha=0.0,
        dt=dt,
        z_final=z_final,
        num_time_samples=num_samples,
        pulse_period=float(num_samples) * dt,
        omega=None,
        starting_step_size=1e-5,
        max_step_size=1e-2,
        min_step_size=1e-8,
        error_tolerance=1e-8,
        honor_solver_controls=True,
    )
    storage_kwargs = db.storage_kwargs(
        example_name=example_name,
        run_group=run_group,
        case_key=case.case_key,
        chunk_records=8,
    )
    z_axis, records = runner.propagate_temporal_records(
        field0,
        sim_cfg,
        num_records=num_records,
        exec_options=exec_options,
        **storage_kwargs,
    )
    db.save_case_from_solver_meta(
        example_name=example_name,
        run_group=run_group,
        case_key=case.case_key,
        solver_meta=runner.last_meta,
        meta={
            "num_samples": int(num_samples),
            "dt": float(dt),
            "pulse_width": float(pulse_width),
            "z_final": float(z_final),
            "gamma": float(case.gamma),
            "beta2": float(case.beta2),
            "chirp": float(case.chirp),
            "num_records": int(num_records),
        },
    )
    return time_axis, np.asarray(z_axis, dtype=np.float64), np.asarray(records, dtype=np.complex128), float(case.chirp)


def _run(args: argparse.Namespace) -> float:
    db = ExampleRunDB(args.db_path)
    example_name = "spm_dispersion_comparison_rk4ip"
    cases = (
        CaseConfig(
            case_key="spm",
            gamma=80.0,
            beta2=0.0,
        ),
        CaseConfig(
            case_key="dispersion",
            gamma=0.0,
            beta2=0.20,
            chirp=12.0,
        ),
    )

    num_samples = 2**12
    dt = 0.01
    pulse_width = 0.50
    z_final = 0.8
    num_records = 120
    exec_options = SimulationOptions(backend="auto", fft_backend="auto")

    runner = NloExampleRunner()
    _configure_runtime_logging(runner)

    if args.replot:
        run_group = db.resolve_replot_group(
            example_name,
            args.run_group,
            required_case_keys=[case.case_key for case in cases],
        )
    else:
        run_group = db.begin_group(example_name, args.run_group)

    case_outputs: dict[str, dict[str, np.ndarray | str]] = {}
    for case in cases:
        time_axis, z_axis, records, chirp = _run_case(
            db=db,
            runner=runner,
            args=args,
            example_name=example_name,
            run_group=run_group,
            case=case,
            num_samples=num_samples,
            dt=dt,
            pulse_width=pulse_width,
            z_final=z_final,
            num_records=num_records,
            exec_options=exec_options,
        )
        case_outputs[case.case_key] = {
            "row_label": _row_label(case, chirp=chirp),
            "annotation": _row_annotation(case, chirp=chirp),
            "time_axis": time_axis,
            "z_axis": z_axis,
            "time_map": np.abs(records) ** 2,
            "frequency_map": _frequency_intensity_map(records),
        }

    spm_output = case_outputs["spm"]
    dispersion_output = case_outputs["dispersion"]
    time_axis = np.asarray(spm_output["time_axis"], dtype=np.float64)
    z_axis = np.asarray(spm_output["z_axis"], dtype=np.float64)
    if not np.allclose(z_axis, np.asarray(dispersion_output["z_axis"], dtype=np.float64), rtol=0.0, atol=1e-12):
        raise RuntimeError("SPM and dispersion cases produced incompatible propagation axes.")

    frequency_axis = np.fft.fftshift(np.fft.fftfreq(time_axis.size, d=float(time_axis[1] - time_axis[0])))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = plot_frequency_time_propagation_grid(
        frequency_axis,
        time_axis,
        z_axis,
        np.asarray(spm_output["frequency_map"], dtype=np.float64),
        np.asarray(spm_output["time_map"], dtype=np.float64),
        np.asarray(dispersion_output["frequency_map"], dtype=np.float64),
        np.asarray(dispersion_output["time_map"], dtype=np.float64),
        output_dir / "spm_dispersion_frequency_time_propagation.png",
        upper_row_label=str(spm_output["row_label"]),
        lower_row_label=str(dispersion_output["row_label"]),
        upper_left_annotation="SPM",
        lower_left_annotation=str(dispersion_output["annotation"]),
    )

    print(f"SPM/dispersion comparison example completed (run_group={run_group}).")
    if saved is not None:
        print(f"Saved comparison figure: {saved}")
    print(f"SPM peak spectral intensity   = {float(np.max(spm_output['frequency_map'])):.6e}")
    print(f"SPM peak temporal intensity   = {float(np.max(spm_output['time_map'])):.6e}")
    print(f"Dispersion peak spectral intensity = {float(np.max(dispersion_output['frequency_map'])):.6e}")
    print(f"Dispersion peak temporal intensity = {float(np.max(dispersion_output['time_map'])):.6e}")

    return 0.0


class SpmDispersionComparisonApp(ExampleAppBase):
    example_slug = "spm_dispersion_comparison"
    description = "Four-panel SPM vs drifted-dispersion propagation comparison."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return SpmDispersionComparisonApp.from_cli().run()


if __name__ == "__main__":
    main()
