"""
Self-phase modulation (SPM) analytical validation example.

For pure Kerr nonlinearity (no dispersion/loss), compares numerical propagation
against the closed-form SPM phase evolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.plotting import (
    plot_intensity_colormap_vs_propagation,
    plot_phase_shift_comparison,
    plot_total_error_over_propagation,
)
from backend.runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from backend.storage import ExampleRunDB


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

def main() -> float:
    parser = build_example_parser(
        example_slug="spm",
        description="Self-phase modulation validation with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "spm_rk4ip"
    case_key = "default"

    n = 2**10
    dt = 0.01
    gamma = 80.0
    z_final = 0.5
    pulse_width = 0.2
    t = centered_time_grid(n, dt)
    # Purely temporal Gaussian pulse.
    a0 = np.exp(-((t / pulse_width) ** 2)).astype(np.complex128)
    omega = np.zeros(n, dtype=np.float64)

    runner = NloExampleRunner()
    _configure_runtime_logging(runner)
    exec_options = SimulationOptions(backend="auto", fft_backend="auto")

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        meta = loaded.meta
        n = int(meta["n"])
        dt = float(meta["dt"])
        gamma = float(meta["gamma"])
        z_final = float(meta["z_final"])
        pulse_width = float(meta["pulse_width"])
        t = centered_time_grid(n, dt)
        a0 = np.exp(-((t / pulse_width) ** 2)).astype(np.complex128)
        records = np.asarray(loaded.records, dtype=np.complex128)
        z_axis = np.asarray(loaded.z_axis, dtype=np.float64)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        sim_cfg = TemporalSimulationConfig(
            gamma=gamma,
            beta2=0.0,
            alpha=0.0,
            dt=dt,
            z_final=z_final,
            num_time_samples=n,
            pulse_period=n * dt,
            omega=omega,
            starting_step_size=1e-3,
            max_step_size=1e-3,
            min_step_size=1e-3,
            error_tolerance=1e-6,
            honor_solver_controls=True,
        )
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=8,
        )
        z_records, a_records = runner.propagate_temporal_records(
            a0,
            sim_cfg,
            num_records=100,
            exec_options=exec_options,
            **storage_kwargs,
        )
        records = np.asarray(a_records, dtype=np.complex128)
        z_axis = np.asarray(z_records, dtype=np.float64)
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            solver_meta=runner.last_meta,
            meta={
                "n": int(n),
                "dt": float(dt),
                "gamma": float(gamma),
                "z_final": float(z_final),
                "pulse_width": float(pulse_width),
            },
        )

    error_curve = np.empty(z_axis.size, dtype=np.float64)
    for i, z in enumerate(z_axis):
        a_ref = a0 * np.exp(1j * gamma * (np.abs(a0) ** 2) * float(z))
        error_curve[i] = _relative_l2_error(records[i], a_ref)

    a_final = records[-1]
    final_error = error_curve[-1]

    # Phase shift relative to initial phase.
    phase_num = np.angle(a_final * np.conj(a0))
    phase_ref = gamma * (np.abs(a0) ** 2) * z_final
    phase_error_wrapped = np.angle(np.exp(1j * (phase_num - phase_ref)))
    intensity0 = np.abs(a0) ** 2
    support_mask = intensity0 >= (1e-4 * float(np.max(intensity0)))
    phase_error_abs = np.abs(phase_error_wrapped)
    max_wrapped_phase_error = float(np.max(phase_error_abs))
    max_wrapped_phase_error_support = float(np.max(phase_error_abs[support_mask]))
    rms_wrapped_phase_error_support = float(np.sqrt(np.mean((phase_error_wrapped[support_mask]) ** 2)))

    power0 = float(np.sum(np.abs(a0) ** 2))
    power1 = float(np.sum(np.abs(a_final) ** 2))
    power_drift = abs(power1 - power0) / max(power0, 1e-15)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    time_intensity_map = np.abs(records) ** 2
    spectra = np.fft.fftshift(np.fft.fft(records, axis=1), axes=1)
    freq_intensity_map = np.abs(spectra) ** 2
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n, d=dt))

    saved = []
    p1 = plot_phase_shift_comparison(
        t,
        phase_num,
        phase_ref,
        support_mask,
        output_dir / "spm_final_phase_shift_comparison.png",
    )
    if p1 is not None:
        saved.append(p1)
    p2 = plot_intensity_colormap_vs_propagation(
        t,
        z_axis,
        time_intensity_map,
        output_dir / "spm_time_intensity_propagation.png",
        x_label="Time t",
        title="SPM: Temporal Intensity Propagation",
        colorbar_label="Normalized intensity",
    )
    if p2 is not None:
        saved.append(p2)
    p3 = plot_intensity_colormap_vs_propagation(
        freq_axis,
        z_axis,
        freq_intensity_map,
        output_dir / "spm_frequency_intensity_propagation.png",
        x_label="Frequency detuning (1/time)",
        title="SPM: Spectral Intensity Propagation",
        colorbar_label="Normalized spectral intensity",
    )
    if p3 is not None:
        saved.append(p3)
    p4 = plot_total_error_over_propagation(
        z_axis,
        error_curve,
        output_dir / "spm_error_over_propagation.png",
        title="SPM: Relative Error Over Propagation",
        y_label="Relative L2 error",
    )
    if p4 is not None:
        saved.append(p4)

    print(f"SPM analytical validation summary (run_group={run_group}):")
    print(f"  final relative L2 error = {final_error:.6e}")
    print(f"  relative power drift    = {power_drift:.6e}")
    print(f"  max wrapped phase error (all samples)     = {max_wrapped_phase_error:.6e}")
    print(f"  max wrapped phase error (pulse support)   = {max_wrapped_phase_error_support:.6e}")
    print(f"  rms wrapped phase error (pulse support)   = {rms_wrapped_phase_error_support:.6e}")
    if saved:
        print("saved plots:")
        for path in saved:
            print(f"  {path}")

    return final_error


if __name__ == "__main__":
    main()
