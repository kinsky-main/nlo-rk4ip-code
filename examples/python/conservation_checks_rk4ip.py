"""
Momentum, energy, and Hamiltonian conservation checks for a lossless NLSE run.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.plotting import plot_three_curve_drift
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


def _spectral_time_derivative(a_t: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.fft.ifft(1j * omega * np.fft.fft(a_t))


def _compute_invariants(
    a_t: np.ndarray,
    omega: np.ndarray,
    dt: float,
    beta2: float,
    gamma: float,
) -> tuple[float, float, float]:
    da_dt = _spectral_time_derivative(a_t, omega)

    energy = float(np.sum(np.abs(a_t) ** 2) * dt)
    momentum = float(np.imag(np.sum(np.conj(a_t) * da_dt)) * dt)
    hamiltonian = float(
        (0.5 * beta2 * np.sum(np.abs(da_dt) ** 2) - 0.5 * gamma * np.sum(np.abs(a_t) ** 4)) * dt
    )
    return energy, momentum, hamiltonian

def main() -> tuple[float, float, float]:
    beta2 = -0.01
    gamma = 0.01
    alpha = 0.0
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    p0 = abs(beta2) / (gamma * t0 * t0)
    ld = (t0 * t0) / abs(beta2)
    z_final = 0.2 * ld

    parser = build_example_parser(
        example_slug="conservation_checks",
        description="Conservation checks with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "conservation_checks_rk4ip"
    case_key = "default"

    n = 2**10
    dt = (16.0 * t0) / n
    carrier = 12.0
    t_phys = centered_time_grid(n, dt)
    tau = t_phys / t0
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)

    # Apply carrier tilt so momentum is non-zero and measurable.
    a0 = (math.sqrt(p0) / np.cosh(tau)) * np.exp(-1j * carrier * t_phys)
    a0 = np.asarray(a0, dtype=np.complex128)

    runner = NloExampleRunner()
    _configure_runtime_logging(runner)
    exec_options = SimulationOptions(backend="auto", fft_backend="auto")

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        meta = loaded.meta
        n = int(meta["n"])
        dt = float(meta["dt"])
        beta2 = float(meta["beta2"])
        gamma = float(meta["gamma"])
        omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)
        z_axis = np.asarray(loaded.z_axis, dtype=np.float64)
        records = np.asarray(loaded.records, dtype=np.complex128)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        sim_cfg = TemporalSimulationConfig(
            gamma=gamma,
            beta2=beta2,
            alpha=alpha,
            dt=dt,
            z_final=z_final,
            num_time_samples=n,
            pulse_period=n * dt,
            omega=omega,
            starting_step_size=z_final / 1000.0,
            max_step_size=z_final / 600.0,
            min_step_size=z_final / 15000.0,
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
        z_axis = np.asarray(z_records, dtype=np.float64)
        records = np.asarray(a_records, dtype=np.complex128)
        db.save_case_from_solver_meta(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            solver_meta=runner.last_meta,
            meta={
                "n": int(n),
                "dt": float(dt),
                "beta2": float(beta2),
                "gamma": float(gamma),
            },
        )

    energy = np.empty(z_axis.size, dtype=np.float64)
    momentum = np.empty(z_axis.size, dtype=np.float64)
    hamiltonian = np.empty(z_axis.size, dtype=np.float64)
    for i in range(z_axis.size):
        energy[i], momentum[i], hamiltonian[i] = _compute_invariants(records[i], omega, dt, beta2, gamma)

    energy_drift = (energy - energy[0]) / max(abs(energy[0]), 1e-15)
    momentum_drift = (momentum - momentum[0]) / max(abs(momentum[0]), 1e-15)
    hamiltonian_drift = (hamiltonian - hamiltonian[0]) / max(abs(hamiltonian[0]), 1e-15)

    max_energy_drift = float(np.max(np.abs(energy_drift)))
    max_momentum_drift = float(np.max(np.abs(momentum_drift)))
    max_hamiltonian_drift = float(np.max(np.abs(hamiltonian_drift)))

    output_dir = args.output_dir
    saved = plot_three_curve_drift(
        z_axis,
        energy_drift,
        momentum_drift,
        hamiltonian_drift,
        output_dir / "energy_conservation_relative_invariant_drift.png",
        label_a="Energy drift",
        label_b="Momentum drift",
        label_c="Hamiltonian drift",
    )

    print(f"conservation-check summary (run_group={run_group}):")
    print(f"  max |energy drift|      = {max_energy_drift:.6e}")
    print(f"  max |momentum drift|    = {max_momentum_drift:.6e}")
    print(f"  max |hamiltonian drift| = {max_hamiltonian_drift:.6e}")
    if saved is not None:
        print(f"saved plot: {saved}")

    return max_energy_drift, max_momentum_drift, max_hamiltonian_drift


if __name__ == "__main__":
    main()
