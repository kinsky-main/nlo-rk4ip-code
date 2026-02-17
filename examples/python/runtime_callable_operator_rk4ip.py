"""
Runtime operator callable example using the unified example runner backend.
"""

from __future__ import annotations

import numpy as np
from backend.runner import NloExampleRunner, SimulationOptions, TemporalSimulationConfig


def centered_time_grid(num_samples: int, delta_time: float) -> np.ndarray:
    return (np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)) * delta_time


def main() -> None:
    from nlolib_ctypes import RuntimeOperators

    n = 1024
    dt = 0.01
    beta2 = 0.05
    scale = beta2 / 2.0
    z_final = 1.0
    t = centered_time_grid(n, dt)
    field0 = np.exp(-((t / 0.20) ** 2)) * np.exp((-1.0j) * 12.0 * t)

    runtime = RuntimeOperators(
        dispersion_fn=lambda w: np.exp((1.0j * scale) * (w * w)),
        nonlinear_expr=None,
    )

    sim_cfg = TemporalSimulationConfig(
        gamma=0.0,
        beta2=0.0,
        alpha=0.0,
        dt=dt,
        z_final=z_final,
        num_time_samples=n,
        runtime=runtime,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
    )

    runner = NloExampleRunner()
    z_records, records = runner.propagate_temporal_records(
        field0.astype(np.complex128),
        sim_cfg,
        num_records=2,
        exec_options=SimulationOptions(backend="cpu", fft_backend="fftw"),
    )

    print("runtime callable example completed.")
    print(f"z records: {z_records}")
    print(f"initial power={np.sum(np.abs(records[0]) ** 2):.6e}")
    print(f"final power={np.sum(np.abs(records[-1]) ** 2):.6e}")


if __name__ == "__main__":
    main()
