"""
Minimal flattened-XY GRIN propagation example using the CFFI API.
"""

from __future__ import annotations

import numpy as np
from backend.runner import NloExampleRunner, SimulationOptions


def main() -> None:
    runner = NloExampleRunner()

    nx = 1024
    ny = 1024
    nxy = nx * ny
    dx = 0.5
    dy = 0.5

    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    w0 = 8.0
    field0 = np.exp(-((xx * xx + yy * yy) / (w0 * w0))).astype(np.complex128)
    field0_flat = field0.reshape(-1)

    num_records = 8
    exec_opts = SimulationOptions(
        backend="auto",
        fft_backend="auto",
        device_heap_fraction=0.70,
    )
    _z_records, records = runner.propagate_flattened_xy_records(
        field0_flat=field0_flat,
        nx=nx,
        ny=ny,
        num_records=num_records,
        propagation_distance=0.25,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=5e-5,
        error_tolerance=1e-7,
        delta_x=dx,
        delta_y=dy,
        grin_gx=2.0e-4,
        grin_gy=2.0e-4,
        gamma=0.0,
        alpha=0.0,
        exec_options=exec_opts,
    )

    in_power = float(np.sum(np.abs(records[0]) ** 2))
    out_power = float(np.sum(np.abs(records[-1]) ** 2))
    print(f"GRIN XY propagation completed: records={num_records}, shape=({ny}, {nx})")
    print(f"Power trend: z0={in_power:.6e}, z_end={out_power:.6e}")


if __name__ == "__main__":
    main()
