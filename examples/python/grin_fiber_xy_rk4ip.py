"""
Minimal flattened-XY GRIN propagation example using the ctypes API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))


def main() -> None:
    from nlolib_ctypes import NLolib, default_execution_options, prepare_sim_config

    api = NLolib()

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

    cfg = prepare_sim_config(
        nxy,
        gamma=0.0,
        betas=[],
        alpha=0.0,
        propagation_distance=0.25,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=5e-5,
        error_tolerance=1e-7,
        pulse_period=float(nx),
        delta_time=1.0,
        frequency_grid=[0j] * nxy,
        spatial_nx=nx,
        spatial_ny=ny,
        delta_x=dx,
        delta_y=dy,
        grin_gx=2.0e-4,
        grin_gy=2.0e-4,
    )

    opts = default_execution_options()
    num_records = 8
    records = np.asarray(api.propagate(cfg, field0_flat, num_records, opts), dtype=np.complex128)
    records = records.reshape(num_records, ny, nx)

    in_power = float(np.sum(np.abs(records[0]) ** 2))
    out_power = float(np.sum(np.abs(records[-1]) ** 2))
    print(f"GRIN XY propagation completed: records={num_records}, shape=({ny}, {nx})")
    print(f"Power trend: z0={in_power:.6e}, z_end={out_power:.6e}")


if __name__ == "__main__":
    main()
