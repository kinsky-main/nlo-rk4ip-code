"""
Minimal flattened-XY GRIN propagation example using the CFFI API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))


def _write_complex_buffer(dst, values: np.ndarray) -> None:
    for i, val in enumerate(values):
        dst[i].re = float(val.real)
        dst[i].im = float(val.imag)


def _read_complex_buffer(src, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        out[i] = complex(src[i].re, src[i].im)
    return out


def main() -> None:
    try:
        from nlolib_cffi import ffi, load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nlolib_cffi/cffi is not available. Install cffi and ensure "
            "PYTHONPATH includes the repo's python/ directory."
        ) from exc

    lib = load()

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

    cfg = ffi.new("sim_config*")
    cfg.nonlinear.gamma = 0.0
    cfg.dispersion.num_dispersion_terms = 0
    cfg.dispersion.alpha = 0.0
    cfg.propagation.propagation_distance = 0.25
    cfg.propagation.starting_step_size = 1e-3
    cfg.propagation.max_step_size = 2e-3
    cfg.propagation.min_step_size = 5e-5
    cfg.propagation.error_tolerance = 1e-7
    cfg.time.pulse_period = float(nx)
    cfg.time.delta_time = 1.0

    freq = ffi.new("nlo_complex[]", nxy)
    for i in range(nxy):
        freq[i].re = 0.0
        freq[i].im = 0.0
    cfg.frequency.frequency_grid = freq

    cfg.spatial.nx = nx
    cfg.spatial.ny = ny
    cfg.spatial.delta_x = dx
    cfg.spatial.delta_y = dy
    cfg.spatial.grin_gx = 2.0e-4
    cfg.spatial.grin_gy = 2.0e-4
    cfg.spatial.spatial_frequency_grid = ffi.NULL
    cfg.spatial.grin_potential_phase_grid = ffi.NULL

    inp = ffi.new("nlo_complex[]", nxy)
    out = ffi.new("nlo_complex[]", nxy * 8)
    _write_complex_buffer(inp, field0_flat)

    opts = ffi.new("nlo_execution_options*")
    opts.backend_type = 2  # NLO_VECTOR_BACKEND_AUTO
    opts.fft_backend = 0  # NLO_FFT_BACKEND_AUTO
    opts.device_heap_fraction = 0.70
    opts.record_ring_target = 0
    opts.forced_device_budget_bytes = 0
    opts.vulkan.physical_device = ffi.NULL
    opts.vulkan.device = ffi.NULL
    opts.vulkan.queue = ffi.NULL
    opts.vulkan.queue_family_index = 0
    opts.vulkan.command_pool = ffi.NULL
    opts.vulkan.descriptor_set_budget_bytes = 0
    opts.vulkan.descriptor_set_count_override = 0

    num_records = 8
    status = int(lib.nlolib_propagate(cfg, nxy, inp, num_records, out, opts))
    if status != 0:
        raise RuntimeError(f"nlolib_propagate failed with status={status}.")

    records = _read_complex_buffer(out, nxy * num_records).reshape(num_records, ny, nx)
    in_power = float(np.sum(np.abs(records[0]) ** 2))
    out_power = float(np.sum(np.abs(records[-1]) ** 2))
    print(f"GRIN XY propagation completed: records={num_records}, shape=({ny}, {nx})")
    print(f"Power trend: z0={in_power:.6e}, z_end={out_power:.6e}")


if __name__ == "__main__":
    main()
