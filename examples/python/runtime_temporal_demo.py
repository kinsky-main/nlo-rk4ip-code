"""
MATLAB-parity temporal runtime-operator demo via Python ctypes bindings.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from backend.runner import centered_time_grid


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

from nlolib_ctypes import (  # noqa: E402
    NLO_FFT_BACKEND_VKFFT,
    NLO_VECTOR_BACKEND_VULKAN,
    NLolib,
    RuntimeOperators,
    default_execution_options,
    prepare_sim_config,
)

def main() -> None:
    n = 512
    dt = 0.02
    beta2 = 0.05

    t = centered_time_grid(n, dt)
    field0 = np.exp(-((t / 0.25) ** 2)) * np.exp((-1.0j) * 8.0 * t)
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)

    runtime = RuntimeOperators(
        dispersion_factor_fn=lambda A, w: (1.0j * (beta2 / 2.0)) * (w * w),
        constants=[beta2 / 2.0, 0.0, 0.01],
    )

    cfg = prepare_sim_config(
        n,
        propagation_distance=0.25,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=n * dt,
        delta_time=dt,
        frequency_grid=[complex(float(w), 0.0) for w in omega],
        runtime=runtime,
    )

    opts = default_execution_options(
        backend_type=NLO_VECTOR_BACKEND_VULKAN,
        fft_backend=NLO_FFT_BACKEND_VKFFT,
    )
    api = NLolib()
    records = np.asarray(
        api.propagate(cfg, field0.tolist(), 2, opts),
        dtype=np.complex128,
    )

    final_field = records[-1]
    print(f"runtime_temporal_demo_py: propagated {n} samples.")
    print(f"initial power={np.sum(np.abs(field0) ** 2):.6e} final power={np.sum(np.abs(final_field) ** 2):.6e}")


if __name__ == "__main__":
    main()
