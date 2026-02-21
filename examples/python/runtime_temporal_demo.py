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
    NLO_VECTOR_BACKEND_AUTO,
    NLolib,
    OperatorSpec,
    PulseSpec,
    default_execution_options,
)

def main() -> None:
    n = 512
    dt = 0.02
    beta2 = 0.05

    t = centered_time_grid(n, dt)
    field0 = np.exp(-((t / 0.25) ** 2)) * np.exp((-1.0j) * 8.0 * t)
    omega = 2.0 * math.pi * np.fft.fftfreq(n, d=dt)

    pulse = PulseSpec(
        samples=field0.tolist(),
        delta_time=dt,
        pulse_period=n * dt,
        frequency_grid=[complex(float(w), 0.0) for w in omega],
    )
    linear_operator = OperatorSpec(
        fn=lambda A, w: (1.0j * (beta2 / 2.0)) * (w * w),
    )
    nonlinear_operator = OperatorSpec(
        expr="i*gamma*I + i*V",
        params={"gamma": 0.01},
    )

    opts = default_execution_options(
        backend_type=NLO_VECTOR_BACKEND_AUTO,
        fft_backend=NLO_FFT_BACKEND_VKFFT,
    )
    api = NLolib()
    result = api.simulate(
        pulse,
        linear_operator,
        nonlinear_operator,
        propagation_distance=0.25,
        output="dense",
        preset="accuracy",
        records=2,
        exec_options=opts,
    )
    records = np.asarray(result.records, dtype=np.complex128)

    final_field = records[-1]
    print(f"runtime_temporal_demo_py: propagated {n} samples.")
    print(f"initial power={np.sum(np.abs(field0) ** 2):.6e} final power={np.sum(np.abs(final_field) ** 2):.6e}")


if __name__ == "__main__":
    main()
