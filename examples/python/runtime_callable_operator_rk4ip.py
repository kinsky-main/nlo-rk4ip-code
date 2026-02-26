"""
Runtime operator callable example using the unified example runner backend.
"""

from __future__ import annotations

import numpy as np
from backend.runner import centered_time_grid
from nlolib_ctypes import (
    NLO_FFT_BACKEND_VKFFT,
    NLO_VECTOR_BACKEND_AUTO,
    NLolib,
    OperatorSpec,
    PulseSpec,
    default_execution_options,
)


def main() -> None:
    n = 1024
    dt = 0.01
    beta2 = 0.05
    scale = beta2 / 2.0
    z_final = 1.0
    t = centered_time_grid(n, dt)
    field0 = np.exp(-((t / 0.20) ** 2)) * np.exp((-1.0j) * 12.0 * t)

    pulse = PulseSpec(
        samples=field0.astype(np.complex128).tolist(),
        delta_time=dt,
        pulse_period=n * dt,
    )
    linear_operator = OperatorSpec(
        fn=lambda A, w: (1.0j * scale) * (w * w),
    )
    nonlinear_operator = OperatorSpec(
        fn=lambda A, I: (1.0j * 0.0) * I,
    )
    exec_options = default_execution_options(
        backend_type=NLO_VECTOR_BACKEND_AUTO,
        fft_backend=NLO_FFT_BACKEND_VKFFT,
    )
    api = NLolib()
    result = api.propagate(
        pulse,
        linear_operator,
        nonlinear_operator,
        propagation_distance=z_final,
        output="dense",
        preset="accuracy",
        records=2,
        exec_options=exec_options,
    )
    z_records = np.asarray(result.z_axis, dtype=np.float64)
    records = np.asarray(result.records, dtype=np.complex128)

    print("runtime callable example completed.")
    print(f"z records: {z_records}")
    print(f"initial power={np.sum(np.abs(records[0]) ** 2):.6e}")
    print(f"final power={np.sum(np.abs(records[-1]) ** 2):.6e}")


if __name__ == "__main__":
    main()
