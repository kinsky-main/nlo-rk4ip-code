"""
Runtime operator callable example using the unified example runner backend.
"""

from __future__ import annotations

import argparse

import numpy as np
from backend.cli import build_example_parser
from backend.runner import centered_time_grid
from backend.storage import ExampleRunDB
from nlolib_ctypes import (
    NLO_FFT_BACKEND_VKFFT,
    NLO_VECTOR_BACKEND_AUTO,
    NLolib,
    OperatorSpec,
    PulseSpec,
    default_execution_options,
)


def main() -> None:
    parser = build_example_parser(
        example_slug="runtime_callable_operator",
        description="Runtime callable operator example with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "runtime_callable_operator_rk4ip"
    case_key = "default"

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        z_records = np.asarray(loaded.z_axis, dtype=np.float64)
        records = np.asarray(loaded.records, dtype=np.complex128)
    else:
        run_group = db.begin_group(example_name, args.run_group)
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
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=2,
        )
        result = api.propagate(
            pulse,
            linear_operator,
            nonlinear_operator,
            propagation_distance=z_final,
            output="dense",
            preset="accuracy",
            records=2,
            exec_options=exec_options,
            sqlite_path=storage_kwargs["sqlite_path"],
            run_id=storage_kwargs["run_id"],
            chunk_records=storage_kwargs["chunk_records"],
            sqlite_max_bytes=storage_kwargs["sqlite_max_bytes"],
            log_final_output_field_to_db=storage_kwargs["log_final_output_field_to_db"],
        )
        z_records = np.asarray(result.z_axis, dtype=np.float64)
        records = np.asarray(result.records, dtype=np.complex128)
        db.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            run_id=str(result.meta["storage_result"]["run_id"]),
            meta={},
        )

    print(f"runtime callable example completed (run_group={run_group}).")
    print(f"z records: {z_records}")
    print(f"initial power={np.sum(np.abs(records[0]) ** 2):.6e}")
    print(f"final power={np.sum(np.abs(records[-1]) ** 2):.6e}")


if __name__ == "__main__":
    main()
