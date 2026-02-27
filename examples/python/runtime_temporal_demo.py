"""
MATLAB-parity temporal runtime-operator demo via Python ctypes bindings.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.runner import centered_time_grid
from backend.storage import ExampleRunDB


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
    parser = build_example_parser(
        example_slug="runtime_temporal_demo",
        description="Runtime temporal demo with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "runtime_temporal_demo"
    case_key = "default"

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        meta = loaded.meta
        n = int(meta["n"])
        dt = float(meta["dt"])
        t = centered_time_grid(n, dt)
        field0 = np.exp(-((t / 0.25) ** 2)) * np.exp((-1.0j) * 8.0 * t)
        records = np.asarray(loaded.records, dtype=np.complex128)
        runtime_logs = ""
    else:
        run_group = db.begin_group(example_name, args.run_group)
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
        exec_options = default_execution_options(
            backend_type=NLO_VECTOR_BACKEND_AUTO,
            fft_backend=NLO_FFT_BACKEND_VKFFT,
        )
        api = NLolib()
        api.set_log_buffer(256 * 1024)
        api.set_log_level(2)
        api.set_progress_options(enabled=True, milestone_percent=10, emit_on_step_adjust=False)
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
            propagation_distance=0.25,
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
        records = np.asarray(result.records, dtype=np.complex128)
        db.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            run_id=str(result.meta["storage_result"]["run_id"]),
            meta={
                "n": int(n),
                "dt": float(dt),
            },
        )
        runtime_logs = api.read_log_buffer(consume=True)

    final_field = records[-1]
    print(f"runtime_temporal_demo_py: propagated {n} samples (run_group={run_group}).")
    print(f"initial power={np.sum(np.abs(field0) ** 2):.6e} final power={np.sum(np.abs(final_field) ** 2):.6e}")
    if runtime_logs:
        print(runtime_logs, end="" if runtime_logs.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
