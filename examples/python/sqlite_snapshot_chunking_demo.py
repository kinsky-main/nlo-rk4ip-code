"""
Demonstrate SQLite-backed snapshot chunking for large record counts.
"""

from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

from nlolib_ctypes import NLolib, OperatorSpec, PulseSpec


def main() -> None:
    api = NLolib()
    if not api.storage_is_available():
        raise RuntimeError(
            "This nlolib build does not include SQLite storage support. "
            "Reconfigure/build with SQLite available."
        )

    nt = 128
    num_records = 24
    t = [((i - (nt / 2.0)) / 16.0) for i in range(nt)]
    field0 = [complex(math.exp(-(u * u)), 0.0) for u in t]
    freq = [complex(i, 0.0) for i in range(nt)]

    pulse = PulseSpec(
        samples=field0,
        delta_time=1.0 / nt,
        pulse_period=1.0,
        frequency_grid=freq,
    )
    linear = OperatorSpec(expr="0")
    nonlinear = OperatorSpec(expr="0")

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "snapshot_chunking_demo.db"
    if db_path.exists():
        db_path.unlink()

    result = api.simulate(
        pulse,
        linear,
        nonlinear,
        propagation_distance=0.2,
        records=num_records,
        output="dense",
        preset="balanced",
        sqlite_path=str(db_path),
        chunk_records=6,
        sqlite_max_bytes=50 * 1024 * 1024 * 1024,
        return_records=False,
    )
    storage_result = result.meta.get("storage_result", {})

    run_id = str(storage_result.get("run_id", ""))
    print(f"sqlite db: {db_path}")
    print(f"run_id: {run_id}")
    print(
        "storage summary: "
        f"captured={storage_result.get('records_captured', 0)} "
        f"spilled={storage_result.get('records_spilled', 0)} "
        f"chunks={storage_result.get('chunks_written', 0)} "
        f"truncated={int(bool(storage_result.get('truncated', False)))} "
        f"db_size_bytes={storage_result.get('db_size_bytes', 0)}"
    )

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        run_rows = cur.execute("SELECT COUNT(*) FROM io_runs").fetchone()[0]
        chunk_rows = cur.execute("SELECT COUNT(*) FROM io_record_chunks").fetchone()[0]
        system_rows = cur.execute("SELECT COUNT(*) FROM io_system_config").fetchone()[0]
        config_rows = cur.execute("SELECT COUNT(*) FROM io_sim_config").fetchone()[0]
        print(
            "db rows: "
            f"runs={run_rows} chunks={chunk_rows} "
            f"system_configs={system_rows} sim_configs={config_rows}"
        )


if __name__ == "__main__":
    main()

