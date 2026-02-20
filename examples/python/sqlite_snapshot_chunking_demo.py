"""
Demonstrate SQLite-backed snapshot chunking for large record counts.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from nlolib_ctypes import NLolib, prepare_sim_config


def _decode_run_id(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


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

    cfg = prepare_sim_config(
        nt,
        propagation_distance=0.2,
        starting_step_size=0.01,
        max_step_size=0.02,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=1.0 / nt,
        frequency_grid=freq,
    )

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "snapshot_chunking_demo.db"
    if db_path.exists():
        db_path.unlink()

    _, storage_result = api.propagate_with_storage(
        cfg,
        field0,
        num_records,
        sqlite_path=str(db_path),
        chunk_records=6,
        sqlite_max_bytes=50 * 1024 * 1024 * 1024,
        return_records=False,
    )

    run_id = _decode_run_id(bytes(storage_result.run_id))
    print(f"sqlite db: {db_path}")
    print(f"run_id: {run_id}")
    print(
        "storage summary: "
        f"captured={storage_result.records_captured} "
        f"spilled={storage_result.records_spilled} "
        f"chunks={storage_result.chunks_written} "
        f"truncated={storage_result.truncated} "
        f"db_size_bytes={storage_result.db_size_bytes}"
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

