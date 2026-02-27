"""
Demonstrate SQLite-backed snapshot chunking for large record counts.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

from backend.cli import build_example_parser
from backend.storage import ExampleRunDB

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

from nlolib_ctypes import NLolib, OperatorSpec, PulseSpec


def main() -> None:
    parser = build_example_parser(
        example_slug="sqlite_snapshot_chunking_demo",
        description="SQLite snapshot chunking demo with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "sqlite_snapshot_chunking_demo"
    case_key = "default"

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

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        run_id = loaded.run_id
        print(f"sqlite db: {db.db_path}")
        print(f"run_id: {run_id}")
        print(f"run_group: {run_group}")
    else:
        run_group = db.begin_group(example_name, args.run_group)
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=6,
            sqlite_max_bytes=50 * 1024 * 1024 * 1024,
        )
        result = api.propagate(
            pulse,
            linear,
            nonlinear,
            propagation_distance=0.2,
            records=num_records,
            output="dense",
            preset="balanced",
            sqlite_path=storage_kwargs["sqlite_path"],
            run_id=storage_kwargs["run_id"],
            chunk_records=storage_kwargs["chunk_records"],
            sqlite_max_bytes=storage_kwargs["sqlite_max_bytes"],
            return_records=False,
        )
        storage_result = result.meta.get("storage_result", {})
        run_id = str(storage_result.get("run_id", ""))
        db.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            run_id=run_id,
            meta={"num_records": int(num_records)},
        )
        print(f"sqlite db: {db.db_path}")
        print(f"run_id: {run_id}")
        print(f"run_group: {run_group}")
        print(
            "storage summary: "
            f"captured={storage_result.get('records_captured', 0)} "
            f"spilled={storage_result.get('records_spilled', 0)} "
            f"chunks={storage_result.get('chunks_written', 0)} "
            f"truncated={int(bool(storage_result.get('truncated', False)))} "
            f"db_size_bytes={storage_result.get('db_size_bytes', 0)}"
        )

    with sqlite3.connect(db.db_path) as con:
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

        row = cur.execute(
            "SELECT records_written, chunks_written, truncated FROM io_runs WHERE run_id=?;",
            (run_id,),
        ).fetchone()
        if row is not None:
            print(
                "run row: "
                f"records_written={int(row[0])} chunks_written={int(row[1])} truncated={int(row[2])}"
            )


if __name__ == "__main__":
    main()

