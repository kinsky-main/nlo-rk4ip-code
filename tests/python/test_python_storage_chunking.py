import math
import sqlite3
import tempfile
from pathlib import Path

from nlolib_ctypes import (
    NLO_VECTOR_BACKEND_CPU,
    NLolib,
    OperatorSpec,
    PulseSpec,
    default_execution_options,
    prepare_sim_config,
)


def _table_exists(cur: sqlite3.Cursor, table_name: str) -> bool:
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _base_case(api: NLolib, db_path: Path) -> None:
    n = 64
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.1,
        starting_step_size=0.01,
        max_step_size=0.02,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=1.0 / n,
        frequency_grid=[complex(i, 0.0) for i in range(n)],
    )
    field0 = [complex(math.exp(-(((i - (n / 2.0)) / 10.0) ** 2)), 0.0) for i in range(n)]
    num_records = 10

    result_obj = api.propagate(
        cfg,
        field0,
        num_records,
        sqlite_path=str(db_path),
        chunk_records=4,
        return_records=False,
    )
    storage = result_obj.meta.get("storage_result", {})

    assert int(storage.get("records_captured", 0)) == num_records
    assert int(storage.get("records_spilled", 0)) == num_records
    assert int(storage.get("chunks_written", 0)) > 0
    assert int(storage.get("truncated", 0)) == 0

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        assert _table_exists(cur, "io_runs")
        assert _table_exists(cur, "io_record_chunks")
        assert cur.execute("SELECT COUNT(*) FROM io_runs").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM io_record_chunks").fetchone()[0] == int(storage.get("chunks_written", 0))
        if _table_exists(cur, "io_final_output_fields"):
            assert cur.execute("SELECT COUNT(*) FROM io_final_output_fields").fetchone()[0] == 0


def _cap_case(api: NLolib, db_path: Path) -> None:
    n = 64
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.1,
        starting_step_size=0.01,
        max_step_size=0.02,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=1.0 / n,
        frequency_grid=[complex(i, 0.0) for i in range(n)],
    )
    field0 = [complex(math.exp(-(((i - (n / 2.0)) / 10.0) ** 2)), 0.0) for i in range(n)]

    result_obj = api.propagate(
        cfg,
        field0,
        10,
        sqlite_path=str(db_path),
        chunk_records=2,
        sqlite_max_bytes=1024,
        return_records=False,
    )
    storage = result_obj.meta.get("storage_result", {})

    assert int(storage.get("truncated", 0)) == 1


def _legacy_ntmax_exceed_case(api: NLolib, db_path: Path) -> None:
    legacy_ntmax = 1 << 20
    n = 1
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.02,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=1.0 / n,
        frequency_grid=[complex(i, 0.0) for i in range(n)],
    )
    field0 = [complex(1.0, 0.0)] * n

    try:
        result_obj = api.propagate(
            cfg,
            field0,
            legacy_ntmax + 1,
            sqlite_path=str(db_path),
            chunk_records=1,
            exec_options=default_execution_options(NLO_VECTOR_BACKEND_CPU),
            return_records=False,
        )
        storage = result_obj.meta.get("storage_result", {})
        assert int(storage.get("records_captured", 0)) >= 1
    except RuntimeError as exc:
        # Current builds reject oversize record requests explicitly.
        assert "status=1" in str(exc)


def _final_output_logging_case(api: NLolib, db_path: Path) -> None:
    n = 32
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.1,
        starting_step_size=0.01,
        max_step_size=0.02,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=1.0 / n,
        frequency_grid=[complex(i, 0.0) for i in range(n)],
    )
    field0 = [complex(math.exp(-(((i - (n / 2.0)) / 8.0) ** 2)), 0.0) for i in range(n)]

    _ = api.propagate(
        cfg,
        field0,
        4,
        sqlite_path=str(db_path),
        chunk_records=2,
        log_final_output_field_to_db=True,
        return_records=False,
    )

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        if _table_exists(cur, "io_final_output_fields"):
            row = cur.execute(
                "SELECT num_time_samples, length(payload) FROM io_final_output_fields LIMIT 1"
            ).fetchone()
            assert row is not None
            assert row[0] == n
            assert row[1] == n * 16


def _simulate_storage_facade_case(api: NLolib, db_path: Path) -> None:
    n = 48
    pulse = PulseSpec(
        samples=[complex(math.exp(-(((i - (n / 2.0)) / 9.0) ** 2)), 0.0) for i in range(n)],
        delta_time=1.0 / n,
        pulse_period=1.0,
        frequency_grid=[complex(i, 0.0) for i in range(n)],
    )
    result = api.propagate(
        pulse,
        OperatorSpec(expr="0"),
        OperatorSpec(expr="0"),
        propagation_distance=0.1,
        records=6,
        sqlite_path=str(db_path),
        chunk_records=3,
        return_records=False,
    )

    assert result.meta.get("storage_enabled") is True
    assert result.meta.get("records_returned") is False
    storage = result.meta.get("storage_result")
    assert isinstance(storage, dict)
    assert int(storage.get("records_captured", 0)) == 6
    assert int(storage.get("records_spilled", 0)) == 6
    assert int(storage.get("chunks_written", 0)) > 0

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        assert _table_exists(cur, "io_runs")
        assert _table_exists(cur, "io_record_chunks")
        assert cur.execute("SELECT COUNT(*) FROM io_runs").fetchone()[0] == 1


def main() -> None:
    api = NLolib()
    if not api.storage_is_available():
        print("test_python_storage_chunking: storage unavailable, skipping.")
        return

    tmp = Path(tempfile.mkdtemp(prefix="nlolib-storage-"))
    base_db = tmp / "base.db"
    cap_db = tmp / "cap.db"
    ntmax_db = tmp / "ntmax_exceed.db"
    final_output_db = tmp / "final_output.db"
    facade_db = tmp / "facade.db"
    _base_case(api, base_db)
    _cap_case(api, cap_db)
    _legacy_ntmax_exceed_case(api, ntmax_db)
    _final_output_logging_case(api, final_output_db)
    _simulate_storage_facade_case(api, facade_db)
    print("test_python_storage_chunking: passed")


if __name__ == "__main__":
    main()
