import math
import sqlite3
import tempfile
from pathlib import Path

from nlolib_ctypes import (
    NLO_VECTOR_BACKEND_CPU,
    NLolib,
    default_execution_options,
    prepare_sim_config,
)


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

    _, result = api.propagate_with_storage(
        cfg,
        field0,
        num_records,
        sqlite_path=str(db_path),
        chunk_records=4,
        return_records=False,
    )

    assert result.records_captured == num_records
    assert result.records_spilled == num_records
    assert result.chunks_written > 0
    assert result.truncated == 0

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        assert cur.execute("SELECT COUNT(*) FROM io_runs").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM io_record_chunks").fetchone()[0] == result.chunks_written


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

    _, result = api.propagate_with_storage(
        cfg,
        field0,
        10,
        sqlite_path=str(db_path),
        chunk_records=2,
        sqlite_max_bytes=1024,
        return_records=False,
    )

    assert result.truncated == 1


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

    _, result = api.propagate_with_storage(
        cfg,
        field0,
        legacy_ntmax + 1,
        sqlite_path=str(db_path),
        chunk_records=1,
        exec_options=default_execution_options(NLO_VECTOR_BACKEND_CPU),
        return_records=False,
    )
    assert result.records_captured >= 1


def main() -> None:
    api = NLolib()
    if not api.storage_is_available():
        print("test_python_storage_chunking: storage unavailable, skipping.")
        return

    tmp = Path(tempfile.mkdtemp(prefix="nlolib-storage-"))
    base_db = tmp / "base.db"
    cap_db = tmp / "cap.db"
    ntmax_db = tmp / "ntmax_exceed.db"
    _base_case(api, base_db)
    _cap_case(api, cap_db)
    _legacy_ntmax_exceed_case(api, ntmax_db)
    print("test_python_storage_chunking: passed")


if __name__ == "__main__":
    main()
