import math
import sqlite3
import sys
import tempfile
from pathlib import Path

from nlolib import (
    VECTOR_BACKEND_CPU,
    NLolib,
    OperatorSpec,
    PulseSpec,
    default_execution_options,
    prepare_sim_config,
)
from nlolib._binding import NloPhysicsConfig, NloSimulationConfig
from nlolib._executor import _DENSE_OUTPUT_MAX_BYTES, _COMPLEX_BYTES, PropagationExecutor
from nlolib._requests import _NormalizedPropagateRequest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PYTHON_DIR = REPO_ROOT / "examples" / "python"
if str(EXAMPLE_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_PYTHON_DIR))

from backend.storage import ExampleRunDB


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
            exec_options=default_execution_options(VECTOR_BACKEND_CPU),
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


class _HugeInput:
    def __len__(self) -> int:
        return (_DENSE_OUTPUT_MAX_BYTES // _COMPLEX_BYTES) + 1

    def __iter__(self):
        raise AssertionError("dense output validation should run before input conversion")


def _dense_output_guard_case() -> None:
    executor = PropagationExecutor(object(), lambda: True)
    request = _NormalizedPropagateRequest(
        sim_cfg=NloSimulationConfig(),
        phys_cfg=NloPhysicsConfig(),
        input_seq=_HugeInput(),  # type: ignore[arg-type]
        num_records=1,
        exec_options=None,
        sqlite_path="guard.db",
        run_id=None,
        sqlite_max_bytes=0,
        chunk_records=0,
        cap_policy=0,
        log_final_output_field_to_db=False,
        return_records=True,
        capture_step_history=False,
        step_history_capacity=0,
        output_label="final",
        explicit_record_z=None,
        progress_callback=None,
        meta_overrides={},
    )

    try:
        executor.execute(request)
        raise AssertionError("expected dense output guard to reject oversized request")
    except ValueError as exc:
        message = str(exc)
        assert "return_records=False" in message
        assert "SQLite storage" in message


def _example_db_tensor_reload_case(api: NLolib, db_path: Path) -> None:
    db = ExampleRunDB(db_path)
    example_name = "test_python_storage_chunking"
    run_group = db.begin_group(example_name, "tensor-reload")
    case_key = "tensor"
    storage_kwargs = db.storage_kwargs(
        example_name=example_name,
        run_group=run_group,
        case_key=case_key,
        chunk_records=2,
    )

    nt = 4
    nx = 3
    ny = 2
    n = nt * nx * ny
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.05,
        starting_step_size=0.01,
        max_step_size=0.02,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=1.0 / nt,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        frequency_grid=[complex(i, 0.0) for i in range(nt)],
        potential_grid=[0j] * (nx * ny),
    )
    field0 = [complex(math.exp(-(((i % nt) - (nt / 2.0)) ** 2)), 0.0) for i in range(n)]
    result = api.propagate(
        cfg,
        field0,
        3,
        default_execution_options(VECTOR_BACKEND_CPU),
        return_records=False,
        **storage_kwargs,
    )
    assert result.meta.get("records_returned") is False
    db.save_case_from_solver_meta(
        example_name=example_name,
        run_group=run_group,
        case_key=case_key,
        solver_meta=result.meta,
        meta={"nt": nt, "nx": nx, "ny": ny},
    )

    loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
    assert loaded.records.shape == (3, n)
    assert loaded.z_axis.shape == (3,)
    assert loaded.num_time_samples == n
    assert loaded.requested_records == 3
    assert loaded.records[-1].shape == (n,)


def _small_dense_return_case(api: NLolib) -> None:
    n = 40
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
    field0 = [complex(math.exp(-(((i - (n / 2.0)) / 7.0) ** 2)), 0.0) for i in range(n)]

    result = api.propagate(
        cfg,
        field0,
        6,
        return_records=True,
    )

    assert result.meta.get("storage_enabled") is False
    assert result.meta.get("records_returned") is True
    assert int(result.meta.get("records_written", 0)) == 6
    assert len(result.records) == 6
    assert len(result.final) == n


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
    _dense_output_guard_case()
    _example_db_tensor_reload_case(api, tmp / "example_tensor.db")
    _small_dense_return_case(api)
    print("test_python_storage_chunking: passed")


if __name__ == "__main__":
    main()
