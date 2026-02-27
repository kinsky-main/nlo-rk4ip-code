"""SQLite-backed run grouping and replay helpers for Python examples."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaseListing:
    case_key: str
    run_id: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class LoadedCase:
    run_id: str
    case_key: str
    meta: dict[str, Any]
    records: np.ndarray
    z_axis: np.ndarray
    requested_records: int
    num_time_samples: int
    z_end: float


class ExampleRunDB:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def ensure_schema(self) -> None:
        schema = (
            "PRAGMA foreign_keys=ON;"
            "CREATE TABLE IF NOT EXISTS ex_run_groups ("
            "  example_name TEXT NOT NULL,"
            "  run_group TEXT NOT NULL,"
            "  created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
            "  PRIMARY KEY(example_name, run_group)"
            ");"
            "CREATE TABLE IF NOT EXISTS ex_case_runs ("
            "  example_name TEXT NOT NULL,"
            "  run_group TEXT NOT NULL,"
            "  case_key TEXT NOT NULL,"
            "  run_id TEXT NOT NULL,"
            "  meta_json TEXT NOT NULL DEFAULT '{}',"
            "  created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
            "  PRIMARY KEY(example_name, run_group, case_key),"
            "  UNIQUE(run_id),"
            "  FOREIGN KEY(example_name, run_group) "
            "    REFERENCES ex_run_groups(example_name, run_group) ON DELETE CASCADE"
            ");"
            "CREATE TABLE IF NOT EXISTS ex_step_history ("
            "  run_id TEXT PRIMARY KEY,"
            "  event_count INTEGER NOT NULL,"
            "  dropped INTEGER NOT NULL,"
            "  capacity INTEGER NOT NULL,"
            "  step_index_blob BLOB NOT NULL,"
            "  z_blob BLOB NOT NULL,"
            "  step_size_blob BLOB NOT NULL,"
            "  next_step_size_blob BLOB NOT NULL,"
            "  error_blob BLOB NOT NULL"
            ");"
        )
        with self._connect() as con:
            con.executescript(schema)

    @staticmethod
    def new_run_group_id() -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y%m%dT%H%M%S.%fZ")

    @staticmethod
    def make_run_id(example_name: str, run_group: str, case_key: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        base = f"{example_name}|{run_group}|{case_key}|{stamp}"
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
        prefix = "".join(ch for ch in example_name.lower() if ch.isalnum())[:18]
        return f"{prefix}-{stamp[-10:]}-{digest}"

    def begin_group(self, example_name: str, run_group: str | None = None) -> str:
        resolved = run_group if run_group else self.new_run_group_id()
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO ex_run_groups(example_name, run_group) VALUES(?, ?);",
                (example_name, resolved),
            )
        return resolved

    def latest_run_group(self, example_name: str) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT run_group FROM ex_run_groups WHERE example_name=? "
                "ORDER BY created_utc DESC, run_group DESC LIMIT 1;",
                (example_name,),
            ).fetchone()
        if row is None:
            return None
        return str(row["run_group"])

    def resolve_replot_group(self, example_name: str, run_group: str | None) -> str:
        if run_group:
            return run_group
        latest = self.latest_run_group(example_name)
        if latest is None:
            raise RuntimeError(
                f"no stored run groups found for example '{example_name}' in DB: {self.db_path}"
            )
        return latest

    def storage_kwargs(
        self,
        *,
        example_name: str,
        run_group: str,
        case_key: str,
        chunk_records: int = 0,
        sqlite_max_bytes: int = 0,
        log_final_output_field_to_db: bool = False,
    ) -> dict[str, Any]:
        return {
            "sqlite_path": str(self.db_path),
            "run_id": self.make_run_id(example_name, run_group, case_key),
            "chunk_records": int(chunk_records),
            "sqlite_max_bytes": int(sqlite_max_bytes),
            "log_final_output_field_to_db": bool(log_final_output_field_to_db),
        }

    def save_case(
        self,
        *,
        example_name: str,
        run_group: str,
        case_key: str,
        run_id: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.begin_group(example_name, run_group)
        payload = json.dumps(meta if meta is not None else {}, sort_keys=True)
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO ex_case_runs("
                "example_name, run_group, case_key, run_id, meta_json"
                ") VALUES(?,?,?,?,?);",
                (example_name, run_group, case_key, run_id, payload),
            )

    def save_case_from_solver_meta(
        self,
        *,
        example_name: str,
        run_group: str,
        case_key: str,
        solver_meta: dict[str, Any],
        meta: dict[str, Any] | None = None,
        save_step_history: bool = False,
    ) -> str:
        storage_result = solver_meta.get("storage_result")
        if not isinstance(storage_result, dict):
            raise RuntimeError("solver meta does not contain storage_result.")
        run_id = str(storage_result.get("run_id", "")).strip()
        if not run_id:
            raise RuntimeError("solver storage_result did not provide a run_id.")
        self.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            run_id=run_id,
            meta=meta,
        )
        if save_step_history:
            self.save_step_history(run_id=run_id, step_history=solver_meta.get("step_history"))
        return run_id

    def list_cases(self, *, example_name: str, run_group: str) -> list[CaseListing]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT case_key, run_id, meta_json "
                "FROM ex_case_runs WHERE example_name=? AND run_group=? "
                "ORDER BY case_key ASC;",
                (example_name, run_group),
            ).fetchall()
        out: list[CaseListing] = []
        for row in rows:
            out.append(
                CaseListing(
                    case_key=str(row["case_key"]),
                    run_id=str(row["run_id"]),
                    meta=self._decode_meta_json(row["meta_json"]),
                )
            )
        return out

    def load_case(self, *, example_name: str, run_group: str, case_key: str) -> LoadedCase:
        with self._connect() as con:
            row = con.execute(
                "SELECT c.run_id, c.meta_json, r.num_recorded_samples, r.num_time_samples, s.z_end "
                "FROM ex_case_runs c "
                "JOIN io_runs r ON r.run_id = c.run_id "
                "JOIN io_sim_config s ON s.config_hash = r.config_hash "
                "WHERE c.example_name=? AND c.run_group=? AND c.case_key=?;",
                (example_name, run_group, case_key),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"missing case '{case_key}' for example='{example_name}', run_group='{run_group}'."
                )

            run_id = str(row["run_id"])
            num_time_samples = int(row["num_time_samples"])
            requested_records = int(row["num_recorded_samples"])
            z_end = float(row["z_end"])
            records = self._load_records(con, run_id, num_time_samples)

        loaded_records = int(records.shape[0])
        if loaded_records == 1:
            z_axis = np.asarray([z_end], dtype=np.float64)
        else:
            z_axis = np.linspace(0.0, z_end, loaded_records, dtype=np.float64)

        return LoadedCase(
            run_id=run_id,
            case_key=case_key,
            meta=self._decode_meta_json(row["meta_json"]),
            records=records,
            z_axis=z_axis,
            requested_records=requested_records,
            num_time_samples=num_time_samples,
            z_end=z_end,
        )

    def load_step_history(self, *, run_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT event_count, dropped, capacity, step_index_blob, z_blob, "
                "step_size_blob, next_step_size_blob, error_blob "
                "FROM ex_step_history WHERE run_id=?;",
                (run_id,),
            ).fetchone()
        if row is None:
            return None

        count = int(row["event_count"])
        out = {
            "step_index": np.frombuffer(bytes(row["step_index_blob"]), dtype=np.int64, count=count).tolist(),
            "z": np.frombuffer(bytes(row["z_blob"]), dtype=np.float64, count=count).tolist(),
            "step_size": np.frombuffer(bytes(row["step_size_blob"]), dtype=np.float64, count=count).tolist(),
            "next_step_size": np.frombuffer(
                bytes(row["next_step_size_blob"]), dtype=np.float64, count=count
            ).tolist(),
            "error": np.frombuffer(bytes(row["error_blob"]), dtype=np.float64, count=count).tolist(),
            "dropped": int(row["dropped"]),
            "capacity": int(row["capacity"]),
        }
        return out

    def save_step_history(self, *, run_id: str, step_history: Any) -> None:
        if not isinstance(step_history, dict):
            raise RuntimeError("step_history metadata is missing or malformed.")

        step_index_raw = step_history.get("step_index", [])
        z_raw = step_history.get("z", [])
        step_size_raw = step_history.get("step_size", [])
        next_step_size_raw = step_history.get("next_step_size", [])
        error_raw = step_history.get("error", [])

        step_index = np.asarray(step_index_raw, dtype=np.int64).reshape(-1)
        z = np.asarray(z_raw, dtype=np.float64).reshape(-1)
        step_size = np.asarray(step_size_raw, dtype=np.float64).reshape(-1)
        next_step_size = np.asarray(next_step_size_raw, dtype=np.float64).reshape(-1)
        error = np.asarray(error_raw, dtype=np.float64).reshape(-1)

        count = min(step_index.size, z.size, step_size.size, next_step_size.size, error.size)
        step_index = step_index[:count]
        z = z[:count]
        step_size = step_size[:count]
        next_step_size = next_step_size[:count]
        error = error[:count]

        dropped = int(step_history.get("dropped", 0))
        capacity = int(step_history.get("capacity", 0))

        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO ex_step_history("
                "run_id, event_count, dropped, capacity, step_index_blob, z_blob, "
                "step_size_blob, next_step_size_blob, error_blob"
                ") VALUES(?,?,?,?,?,?,?,?,?);",
                (
                    run_id,
                    count,
                    dropped,
                    capacity,
                    step_index.tobytes(order="C"),
                    z.tobytes(order="C"),
                    step_size.tobytes(order="C"),
                    next_step_size.tobytes(order="C"),
                    error.tobytes(order="C"),
                ),
            )

    @staticmethod
    def _decode_meta_json(raw_meta: Any) -> dict[str, Any]:
        if raw_meta is None:
            return {}
        text = str(raw_meta)
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, dict):
            return decoded
        return {}

    @staticmethod
    def _load_records(con: sqlite3.Connection, run_id: str, num_time_samples: int) -> np.ndarray:
        rows = con.execute(
            "SELECT record_start, record_count, payload FROM io_record_chunks "
            "WHERE run_id=? ORDER BY record_start ASC;",
            (run_id,),
        ).fetchall()
        if not rows:
            raise RuntimeError(f"run_id '{run_id}' has no io_record_chunks.")

        max_records = 0
        for row in rows:
            start = int(row["record_start"])
            count = int(row["record_count"])
            max_records = max(max_records, start + count)
        if max_records <= 0:
            raise RuntimeError(f"run_id '{run_id}' has invalid chunk metadata.")

        out = np.zeros((max_records, int(num_time_samples)), dtype=np.complex128)
        for row in rows:
            start = int(row["record_start"])
            count = int(row["record_count"])
            payload = bytes(row["payload"])
            expected_f64 = int(count) * int(num_time_samples) * 2
            decoded = np.frombuffer(payload, dtype=np.float64)
            if decoded.size != expected_f64:
                raise RuntimeError(
                    f"run_id '{run_id}' chunk decode mismatch: expected {expected_f64} float64 values, "
                    f"got {decoded.size}."
                )
            records = decoded.view(np.complex128).reshape(count, int(num_time_samples))
            out[start : start + count, :] = records
        return out
