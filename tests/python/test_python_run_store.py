"""Stored runs through the Python binding: key, merge into the shared database, contract row."""

import math
import os
import re
import sqlite3
import tempfile

import nlolib


def main() -> None:
    nlolib.set_progress_options(enabled=False)
    root = tempfile.mkdtemp()
    db = os.path.join(root, "nlolib", "nlolib.sqlite")
    nt, dt = 128, 0.05
    samples = [complex(1.0 / math.cosh(((i - (nt - 1) / 2) * dt) / 0.3)) for i in range(nt)]
    pulse = nlolib.PulseSpec(samples=samples, delta_time=dt)
    linear = nlolib.OperatorSpec(fn=lambda A, w: 1j * (-0.04) * w * w)
    nonlinear = nlolib.OperatorSpec(fn=lambda A, I: 1j * 1.0 * A * I)
    exec_options = nlolib.default_execution_options(nlolib.VECTOR_BACKEND_CPU)
    kwargs = dict(propagation_distance=0.3, records=5, sqlite_path=db, exec_options=exec_options)

    key = nlolib.storage_run_key(pulse, linear, nonlinear, **kwargs)
    assert re.fullmatch(r"[0-9a-f]{16}", key), key
    assert not nlolib.storage_has_run(db, key)
    raw_key = nlolib.storage_run_key(pulse, linear, nonlinear, record_mode="raw", **kwargs)
    assert raw_key != key, "record mode must be part of the key"

    result = nlolib.propagate(pulse, linear, nonlinear, **kwargs)
    stored = result.meta["storage_result"]
    assert stored["key"] == key and stored["merged"], stored
    assert nlolib.storage_has_run(db, key)
    assert nlolib.storage_recover_staging(db) == {"merged": 0, "discarded": 0, "skipped": 0}

    deferred_kwargs = dict(kwargs, propagation_distance=0.2, defer_merge=True)
    deferred = nlolib.propagate(pulse, linear, nonlinear, **deferred_kwargs)
    dstored = deferred.meta["storage_result"]
    assert dstored["deferred"] and not dstored["merged"], dstored
    assert not nlolib.storage_has_run(db, dstored["key"])
    assert nlolib.storage_recover_staging(db)["merged"] == 1
    assert nlolib.storage_has_run(db, dstored["key"])

    con = sqlite3.connect(db)
    try:
        info = con.execute("PRAGMA table_info(runs)").fetchall()
        assert [(c[1], c[2], c[3], c[5]) for c in info[:6]] == [
            ("key", "TEXT", 0, 1),
            ("kind", "TEXT", 1, 0),
            ("file", "TEXT", 1, 0),
            ("created_at", "TEXT", 0, 0),
            ("git_commit", "TEXT", 0, 0),
            ("julia_version", "TEXT", 0, 0),
        ]
        params = [c[1] for c in info[6:]]
        assert params and all(p.startswith("nlolib_") and p == p.lower() for p in params)
        row = con.execute("SELECT key, kind, file, created_at, julia_version, nlolib_binding FROM runs WHERE key=?", (key,)).fetchone()
        assert row[0] == key and row[1] == "nlolib" and row[2] == "nlolib/nlolib.sqlite"
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{3})?", row[3]), row[3]
        assert row[4] is None and row[5] == "python"
        for types in con.execute("SELECT " + ", ".join(f'typeof("{p}")' for p in params) + " FROM runs"):
            assert "blob" not in types
        assert con.execute("SELECT COUNT(*) FROM io_record_maps").fetchone()[0] == 10
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        con.close()
    print("test_python_run_store: ok")


if __name__ == "__main__":
    main()
