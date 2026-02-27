"""Shared command-line parsing helpers for Python examples."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_example_parser(
    *,
    example_slug: str,
    description: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Replot from an existing DB run group instead of running the solver.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / example_slug / "runs.sqlite3",
        help="SQLite path used for solver run storage and replot reads.",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help="Optional run-group ID. Defaults to generated ID (run mode) or latest (replot mode).",
    )
    return parser
