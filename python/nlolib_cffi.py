"""
Minimal CFFI (ABI mode) bindings for NLOLib.
"""

from __future__ import annotations

import ctypes.util
import os
from pathlib import Path

from cffi import FFI

NT_MAX = 1 << 20


_CDEF_PATH = Path(__file__).with_name("nlolib.cdef.h")
_CDEF_TEXT = _CDEF_PATH.read_text(encoding="utf-8")

ffi = FFI()
ffi.cdef(_CDEF_TEXT)


def _candidate_library_paths() -> list[str]:
    candidates: list[str] = []
    env_path = os.environ.get("NLOLIB_LIBRARY")
    if env_path:
        candidates.append(env_path)

    here = Path(__file__).resolve().parent
    root = here.parent
    if os.name == "nt":
        candidates.extend([
            str(here / "nlolib.dll"),
            str(here / "Debug" / "nlolib.dll"),
            str(here / "Release" / "nlolib.dll"),
            str(here / "RelWithDebInfo" / "nlolib.dll"),
            str(here / "MinSizeRel" / "nlolib.dll"),
            str(root / "nlolib.dll"),
        ])
    elif os.name == "posix":
        candidates.extend([
            str(here / "libnlolib.so"),
            str(root / "libnlolib.so"),
            str(here / "libnlolib.dylib"),
            str(root / "libnlolib.dylib"),
        ])

    found = ctypes.util.find_library("nlolib")
    if found:
        candidates.append(found)

    return candidates


def load(path: str | None = None):
    """
    Load the shared library and return the CFFI lib handle.

    Set NLOLIB_LIBRARY to override discovery.
    """
    if path is None:
        for candidate in _candidate_library_paths():
            try:
                return ffi.dlopen(candidate)
            except OSError:
                continue
        raise OSError(
            "Unable to locate NLOLib shared library. "
            "Set NLOLIB_LIBRARY to the full path."
        )

    return ffi.dlopen(path)


__all__ = ["ffi", "load", "NT_MAX"]
