import importlib.util
import pathlib
import re

import nlolib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_package_surface() -> None:
    assert hasattr(nlolib, "NLolib")
    assert hasattr(nlolib, "propagate")
    assert hasattr(nlolib, "query_runtime_limits")
    assert hasattr(nlolib, "NloPerfProfileSnapshot")
    assert hasattr(nlolib, "perf_profile_set_enabled")
    assert hasattr(nlolib, "perf_profile_is_enabled")
    assert hasattr(nlolib, "perf_profile_reset")
    assert hasattr(nlolib, "perf_profile_read")
    assert hasattr(nlolib, "PulseSpec")
    assert hasattr(nlolib, "OperatorSpec")
    assert hasattr(nlolib, "RuntimeOperators")
    assert hasattr(nlolib, "translate_callable")
    assert hasattr(nlolib.NLolib, "perf_profile_set_enabled")
    assert hasattr(nlolib.NLolib, "perf_profile_is_enabled")
    assert hasattr(nlolib.NLolib, "perf_profile_reset")
    assert hasattr(nlolib.NLolib, "perf_profile_read")
    assert "._legacy_impl" not in nlolib.NLolib.__module__
    assert "._legacy_impl" not in nlolib.propagate.__module__
    assert importlib.util.find_spec("nlolib._legacy_impl") is None
    assert importlib.util.find_spec("nlolib_ctypes") is None


def test_version_attribute() -> None:
    """nlolib.__version__ must name the version this tree builds.

    Compared against CMakeLists.txt, the source of truth, rather than against
    pyproject.toml that __version__ reads: that way this fails if the attribute
    stops tracking the project version, while test_version_copies_match is what
    fails if the copies themselves drift.
    """
    assert hasattr(nlolib, "__version__")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"project\([^)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", cmake)
    assert match, "could not parse the project version from CMakeLists.txt"
    assert nlolib.__version__ == match.group(1), (
        f"__version__ is {nlolib.__version__}, expected {match.group(1)}"
    )
    # Never exported: `from nlolib import *` must not rebind a caller's own.
    assert "__version__" not in nlolib.__all__


def main() -> None:
    test_package_surface()
    test_version_attribute()
    print("test_python_package_surface: package exports validated.")


if __name__ == "__main__":
    main()
