import importlib.util
import pathlib
import re

import nlolib

ROOT = pathlib.Path(__file__).resolve().parents[1]


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
    """nlolib.__version__ must name the version this tree packages.

    pyproject.toml is the source of truth for the Python package version; this
    fails if the attribute stops tracking it.
    """
    assert hasattr(nlolib, "__version__")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', pyproject)
    assert match, "could not parse the package version from pyproject.toml"
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
