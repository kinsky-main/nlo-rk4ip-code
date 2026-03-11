import nlolib


def test_package_surface() -> None:
    assert hasattr(nlolib, "NLolib")
    assert hasattr(nlolib, "propagate")
    assert hasattr(nlolib, "query_runtime_limits")
    assert hasattr(nlolib, "PulseSpec")
    assert hasattr(nlolib, "OperatorSpec")
    assert hasattr(nlolib, "RuntimeOperators")
    assert hasattr(nlolib, "translate_callable")
    assert "._legacy_impl" not in nlolib.NLolib.__module__
    assert "._legacy_impl" not in nlolib.propagate.__module__


def main() -> None:
    test_package_surface()
    print("test_python_package_surface: package exports validated.")


if __name__ == "__main__":
    main()
