import math

from runtime_expr import (
    RUNTIME_CONTEXT_DISPERSION_FACTOR,
    RUNTIME_CONTEXT_NONLINEAR,
    translate_callable,
)


def _assert_raises_value_error(func, expected_substring):
    try:
        func()
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert expected_substring in str(exc), f"missing '{expected_substring}' in '{exc}'"


def test_numpy_functions_translate():
    alpha = 0.25
    # Translation is syntax-based, so NumPy-style qualified calls are validated
    # without requiring numpy to be installed in the test environment.
    translated = translate_callable(
        lambda A, w: 1j * (np.sin(w) + alpha * np.cos(w)),  # noqa: E731
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
    )
    assert "sin(" in translated.expression
    assert "cos(" in translated.expression
    assert translated.constants == [alpha]


def test_math_functions_translate():
    offset = 1.0
    translated = translate_callable(
        lambda A, w: 1j * (math.sqrt((w * w) + offset) + math.log((w * w) + offset)),  # noqa: E731
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
    )
    assert "sqrt(" in translated.expression
    assert "log(" in translated.expression
    assert translated.constants == [offset]


def test_beta_sum_translation():
    beta2 = 0.5
    beta3 = -0.02
    beta4 = 0.003
    translated = translate_callable(
        lambda A, w: 1j * (beta2 * (w**2) + beta3 * (w**3) + beta4 * (w**4)),  # noqa: E731
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
    )
    assert "^2" in translated.expression
    assert "^3" in translated.expression
    assert "^4" in translated.expression
    assert translated.constants == [beta2, beta3, beta4]


def test_diffraction_factor_translation():
    beta_t = -0.015
    translated = translate_callable(
        lambda A, w: 1j * beta_t * w,  # noqa: E731
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
    )
    assert translated.expression.count("w") > 0
    assert translated.constants == [beta_t]


def test_raman_like_nonlinear_translation():
    gamma = 0.01
    f_r = 0.18
    translated = translate_callable(
        lambda A, I, V: 1j * A * (gamma * (1.0 - f_r) * I + gamma * f_r * V),  # noqa: E731
        RUNTIME_CONTEXT_NONLINEAR,
    )
    assert "A" in translated.expression
    assert "I" in translated.expression
    assert "V" in translated.expression
    assert translated.constants == [gamma, f_r]


def test_rejects_unsupported_numpy_call():
    _assert_raises_value_error(
        lambda: translate_callable(  # noqa: E731
            lambda A, w: np.tanh(w),  # noqa: E731
            RUNTIME_CONTEXT_DISPERSION_FACTOR,
        ),
        "unsupported function call",
    )


def test_rejects_invalid_signature():
    _assert_raises_value_error(
        lambda: translate_callable(  # noqa: E731
            lambda: 0.0,  # noqa: E731
            RUNTIME_CONTEXT_DISPERSION_FACTOR,
        ),
        "dispersion_factor callable",
    )


def test_rejects_disallowed_complex_literal():
    _assert_raises_value_error(
        lambda: translate_callable(  # noqa: E731
            lambda A, w: (2.0 + 3.0j) * w,  # noqa: E731
            RUNTIME_CONTEXT_DISPERSION_FACTOR,
        ),
        "complex literals",
    )


def main():
    test_numpy_functions_translate()
    test_math_functions_translate()
    test_beta_sum_translation()
    test_diffraction_factor_translation()
    test_raman_like_nonlinear_translation()
    test_rejects_unsupported_numpy_call()
    test_rejects_invalid_signature()
    test_rejects_disallowed_complex_literal()
    print("test_python_runtime_expr_translation: callable translation coverage validated.")


if __name__ == "__main__":
    main()
