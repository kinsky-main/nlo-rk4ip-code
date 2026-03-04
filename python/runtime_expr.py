"""
Callable-to-runtime-expression translation utilities.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Mapping

RUNTIME_CONTEXT_DISPERSION = "dispersion"
RUNTIME_CONTEXT_DISPERSION_FACTOR = "dispersion_factor"
RUNTIME_CONTEXT_NONLINEAR = "nonlinear"
RUNTIME_CONTEXT_LINEAR_FACTOR = "linear_factor"
RUNTIME_CONTEXT_LINEAR = "linear"
RUNTIME_CONTEXT_POTENTIAL = "potential"

_ALLOWED_FUNCS = {"exp", "log", "sqrt", "sin", "cos"}

_CONTEXT_ALLOWED_SYMBOLS: dict[str, set[str]] = {
    RUNTIME_CONTEXT_DISPERSION_FACTOR: {"A", "w", "wt", "kx", "ky", "x", "y", "t"},
    RUNTIME_CONTEXT_DISPERSION: {"A", "D", "h", "w", "wt", "kx", "ky", "x", "y", "t", "V"},
    RUNTIME_CONTEXT_NONLINEAR: {"A", "I", "V"},
    RUNTIME_CONTEXT_LINEAR_FACTOR: {"A", "wt", "kx", "ky", "x", "y", "t"},
    RUNTIME_CONTEXT_LINEAR: {"A", "D", "h", "wt", "kx", "ky", "x", "y", "t", "V"},
    RUNTIME_CONTEXT_POTENTIAL: {"x", "y", "t", "wt", "kx", "ky"},
}

_CONTEXT_DEFAULT_PARAM_SYMBOLS: dict[str, list[str]] = {
    RUNTIME_CONTEXT_DISPERSION_FACTOR: ["A", "w", "wt", "kx", "ky", "x", "y", "t"],
    RUNTIME_CONTEXT_DISPERSION: ["A", "D", "h", "w", "wt", "kx", "ky", "x", "y", "t", "V"],
    RUNTIME_CONTEXT_NONLINEAR: ["A", "I", "V"],
    RUNTIME_CONTEXT_LINEAR_FACTOR: ["A", "wt", "kx", "ky", "x", "y", "t"],
    RUNTIME_CONTEXT_LINEAR: ["A", "D", "h", "wt", "kx", "ky", "x", "y", "t", "V"],
    RUNTIME_CONTEXT_POTENTIAL: ["x", "y", "t", "wt", "kx", "ky"],
}

_CONTEXT_MIN_PARAMS: dict[str, int] = {
    RUNTIME_CONTEXT_DISPERSION_FACTOR: 1,
    RUNTIME_CONTEXT_DISPERSION: 1,
    RUNTIME_CONTEXT_NONLINEAR: 1,
    RUNTIME_CONTEXT_LINEAR_FACTOR: 1,
    RUNTIME_CONTEXT_LINEAR: 1,
    RUNTIME_CONTEXT_POTENTIAL: 1,
}


@dataclass
class TranslatedExpression:
    expression: str
    constants: list[float]


def _is_real_scalar(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _format_real(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("runtime expression constants must be finite real scalars")
    return format(float(value), ".17g")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _resolve_attribute_chain(node: ast.Attribute, scope: Mapping[str, Any]) -> Any:
    chain: list[str] = [node.attr]
    cursor: ast.AST = node.value
    while isinstance(cursor, ast.Attribute):
        chain.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        raise ValueError("unsupported attribute chain in runtime callable")

    base_name = cursor.id
    if base_name not in scope:
        raise ValueError(f"unresolved attribute base '{base_name}' in runtime callable")
    value = scope[base_name]
    for attr in reversed(chain):
        if not hasattr(value, attr):
            raise ValueError(f"unresolved attribute '{attr}' in runtime callable")
        value = getattr(value, attr)
    return value


def _extract_expr_from_callable(func: Callable[..., Any]) -> ast.AST:
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as exc:
        raise ValueError(
            "unable to inspect callable source; pass an explicit expression string instead"
        ) from exc

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            return node.body

    fn_name = getattr(func, "__name__", None)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    return stmt.value
            raise ValueError("function callable must contain a return expression")

    raise ValueError("could not extract expression body from callable")


class _ExpressionTranslator:
    def __init__(
        self,
        context: str,
        symbol_map: Mapping[str, str],
        captured_scope: Mapping[str, Any],
        constant_bindings: Mapping[str, float] | None,
        auto_capture: bool,
    ):
        self._context = context
        self._symbol_map = dict(symbol_map)
        self._captured_scope = dict(captured_scope)
        self._constant_bindings = dict(constant_bindings or {})
        self._auto_capture = auto_capture
        self._constants: list[float] = []
        self._constants_by_name: dict[str, int] = {}

    def _require_symbol_allowed(self, symbol: str) -> None:
        allowed = _CONTEXT_ALLOWED_SYMBOLS.get(self._context, set())
        if symbol in allowed:
            return
        raise ValueError(f"'{symbol}' is not valid in {self._context} runtime expressions")

    def _add_named_constant(self, name: str, value: float) -> str:
        if name in self._constants_by_name:
            idx = self._constants_by_name[name]
        else:
            idx = len(self._constants)
            self._constants_by_name[name] = idx
            self._constants.append(float(value))
        return f"c{idx}"

    def _resolve_name_constant(self, name: str) -> str:
        if name in self._constant_bindings:
            value = self._constant_bindings[name]
            if not _is_real_scalar(value):
                raise ValueError(f"constant binding '{name}' must be a finite real scalar")
            return self._add_named_constant(f"binding:{name}", float(value))

        if not self._auto_capture or name not in self._captured_scope:
            raise ValueError(
                f"unknown identifier '{name}' in runtime callable; "
                "add it to constant_bindings or use an explicit expression string"
            )

        value = self._captured_scope[name]
        if isinstance(value, complex):
            if value.real == 0.0 and value.imag == 1.0:
                return "i"
            if value.real == 0.0 and value.imag == -1.0:
                return "(-i)"
            raise ValueError(
                f"captured complex identifier '{name}' is unsupported; use real scalars or 'i'"
            )

        if not _is_real_scalar(value):
            raise ValueError(
                f"captured identifier '{name}' must be a finite real scalar"
            )
        return self._add_named_constant(name, float(value))

    def _translate_constant(self, value: Any) -> str:
        if isinstance(value, complex):
            if value.real == 0.0 and value.imag == 1.0:
                return "i"
            if value.real == 0.0 and value.imag == -1.0:
                return "(-i)"
            if value.imag == 0.0:
                return _format_real(float(value.real))
            raise ValueError("complex literals other than +/-1j are unsupported")
        if _is_real_scalar(value):
            return _format_real(float(value))
        raise ValueError(f"unsupported literal in runtime callable: {value!r}")

    def translate(self, node: ast.AST) -> str:
        if isinstance(node, ast.BinOp):
            left = self.translate(node.left)
            right = self.translate(node.right)
            if isinstance(node.op, ast.Add):
                return f"({left}+{right})"
            if isinstance(node.op, ast.Sub):
                return f"({left}-{right})"
            if isinstance(node.op, ast.Mult):
                return f"({left}*{right})"
            if isinstance(node.op, ast.Div):
                return f"({left}/{right})"
            if isinstance(node.op, ast.Pow):
                return f"({left}^{right})"
            raise ValueError("unsupported binary operator in runtime callable")

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                operand = self.translate(node.operand)
                return f"(-{operand})"
            if isinstance(node.op, ast.UAdd):
                return self.translate(node.operand)
            raise ValueError("unsupported unary operator in runtime callable")

        if isinstance(node, ast.Call):
            fn_name = _call_name(node.func)
            if fn_name is None or fn_name not in _ALLOWED_FUNCS:
                raise ValueError("unsupported function call in runtime callable")
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"{fn_name}() requires exactly one positional argument")
            arg_expr = self.translate(node.args[0])
            return f"{fn_name}({arg_expr})"

        if isinstance(node, ast.Name):
            if node.id in self._symbol_map:
                symbol = self._symbol_map[node.id]
                self._require_symbol_allowed(symbol)
                return symbol

            if node.id == "i":
                return node.id
            if node.id in {"w", "wt", "A", "I", "D", "V", "h", "kx", "ky", "x", "y", "t"}:
                self._require_symbol_allowed(node.id)
                return node.id

            return self._resolve_name_constant(node.id)

        if isinstance(node, ast.Attribute):
            value = _resolve_attribute_chain(node, self._captured_scope)
            return self._translate_constant(value)

        if isinstance(node, ast.Constant):
            return self._translate_constant(node.value)

        raise ValueError(
            f"unsupported expression node '{type(node).__name__}' in runtime callable"
        )

    @property
    def constants(self) -> list[float]:
        return list(self._constants)


def translate_callable(
    func: Callable[..., Any],
    context: str,
    *,
    constant_bindings: Mapping[str, float] | None = None,
    auto_capture: bool = True,
) -> TranslatedExpression:
    if context not in _CONTEXT_ALLOWED_SYMBOLS:
        raise ValueError(
            "context must be one of: "
            "'dispersion_factor', 'dispersion', 'nonlinear', "
            "'linear_factor', 'linear', or 'potential'"
        )

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    for param in params:
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError("runtime callables must only use positional parameters")

    min_params = _CONTEXT_MIN_PARAMS[context]
    default_symbols = _CONTEXT_DEFAULT_PARAM_SYMBOLS[context]
    if len(params) < min_params or len(params) > len(default_symbols):
        raise ValueError(
            f"{context} callable must take between {min_params} and "
            f"{len(default_symbols)} positional arguments"
        )

    allowed_symbols = _CONTEXT_ALLOWED_SYMBOLS[context]
    symbol_map: dict[str, str] = {}
    used_symbols: set[str] = set()
    default_idx = 0
    for param in params:
        if param.name in allowed_symbols and param.name not in used_symbols:
            symbol = param.name
        else:
            while default_idx < len(default_symbols) and default_symbols[default_idx] in used_symbols:
                default_idx += 1
            if default_idx >= len(default_symbols):
                raise ValueError(f"unsupported parameter '{param.name}' for {context} callable")
            symbol = default_symbols[default_idx]
            default_idx += 1
        symbol_map[param.name] = symbol
        used_symbols.add(symbol)

    closure_vars = inspect.getclosurevars(func)
    captured_scope: dict[str, Any] = {}
    captured_scope.update(closure_vars.builtins)
    captured_scope.update(closure_vars.globals)
    captured_scope.update(closure_vars.nonlocals)

    expr_node = _extract_expr_from_callable(func)
    translator = _ExpressionTranslator(
        context=context,
        symbol_map=symbol_map,
        captured_scope=captured_scope,
        constant_bindings=constant_bindings,
        auto_capture=auto_capture,
    )
    expression = translator.translate(expr_node)
    return TranslatedExpression(expression=expression, constants=translator.constants)
