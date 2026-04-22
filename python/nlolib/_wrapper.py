"""
Flat module-level convenience wrapper for the nlolib package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ._binding import NLOLIB_LOG_LEVEL_INFO, NLOLIB_PROGRESS_STREAM_STDERR, NLOLIB_STATUS_OK
from ._client import NLolib
from ._models import OperatorSpec, PropagationResult, PulseSpec

_DEFAULT_API: NLolib | None = None


def _api() -> NLolib:
    global _DEFAULT_API
    if _DEFAULT_API is None:
        _DEFAULT_API = NLolib()
    return _DEFAULT_API


def _enforce_callable_operator(name: str, operator: OperatorSpec) -> None:
    if not isinstance(operator, OperatorSpec):
        raise TypeError(f"{name} must be an OperatorSpec")
    if operator.fn is None:
        raise ValueError(f"{name}.fn is required")
    if operator.expr is not None:
        raise ValueError(f"{name}.expr is not supported; use callable fn only")


def _normalize_solver_aliases(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    if "t_span" in out:
        t_span = out.pop("t_span")
        if not isinstance(t_span, Sequence) or len(t_span) != 2:
            raise ValueError("t_span must be a 2-tuple/list")
        t0 = float(t_span[0])
        t1 = float(t_span[1])
        if t0 != 0.0:
            raise ValueError("t_span start must be 0.0 for propagate")
        out["propagation_distance"] = float(t1 - t0)

    if "first_step" in out and "starting_step_size" not in out:
        out["starting_step_size"] = float(out.pop("first_step"))
    if "max_step" in out and "max_step_size" not in out:
        out["max_step_size"] = float(out.pop("max_step"))
    if "min_step" in out and "min_step_size" not in out:
        out["min_step_size"] = float(out.pop("min_step"))
    if "rtol" in out and "error_tolerance" not in out:
        out["error_tolerance"] = float(out.pop("rtol"))
    if "output_mode" in out and "output" not in out:
        out["output"] = out.pop("output_mode")
    return out


def _evaluate_events(
    events: Callable[[float, list[complex]], float] | Sequence[Callable[[float, list[complex]], float]] | None,
    z_axis: list[float],
    records: list[list[complex]],
) -> tuple[list[list[float]], list[list[list[complex]]], int]:
    if events is None or len(z_axis) < 2 or len(records) < 2:
        return [], [], 0
    event_list = [events] if callable(events) else list(events)
    t_events: list[list[float]] = [[] for _ in event_list]
    y_events: list[list[list[complex]]] = [[] for _ in event_list]
    terminal_hit = 0
    for event_idx, event in enumerate(event_list):
        direction = float(getattr(event, "direction", 0.0))
        terminal = bool(getattr(event, "terminal", False))
        for i in range(1, len(z_axis)):
            z0 = float(z_axis[i - 1])
            z1 = float(z_axis[i])
            y0 = records[i - 1]
            y1 = records[i]
            f0 = float(event(z0, y0))
            f1 = float(event(z1, y1))
            crossed = (f0 == 0.0) or (f1 == 0.0) or (f0 < 0.0 < f1) or (f1 < 0.0 < f0)
            if not crossed:
                continue
            slope = f1 - f0
            if direction > 0.0 and slope <= 0.0:
                continue
            if direction < 0.0 and slope >= 0.0:
                continue
            alpha = 0.0 if slope == 0.0 else (0.0 - f0) / slope
            alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
            ze = z0 + (z1 - z0) * alpha
            ye = [
                complex(y0k.real + (y1k.real - y0k.real) * alpha, y0k.imag + (y1k.imag - y0k.imag) * alpha)
                for y0k, y1k in zip(y0, y1)
            ]
            t_events[event_idx].append(float(ze))
            y_events[event_idx].append(ye)
            if terminal:
                terminal_hit = 1
                return t_events, y_events, terminal_hit
    return t_events, y_events, terminal_hit


def _build_dense_sol(z_axis: list[float], records: list[list[complex]]) -> Callable[[float], list[complex]]:
    def sol(z: float) -> list[complex]:
        if len(z_axis) <= 0:
            return []
        if z <= z_axis[0]:
            return list(records[0])
        if z >= z_axis[-1]:
            return list(records[-1])
        for i in range(1, len(z_axis)):
            z0 = float(z_axis[i - 1])
            z1 = float(z_axis[i])
            if z <= z1:
                alpha = (float(z) - z0) / (z1 - z0) if z1 > z0 else 0.0
                return [
                    complex(
                        y0.real + (y1.real - y0.real) * alpha,
                        y0.imag + (y1.imag - y0.imag) * alpha,
                    )
                    for y0, y1 in zip(records[i - 1], records[i])
                ]
        return list(records[-1])

    return sol


@dataclass
class PropagateResult:
    records: list[list[complex]]
    z_axis: list[float]
    final: list[complex]
    meta: dict[str, Any]
    status: int
    message: str
    t_events: list[list[float]]
    y_events: list[list[list[complex]]]
    sol: Callable[[float], list[complex]] | None = None


def propagate(
    pulse: PulseSpec,
    linear_operator: OperatorSpec,
    nonlinear_operator: OperatorSpec,
    **kwargs: Any,
) -> PropagateResult:
    """Propagate a pulse with deterministic solver semantics."""
    _enforce_callable_operator("linear_operator", linear_operator)
    _enforce_callable_operator("nonlinear_operator", nonlinear_operator)
    options = _normalize_solver_aliases(kwargs)
    dense_output = bool(options.pop("dense_output", False))
    events = options.pop("events", None)
    result: PropagationResult = _api().propagate(
        pulse,
        linear_operator,
        nonlinear_operator,
        **options,
    )
    t_events, y_events, terminal_hit = _evaluate_events(events, result.z_axis, result.records)
    records = list(result.records)
    z_axis = list(result.z_axis)
    final = list(result.final)
    status = int(result.meta.get("status", NLOLIB_STATUS_OK))
    message = str(result.meta.get("message", "propagate completed"))
    if terminal_hit != 0:
        event_z = None
        event_y = None
        for i, series in enumerate(t_events):
            if len(series) > 0:
                candidate = float(series[0])
                if event_z is None or candidate < event_z:
                    event_z = candidate
                    event_y = y_events[i][0]
        if event_z is not None and event_y is not None:
            kept_records: list[list[complex]] = []
            kept_z: list[float] = []
            for z, y in zip(z_axis, records):
                if float(z) <= float(event_z):
                    kept_z.append(float(z))
                    kept_records.append(list(y))
            if len(kept_z) <= 0 or abs(kept_z[-1] - float(event_z)) > 1e-15:
                kept_z.append(float(event_z))
                kept_records.append(list(event_y))
            z_axis = kept_z
            records = kept_records
            final = list(records[-1]) if len(records) > 0 else []
        status = 1
        message = "A termination event occurred."
    sol = _build_dense_sol(z_axis, records) if dense_output else None
    return PropagateResult(
        records=records,
        z_axis=z_axis,
        final=final,
        meta=result.meta,
        status=status,
        message=message,
        t_events=t_events,
        y_events=y_events,
        sol=sol,
    )


def query_runtime_limits(*args: Any, **kwargs: Any):
    return _api().query_runtime_limits(*args, **kwargs)


def storage_is_available() -> bool:
    return _api().storage_is_available()


def perf_profile_set_enabled(enabled: bool = True) -> None:
    _api().perf_profile_set_enabled(enabled=enabled)


def perf_profile_is_enabled() -> bool:
    return _api().perf_profile_is_enabled()


def perf_profile_reset() -> None:
    _api().perf_profile_reset()


def perf_profile_read():
    return _api().perf_profile_read()


def set_log_level(level: int = NLOLIB_LOG_LEVEL_INFO) -> None:
    _api().set_log_level(level)


def set_log_buffer(capacity_bytes: int = 256 * 1024) -> None:
    _api().set_log_buffer(capacity_bytes)


def clear_log_buffer() -> None:
    _api().clear_log_buffer()


def read_log_buffer(consume: bool = True, max_bytes: int = 256 * 1024) -> str:
    return _api().read_log_buffer(consume=consume, max_bytes=max_bytes)


def set_progress_options(
    enabled: bool = True,
    milestone_percent: int = 5,
    emit_on_step_adjust: bool = False,
) -> None:
    _api().set_progress_options(
        enabled=enabled,
        milestone_percent=milestone_percent,
        emit_on_step_adjust=emit_on_step_adjust,
    )


def set_progress_stream(stream_mode: int = NLOLIB_PROGRESS_STREAM_STDERR) -> None:
    _api().set_progress_stream(stream_mode=stream_mode)
