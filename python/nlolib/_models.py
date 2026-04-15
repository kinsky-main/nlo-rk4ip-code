"""
Public Python-side model types for the nlolib wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ._binding import NONLINEAR_MODEL_EXPR, TENSOR_LAYOUT_XYT_T_FAST


@dataclass
class RuntimeOperators:
    linear_factor_expr: str | None = None
    linear_expr: str | None = None
    potential_expr: str | None = None
    dispersion_factor_expr: str | None = None
    dispersion_expr: str | None = None
    nonlinear_expr: str | None = None
    linear_factor_fn: Callable[..., object] | None = None
    linear_fn: Callable[..., object] | None = None
    potential_fn: Callable[..., object] | None = None
    dispersion_factor_fn: Callable[..., object] | None = None
    dispersion_fn: Callable[..., object] | None = None
    nonlinear_fn: Callable[..., object] | None = None
    nonlinear_model: int = NONLINEAR_MODEL_EXPR
    nonlinear_gamma: float = 0.0
    raman_fraction: float = 0.0
    raman_tau1: float = 0.0122
    raman_tau2: float = 0.0320
    shock_omega0: float = 0.0
    raman_response_time: Sequence[complex] | None = None
    constants: Sequence[float] = ()
    constant_bindings: Mapping[str, float] | None = None
    auto_capture_constants: bool = True


@dataclass
class PulseSpec:
    samples: Sequence[complex]
    delta_time: float
    pulse_period: float | None = None
    frequency_grid: Sequence[complex] | None = None
    tensor_nt: int | None = None
    tensor_nx: int | None = None
    tensor_ny: int | None = None
    tensor_layout: int = TENSOR_LAYOUT_XYT_T_FAST
    delta_x: float = 1.0
    delta_y: float = 1.0
    spatial_frequency_grid: Sequence[complex] | None = None
    potential_grid: Sequence[complex] | None = None


@dataclass
class OperatorSpec:
    expr: str | None = None
    fn: Callable[..., object] | None = None
    params: Mapping[str, float] | Sequence[float] | None = None


@dataclass
class PropagationResult:
    records: list[list[complex]]
    z_axis: list[float]
    final: list[complex]
    meta: dict[str, Any]


@dataclass(frozen=True)
class ProgressInfo:
    event_type: int
    step_index: int
    reject_attempt: int
    z: float
    z_end: float
    percent: float
    step_size: float
    next_step_size: float
    error: float
    elapsed_seconds: float
    eta_seconds: float


class PropagationAbortedError(RuntimeError):
    def __init__(self, result: PropagationResult):
        super().__init__("nlolib_propagate aborted by progress callback")
        self.result = result
