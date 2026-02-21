"""
ctypes bindings and high-level Python API for NLOLib.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .runtime_expr import (  # type: ignore[attr-defined]
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
        RUNTIME_CONTEXT_DISPERSION,
        RUNTIME_CONTEXT_NONLINEAR,
        translate_callable,
    )
except ImportError:
    from runtime_expr import (
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
        RUNTIME_CONTEXT_DISPERSION,
        RUNTIME_CONTEXT_NONLINEAR,
        translate_callable,
    )

NT_MAX = ctypes.c_size_t(-1).value  # Unbounded sentinel; prefer query_runtime_limits().
LEGACY_NT_MAX = 1 << 20
NLO_RUNTIME_OPERATOR_CONSTANTS_MAX = 16
NLO_STORAGE_RUN_ID_MAX = 64

NLO_VECTOR_BACKEND_CPU = 0
NLO_VECTOR_BACKEND_VULKAN = 1
NLO_VECTOR_BACKEND_AUTO = 2

NLO_FFT_BACKEND_AUTO = 0
NLO_FFT_BACKEND_FFTW = 1
NLO_FFT_BACKEND_VKFFT = 2

NLOLIB_STATUS_OK = 0
NLOLIB_STATUS_INVALID_ARGUMENT = 1
NLOLIB_STATUS_ALLOCATION_FAILED = 2
NLOLIB_STATUS_NOT_IMPLEMENTED = 3

NLOLIB_LOG_LEVEL_ERROR = 0
NLOLIB_LOG_LEVEL_WARN = 1
NLOLIB_LOG_LEVEL_INFO = 2
NLOLIB_LOG_LEVEL_DEBUG = 3

NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES = 0
NLO_STORAGE_DB_CAP_POLICY_FAIL = 1


class NloComplex(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


VkPhysicalDevice = ctypes.c_void_p
VkDevice = ctypes.c_void_p
VkQueue = ctypes.c_void_p
VkCommandPool = ctypes.c_void_p


class NloVkBackendConfig(ctypes.Structure):
    _fields_ = [
        ("physical_device", VkPhysicalDevice),
        ("device", VkDevice),
        ("queue", VkQueue),
        ("queue_family_index", ctypes.c_uint32),
        ("command_pool", VkCommandPool),
        ("descriptor_set_budget_bytes", ctypes.c_size_t),
        ("descriptor_set_count_override", ctypes.c_uint32),
    ]


class PropagationParams(ctypes.Structure):
    _fields_ = [
        ("starting_step_size", ctypes.c_double),
        ("max_step_size", ctypes.c_double),
        ("min_step_size", ctypes.c_double),
        ("error_tolerance", ctypes.c_double),
        ("propagation_distance", ctypes.c_double),
    ]


class TimeGrid(ctypes.Structure):
    _fields_ = [
        ("nt", ctypes.c_size_t),
        ("pulse_period", ctypes.c_double),
        ("delta_time", ctypes.c_double),
    ]


class FrequencyGrid(ctypes.Structure):
    _fields_ = [("frequency_grid", ctypes.POINTER(NloComplex))]


class SpatialGrid(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_size_t),
        ("ny", ctypes.c_size_t),
        ("delta_x", ctypes.c_double),
        ("delta_y", ctypes.c_double),
        ("spatial_frequency_grid", ctypes.POINTER(NloComplex)),
        ("potential_grid", ctypes.POINTER(NloComplex)),
    ]


class RuntimeOperatorParams(ctypes.Structure):
    _fields_ = [
        ("dispersion_factor_expr", ctypes.c_char_p),
        ("dispersion_expr", ctypes.c_char_p),
        ("transverse_factor_expr", ctypes.c_char_p),
        ("transverse_expr", ctypes.c_char_p),
        ("nonlinear_expr", ctypes.c_char_p),
        ("num_constants", ctypes.c_size_t),
        ("constants", ctypes.c_double * NLO_RUNTIME_OPERATOR_CONSTANTS_MAX),
    ]


class SimConfig(ctypes.Structure):
    _fields_ = [
        ("propagation", PropagationParams),
        ("time", TimeGrid),
        ("frequency", FrequencyGrid),
        ("spatial", SpatialGrid),
        ("runtime", RuntimeOperatorParams),
    ]


class NloExecutionOptions(ctypes.Structure):
    _fields_ = [
        ("backend_type", ctypes.c_int),
        ("fft_backend", ctypes.c_int),
        ("device_heap_fraction", ctypes.c_double),
        ("record_ring_target", ctypes.c_size_t),
        ("forced_device_budget_bytes", ctypes.c_size_t),
        ("vulkan", NloVkBackendConfig),
    ]


class NloRuntimeLimits(ctypes.Structure):
    _fields_ = [
        ("max_num_time_samples_runtime", ctypes.c_size_t),
        ("max_num_recorded_samples_in_memory", ctypes.c_size_t),
        ("max_num_recorded_samples_with_storage", ctypes.c_size_t),
        ("estimated_required_working_set_bytes", ctypes.c_size_t),
        ("estimated_device_budget_bytes", ctypes.c_size_t),
        ("storage_available", ctypes.c_int),
    ]


class NloStorageOptions(ctypes.Structure):
    _fields_ = [
        ("sqlite_path", ctypes.c_char_p),
        ("run_id", ctypes.c_char_p),
        ("sqlite_max_bytes", ctypes.c_size_t),
        ("chunk_records", ctypes.c_size_t),
        ("cap_policy", ctypes.c_int),
        ("log_final_output_field_to_db", ctypes.c_int),
    ]


class NloStorageResult(ctypes.Structure):
    _fields_ = [
        ("run_id", ctypes.c_char * NLO_STORAGE_RUN_ID_MAX),
        ("records_captured", ctypes.c_size_t),
        ("records_spilled", ctypes.c_size_t),
        ("chunks_written", ctypes.c_size_t),
        ("db_size_bytes", ctypes.c_size_t),
        ("truncated", ctypes.c_int),
    ]


def _candidate_library_paths() -> list[str]:
    candidates: list[str] = []
    env_path = os.environ.get("NLOLIB_LIBRARY")
    if env_path:
        candidates.append(env_path)

    here = Path(__file__).resolve().parent
    package_dir = here / "nlolib"
    root = here.parent
    if os.name == "nt":
        candidates.extend(
            [
                str(package_dir / "nlolib.dll"),
                str(here / "nlolib.dll"),
                str(here / "Debug" / "nlolib.dll"),
                str(here / "Release" / "nlolib.dll"),
                str(here / "RelWithDebInfo" / "nlolib.dll"),
                str(here / "MinSizeRel" / "nlolib.dll"),
                str(root / "build" / "src" / "Release" / "nlolib.dll"),
                str(root / "build" / "src" / "Debug" / "nlolib.dll"),
                str(root / "build" / "src" / "RelWithDebInfo" / "nlolib.dll"),
                str(root / "build" / "src" / "MinSizeRel" / "nlolib.dll"),
                str(root / "build" / "examples" / "Release" / "nlolib.dll"),
                str(root / "build" / "examples" / "Debug" / "nlolib.dll"),
                str(root / "build" / "examples" / "RelWithDebInfo" / "nlolib.dll"),
                str(root / "build" / "examples" / "MinSizeRel" / "nlolib.dll"),
                str(root / "nlolib.dll"),
            ]
        )
    elif os.name == "posix":
        candidates.extend(
            [
                str(package_dir / "libnlolib.so"),
                str(here / "libnlolib.so"),
                str(root / "libnlolib.so"),
                str(package_dir / "libnlolib.dylib"),
                str(here / "libnlolib.dylib"),
                str(root / "libnlolib.dylib"),
            ]
        )

    found = ctypes.util.find_library("nlolib")
    if found:
        candidates.append(found)

    return candidates


def load(path: str | None = None) -> ctypes.CDLL:
    """
    Load and configure the NLOLib shared library.

    Set NLOLIB_LIBRARY to override discovery.
    """
    lib_path: str | None = path
    env_override = os.environ.get("NLOLIB_LIBRARY")
    if lib_path is None and env_override:
        lib_path = env_override

    if lib_path is not None:
        lib = ctypes.CDLL(lib_path)
    else:
        preferred_lib = None
        preferred_path = None
        fallback_lib = None
        fallback_path = None
        for candidate in _candidate_library_paths():
            try:
                loaded = ctypes.CDLL(candidate)
            except OSError:
                continue

            if hasattr(loaded, "nlolib_query_runtime_limits"):
                preferred_lib = loaded
                preferred_path = candidate
                break

            if fallback_lib is None and hasattr(loaded, "nlolib_propagate"):
                fallback_lib = loaded
                fallback_path = candidate

        if preferred_lib is not None:
            lib = preferred_lib
            lib_path = preferred_path
        elif fallback_lib is not None:
            lib = fallback_lib
            lib_path = fallback_path
        else:
            raise OSError(
                "Unable to locate NLOLib shared library. "
                "Set NLOLIB_LIBRARY to the full path."
            )

    lib.nlolib_propagate.argtypes = [
        ctypes.POINTER(SimConfig),
        ctypes.c_size_t,
        ctypes.POINTER(NloComplex),
        ctypes.c_size_t,
        ctypes.POINTER(NloComplex),
        ctypes.POINTER(NloExecutionOptions),
    ]
    lib.nlolib_propagate.restype = ctypes.c_int
    try:
        lib.nlolib_query_runtime_limits.argtypes = [
            ctypes.POINTER(SimConfig),
            ctypes.POINTER(NloExecutionOptions),
            ctypes.POINTER(NloRuntimeLimits),
        ]
        lib.nlolib_query_runtime_limits.restype = ctypes.c_int
        lib._has_query_runtime_limits = True
    except AttributeError:
        lib._has_query_runtime_limits = False

    try:
        lib.nlolib_propagate_with_storage.argtypes = [
            ctypes.POINTER(SimConfig),
            ctypes.c_size_t,
            ctypes.POINTER(NloComplex),
            ctypes.c_size_t,
            ctypes.POINTER(NloComplex),
            ctypes.POINTER(NloExecutionOptions),
            ctypes.POINTER(NloStorageOptions),
            ctypes.POINTER(NloStorageResult),
        ]
        lib.nlolib_propagate_with_storage.restype = ctypes.c_int
        lib._has_propagate_with_storage = True
    except AttributeError:
        lib._has_propagate_with_storage = False

    try:
        lib.nlolib_storage_is_available.argtypes = []
        lib.nlolib_storage_is_available.restype = ctypes.c_int
        lib._has_storage_is_available = True
    except AttributeError:
        lib._has_storage_is_available = False

    try:
        lib.nlolib_set_log_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.nlolib_set_log_file.restype = ctypes.c_int
        lib._has_set_log_file = True
    except AttributeError:
        lib._has_set_log_file = False

    try:
        lib.nlolib_set_log_buffer.argtypes = [ctypes.c_size_t]
        lib.nlolib_set_log_buffer.restype = ctypes.c_int
        lib._has_set_log_buffer = True
    except AttributeError:
        lib._has_set_log_buffer = False

    try:
        lib.nlolib_clear_log_buffer.argtypes = []
        lib.nlolib_clear_log_buffer.restype = ctypes.c_int
        lib._has_clear_log_buffer = True
    except AttributeError:
        lib._has_clear_log_buffer = False

    try:
        lib.nlolib_read_log_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.nlolib_read_log_buffer.restype = ctypes.c_int
        lib._has_read_log_buffer = True
    except AttributeError:
        lib._has_read_log_buffer = False

    try:
        lib.nlolib_set_log_level.argtypes = [ctypes.c_int]
        lib.nlolib_set_log_level.restype = ctypes.c_int
        lib._has_set_log_level = True
    except AttributeError:
        lib._has_set_log_level = False

    try:
        lib.nlolib_set_progress_options.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.nlolib_set_progress_options.restype = ctypes.c_int
        lib._has_set_progress_options = True
    except AttributeError:
        lib._has_set_progress_options = False

    lib._nlo_loaded_path = str(lib_path) if lib_path is not None else ""
    return lib


def make_complex_array(values: Sequence[complex] | int) -> ctypes.Array:
    """
    Build a ctypes complex array from python values or allocate by length.
    """
    if isinstance(values, int):
        return (NloComplex * values)()

    out = (NloComplex * len(values))()
    for i, val in enumerate(values):
        out[i].re = float(val.real)
        out[i].im = float(val.imag)
    return out


def complex_array_to_list(values: ctypes.Array, count: int) -> list[complex]:
    """
    Convert ctypes complex array to python complex list.
    """
    out = [0j] * count
    for i in range(count):
        out[i] = complex(values[i].re, values[i].im)
    return out


def default_execution_options(
    backend_type: int = NLO_VECTOR_BACKEND_AUTO,
    fft_backend: int = NLO_FFT_BACKEND_AUTO,
) -> NloExecutionOptions:
    """
    Build default execution options matching C defaults.
    """
    options = NloExecutionOptions()
    options.backend_type = int(backend_type)
    options.fft_backend = int(fft_backend)
    options.device_heap_fraction = 0.70
    options.record_ring_target = 0
    options.forced_device_budget_bytes = 0
    options.vulkan.physical_device = None
    options.vulkan.device = None
    options.vulkan.queue = None
    options.vulkan.queue_family_index = 0
    options.vulkan.command_pool = None
    options.vulkan.descriptor_set_budget_bytes = 0
    options.vulkan.descriptor_set_count_override = 0
    return options


def default_storage_options(
    *,
    sqlite_path: str,
    run_id: str | None = None,
    sqlite_max_bytes: int = 0,
    chunk_records: int = 0,
    cap_policy: int = NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES,
    log_final_output_field_to_db: bool = False,
) -> tuple[NloStorageOptions, list[bytes]]:
    """
    Build storage options and return keepalive byte buffers for ctypes strings.
    """
    opts = NloStorageOptions()
    keepalive: list[bytes] = []

    if not sqlite_path:
        raise ValueError("sqlite_path must be a non-empty string")
    path_b = sqlite_path.encode("utf-8")
    keepalive.append(path_b)
    opts.sqlite_path = ctypes.c_char_p(path_b)

    if run_id is not None and run_id != "":
        run_b = run_id.encode("utf-8")
        keepalive.append(run_b)
        opts.run_id = ctypes.c_char_p(run_b)
    else:
        opts.run_id = None

    opts.sqlite_max_bytes = int(sqlite_max_bytes)
    opts.chunk_records = int(chunk_records)
    opts.cap_policy = int(cap_policy)
    opts.log_final_output_field_to_db = int(bool(log_final_output_field_to_db))
    return opts, keepalive


@dataclass
class RuntimeOperators:
    dispersion_factor_expr: str | None = None
    dispersion_expr: str | None = None
    transverse_factor_expr: str | None = None
    transverse_expr: str | None = None
    nonlinear_expr: str | None = None
    dispersion_factor_fn: Callable[..., object] | None = None
    dispersion_fn: Callable[..., object] | None = None
    transverse_factor_fn: Callable[..., object] | None = None
    transverse_fn: Callable[..., object] | None = None
    nonlinear_fn: Callable[..., object] | None = None
    constants: Sequence[float] = ()
    constant_bindings: Mapping[str, float] | None = None
    auto_capture_constants: bool = True


@dataclass
class PulseSpec:
    samples: Sequence[complex]
    delta_time: float
    pulse_period: float | None = None
    frequency_grid: Sequence[complex] | None = None
    time_nt: int | None = None
    spatial_nx: int | None = None
    spatial_ny: int | None = None
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


_LINEAR_OPERATOR_PRESETS: dict[str, dict[str, object]] = {
    "none": {"expr": "0", "params": {}},
    "gvd": {"expr": "i*beta2*w*w-loss", "params": {"beta2": -0.5, "loss": 0.0}},
}

_TRANSVERSE_OPERATOR_PRESETS: dict[str, dict[str, object]] = {
    "none": {"expr": "0", "params": {}},
    "diffraction": {"expr": "i*beta_t*w", "params": {"beta_t": 1.0}},
    "default": {"expr": "i*beta_t*w", "params": {"beta_t": 1.0}},
}

_NONLINEAR_OPERATOR_PRESETS: dict[str, dict[str, object]] = {
    "none": {"expr": "0", "params": {}},
    "kerr": {"expr": "i*gamma*I", "params": {"gamma": 1.0}},
}


def _default_frequency_grid(num_time_samples: int, delta_time: float) -> list[complex]:
    two_pi = 2.0 * 3.141592653589793
    omega_step = two_pi / (float(num_time_samples) * delta_time)
    positive_limit = (num_time_samples - 1) // 2
    out: list[complex] = [0j] * num_time_samples
    for i in range(num_time_samples):
        if i <= positive_limit:
            omega = float(i) * omega_step
        else:
            omega = -(float(num_time_samples - i) * omega_step)
        out[i] = complex(omega, 0.0)
    return out


def _normalize_pulse_spec(pulse: PulseSpec | Mapping[str, object]) -> PulseSpec:
    if isinstance(pulse, PulseSpec):
        spec = pulse
    elif isinstance(pulse, Mapping):
        if "samples" not in pulse or "delta_time" not in pulse:
            raise ValueError("pulse mapping must define 'samples' and 'delta_time'")
        spec = PulseSpec(
            samples=pulse["samples"],  # type: ignore[arg-type]
            delta_time=float(pulse["delta_time"]),  # type: ignore[arg-type]
            pulse_period=float(pulse["pulse_period"]) if pulse.get("pulse_period") is not None else None,
            frequency_grid=pulse.get("frequency_grid"),  # type: ignore[arg-type]
            time_nt=int(pulse["time_nt"]) if pulse.get("time_nt") is not None else None,
            spatial_nx=int(pulse["spatial_nx"]) if pulse.get("spatial_nx") is not None else None,
            spatial_ny=int(pulse["spatial_ny"]) if pulse.get("spatial_ny") is not None else None,
            delta_x=float(pulse.get("delta_x", 1.0)),
            delta_y=float(pulse.get("delta_y", 1.0)),
            spatial_frequency_grid=pulse.get("spatial_frequency_grid"),  # type: ignore[arg-type]
            potential_grid=pulse.get("potential_grid"),  # type: ignore[arg-type]
        )
    else:
        raise ValueError("pulse must be a PulseSpec or mapping")
    if len(spec.samples) == 0:
        raise ValueError("pulse.samples must be non-empty")
    if spec.delta_time <= 0.0:
        raise ValueError("pulse.delta_time must be > 0")
    if spec.time_nt is not None and int(spec.time_nt) <= 0:
        raise ValueError("pulse.time_nt must be > 0 when provided")
    return spec


def _validate_coupled_pulse_spec(pulse: PulseSpec) -> None:
    if pulse.time_nt is None or int(pulse.time_nt) <= 0:
        raise ValueError("pulse.time_nt must be > 0 for coupled transverse simulations")
    if pulse.spatial_nx is None or int(pulse.spatial_nx) <= 0:
        raise ValueError("pulse.spatial_nx must be > 0 for coupled transverse simulations")
    if pulse.spatial_ny is None or int(pulse.spatial_ny) <= 0:
        raise ValueError("pulse.spatial_ny must be > 0 for coupled transverse simulations")

    nt = int(pulse.time_nt)
    nx = int(pulse.spatial_nx)
    ny = int(pulse.spatial_ny)
    total = nt * nx * ny
    if len(pulse.samples) != total:
        raise ValueError("len(pulse.samples) must equal pulse.time_nt * pulse.spatial_nx * pulse.spatial_ny")

    xy = nx * ny
    if pulse.spatial_frequency_grid is None:
        raise ValueError("pulse.spatial_frequency_grid is required for coupled transverse simulations")
    if len(pulse.spatial_frequency_grid) not in {xy, total}:
        raise ValueError(
            "pulse.spatial_frequency_grid length must match pulse.spatial_nx*pulse.spatial_ny "
            "(or full-volume length)"
        )
    if pulse.potential_grid is not None and len(pulse.potential_grid) not in {xy, total}:
        raise ValueError(
            "pulse.potential_grid length must match pulse.spatial_nx*pulse.spatial_ny "
            "(or full-volume length)"
        )


def _solver_profile_defaults(profile: str, propagation_distance: float) -> dict[str, float | int]:
    if propagation_distance <= 0.0:
        raise ValueError("propagation_distance must be > 0")

    if profile == "balanced":
        return {
            "starting_step_size": propagation_distance / 200.0,
            "max_step_size": propagation_distance / 25.0,
            "min_step_size": propagation_distance / 20000.0,
            "error_tolerance": 1e-6,
            "records": 128,
        }
    if profile == "fast":
        return {
            "starting_step_size": propagation_distance / 120.0,
            "max_step_size": propagation_distance / 12.0,
            "min_step_size": propagation_distance / 4000.0,
            "error_tolerance": 5e-6,
            "records": 64,
        }
    if profile == "accuracy":
        return {
            "starting_step_size": propagation_distance / 400.0,
            "max_step_size": propagation_distance / 50.0,
            "min_step_size": propagation_distance / 80000.0,
            "error_tolerance": 1e-7,
            "records": 192,
        }
    raise ValueError(f"unsupported preset '{profile}'")


def _parameterize_expression(
    expression: str,
    params: Mapping[str, float] | Sequence[float] | None,
    offset: int,
) -> tuple[str, list[float]]:
    if params is None:
        return expression, []

    if isinstance(params, Mapping):
        rewritten = expression
        constants: list[float] = []
        for idx, (name, value) in enumerate(params.items()):
            if not re.match(r"^[A-Za-z_]\w*$", name):
                raise ValueError(f"invalid parameter name '{name}'")
            rewritten = re.sub(rf"\b{re.escape(name)}\b", f"c{offset + idx}", rewritten)
            constants.append(float(value))
        return rewritten, constants

    constants = [float(value) for value in params]
    return expression, constants


def _coerce_operator_spec(operator: str | OperatorSpec | Mapping[str, object]) -> OperatorSpec:
    if isinstance(operator, OperatorSpec):
        return operator
    if isinstance(operator, str):
        return OperatorSpec(expr=operator)
    if isinstance(operator, Mapping):
        expr = operator.get("expr")
        fn = operator.get("fn")
        params = operator.get("params")
        return OperatorSpec(
            expr=str(expr) if expr is not None else None,
            fn=fn if callable(fn) else None,
            params=params if params is None or isinstance(params, (Mapping, Sequence)) else None,
        )
    raise ValueError("operator must be a preset string, OperatorSpec, or mapping")


def _resolve_operator_spec(
    context: str,
    operator: str | OperatorSpec | Mapping[str, object],
    offset: int,
) -> tuple[str | None, Callable[..., object] | None, list[float], Mapping[str, float] | None]:
    if context == "linear":
        presets = _LINEAR_OPERATOR_PRESETS
    elif context == "transverse":
        presets = _TRANSVERSE_OPERATOR_PRESETS
    else:
        presets = _NONLINEAR_OPERATOR_PRESETS

    spec = _coerce_operator_spec(operator)
    params = spec.params
    expr = spec.expr
    fn = spec.fn

    if expr is not None and fn is not None:
        raise ValueError(f"{context} operator cannot define both expr and fn")

    if fn is None and expr is not None and expr in presets:
        preset_entry = presets[expr]
        expr = str(preset_entry["expr"])
        params = preset_entry.get("params")  # type: ignore[assignment]

    if expr is None and fn is None:
        raise ValueError(f"{context} operator must define expr/fn or a known preset")

    if fn is not None:
        if params is not None and not isinstance(params, Mapping):
            raise ValueError(f"{context} callable operator params must be a mapping")
        bindings = {name: float(value) for name, value in params.items()} if isinstance(params, Mapping) else None
        return None, fn, [], bindings

    assert expr is not None
    resolved_expr, constants = _parameterize_expression(expr, params, offset)
    return resolved_expr, None, constants, None


def _decode_storage_run_id(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _storage_result_to_meta(storage_result: NloStorageResult) -> dict[str, object]:
    return {
        "run_id": _decode_storage_run_id(bytes(storage_result.run_id)),
        "records_captured": int(storage_result.records_captured),
        "records_spilled": int(storage_result.records_spilled),
        "chunks_written": int(storage_result.chunks_written),
        "db_size_bytes": int(storage_result.db_size_bytes),
        "truncated": bool(storage_result.truncated),
    }


def _shift_constant_indices(expression: str, offset: int) -> str:
    if offset == 0:
        return expression

    def repl(match: re.Match[str]) -> str:
        return f"c{int(match.group(1)) + offset}"

    return re.sub(r"\bc(\d+)\b", repl, expression)


@dataclass
class PreparedSimConfig:
    config: SimConfig
    keepalive: list[object]

    @property
    def ptr(self) -> ctypes.POINTER(SimConfig):
        return ctypes.pointer(self.config)


def prepare_sim_config(
    num_time_samples: int,
    *,
    propagation_distance: float,
    starting_step_size: float,
    max_step_size: float,
    min_step_size: float,
    error_tolerance: float,
    pulse_period: float,
    delta_time: float,
    time_nt: int | None = None,
    frequency_grid: Sequence[complex],
    spatial_nx: int | None = None,
    spatial_ny: int | None = None,
    delta_x: float = 1.0,
    delta_y: float = 1.0,
    spatial_frequency_grid: Sequence[complex] | None = None,
    potential_grid: Sequence[complex] | None = None,
    runtime: RuntimeOperators | None = None,
) -> PreparedSimConfig:
    """
    Build a fully-initialized sim_config plus keepalive storage.
    """
    if num_time_samples <= 0:
        raise ValueError("num_time_samples must be > 0")
    if time_nt is not None and int(time_nt) > 0:
        time_nt_int = int(time_nt)
        if len(frequency_grid) not in {time_nt_int, num_time_samples}:
            raise ValueError(
                "frequency_grid length must match time_nt (or num_time_samples for full-volume grids)"
            )
    elif len(frequency_grid) != num_time_samples:
        raise ValueError("frequency_grid length must match num_time_samples")

    cfg = SimConfig()
    keepalive: list[object] = []

    cfg.propagation.propagation_distance = float(propagation_distance)
    cfg.propagation.starting_step_size = float(starting_step_size)
    cfg.propagation.max_step_size = float(max_step_size)
    cfg.propagation.min_step_size = float(min_step_size)
    cfg.propagation.error_tolerance = float(error_tolerance)

    cfg.time.nt = int(time_nt) if time_nt is not None else 0
    cfg.time.pulse_period = float(pulse_period)
    cfg.time.delta_time = float(delta_time)

    freq_arr = make_complex_array(frequency_grid)
    keepalive.append(freq_arr)
    cfg.frequency.frequency_grid = ctypes.cast(freq_arr, ctypes.POINTER(NloComplex))

    nx = int(spatial_nx) if spatial_nx is not None else (1 if cfg.time.nt > 0 else int(num_time_samples))
    ny = int(spatial_ny) if spatial_ny is not None else 1
    if cfg.time.nt > 0:
        if nx <= 0 or ny <= 0:
            raise ValueError("spatial_nx and spatial_ny must be positive when time_nt is set")
        if (cfg.time.nt * nx * ny) != int(num_time_samples):
            raise ValueError("time_nt * spatial_nx * spatial_ny must match num_time_samples")
    cfg.spatial.nx = nx
    cfg.spatial.ny = ny
    cfg.spatial.delta_x = float(delta_x)
    cfg.spatial.delta_y = float(delta_y)

    if cfg.time.nt > 0:
        xy_points = nx * ny
        if spatial_frequency_grid is not None and len(spatial_frequency_grid) not in {
            xy_points,
            num_time_samples,
        }:
            raise ValueError(
                "spatial_frequency_grid length must match spatial_nx*spatial_ny "
                "(or num_time_samples for full-volume grids)"
            )
        if potential_grid is not None and len(potential_grid) not in {
            xy_points,
            num_time_samples,
        }:
            raise ValueError(
                "potential_grid length must match spatial_nx*spatial_ny "
                "(or num_time_samples for full-volume grids)"
            )

    if spatial_frequency_grid is not None:
        spatial_arr = make_complex_array(spatial_frequency_grid)
        keepalive.append(spatial_arr)
        cfg.spatial.spatial_frequency_grid = ctypes.cast(
            spatial_arr, ctypes.POINTER(NloComplex)
        )
    else:
        cfg.spatial.spatial_frequency_grid = None

    if potential_grid is not None:
        potential_arr = make_complex_array(potential_grid)
        keepalive.append(potential_arr)
        cfg.spatial.potential_grid = ctypes.cast(
            potential_arr, ctypes.POINTER(NloComplex)
        )
    else:
        cfg.spatial.potential_grid = None

    if runtime is not None:
        constants = [float(value) for value in runtime.constants]
        constant_bindings = runtime.constant_bindings
        auto_capture = bool(runtime.auto_capture_constants)

        if runtime.dispersion_factor_expr and runtime.dispersion_factor_fn is not None:
            raise ValueError(
                "runtime.dispersion_factor_expr and runtime.dispersion_factor_fn are mutually exclusive"
            )
        if runtime.dispersion_expr and runtime.dispersion_fn is not None:
            raise ValueError("runtime.dispersion_expr and runtime.dispersion_fn are mutually exclusive")
        if runtime.transverse_factor_expr and runtime.transverse_factor_fn is not None:
            raise ValueError(
                "runtime.transverse_factor_expr and runtime.transverse_factor_fn are mutually exclusive"
            )
        if runtime.transverse_expr and runtime.transverse_fn is not None:
            raise ValueError("runtime.transverse_expr and runtime.transverse_fn are mutually exclusive")
        if runtime.nonlinear_expr and runtime.nonlinear_fn is not None:
            raise ValueError("runtime.nonlinear_expr and runtime.nonlinear_fn are mutually exclusive")

        dispersion_factor_expr = runtime.dispersion_factor_expr
        if runtime.dispersion_factor_fn is not None:
            translated = translate_callable(
                runtime.dispersion_factor_fn,
                RUNTIME_CONTEXT_DISPERSION_FACTOR,
                constant_bindings=constant_bindings,
                auto_capture=auto_capture,
            )
            shifted = _shift_constant_indices(translated.expression, len(constants))
            constants.extend(translated.constants)
            dispersion_factor_expr = shifted

        dispersion_expr = runtime.dispersion_expr
        if runtime.dispersion_fn is not None:
            translated = translate_callable(
                runtime.dispersion_fn,
                RUNTIME_CONTEXT_DISPERSION,
                constant_bindings=constant_bindings,
                auto_capture=auto_capture,
            )
            shifted = _shift_constant_indices(translated.expression, len(constants))
            constants.extend(translated.constants)
            dispersion_expr = shifted

        transverse_factor_expr = runtime.transverse_factor_expr
        if runtime.transverse_factor_fn is not None:
            translated = translate_callable(
                runtime.transverse_factor_fn,
                RUNTIME_CONTEXT_DISPERSION_FACTOR,
                constant_bindings=constant_bindings,
                auto_capture=auto_capture,
            )
            shifted = _shift_constant_indices(translated.expression, len(constants))
            constants.extend(translated.constants)
            transverse_factor_expr = shifted

        transverse_expr = runtime.transverse_expr
        if runtime.transverse_fn is not None:
            translated = translate_callable(
                runtime.transverse_fn,
                RUNTIME_CONTEXT_DISPERSION,
                constant_bindings=constant_bindings,
                auto_capture=auto_capture,
            )
            shifted = _shift_constant_indices(translated.expression, len(constants))
            constants.extend(translated.constants)
            transverse_expr = shifted

        nonlinear_expr = runtime.nonlinear_expr
        if runtime.nonlinear_fn is not None:
            translated = translate_callable(
                runtime.nonlinear_fn,
                RUNTIME_CONTEXT_NONLINEAR,
                constant_bindings=constant_bindings,
                auto_capture=auto_capture,
            )
            shifted = _shift_constant_indices(translated.expression, len(constants))
            constants.extend(translated.constants)
            nonlinear_expr = shifted

        if len(constants) > NLO_RUNTIME_OPERATOR_CONSTANTS_MAX:
            raise ValueError(
                "runtime.constants length exceeds "
                f"{NLO_RUNTIME_OPERATOR_CONSTANTS_MAX}"
            )

        if dispersion_factor_expr:
            disp_factor_bytes = dispersion_factor_expr.encode("utf-8")
            keepalive.append(disp_factor_bytes)
            cfg.runtime.dispersion_factor_expr = ctypes.c_char_p(disp_factor_bytes)
        else:
            cfg.runtime.dispersion_factor_expr = None

        if dispersion_expr:
            disp_bytes = dispersion_expr.encode("utf-8")
            keepalive.append(disp_bytes)
            cfg.runtime.dispersion_expr = ctypes.c_char_p(disp_bytes)
        else:
            cfg.runtime.dispersion_expr = None

        if transverse_factor_expr:
            trans_factor_bytes = transverse_factor_expr.encode("utf-8")
            keepalive.append(trans_factor_bytes)
            cfg.runtime.transverse_factor_expr = ctypes.c_char_p(trans_factor_bytes)
        else:
            cfg.runtime.transverse_factor_expr = None

        if transverse_expr:
            trans_bytes = transverse_expr.encode("utf-8")
            keepalive.append(trans_bytes)
            cfg.runtime.transverse_expr = ctypes.c_char_p(trans_bytes)
        else:
            cfg.runtime.transverse_expr = None

        if nonlinear_expr:
            nonlin_bytes = nonlinear_expr.encode("utf-8")
            keepalive.append(nonlin_bytes)
            cfg.runtime.nonlinear_expr = ctypes.c_char_p(nonlin_bytes)
        else:
            cfg.runtime.nonlinear_expr = None

        cfg.runtime.num_constants = len(constants)
        for i, constant in enumerate(constants):
            cfg.runtime.constants[i] = float(constant)
    else:
        cfg.runtime.dispersion_factor_expr = None
        cfg.runtime.dispersion_expr = None
        cfg.runtime.transverse_factor_expr = None
        cfg.runtime.transverse_expr = None
        cfg.runtime.nonlinear_expr = None
        cfg.runtime.num_constants = 0

    return PreparedSimConfig(config=cfg, keepalive=keepalive)


class NLolib:
    """
    High-level nlolib API wrapper.
    """

    def __init__(self, path: str | None = None):
        self.lib = load(path)

    def storage_is_available(self) -> bool:
        """
        Return whether SQLite-backed storage is available in the loaded library.
        """
        if not bool(getattr(self.lib, "_has_storage_is_available", False)):
            return False
        return bool(int(self.lib.nlolib_storage_is_available()))

    def set_log_file(self, path: str | None, append: bool = False) -> None:
        """
        Configure optional file sink for runtime logs.
        """
        if not bool(getattr(self.lib, "_has_set_log_file", False)):
            raise RuntimeError("nlolib_set_log_file is unavailable in the loaded nlolib build")
        encoded = path.encode("utf-8") if path else None
        status = int(self.lib.nlolib_set_log_file(encoded, int(bool(append))))
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_set_log_file failed with status={status}")

    def set_log_buffer(self, capacity_bytes: int = 256 * 1024) -> None:
        """
        Configure in-memory ring buffer sink for runtime logs.
        """
        if not bool(getattr(self.lib, "_has_set_log_buffer", False)):
            raise RuntimeError("nlolib_set_log_buffer is unavailable in the loaded nlolib build")
        status = int(self.lib.nlolib_set_log_buffer(int(capacity_bytes)))
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_set_log_buffer failed with status={status}")

    def clear_log_buffer(self) -> None:
        """
        Clear buffered runtime logs.
        """
        if not bool(getattr(self.lib, "_has_clear_log_buffer", False)):
            raise RuntimeError("nlolib_clear_log_buffer is unavailable in the loaded nlolib build")
        status = int(self.lib.nlolib_clear_log_buffer())
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_clear_log_buffer failed with status={status}")

    def read_log_buffer(self, consume: bool = True, max_bytes: int = 256 * 1024) -> str:
        """
        Read buffered runtime logs as UTF-8 text.
        """
        if not bool(getattr(self.lib, "_has_read_log_buffer", False)):
            raise RuntimeError("nlolib_read_log_buffer is unavailable in the loaded nlolib build")
        if max_bytes < 2:
            raise ValueError("max_bytes must be >= 2")

        out = ctypes.create_string_buffer(max_bytes)
        written = ctypes.c_size_t(0)
        status = int(
            self.lib.nlolib_read_log_buffer(
                ctypes.cast(out, ctypes.c_void_p),
                ctypes.c_size_t(max_bytes),
                ctypes.byref(written),
                int(bool(consume)),
            )
        )
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_read_log_buffer failed with status={status}")
        return out.raw[: int(written.value)].decode("utf-8", errors="replace")

    def set_log_level(self, level: int = NLOLIB_LOG_LEVEL_INFO) -> None:
        """
        Set global runtime log level.
        """
        if not bool(getattr(self.lib, "_has_set_log_level", False)):
            raise RuntimeError("nlolib_set_log_level is unavailable in the loaded nlolib build")
        status = int(self.lib.nlolib_set_log_level(int(level)))
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_set_log_level failed with status={status}")

    def set_progress_options(
        self,
        enabled: bool = True,
        milestone_percent: int = 5,
        emit_on_step_adjust: bool = False,
    ) -> None:
        """
        Configure runtime progress logging options.
        """
        if not bool(getattr(self.lib, "_has_set_progress_options", False)):
            raise RuntimeError("nlolib_set_progress_options is unavailable in the loaded nlolib build")
        status = int(
            self.lib.nlolib_set_progress_options(
                int(bool(enabled)),
                int(milestone_percent),
                int(bool(emit_on_step_adjust)),
            )
        )
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_set_progress_options failed with status={status}")

    def query_runtime_limits(
        self,
        config: PreparedSimConfig | SimConfig | None = None,
        exec_options: NloExecutionOptions | None = None,
    ) -> NloRuntimeLimits:
        """
        Query runtime-derived solver limits for current backend/options.
        """
        if not bool(getattr(self.lib, "_has_query_runtime_limits", False)):
            out = NloRuntimeLimits()
            # Conservative fallback for legacy libraries that predate
            # nlolib_query_runtime_limits and enforce NT_MAX in C.
            out.max_num_time_samples_runtime = int(LEGACY_NT_MAX)
            out.max_num_recorded_samples_in_memory = int(LEGACY_NT_MAX)
            out.max_num_recorded_samples_with_storage = int(LEGACY_NT_MAX)
            out.estimated_required_working_set_bytes = 0
            out.estimated_device_budget_bytes = 0
            out.storage_available = 1 if self.storage_is_available() else 0
            return out

        cfg = None
        if config is not None:
            cfg = config.config if isinstance(config, PreparedSimConfig) else config
        cfg_ptr = ctypes.pointer(cfg) if cfg is not None else None
        opts_ptr = ctypes.pointer(exec_options) if exec_options is not None else None
        out = NloRuntimeLimits()
        status = int(self.lib.nlolib_query_runtime_limits(cfg_ptr, opts_ptr, ctypes.pointer(out)))
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_query_runtime_limits failed with status={status}")
        return out

    def propagate(
        self,
        config: PreparedSimConfig | SimConfig,
        input_field: Sequence[complex],
        num_recorded_samples: int,
        exec_options: NloExecutionOptions | None = None,
    ) -> list[list[complex]]:
        """
        Propagate an input field and return record-major complex samples.
        """
        if num_recorded_samples <= 0:
            raise ValueError("num_recorded_samples must be > 0")

        n = len(input_field)
        if n == 0:
            raise ValueError("input_field must be non-empty")

        in_arr = make_complex_array(input_field)
        out_arr = make_complex_array(n * int(num_recorded_samples))

        cfg = config.config if isinstance(config, PreparedSimConfig) else config
        cfg_ptr = ctypes.pointer(cfg)
        opts_ptr = ctypes.pointer(exec_options) if exec_options is not None else None

        status = int(
            self.lib.nlolib_propagate(
                cfg_ptr,
                n,
                ctypes.cast(in_arr, ctypes.POINTER(NloComplex)),
                int(num_recorded_samples),
                ctypes.cast(out_arr, ctypes.POINTER(NloComplex)),
                opts_ptr,
            )
        )
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_propagate failed with status={status}")

        flat = complex_array_to_list(out_arr, n * int(num_recorded_samples))
        records: list[list[complex]] = []
        for i in range(int(num_recorded_samples)):
            start = i * n
            records.append(flat[start : start + n])
        return records

    def simulate(
        self,
        pulse: PulseSpec | Mapping[str, object],
        linear_operator: str | OperatorSpec | Mapping[str, object] = "gvd",
        nonlinear_operator: str | OperatorSpec | Mapping[str, object] = "kerr",
        *,
        transverse_operator: str | OperatorSpec | Mapping[str, object] = "none",
        propagation_distance: float,
        output: str = "dense",
        preset: str = "balanced",
        records: int | None = None,
        exec_options: NloExecutionOptions | None = None,
        sqlite_path: str | None = None,
        run_id: str | None = None,
        sqlite_max_bytes: int = 0,
        chunk_records: int = 0,
        cap_policy: int = NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES,
        log_final_output_field_to_db: bool = False,
        return_records: bool = True,
    ) -> PropagationResult:
        """
        High-level propagation facade using pulse/operator specs with defaults.
        """
        pulse_spec = _normalize_pulse_spec(pulse)
        profile = _solver_profile_defaults(preset, float(propagation_distance))
        num_records = int(records) if records is not None else int(profile["records"])
        if output == "final":
            num_records = 1
        elif output != "dense":
            raise ValueError("output must be 'dense' or 'final'")
        if num_records <= 0:
            raise ValueError("records must be > 0")

        linear_expr, linear_fn, linear_constants, linear_bindings = _resolve_operator_spec(
            "linear", linear_operator, 0
        )
        transverse_requested = not (
            isinstance(transverse_operator, str) and transverse_operator.strip().lower() == "none"
        )
        if transverse_requested:
            _validate_coupled_pulse_spec(pulse_spec)
            transverse_expr, transverse_fn, transverse_constants, transverse_bindings = _resolve_operator_spec(
                "transverse", transverse_operator, len(linear_constants)
            )
            if transverse_fn is not None:
                raise ValueError("transverse callable operators are not supported in the high-level facade")
        else:
            transverse_expr = None
            transverse_fn = None
            transverse_constants = []
            transverse_bindings = None
        nonlinear_expr, nonlinear_fn, nonlinear_constants, nonlinear_bindings = _resolve_operator_spec(
            "nonlinear", nonlinear_operator, len(linear_constants) + len(transverse_constants)
        )

        binding_map: dict[str, float] = {}
        for source in (linear_bindings, transverse_bindings, nonlinear_bindings):
            if source is None:
                continue
            for key, value in source.items():
                existing = binding_map.get(key)
                if existing is not None and existing != value:
                    raise ValueError(f"conflicting callable param '{key}' across operators")
                binding_map[key] = value

        num_time_samples = len(pulse_spec.samples)
        temporal_samples = (
            int(pulse_spec.time_nt)
            if pulse_spec.time_nt is not None and int(pulse_spec.time_nt) > 0
            else num_time_samples
        )
        pulse_period = (
            float(pulse_spec.pulse_period)
            if pulse_spec.pulse_period is not None
            else float(pulse_spec.delta_time) * float(temporal_samples)
        )
        frequency_grid = (
            list(pulse_spec.frequency_grid)
            if pulse_spec.frequency_grid is not None
            else _default_frequency_grid(temporal_samples, pulse_spec.delta_time)
        )

        runtime = RuntimeOperators(
            dispersion_factor_expr=linear_expr,
            transverse_factor_expr=transverse_expr,
            transverse_expr=("exp(h*D)" if transverse_requested else None),
            nonlinear_expr=nonlinear_expr,
            dispersion_factor_fn=linear_fn,
            transverse_factor_fn=transverse_fn,
            nonlinear_fn=nonlinear_fn,
            constants=[*linear_constants, *transverse_constants, *nonlinear_constants],
            constant_bindings=binding_map if binding_map else None,
            auto_capture_constants=(not binding_map),
        )

        config = prepare_sim_config(
            num_time_samples,
            propagation_distance=float(propagation_distance),
            starting_step_size=float(profile["starting_step_size"]),
            max_step_size=float(profile["max_step_size"]),
            min_step_size=float(profile["min_step_size"]),
            error_tolerance=float(profile["error_tolerance"]),
            pulse_period=float(pulse_period),
            delta_time=float(pulse_spec.delta_time),
            time_nt=pulse_spec.time_nt,
            frequency_grid=frequency_grid,
            spatial_nx=pulse_spec.spatial_nx,
            spatial_ny=pulse_spec.spatial_ny,
            delta_x=float(pulse_spec.delta_x),
            delta_y=float(pulse_spec.delta_y),
            spatial_frequency_grid=pulse_spec.spatial_frequency_grid,
            potential_grid=pulse_spec.potential_grid,
            runtime=runtime,
        )

        storage_result_meta: dict[str, object] | None = None
        if sqlite_path is None:
            out_records = self.propagate(config, pulse_spec.samples, num_records, exec_options)
        else:
            storage_records, storage_result = self.propagate_with_storage(
                config,
                pulse_spec.samples,
                num_records,
                sqlite_path=sqlite_path,
                run_id=run_id,
                sqlite_max_bytes=sqlite_max_bytes,
                chunk_records=chunk_records,
                cap_policy=cap_policy,
                log_final_output_field_to_db=log_final_output_field_to_db,
                exec_options=exec_options,
                return_records=bool(return_records),
            )
            storage_result_meta = _storage_result_to_meta(storage_result)
            out_records = storage_records if storage_records is not None else []

        if num_records == 1:
            z_axis = [float(propagation_distance)]
        else:
            z_axis = [
                (float(propagation_distance) * float(i)) / float(num_records - 1)
                for i in range(num_records)
            ]

        if len(out_records) == 0:
            final: list[complex] = []
        else:
            final = list(out_records[-1])

        meta: dict[str, Any] = {
            "preset": preset,
            "output": output,
            "records": num_records,
            "coupled": bool(transverse_requested),
            "storage_enabled": bool(sqlite_path is not None),
            "records_returned": bool(len(out_records) > 0),
            "backend_requested": (
                int(exec_options.backend_type)
                if exec_options is not None
                else int(NLO_VECTOR_BACKEND_AUTO)
            ),
        }
        if storage_result_meta is not None:
            meta["storage_result"] = storage_result_meta

        return PropagationResult(
            records=out_records,
            z_axis=z_axis,
            final=final,
            meta=meta,
        )

    def propagate_with_storage(
        self,
        config: PreparedSimConfig | SimConfig,
        input_field: Sequence[complex],
        num_recorded_samples: int,
        *,
        sqlite_path: str,
        run_id: str | None = None,
        sqlite_max_bytes: int = 0,
        chunk_records: int = 0,
        cap_policy: int = NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES,
        log_final_output_field_to_db: bool = False,
        exec_options: NloExecutionOptions | None = None,
        return_records: bool = False,
    ) -> tuple[list[list[complex]] | None, NloStorageResult]:
        """
        Propagate and persist snapshot chunks into SQLite.
        """
        if not bool(getattr(self.lib, "_has_propagate_with_storage", False)):
            raise RuntimeError(
                "nlolib_propagate_with_storage is unavailable in the loaded nlolib build"
            )

        if num_recorded_samples <= 0:
            raise ValueError("num_recorded_samples must be > 0")

        n = len(input_field)
        if n == 0:
            raise ValueError("input_field must be non-empty")

        if not self.storage_is_available():
            raise RuntimeError("SQLite storage is not available in this nlolib build")

        in_arr = make_complex_array(input_field)
        out_arr = make_complex_array(n * int(num_recorded_samples)) if return_records else None

        cfg = config.config if isinstance(config, PreparedSimConfig) else config
        cfg_ptr = ctypes.pointer(cfg)
        opts_ptr = ctypes.pointer(exec_options) if exec_options is not None else None

        storage_opts, storage_keepalive = default_storage_options(
            sqlite_path=sqlite_path,
            run_id=run_id,
            sqlite_max_bytes=sqlite_max_bytes,
            chunk_records=chunk_records,
            cap_policy=cap_policy,
            log_final_output_field_to_db=log_final_output_field_to_db,
        )
        storage_result = NloStorageResult()
        _ = storage_keepalive  # Keep ctypes-backed strings alive for call duration.

        status = int(
            self.lib.nlolib_propagate_with_storage(
                cfg_ptr,
                n,
                ctypes.cast(in_arr, ctypes.POINTER(NloComplex)),
                int(num_recorded_samples),
                ctypes.cast(out_arr, ctypes.POINTER(NloComplex)) if out_arr is not None else None,
                opts_ptr,
                ctypes.pointer(storage_opts),
                ctypes.pointer(storage_result),
            )
        )
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_propagate_with_storage failed with status={status}")

        if out_arr is None:
            return None, storage_result

        flat = complex_array_to_list(out_arr, n * int(num_recorded_samples))
        records: list[list[complex]] = []
        for i in range(int(num_recorded_samples)):
            start = i * n
            records.append(flat[start : start + n])
        return records, storage_result


__all__ = [
    "NT_MAX",
    "NLO_RUNTIME_OPERATOR_CONSTANTS_MAX",
    "NLO_STORAGE_RUN_ID_MAX",
    "NLO_VECTOR_BACKEND_CPU",
    "NLO_VECTOR_BACKEND_VULKAN",
    "NLO_VECTOR_BACKEND_AUTO",
    "NLO_FFT_BACKEND_AUTO",
    "NLO_FFT_BACKEND_FFTW",
    "NLO_FFT_BACKEND_VKFFT",
    "NLOLIB_STATUS_OK",
    "NLOLIB_STATUS_INVALID_ARGUMENT",
    "NLOLIB_STATUS_ALLOCATION_FAILED",
    "NLOLIB_STATUS_NOT_IMPLEMENTED",
    "NLOLIB_LOG_LEVEL_ERROR",
    "NLOLIB_LOG_LEVEL_WARN",
    "NLOLIB_LOG_LEVEL_INFO",
    "NLOLIB_LOG_LEVEL_DEBUG",
    "NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES",
    "NLO_STORAGE_DB_CAP_POLICY_FAIL",
    "NloComplex",
    "NloExecutionOptions",
    "NloRuntimeLimits",
    "NloStorageOptions",
    "NloStorageResult",
    "NloVkBackendConfig",
    "OperatorSpec",
    "PropagationResult",
    "PulseSpec",
    "PreparedSimConfig",
    "RuntimeOperators",
    "NLolib",
    "complex_array_to_list",
    "default_execution_options",
    "default_storage_options",
    "load",
    "make_complex_array",
    "prepare_sim_config",
]
