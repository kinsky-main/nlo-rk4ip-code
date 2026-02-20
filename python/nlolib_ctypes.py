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
from typing import Callable, Mapping, Sequence

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

NT_MAX = 1 << 20
NLO_RUNTIME_OPERATOR_CONSTANTS_MAX = 16

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


def _candidate_library_paths() -> list[str]:
    candidates: list[str] = []
    env_path = os.environ.get("NLOLIB_LIBRARY")
    if env_path:
        candidates.append(env_path)

    here = Path(__file__).resolve().parent
    root = here.parent
    if os.name == "nt":
        candidates.extend(
            [
                str(here / "nlolib.dll"),
                str(here / "Debug" / "nlolib.dll"),
                str(here / "Release" / "nlolib.dll"),
                str(here / "RelWithDebInfo" / "nlolib.dll"),
                str(here / "MinSizeRel" / "nlolib.dll"),
                str(root / "nlolib.dll"),
            ]
        )
    elif os.name == "posix":
        candidates.extend(
            [
                str(here / "libnlolib.so"),
                str(root / "libnlolib.so"),
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
    if lib_path is None:
        for candidate in _candidate_library_paths():
            try:
                lib = ctypes.CDLL(candidate)
                lib_path = candidate
                break
            except OSError:
                continue
        else:
            raise OSError(
                "Unable to locate NLOLib shared library. "
                "Set NLOLIB_LIBRARY to the full path."
            )

        if lib_path is None:
            raise OSError("Unable to locate NLOLib shared library.")
    else:
        lib = ctypes.CDLL(lib_path)

    lib.nlolib_propagate.argtypes = [
        ctypes.POINTER(SimConfig),
        ctypes.c_size_t,
        ctypes.POINTER(NloComplex),
        ctypes.c_size_t,
        ctypes.POINTER(NloComplex),
        ctypes.POINTER(NloExecutionOptions),
    ]
    lib.nlolib_propagate.restype = ctypes.c_int
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


@dataclass
class RuntimeOperators:
    dispersion_factor_expr: str | None = None
    dispersion_expr: str | None = None
    nonlinear_expr: str | None = None
    dispersion_factor_fn: Callable[..., object] | None = None
    dispersion_fn: Callable[..., object] | None = None
    nonlinear_fn: Callable[..., object] | None = None
    constants: Sequence[float] = ()
    constant_bindings: Mapping[str, float] | None = None
    auto_capture_constants: bool = True


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
    if len(frequency_grid) != num_time_samples:
        raise ValueError("frequency_grid length must match num_time_samples")

    cfg = SimConfig()
    keepalive: list[object] = []

    cfg.propagation.propagation_distance = float(propagation_distance)
    cfg.propagation.starting_step_size = float(starting_step_size)
    cfg.propagation.max_step_size = float(max_step_size)
    cfg.propagation.min_step_size = float(min_step_size)
    cfg.propagation.error_tolerance = float(error_tolerance)

    cfg.time.pulse_period = float(pulse_period)
    cfg.time.delta_time = float(delta_time)

    freq_arr = make_complex_array(frequency_grid)
    keepalive.append(freq_arr)
    cfg.frequency.frequency_grid = ctypes.cast(freq_arr, ctypes.POINTER(NloComplex))

    nx = int(spatial_nx) if spatial_nx is not None else int(num_time_samples)
    ny = int(spatial_ny) if spatial_ny is not None else 1
    cfg.spatial.nx = nx
    cfg.spatial.ny = ny
    cfg.spatial.delta_x = float(delta_x)
    cfg.spatial.delta_y = float(delta_y)

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
        cfg.runtime.nonlinear_expr = None
        cfg.runtime.num_constants = 0

    return PreparedSimConfig(config=cfg, keepalive=keepalive)


class NLolib:
    """
    High-level nlolib API wrapper.
    """

    def __init__(self, path: str | None = None):
        self.lib = load(path)

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


__all__ = [
    "NT_MAX",
    "NLO_RUNTIME_OPERATOR_CONSTANTS_MAX",
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
    "NloComplex",
    "NloExecutionOptions",
    "NloVkBackendConfig",
    "PreparedSimConfig",
    "RuntimeOperators",
    "NLolib",
    "complex_array_to_list",
    "default_execution_options",
    "load",
    "make_complex_array",
    "prepare_sim_config",
]
