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

NT_MAX = ctypes.c_size_t(-1).value  # Unbounded sentinel; prefer query_runtime_limits().
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
            out.max_num_time_samples_runtime = int(NT_MAX)
            out.max_num_recorded_samples_in_memory = int(NT_MAX)
            out.max_num_recorded_samples_with_storage = int(NT_MAX)
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
    "NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES",
    "NLO_STORAGE_DB_CAP_POLICY_FAIL",
    "NloComplex",
    "NloExecutionOptions",
    "NloRuntimeLimits",
    "NloStorageOptions",
    "NloStorageResult",
    "NloVkBackendConfig",
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
