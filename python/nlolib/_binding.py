"""
Low-level ctypes bindings for the nlolib shared library.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Sequence

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
NLOLIB_STATUS_ABORTED = 4

NLOLIB_LOG_LEVEL_ERROR = 0
NLOLIB_LOG_LEVEL_WARN = 1
NLOLIB_LOG_LEVEL_INFO = 2
NLOLIB_LOG_LEVEL_DEBUG = 3
NLOLIB_PROGRESS_STREAM_STDERR = 0
NLOLIB_PROGRESS_STREAM_STDOUT = 1
NLOLIB_PROGRESS_STREAM_BOTH = 2
NLO_PROGRESS_EVENT_ACCEPTED = 0
NLO_PROGRESS_EVENT_REJECTED = 1
NLO_PROGRESS_EVENT_FINISH = 2

NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES = 0
NLO_STORAGE_DB_CAP_POLICY_FAIL = 1
NLO_PROPAGATE_OUTPUT_DENSE = 0
NLO_PROPAGATE_OUTPUT_FINAL_ONLY = 1

NLO_NONLINEAR_MODEL_EXPR = 0
NLO_NONLINEAR_MODEL_KERR_RAMAN = 1
NLO_TENSOR_LAYOUT_XYT_T_FAST = 0


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
        ("wt_axis", ctypes.POINTER(NloComplex)),
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
        ("kx_axis", ctypes.POINTER(NloComplex)),
        ("ky_axis", ctypes.POINTER(NloComplex)),
        ("potential_grid", ctypes.POINTER(NloComplex)),
    ]


class NloTensor3dDesc(ctypes.Structure):
    _fields_ = [
        ("nt", ctypes.c_size_t),
        ("nx", ctypes.c_size_t),
        ("ny", ctypes.c_size_t),
        ("layout", ctypes.c_int),
    ]


class RuntimeOperatorParams(ctypes.Structure):
    _fields_ = [
        ("linear_factor_expr", ctypes.c_char_p),
        ("linear_expr", ctypes.c_char_p),
        ("potential_expr", ctypes.c_char_p),
        ("dispersion_factor_expr", ctypes.c_char_p),
        ("dispersion_expr", ctypes.c_char_p),
        ("nonlinear_expr", ctypes.c_char_p),
        ("nonlinear_model", ctypes.c_int),
        ("nonlinear_gamma", ctypes.c_double),
        ("raman_fraction", ctypes.c_double),
        ("raman_tau1", ctypes.c_double),
        ("raman_tau2", ctypes.c_double),
        ("shock_omega0", ctypes.c_double),
        ("raman_response_time", ctypes.POINTER(NloComplex)),
        ("raman_response_len", ctypes.c_size_t),
        ("num_constants", ctypes.c_size_t),
        ("constants", ctypes.c_double * NLO_RUNTIME_OPERATOR_CONSTANTS_MAX),
    ]


class NloSimulationConfig(ctypes.Structure):
    _fields_ = [
        ("propagation", PropagationParams),
        ("tensor", NloTensor3dDesc),
        ("time", TimeGrid),
        ("frequency", FrequencyGrid),
        ("spatial", SpatialGrid),
    ]


class NloPhysicsConfig(ctypes.Structure):
    _fields_ = [("runtime", RuntimeOperatorParams)]


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


class NloStepEvent(ctypes.Structure):
    _fields_ = [
        ("step_index", ctypes.c_size_t),
        ("z_current", ctypes.c_double),
        ("step_size", ctypes.c_double),
        ("next_step_size", ctypes.c_double),
        ("error", ctypes.c_double),
    ]


class NloProgressInfo(ctypes.Structure):
    _fields_ = [
        ("event_type", ctypes.c_int),
        ("step_index", ctypes.c_size_t),
        ("reject_attempt", ctypes.c_size_t),
        ("z", ctypes.c_double),
        ("z_end", ctypes.c_double),
        ("percent", ctypes.c_double),
        ("step_size", ctypes.c_double),
        ("next_step_size", ctypes.c_double),
        ("error", ctypes.c_double),
        ("elapsed_seconds", ctypes.c_double),
        ("eta_seconds", ctypes.c_double),
    ]


NloProgressCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(NloProgressInfo),
    ctypes.c_void_p,
)


class NloPropagateOptions(ctypes.Structure):
    _fields_ = [
        ("num_recorded_samples", ctypes.c_size_t),
        ("output_mode", ctypes.c_int),
        ("return_records", ctypes.c_int),
        ("exec_options", ctypes.POINTER(NloExecutionOptions)),
        ("storage_options", ctypes.POINTER(NloStorageOptions)),
        ("explicit_record_z", ctypes.POINTER(ctypes.c_double)),
        ("explicit_record_z_count", ctypes.c_size_t),
        ("progress_callback", NloProgressCallback),
        ("progress_user_data", ctypes.c_void_p),
    ]


class NloPropagateOutput(ctypes.Structure):
    _fields_ = [
        ("output_records", ctypes.POINTER(NloComplex)),
        ("output_record_capacity", ctypes.c_size_t),
        ("records_written", ctypes.POINTER(ctypes.c_size_t)),
        ("storage_result", ctypes.POINTER(NloStorageResult)),
        ("output_step_events", ctypes.POINTER(NloStepEvent)),
        ("output_step_event_capacity", ctypes.c_size_t),
        ("step_events_written", ctypes.POINTER(ctypes.c_size_t)),
        ("step_events_dropped", ctypes.POINTER(ctypes.c_size_t)),
    ]


def load(path: Path | None= None) -> ctypes.CDLL:
    """
    Load and configure the NLOLib shared library.

    Set NLOLIB_LIBRARY to override discovery.
    """
    lib_path: Path | None = path
    env_override = os.environ.get("NLOLIB_LIBRARY")
    if env_override:
        lib_path = Path(env_override)
    elif lib_path is None or not lib_path.exists():
        package_dir = Path(__file__).resolve().parent
        package_root = package_dir.parent
        if os.name == "nt":
            lib_path = package_root.joinpath("Release", "nlolib.dll")
        else:
            lib_path = package_root.joinpath("libnlolib.so")

    if lib_path is None or not lib_path.exists():
        raise OSError(
            "Unable to locate NLOLib shared library. "
            "Set NLOLIB_LIBRARY to the full path."
        )
 
    lib = ctypes.CDLL(str(lib_path))
    dll_dir_handles = []
    if os.name == "nt":
        # Add the library directory to the DLL search path on Windows to ensure dependencies are found.
        # This is needed when the library is not in the same directory as the Python executable.
        dll_dir = str(lib_path.parent)
        try:
            os.add_dll_directory(dll_dir)
            dll_dir_handles.append(dll_dir)  # Keep handle alive
        except Exception as e:
            raise OSError(f"Failed to add '{dll_dir}' to DLL search path: {e}") from e

    print(f"Loaded NLOLib from '{lib_path}'")

    lib.nlolib_propagate.argtypes = [
        ctypes.POINTER(NloSimulationConfig),
        ctypes.POINTER(NloPhysicsConfig),
        ctypes.c_size_t,
        ctypes.POINTER(NloComplex),
        ctypes.POINTER(NloPropagateOptions),
        ctypes.POINTER(NloPropagateOutput),
    ]
    lib.nlolib_propagate.restype = ctypes.c_int
    try:
        lib.nlolib_query_runtime_limits.argtypes = [
            ctypes.POINTER(NloSimulationConfig),
            ctypes.POINTER(NloPhysicsConfig),
            ctypes.POINTER(NloExecutionOptions),
            ctypes.POINTER(NloRuntimeLimits),
        ]
        lib.nlolib_query_runtime_limits.restype = ctypes.c_int
        lib._has_query_runtime_limits = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_query_runtime_limits = False # pyright: ignore[reportAttributeAccessIssue]

    try:
        lib.nlolib_propagate_options_default.argtypes = []
        lib.nlolib_propagate_options_default.restype = NloPropagateOptions
        lib.nlolib_propagate_output_default.argtypes = []
        lib.nlolib_propagate_output_default.restype = NloPropagateOutput
        lib._has_propagate_defaults = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_propagate_defaults = False # pyright: ignore[reportAttributeAccessIssue]

    try:
        lib.nlolib_storage_is_available.argtypes = []
        lib.nlolib_storage_is_available.restype = ctypes.c_int
        lib._has_storage_is_available = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_storage_is_available = False # pyright: ignore[reportAttributeAccessIssue]

    try:
        lib.nlolib_set_log_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.nlolib_set_log_file.restype = ctypes.c_int
        lib._has_set_log_file = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_set_log_file = False # pyright: ignore[reportAttributeAccessIssue]

    try:
        lib.nlolib_set_log_buffer.argtypes = [ctypes.c_size_t]
        lib.nlolib_set_log_buffer.restype = ctypes.c_int
        lib._has_set_log_buffer = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_set_log_buffer = False # pyright: ignore[reportAttributeAccessIssue]
    try:
        lib.nlolib_clear_log_buffer.argtypes = []
        lib.nlolib_clear_log_buffer.restype = ctypes.c_int
        lib._has_clear_log_buffer = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_clear_log_buffer = False # pyright: ignore[reportAttributeAccessIssue]

    try:
        lib.nlolib_read_log_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        lib.nlolib_read_log_buffer.restype = ctypes.c_int
        lib._has_read_log_buffer = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_read_log_buffer = False # pyright: ignore[reportAttributeAccessIssue]

    try:
        lib.nlolib_set_log_level.argtypes = [ctypes.c_int]
        lib.nlolib_set_log_level.restype = ctypes.c_int
        lib._has_set_log_level = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_set_log_level = False # pyright: ignore[reportAttributeAccessIssue]

    try:
        lib.nlolib_set_progress_options.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.nlolib_set_progress_options.restype = ctypes.c_int
        lib._has_set_progress_options = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_set_progress_options = False # pyright: ignore[reportAttributeAccessIssue]

    try:
        lib.nlolib_set_progress_stream.argtypes = [ctypes.c_int]
        lib.nlolib_set_progress_stream.restype = ctypes.c_int
        lib._has_set_progress_stream = True # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        lib._has_set_progress_stream = False # pyright: ignore[reportAttributeAccessIssue]

    lib._nlo_loaded_path = str(lib_path) if lib_path is not None else "" # pyright: ignore[reportAttributeAccessIssue]
    lib._nlo_dll_dir_handles = dll_dir_handles # pyright: ignore[reportAttributeAccessIssue]
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
