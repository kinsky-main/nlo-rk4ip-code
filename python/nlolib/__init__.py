"""User-facing Python API for nlolib."""

import re
from importlib import metadata
from pathlib import Path

from ._binding import (
    NT_MAX,
    FFT_BACKEND_AUTO,
    FFT_BACKEND_FFTW,
    FFT_BACKEND_VKFFT,
    NONLINEAR_MODEL_EXPR,
    NONLINEAR_MODEL_KERR_RAMAN,
    PROPAGATE_OUTPUT_DENSE,
    PROPAGATE_OUTPUT_FINAL_ONLY,
    STORAGE_DB_CAP_POLICY_FAIL,
    STORAGE_DB_CAP_POLICY_STOP_WRITES,
    STORAGE_RUN_ID_MAX,
    TENSOR_LAYOUT_XYT_T_FAST,
    VECTOR_BACKEND_AUTO,
    VECTOR_BACKEND_CPU,
    VECTOR_BACKEND_VULKAN,
    NLO_FFT_BACKEND_AUTO,
    NLO_FFT_BACKEND_FFTW,
    NLO_FFT_BACKEND_VKFFT,
    NLOLIB_LOG_LEVEL_DEBUG,
    NLOLIB_LOG_LEVEL_ERROR,
    NLOLIB_LOG_LEVEL_INFO,
    NLOLIB_LOG_LEVEL_WARN,
    NLOLIB_PROGRESS_STREAM_BOTH,
    NLOLIB_PROGRESS_STREAM_STDERR,
    NLOLIB_PROGRESS_STREAM_STDOUT,
    NLOLIB_STATUS_ALLOCATION_FAILED,
    NLOLIB_STATUS_ABORTED,
    NLOLIB_STATUS_INVALID_ARGUMENT,
    NLOLIB_STATUS_NOT_IMPLEMENTED,
    NLOLIB_STATUS_OK,
    NLO_VECTOR_BACKEND_AUTO,
    NLO_VECTOR_BACKEND_CPU,
    NLO_VECTOR_BACKEND_VULKAN,
    PROGRESS_EVENT_ACCEPTED,
    PROGRESS_EVENT_FINISH,
    PROGRESS_EVENT_REJECTED,
    NloComplex,
    NloExecutionOptions,
    NloPerfProfileSnapshot,
    NloRuntimeLimits,
    NloStorageOptions,
    NloStorageResult,
    NloTensor3dDesc,
    NloVkBackendConfig,
    complex_array_to_list,
    default_execution_options,
    default_storage_options,
    load,
    make_complex_array,
)
from ._client import NLolib
from ._config import PreparedSimConfig, prepare_sim_config
from ._models import (
    OperatorSpec,
    ProgressInfo,
    PropagationAbortedError,
    PropagationResult,
    PulseSpec,
    RuntimeOperators,
)
from ._wrapper import (
    PropagateResult,
    clear_log_buffer,
    perf_profile_is_enabled,
    perf_profile_read,
    perf_profile_reset,
    perf_profile_set_enabled,
    propagate,
    query_runtime_limits,
    read_log_buffer,
    set_log_buffer,
    set_log_level,
    set_progress_options,
    set_progress_stream,
    storage_is_available,
)
from .translate.runtime_expr import translate_callable


def _detect_version() -> str:
    """Report this package's version without storing a copy of it.

    ``pyproject.toml`` is the source of truth for the Python package version,
    which is released independently of the core library. Deriving the
    attribute from it rather than assigning a literal here means there is no
    second copy to drift.

    A source checkout has ``pyproject.toml`` next to the package directory; it
    is read first because an installed nlolib can shadow a checkout on
    ``sys.path``, and the installed distribution metadata would then name a
    version this code is not. The ``name`` check keeps an unrelated
    ``pyproject.toml`` from answering.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if 'name = "nlolib"' in text:
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    try:
        return metadata.version("nlolib")
    except metadata.PackageNotFoundError:
        return "unknown"


# Deliberately not in __all__: `from nlolib import *` should not rebind the
# caller's own __version__.
__version__ = _detect_version()

__all__ = [
    "NT_MAX",
    "STORAGE_RUN_ID_MAX",
    "STORAGE_DB_CAP_POLICY_STOP_WRITES",
    "STORAGE_DB_CAP_POLICY_FAIL",
    "TENSOR_LAYOUT_XYT_T_FAST",
    "VECTOR_BACKEND_CPU",
    "VECTOR_BACKEND_VULKAN",
    "VECTOR_BACKEND_AUTO",
    "NLO_VECTOR_BACKEND_CPU",
    "NLO_VECTOR_BACKEND_VULKAN",
    "NLO_VECTOR_BACKEND_AUTO",
    "FFT_BACKEND_AUTO",
    "FFT_BACKEND_FFTW",
    "FFT_BACKEND_VKFFT",
    "NLO_FFT_BACKEND_AUTO",
    "NLO_FFT_BACKEND_FFTW",
    "NLO_FFT_BACKEND_VKFFT",
    "NONLINEAR_MODEL_EXPR",
    "NONLINEAR_MODEL_KERR_RAMAN",
    "PROPAGATE_OUTPUT_DENSE",
    "PROPAGATE_OUTPUT_FINAL_ONLY",
    "NLOLIB_STATUS_OK",
    "NLOLIB_STATUS_INVALID_ARGUMENT",
    "NLOLIB_STATUS_ALLOCATION_FAILED",
    "NLOLIB_STATUS_NOT_IMPLEMENTED",
    "NLOLIB_STATUS_ABORTED",
    "NLOLIB_LOG_LEVEL_ERROR",
    "NLOLIB_LOG_LEVEL_WARN",
    "NLOLIB_LOG_LEVEL_INFO",
    "NLOLIB_LOG_LEVEL_DEBUG",
    "NLOLIB_PROGRESS_STREAM_STDERR",
    "NLOLIB_PROGRESS_STREAM_STDOUT",
    "NLOLIB_PROGRESS_STREAM_BOTH",
    "PROGRESS_EVENT_ACCEPTED",
    "PROGRESS_EVENT_REJECTED",
    "PROGRESS_EVENT_FINISH",
    "PulseSpec",
    "OperatorSpec",
    "ProgressInfo",
    "PropagationAbortedError",
    "PropagateResult",
    "NLolib",
    "NloComplex",
    "NloExecutionOptions",
    "NloPerfProfileSnapshot",
    "NloRuntimeLimits",
    "NloStorageOptions",
    "NloStorageResult",
    "NloTensor3dDesc",
    "NloVkBackendConfig",
    "PreparedSimConfig",
    "PropagationResult",
    "RuntimeOperators",
    "complex_array_to_list",
    "make_complex_array",
    "prepare_sim_config",
    "default_execution_options",
    "default_storage_options",
    "load",
    "translate_callable",
    "propagate",
    "query_runtime_limits",
    "storage_is_available",
    "perf_profile_set_enabled",
    "perf_profile_is_enabled",
    "perf_profile_reset",
    "perf_profile_read",
    "set_log_level",
    "set_log_buffer",
    "clear_log_buffer",
    "read_log_buffer",
    "set_progress_options",
    "set_progress_stream",
]
