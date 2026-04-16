"""
\file __init__.py
\brief Docs-only public Python API shim for the ``nlolib`` package.
\ingroup python_binding

This shim mirrors the public Python package surface used by the Doxygen site.
It is not imported by the runtime package.
"""

NT_MAX = 0
STORAGE_RUN_ID_MAX = 64
VECTOR_BACKEND_CPU = 0
VECTOR_BACKEND_VULKAN = 1
VECTOR_BACKEND_AUTO = 2
FFT_BACKEND_AUTO = 0
FFT_BACKEND_FFTW = 1
FFT_BACKEND_VKFFT = 2
NONLINEAR_MODEL_EXPR = 0
NONLINEAR_MODEL_KERR_RAMAN = 1
NLOLIB_STATUS_OK = 0
NLOLIB_STATUS_INVALID_ARGUMENT = 1
NLOLIB_STATUS_ALLOCATION_FAILED = 2
NLOLIB_STATUS_NOT_IMPLEMENTED = 3
NLOLIB_STATUS_ABORTED = 4


class RuntimeOperators:
    """\ingroup python_binding
    Public runtime-operator configuration.
    """


class PulseSpec:
    """\ingroup python_binding
    Input pulse and optional tensor-grid metadata.
    """


class OperatorSpec:
    """\ingroup python_binding
    Linear or nonlinear operator specification.
    """


class ProgressInfo:
    """\ingroup python_binding
    Per-event propagation progress metadata.
    """


class PropagationResult:
    """\ingroup python_binding
    High-level propagation result.
    """


class PropagateResult:
    """\ingroup python_binding
    Convenience result returned by ``propagate``.
    """


class PropagationAbortedError(RuntimeError):
    """\ingroup python_binding
    Raised when a progress callback aborts propagation.
    """


class NLolib:
    """\ingroup python_binding
    High-level Python client facade.
    """

    def storage_is_available(self):
        """Return whether SQLite-backed storage is available."""

    def set_log_file(self, path=None, append=False):
        """Configure optional file sink for runtime logs."""

    def set_log_buffer(self, capacity_bytes=262144):
        """Configure in-memory ring buffer sink for runtime logs."""

    def clear_log_buffer(self):
        """Clear buffered runtime logs."""

    def read_log_buffer(self, consume=True, max_bytes=262144):
        """Read buffered runtime logs as UTF-8 text."""

    def set_log_level(self, level=2):
        """Set global runtime log level."""

    def set_progress_options(self, enabled=True, milestone_percent=5, emit_on_step_adjust=False):
        """Configure runtime progress output options."""

    def set_progress_stream(self, stream_mode=0):
        """Configure output stream selection for runtime progress lines."""

    def query_runtime_limits(self, config=None, exec_options=None, physics_config=None):
        """Query runtime-derived solver limits for current backend/options."""

    def propagate(self, primary, *args, **kwargs):
        """Unified high-level propagation entrypoint."""


def propagate(pulse, linear_operator, nonlinear_operator, **kwargs):
    """\ingroup python_binding
    Propagate a pulse through the high-level convenience wrapper.
    """


def query_runtime_limits(*args, **kwargs):
    """\ingroup python_binding
    Query runtime-derived solver limits through the shared default client.
    """


def storage_is_available():
    """\ingroup python_binding
    Return whether SQLite-backed storage is available.
    """


def set_log_level(level=2):
    """\ingroup python_binding
    Set the global runtime log level.
    """


def set_log_buffer(capacity_bytes=262144):
    """\ingroup python_binding
    Configure an in-memory runtime log buffer.
    """


def clear_log_buffer():
    """\ingroup python_binding
    Clear the in-memory runtime log buffer.
    """


def read_log_buffer(consume=True, max_bytes=262144):
    """\ingroup python_binding
    Read buffered runtime logs as UTF-8 text.
    """


def set_progress_options(enabled=True, milestone_percent=5, emit_on_step_adjust=False):
    """\ingroup python_binding
    Configure runtime progress output options.
    """


def set_progress_stream(stream_mode=0):
    """\ingroup python_binding
    Configure the runtime progress output stream.
    """


def translate_callable(func, *, context=None):
    """\ingroup python_binding
    Translate a Python callable into a runtime expression plus constants.
    """
