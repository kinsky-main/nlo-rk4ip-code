"""
Stateful OOP client facade for the nlolib Python wrapper.
"""

from __future__ import annotations

import ctypes
from typing import Any

from ._binding import (
    LEGACY_NT_MAX,
    NLOLIB_LOG_LEVEL_INFO,
    NLOLIB_PROGRESS_STREAM_STDERR,
    NLOLIB_STATUS_OK,
    NloExecutionOptions,
    NloPhysicsConfig,
    NloRuntimeLimits,
    NloSimulationConfig,
    load,
)
from ._config import PreparedSimConfig
from ._executor import PropagationExecutor
from ._models import PropagationResult
from ._requests import PropagateRequestBuilder


class NLolib:
    """
    High-level nlolib API wrapper.
    """

    def __init__(self, path: str | None = None):
        self.lib = load(path)
        self._request_builder = PropagateRequestBuilder()
        self._executor = PropagationExecutor(self.lib, self.storage_is_available)

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
        Configure runtime progress TUI options.
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

    def set_progress_stream(self, stream_mode: int = NLOLIB_PROGRESS_STREAM_STDERR) -> None:
        """
        Configure output stream selection for runtime progress TUI lines.
        """
        if not bool(getattr(self.lib, "_has_set_progress_stream", False)):
            raise RuntimeError("nlolib_set_progress_stream is unavailable in the loaded nlolib build")
        status = int(self.lib.nlolib_set_progress_stream(int(stream_mode)))
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_set_progress_stream failed with status={status}")

    def query_runtime_limits(
        self,
        config: PreparedSimConfig | NloSimulationConfig | None = None,
        exec_options: NloExecutionOptions | None = None,
        physics_config: NloPhysicsConfig | None = None,
    ) -> NloRuntimeLimits:
        """
        Query runtime-derived solver limits for current backend/options.
        """
        if not bool(getattr(self.lib, "_has_query_runtime_limits", False)):
            out = NloRuntimeLimits()
            out.max_num_time_samples_runtime = int(LEGACY_NT_MAX)
            out.max_num_recorded_samples_in_memory = int(LEGACY_NT_MAX)
            out.max_num_recorded_samples_with_storage = int(LEGACY_NT_MAX)
            out.estimated_required_working_set_bytes = 0
            out.estimated_device_budget_bytes = 0
            out.storage_available = 1 if self.storage_is_available() else 0
            return out

        sim_cfg = None
        phys_cfg = None
        if config is not None:
            if isinstance(config, PreparedSimConfig):
                sim_cfg = config.simulation_config
                phys_cfg = config.physics_config
            else:
                sim_cfg = config
        if physics_config is not None:
            phys_cfg = physics_config

        sim_ptr = ctypes.pointer(sim_cfg) if sim_cfg is not None else None
        phys_ptr = ctypes.pointer(phys_cfg) if phys_cfg is not None else None
        opts_ptr = ctypes.pointer(exec_options) if exec_options is not None else None
        out = NloRuntimeLimits()
        status = int(self.lib.nlolib_query_runtime_limits(sim_ptr, phys_ptr, opts_ptr, ctypes.pointer(out)))
        if status != NLOLIB_STATUS_OK:
            raise RuntimeError(f"nlolib_query_runtime_limits failed with status={status}")
        return out

    def propagate(self, primary: Any, *args: Any, **kwargs: Any) -> PropagationResult:
        """
        Unified propagation entrypoint.

        Low-level form:
            propagate(config, input_field, num_recorded_samples, exec_options=None, **storage_kwargs)

        High-level form:
            propagate(
                pulse,
                linear_operator="gvd",
                nonlinear_operator="kerr",
                *,
                propagation_distance=...,
                ...
            )
        """
        if isinstance(primary, (PreparedSimConfig, NloSimulationConfig)):
            request = self._request_builder.from_config(primary, *args, **kwargs)
        else:
            request = self._request_builder.from_pulse(primary, *args, **kwargs)
        return self._executor.execute(request)
