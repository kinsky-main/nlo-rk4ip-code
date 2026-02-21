"""Reusable ctypes runner utilities for Python examples."""

from __future__ import annotations

import math
import sys
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))


def centered_time_grid(num_samples: int, delta_time: float) -> np.ndarray:
    """Return a centered time axis with sample spacing ``delta_time``."""
    return (np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)) * delta_time


def _parse_pointer_value(value: int | str) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            return int(text, 16)
        return int(text, 10)
    return int(value)


def _to_vk_handle(value: int | str | None):
    if value is None:
        return None
    parsed = _parse_pointer_value(value)
    if parsed == 0:
        return None
    return ctypes.c_void_p(parsed)


@dataclass
class SimulationOptions:
    backend: str | dict[str, Any] = "auto"
    fft_backend: str = "auto"
    device_heap_fraction: float = 0.70
    record_ring_target: int = 0
    forced_device_budget_bytes: int = 0

    def backend_type(self) -> str:
        if isinstance(self.backend, str):
            return self.backend.strip().lower()
        if isinstance(self.backend, dict):
            return str(self.backend.get("type", "cpu")).strip().lower()
        return ""

    def to_ctypes(self, nlo):
        opts = nlo.default_execution_options(
            backend_type=nlo.NLO_VECTOR_BACKEND_AUTO,
            fft_backend=nlo.NLO_FFT_BACKEND_AUTO,
        )
        opts.device_heap_fraction = float(self.device_heap_fraction)
        opts.record_ring_target = int(self.record_ring_target)
        opts.forced_device_budget_bytes = int(self.forced_device_budget_bytes)

        fft_backend_map = {
            "auto": nlo.NLO_FFT_BACKEND_AUTO,
            "fftw": nlo.NLO_FFT_BACKEND_FFTW,
            "vkfft": nlo.NLO_FFT_BACKEND_VKFFT,
        }
        fft_backend = str(self.fft_backend).strip().lower()
        if fft_backend not in fft_backend_map:
            raise ValueError("fft_backend must be one of: auto, fftw, vkfft.")
        opts.fft_backend = fft_backend_map[fft_backend]

        backend_cfg = self.backend
        if isinstance(backend_cfg, str):
            backend_type = backend_cfg.strip().lower()
            cfg = {}
        elif isinstance(backend_cfg, dict):
            backend_type = str(backend_cfg.get("type", "cpu")).strip().lower()
            cfg = backend_cfg
        else:
            raise TypeError("backend must be a string or dict when provided.")

        if backend_type == "cpu":
            opts.backend_type = nlo.NLO_VECTOR_BACKEND_CPU
            return opts
        if backend_type == "auto":
            opts.backend_type = nlo.NLO_VECTOR_BACKEND_AUTO
            return opts
        if backend_type != "vulkan":
            raise ValueError("backend type must be one of: 'cpu', 'auto', or 'vulkan'.")

        vk_cfg = cfg.get("vulkan", cfg)
        required = ("physical_device", "device", "queue", "queue_family_index")
        missing = [name for name in required if name not in vk_cfg]
        if missing:
            raise ValueError(
                "vulkan backend requires handle fields: "
                + ", ".join(required)
                + ". Missing: "
                + ", ".join(missing)
            )

        opts.backend_type = nlo.NLO_VECTOR_BACKEND_VULKAN
        opts.vulkan.physical_device = _to_vk_handle(vk_cfg.get("physical_device"))
        opts.vulkan.device = _to_vk_handle(vk_cfg.get("device"))
        opts.vulkan.queue = _to_vk_handle(vk_cfg.get("queue"))
        opts.vulkan.queue_family_index = int(vk_cfg.get("queue_family_index", 0))
        opts.vulkan.command_pool = _to_vk_handle(vk_cfg.get("command_pool"))
        opts.vulkan.descriptor_set_budget_bytes = int(vk_cfg.get("descriptor_set_budget_bytes", 0))
        opts.vulkan.descriptor_set_count_override = int(vk_cfg.get("descriptor_set_count_override", 0))
        return opts


@dataclass
class TemporalSimulationConfig:
    """
    Example convenience config mapped to operator-default runtime constants.

    When ``runtime`` is omitted, ``gamma``, ``beta2``, and ``alpha`` map to
    default constants ``c2``, ``c0``, and ``c1`` respectively.
    """

    gamma: float
    beta2: float
    dt: float
    z_final: float
    num_time_samples: int
    alpha: float = 0.0
    pulse_period: float | None = None
    omega: np.ndarray | None = None
    starting_step_size: float = 1e-4
    max_step_size: float = 1e-2
    min_step_size: float = 1e-6
    error_tolerance: float = 1e-8
    runtime: Any | None = None

    def resolved_pulse_period(self) -> float:
        if self.pulse_period is not None:
            return float(self.pulse_period)
        return float(self.num_time_samples) * float(self.dt)

    def resolved_omega(self) -> np.ndarray:
        if self.omega is not None:
            omega = np.asarray(self.omega, dtype=np.float64)
            if int(omega.size) != int(self.num_time_samples):
                raise ValueError("omega grid size must match num_time_samples.")
            return omega
        return 2.0 * math.pi * np.fft.fftfreq(int(self.num_time_samples), d=float(self.dt))


class NloExampleRunner:
    def __init__(self, library_path: str | None = None):
        try:
            import nlolib_ctypes as nlo
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nlolib_ctypes is not available. Ensure "
                "PYTHONPATH includes the repo's python/ directory."
            ) from exc

        self.nlo = nlo
        self.api = nlo.NLolib(path=library_path)

    @staticmethod
    def _effective_options(
        options: SimulationOptions,
        num_records: int,
    ) -> SimulationOptions:
        backend_type = options.backend_type()
        if options.record_ring_target > 0:
            return options
        if backend_type not in {"auto", "vulkan"}:
            return options

        # Keep GPU prioritized while avoiding oversized descriptor/ring allocations.
        capped_ring = max(1, min(int(num_records), 32))
        return SimulationOptions(
            backend=options.backend,
            fft_backend=options.fft_backend,
            device_heap_fraction=options.device_heap_fraction,
            record_ring_target=capped_ring,
            forced_device_budget_bytes=options.forced_device_budget_bytes,
        )

    @staticmethod
    def _preset_from_error_tolerance(error_tolerance: float) -> str:
        tol = float(error_tolerance)
        if tol <= 1e-7:
            return "accuracy"
        if tol <= 5e-6:
            return "balanced"
        return "fast"

    def propagate_temporal_records(
        self,
        field0: np.ndarray,
        sim_cfg: TemporalSimulationConfig,
        num_records: int,
        exec_options: SimulationOptions | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_records <= 0:
            raise ValueError("num_records must be positive.")

        field = np.asarray(field0, dtype=np.complex128).reshape(-1)
        n = int(field.size)
        if int(sim_cfg.num_time_samples) != n:
            raise ValueError("field length must match sim_cfg.num_time_samples.")

        omega = sim_cfg.resolved_omega()
        runtime_cfg = sim_cfg.runtime

        options = exec_options if exec_options is not None else SimulationOptions()
        effective_options = self._effective_options(options, int(num_records))
        opts = effective_options.to_ctypes(self.nlo)

        if runtime_cfg is not None:
            prepared = self.nlo.prepare_sim_config(
                n,
                propagation_distance=float(sim_cfg.z_final),
                starting_step_size=float(sim_cfg.starting_step_size),
                max_step_size=float(sim_cfg.max_step_size),
                min_step_size=float(sim_cfg.min_step_size),
                error_tolerance=float(sim_cfg.error_tolerance),
                pulse_period=float(sim_cfg.resolved_pulse_period()),
                delta_time=float(sim_cfg.dt),
                frequency_grid=[complex(float(om), 0.0) for om in omega],
                spatial_nx=n,
                spatial_ny=1,
                delta_x=1.0,
                delta_y=1.0,
                runtime=runtime_cfg,
            )
            records = np.asarray(
                self.api.propagate(prepared, field.tolist(), int(num_records), opts),
                dtype=np.complex128,
            ).reshape(int(num_records), n)
            if int(num_records) == 1:
                z_records = np.asarray([float(sim_cfg.z_final)], dtype=np.float64)
            else:
                z_records = np.linspace(0.0, float(sim_cfg.z_final), int(num_records))
            return z_records, records

        pulse = self.nlo.PulseSpec(
            samples=field.tolist(),
            delta_time=float(sim_cfg.dt),
            pulse_period=float(sim_cfg.resolved_pulse_period()),
            frequency_grid=[complex(float(om), 0.0) for om in omega],
        )
        linear_operator = self.nlo.OperatorSpec(
            expr="i*beta2*w*w-loss",
            params={
                "beta2": 0.5 * float(sim_cfg.beta2),
                "loss": 0.5 * float(sim_cfg.alpha),
            },
        )
        nonlinear_operator = self.nlo.OperatorSpec(
            expr="i*gamma*I + i*V",
            params={"gamma": float(sim_cfg.gamma)},
        )
        result = self.api.simulate(
            pulse,
            linear_operator,
            nonlinear_operator,
            propagation_distance=float(sim_cfg.z_final),
            output=("final" if int(num_records) == 1 else "dense"),
            preset=self._preset_from_error_tolerance(sim_cfg.error_tolerance),
            records=int(num_records),
            exec_options=opts,
        )
        z_records = np.asarray(result.z_axis, dtype=np.float64)
        records = np.asarray(result.records, dtype=np.complex128).reshape(int(num_records), n)
        return z_records, records

    def propagate_flattened_xy_records(
        self,
        field0_flat: np.ndarray,
        nx: int,
        ny: int,
        num_records: int,
        propagation_distance: float,
        starting_step_size: float,
        max_step_size: float,
        min_step_size: float,
        error_tolerance: float,
        delta_x: float,
        delta_y: float,
        gamma: float = 0.0,
        alpha: float = 0.0,
        frequency_grid: np.ndarray | None = None,
        spatial_frequency_grid: np.ndarray | None = None,
        potential_grid: np.ndarray | None = None,
        runtime: Any | None = None,
        exec_options: SimulationOptions | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_records <= 0:
            raise ValueError("num_records must be positive.")

        nxy = int(nx) * int(ny)
        field = np.asarray(field0_flat, dtype=np.complex128).reshape(-1)
        if int(field.size) != nxy:
            raise ValueError("field0_flat length must equal nx * ny.")

        freq_values = np.zeros(nxy, dtype=np.complex128)
        if frequency_grid is not None:
            freq_values = np.asarray(frequency_grid, dtype=np.complex128).reshape(-1)
            if int(freq_values.size) != nxy:
                raise ValueError("frequency_grid length must equal nx * ny.")

        spatial_values = None
        if spatial_frequency_grid is not None:
            spatial_values = np.asarray(spatial_frequency_grid, dtype=np.complex128).reshape(-1)
            if int(spatial_values.size) != nxy:
                raise ValueError("spatial_frequency_grid length must equal nx * ny.")

        potential_values = None
        if potential_grid is not None:
            potential_values = np.asarray(potential_grid, dtype=np.complex128).reshape(-1)
            if int(potential_values.size) != nxy:
                raise ValueError("potential_grid length must equal nx * ny.")

        runtime_cfg = runtime

        options = exec_options if exec_options is not None else SimulationOptions()
        effective_options = self._effective_options(options, int(num_records))
        opts = effective_options.to_ctypes(self.nlo)

        if runtime_cfg is not None:
            prepared = self.nlo.prepare_sim_config(
                nxy,
                propagation_distance=float(propagation_distance),
                starting_step_size=float(starting_step_size),
                max_step_size=float(max_step_size),
                min_step_size=float(min_step_size),
                error_tolerance=float(error_tolerance),
                pulse_period=float(nx),
                delta_time=1.0,
                frequency_grid=freq_values.tolist(),
                spatial_nx=int(nx),
                spatial_ny=int(ny),
                delta_x=float(delta_x),
                delta_y=float(delta_y),
                spatial_frequency_grid=(None if spatial_values is None else spatial_values.tolist()),
                potential_grid=(None if potential_values is None else potential_values.tolist()),
                runtime=runtime_cfg,
            )
            records = np.asarray(
                self.api.propagate(prepared, field.tolist(), int(num_records), opts),
                dtype=np.complex128,
            ).reshape(int(num_records), int(ny), int(nx))
            if int(num_records) == 1:
                z_records = np.asarray([float(propagation_distance)], dtype=np.float64)
            else:
                z_records = np.linspace(0.0, float(propagation_distance), int(num_records))
            return z_records, records

        pulse = self.nlo.PulseSpec(
            samples=field.tolist(),
            delta_time=1.0,
            pulse_period=float(nx),
            frequency_grid=freq_values.tolist(),
            spatial_nx=int(nx),
            spatial_ny=int(ny),
            delta_x=float(delta_x),
            delta_y=float(delta_y),
            spatial_frequency_grid=(None if spatial_values is None else spatial_values.tolist()),
            potential_grid=(None if potential_values is None else potential_values.tolist()),
        )
        linear_operator = self.nlo.OperatorSpec(
            expr="i*beta2*w*w-loss",
            params={
                "beta2": 0.0,
                "loss": 0.5 * float(alpha),
            },
        )
        nonlinear_operator = self.nlo.OperatorSpec(
            expr="i*gamma*I + i*V",
            params={"gamma": float(gamma)},
        )
        result = self.api.simulate(
            pulse,
            linear_operator,
            nonlinear_operator,
            propagation_distance=float(propagation_distance),
            output=("final" if int(num_records) == 1 else "dense"),
            preset=self._preset_from_error_tolerance(error_tolerance),
            records=int(num_records),
            exec_options=opts,
        )
        z_records = np.asarray(result.z_axis, dtype=np.float64)
        records = np.asarray(result.records, dtype=np.complex128).reshape(int(num_records), int(ny), int(nx))
        return z_records, records
