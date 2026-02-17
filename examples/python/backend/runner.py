"""Reusable CFFI runner for Python examples."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))


def _parse_pointer_value(value: int | str) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            return int(text, 16)
        return int(text, 10)
    return int(value)


def _to_vk_handle(ffi, value: int | str | None, ctype: str):
    if value is None:
        return ffi.NULL
    parsed = _parse_pointer_value(value)
    if parsed == 0:
        return ffi.NULL
    return ffi.cast(ctype, parsed)


@dataclass
class SimulationOptions:
    backend: str | dict[str, Any] = "auto"
    fft_backend: str = "auto"
    device_heap_fraction: float = 0.70
    record_ring_target: int = 0
    forced_device_budget_bytes: int = 0

    def to_cffi(self, ffi):
        opts = ffi.new("nlo_execution_options*")
        opts.backend_type = 2  # NLO_VECTOR_BACKEND_AUTO
        opts.fft_backend = 0  # NLO_FFT_BACKEND_AUTO
        opts.device_heap_fraction = float(self.device_heap_fraction)
        opts.record_ring_target = int(self.record_ring_target)
        opts.forced_device_budget_bytes = int(self.forced_device_budget_bytes)
        opts.vulkan.physical_device = ffi.NULL
        opts.vulkan.device = ffi.NULL
        opts.vulkan.queue = ffi.NULL
        opts.vulkan.queue_family_index = 0
        opts.vulkan.command_pool = ffi.NULL
        opts.vulkan.descriptor_set_budget_bytes = 0
        opts.vulkan.descriptor_set_count_override = 0

        fft_backend_map = {"auto": 0, "fftw": 1, "vkfft": 2}
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
            opts.backend_type = 0
            return opts
        if backend_type == "auto":
            opts.backend_type = 2
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

        opts.backend_type = 1  # NLO_VECTOR_BACKEND_VULKAN
        opts.vulkan.physical_device = _to_vk_handle(ffi, vk_cfg.get("physical_device"), "VkPhysicalDevice")
        opts.vulkan.device = _to_vk_handle(ffi, vk_cfg.get("device"), "VkDevice")
        opts.vulkan.queue = _to_vk_handle(ffi, vk_cfg.get("queue"), "VkQueue")
        opts.vulkan.queue_family_index = int(vk_cfg.get("queue_family_index", 0))
        opts.vulkan.command_pool = _to_vk_handle(ffi, vk_cfg.get("command_pool"), "VkCommandPool")
        opts.vulkan.descriptor_set_budget_bytes = int(vk_cfg.get("descriptor_set_budget_bytes", 0))
        opts.vulkan.descriptor_set_count_override = int(vk_cfg.get("descriptor_set_count_override", 0))
        return opts


@dataclass
class TemporalSimulationConfig:
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
            from nlolib_cffi import ffi, load
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nlolib_cffi/cffi is not available. Install cffi and ensure "
                "PYTHONPATH includes the repo's python/ directory."
            ) from exc

        self.ffi = ffi
        self.lib = load(library_path)

    @staticmethod
    def _write_complex_buffer(dst, values: np.ndarray) -> None:
        for i, val in enumerate(values):
            dst[i].re = float(val.real)
            dst[i].im = float(val.imag)

    @staticmethod
    def _read_complex_buffer(src, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.complex128)
        for i in range(n):
            out[i] = complex(src[i].re, src[i].im)
        return out

    def _build_temporal_sim_config(self, cfg, sim_cfg: TemporalSimulationConfig, frequency_grid) -> None:
        cfg.nonlinear.gamma = float(sim_cfg.gamma)
        cfg.dispersion.num_dispersion_terms = 3
        cfg.dispersion.betas[0] = 0.0
        cfg.dispersion.betas[1] = 0.0
        cfg.dispersion.betas[2] = float(sim_cfg.beta2)
        cfg.dispersion.alpha = float(sim_cfg.alpha)

        cfg.propagation.propagation_distance = float(sim_cfg.z_final)
        cfg.propagation.starting_step_size = float(sim_cfg.starting_step_size)
        cfg.propagation.max_step_size = float(sim_cfg.max_step_size)
        cfg.propagation.min_step_size = float(sim_cfg.min_step_size)
        cfg.propagation.error_tolerance = float(sim_cfg.error_tolerance)

        cfg.time.pulse_period = float(sim_cfg.resolved_pulse_period())
        cfg.time.delta_time = float(sim_cfg.dt)
        cfg.frequency.frequency_grid = frequency_grid

        cfg.spatial.nx = int(sim_cfg.num_time_samples)
        cfg.spatial.ny = 1
        cfg.spatial.delta_x = 1.0
        cfg.spatial.delta_y = 1.0
        cfg.spatial.grin_gx = 0.0
        cfg.spatial.grin_gy = 0.0
        cfg.spatial.spatial_frequency_grid = self.ffi.NULL
        cfg.spatial.grin_potential_phase_grid = self.ffi.NULL

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
        cfg = self.ffi.new("sim_config*")
        freq = self.ffi.new("nlo_complex[]", n)
        for i, om in enumerate(omega):
            freq[i].re = float(om)
            freq[i].im = 0.0
        self._build_temporal_sim_config(cfg, sim_cfg, freq)

        inp = self.ffi.new("nlo_complex[]", n)
        out = self.ffi.new("nlo_complex[]", n * int(num_records))
        self._write_complex_buffer(inp, field)

        options = exec_options if exec_options is not None else SimulationOptions()
        opts_ptr = options.to_cffi(self.ffi)
        status = int(self.lib.nlolib_propagate(cfg, n, inp, int(num_records), out, opts_ptr))
        if status != 0:
            raise RuntimeError(f"nlolib_propagate failed with status={status}.")

        flat = self._read_complex_buffer(out, n * int(num_records))
        records = flat.reshape(int(num_records), n)
        if int(num_records) == 1:
            z_records = np.asarray([float(sim_cfg.z_final)], dtype=np.float64)
        else:
            z_records = np.linspace(0.0, float(sim_cfg.z_final), int(num_records))
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
        grin_gx: float,
        grin_gy: float,
        gamma: float = 0.0,
        alpha: float = 0.0,
        frequency_grid: np.ndarray | None = None,
        spatial_frequency_grid: np.ndarray | None = None,
        grin_potential_phase_grid: np.ndarray | None = None,
        exec_options: SimulationOptions | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_records <= 0:
            raise ValueError("num_records must be positive.")

        nxy = int(nx) * int(ny)
        field = np.asarray(field0_flat, dtype=np.complex128).reshape(-1)
        if int(field.size) != nxy:
            raise ValueError("field0_flat length must equal nx * ny.")

        cfg = self.ffi.new("sim_config*")
        cfg.nonlinear.gamma = float(gamma)
        cfg.dispersion.num_dispersion_terms = 0
        cfg.dispersion.alpha = float(alpha)
        cfg.propagation.propagation_distance = float(propagation_distance)
        cfg.propagation.starting_step_size = float(starting_step_size)
        cfg.propagation.max_step_size = float(max_step_size)
        cfg.propagation.min_step_size = float(min_step_size)
        cfg.propagation.error_tolerance = float(error_tolerance)
        cfg.time.pulse_period = float(nx)
        cfg.time.delta_time = 1.0

        freq_values = np.zeros(nxy, dtype=np.complex128)
        if frequency_grid is not None:
            freq_values = np.asarray(frequency_grid, dtype=np.complex128).reshape(-1)
            if int(freq_values.size) != nxy:
                raise ValueError("frequency_grid length must equal nx * ny.")
        freq_buffer = self.ffi.new("nlo_complex[]", nxy)
        self._write_complex_buffer(freq_buffer, freq_values)
        cfg.frequency.frequency_grid = freq_buffer

        cfg.spatial.nx = int(nx)
        cfg.spatial.ny = int(ny)
        cfg.spatial.delta_x = float(delta_x)
        cfg.spatial.delta_y = float(delta_y)
        cfg.spatial.grin_gx = float(grin_gx)
        cfg.spatial.grin_gy = float(grin_gy)

        spatial_freq_buffer = self.ffi.NULL
        if spatial_frequency_grid is not None:
            spatial_freq_values = np.asarray(spatial_frequency_grid, dtype=np.complex128).reshape(-1)
            if int(spatial_freq_values.size) != nxy:
                raise ValueError("spatial_frequency_grid length must equal nx * ny.")
            spatial_freq_buffer = self.ffi.new("nlo_complex[]", nxy)
            self._write_complex_buffer(spatial_freq_buffer, spatial_freq_values)
        cfg.spatial.spatial_frequency_grid = spatial_freq_buffer

        grin_phase_buffer = self.ffi.NULL
        if grin_potential_phase_grid is not None:
            grin_phase_values = np.asarray(grin_potential_phase_grid, dtype=np.complex128).reshape(-1)
            if int(grin_phase_values.size) != nxy:
                raise ValueError("grin_potential_phase_grid length must equal nx * ny.")
            grin_phase_buffer = self.ffi.new("nlo_complex[]", nxy)
            self._write_complex_buffer(grin_phase_buffer, grin_phase_values)
        cfg.spatial.grin_potential_phase_grid = grin_phase_buffer

        inp = self.ffi.new("nlo_complex[]", nxy)
        out = self.ffi.new("nlo_complex[]", nxy * int(num_records))
        self._write_complex_buffer(inp, field)

        options = exec_options if exec_options is not None else SimulationOptions()
        opts_ptr = options.to_cffi(self.ffi)
        status = int(self.lib.nlolib_propagate(cfg, nxy, inp, int(num_records), out, opts_ptr))
        if status != 0:
            raise RuntimeError(f"nlolib_propagate failed with status={status}.")

        flat = self._read_complex_buffer(out, nxy * int(num_records))
        records = flat.reshape(int(num_records), int(ny), int(nx))
        if int(num_records) == 1:
            z_records = np.asarray([float(propagation_distance)], dtype=np.float64)
        else:
            z_records = np.linspace(0.0, float(propagation_distance), int(num_records))
        return z_records, records
