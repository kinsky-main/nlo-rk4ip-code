from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PYTHON = REPO_ROOT / "examples" / "python"
if str(EXAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PYTHON))

from backend.metrics import filtered_relative_l2_error, relative_l2_error  # noqa: E402
from backend.reference import (  # noqa: E402
    analytical_initial_condition_error,
    exact_linear_temporal_final,
    peak_intensity_over_records,
    second_order_soliton_normalized_records,
    second_order_soliton_period,
)
from nlolib import (  # noqa: E402
    NLOLIB_LOG_LEVEL_ERROR,
    NLOLIB_LOG_LEVEL_WARN,
    NLO_VECTOR_BACKEND_AUTO,
    NLO_VECTOR_BACKEND_CPU,
    NLolib,
    RuntimeOperators,
    default_execution_options,
    prepare_sim_config,
)


def _omega_grid_unshifted(n: int, dt: float) -> list[float]:
    two_pi = 2.0 * math.pi
    return [
        two_pi
        * (
            float(i) / (float(n) * dt)
            if i <= (n - 1) // 2
            else -float(n - i) / (float(n) * dt)
        )
        for i in range(n)
    ]


def _centered_time_grid(n: int, dt: float) -> np.ndarray:
    mid = n // 2
    return np.asarray([(float(i - mid) * dt) for i in range(n)], dtype=np.float64)


def _sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x)


def _auto_backend_resolves_to_vulkan(api: NLolib) -> bool:
    n = 32
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.01,
        min_step_size=0.01,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=1.0 / float(n),
        frequency_grid=[0j] * n,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 0.0]),
    )
    api.set_log_level(2)
    api.set_log_buffer(64 * 1024)
    api.clear_log_buffer()
    api.propagate(cfg, [0j] * n, 1, default_execution_options(NLO_VECTOR_BACKEND_AUTO))
    logs = api.read_log_buffer(consume=True, max_bytes=64 * 1024)
    return "actual: VULKAN" in logs


def _fundamental_soliton_case(n: int = 256, window_multiple_t0: float = 32.0) -> dict[str, object]:
    beta2 = -0.01
    gamma = 0.01
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    ld = (t0 * t0) / abs(beta2)
    z_final = 0.5 * math.pi * ld
    dt = (float(window_multiple_t0) * t0) / float(n)
    t = _centered_time_grid(n, dt)
    tau = t / t0
    omega = np.asarray(_omega_grid_unshifted(n, dt), dtype=np.float64)
    p0 = abs(beta2) / (gamma * t0 * t0)
    field0 = (math.sqrt(p0) * _sech(tau)).astype(np.complex128)
    reference = (np.exp(0.5j * (z_final / ld)) * field0).astype(np.complex128)
    return {
        "beta2": beta2,
        "gamma": gamma,
        "dt": dt,
        "omega": omega,
        "z_final": z_final,
        "field0": field0,
        "reference": reference,
    }


def test_second_order_reference_matches_known_recurrence_and_midperiod_compression() -> None:
    beta2 = -0.01
    tfwhm = 100e-3
    t0 = tfwhm / (2.0 * math.log(1.0 + math.sqrt(2.0)))
    n = 4096
    dt = (64.0 * t0) / float(n)
    t = _centered_time_grid(n, dt) / t0
    z0 = second_order_soliton_period(beta2, t0)
    z_axis = np.linspace(0.0, z0, 513, dtype=np.float64)

    initial_error = analytical_initial_condition_error(t, beta2, t0)
    assert initial_error <= 1.0e-12, f"second-order analytical initial condition drifted: {initial_error}"

    records = second_order_soliton_normalized_records(t, z_axis, beta2, t0)
    peak_trace = peak_intensity_over_records(records)
    peak_index = int(np.argmax(peak_trace))
    peak_location = float(z_axis[peak_index] / z0)
    peak_intensity = float(peak_trace[peak_index])

    initial_intensity = np.abs(records[0]) ** 2
    final_intensity = np.abs(records[-1]) ** 2
    recurrence_error = relative_l2_error(final_intensity, initial_intensity)

    assert recurrence_error <= 1.0e-12, f"second-order analytical intensity no longer recurs over one period: {recurrence_error}"
    assert abs(peak_location - 0.5) <= 0.01, (
        f"second-order analytical peak compression moved away from mid-period: {peak_location}"
    )
    assert 15.5 <= peak_intensity <= 16.5, (
        f"second-order analytical peak intensity is outside the expected N=2 range: {peak_intensity}"
    )


def test_linear_reference_prefers_current_dispersion_sign(api: NLolib) -> None:
    n = 256
    dt = 0.02
    z_final = 0.35
    beta2 = 0.05
    c0 = 0.5 * beta2
    t = _centered_time_grid(n, dt)
    omega = np.asarray(_omega_grid_unshifted(n, dt), dtype=np.float64)
    field0 = (np.exp(-((t / 0.25) ** 2)) * np.exp((-1.0j) * 12.0 * t)).astype(np.complex128)

    cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=z_final / 8.0,
        max_step_size=z_final / 8.0,
        min_step_size=z_final / 8.0,
        error_tolerance=1e-12,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(v, 0.0) for v in omega],
        runtime=RuntimeOperators(constants=[c0, 0.0, 0.0]),
    )
    final_field = np.asarray(
        api.propagate(cfg, field0.tolist(), 2, default_execution_options(NLO_VECTOR_BACKEND_CPU)).records[1],
        dtype=np.complex128,
    )
    current_ref = exact_linear_temporal_final(field0, z_final, omega, c0)
    opposite_ref = exact_linear_temporal_final(field0, z_final, omega, -c0)
    current_error = relative_l2_error(final_field, current_ref)
    opposite_error = relative_l2_error(final_field, opposite_ref)

    assert current_error <= 1.0e-10, f"CPU linear propagation no longer matches current sign convention: {current_error}"
    assert opposite_error >= (1.0e3 * max(current_error, 1.0e-15)), (
        f"opposite-sign dispersion reference is not clearly worse: "
        f"current={current_error}, opposite={opposite_error}"
    )


def test_zero_operator_identity_fixed_step_cpu_and_auto(api: NLolib) -> None:
    n = 512
    dt = 0.01
    z_final = 0.2
    t = _centered_time_grid(n, dt)
    field0 = (np.exp(-((t / 0.30) ** 2)) * np.exp((-1.0j) * 5.0 * t)).astype(np.complex128)

    cfg = prepare_sim_config(
        n,
        propagation_distance=z_final,
        starting_step_size=z_final / 16.0,
        max_step_size=z_final / 16.0,
        min_step_size=z_final / 16.0,
        error_tolerance=1.0e-12,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[0j] * n,
        runtime=RuntimeOperators(
            dispersion_factor_expr="0",
            nonlinear_expr="0",
            constants=[0.0, 0.0, 0.0, 0.0],
        ),
    )

    cpu_final = np.asarray(
        api.propagate(cfg, field0.tolist(), 2, default_execution_options(NLO_VECTOR_BACKEND_CPU)).records[1],
        dtype=np.complex128,
    )
    cpu_error = relative_l2_error(cpu_final, field0)
    assert cpu_error <= 1.0e-12, f"CPU identity propagation drifted unexpectedly: {cpu_error}"

    if not _auto_backend_resolves_to_vulkan(api):
        print("test_zero_operator_identity_fixed_step_cpu_and_auto: AUTO did not resolve to Vulkan, skipping GPU branch.")
        return

    auto_final = np.asarray(
        api.propagate(cfg, field0.tolist(), 2, default_execution_options(NLO_VECTOR_BACKEND_AUTO)).records[1],
        dtype=np.complex128,
    )
    auto_error = relative_l2_error(auto_final, field0)
    assert auto_error <= 5.0e-6, f"AUTO/Vulkan identity propagation drift exceeded diagnostic bound: {auto_error}"

def main() -> None:
    api = NLolib()
    api.set_log_level(NLOLIB_LOG_LEVEL_ERROR)

    test_second_order_reference_matches_known_recurrence_and_midperiod_compression()
    test_linear_reference_prefers_current_dispersion_sign(api)
    test_zero_operator_identity_fixed_step_cpu_and_auto(api)

    api.set_log_level(NLOLIB_LOG_LEVEL_WARN)
    print("test_python_reference_benchmarks: reference benchmarks validated.")


if __name__ == "__main__":
    main()
