"""
Benchmark runtime string expressions vs callable translation overhead.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from statistics import mean

from nlolib_ctypes import (
    NLO_VECTOR_BACKEND_CPU,
    NLolib,
    RuntimeOperators,
    default_execution_options,
    prepare_sim_config,
)


def _omega_grid_unshifted(n: int, dt: float) -> list[float]:
    two_pi = 2.0 * math.pi
    values = [0.0] * n
    for i in range(n):
        if i <= (n - 1) // 2:
            values[i] = two_pi * (float(i) / (float(n) * dt))
        else:
            values[i] = two_pi * (-float(n - i) / (float(n) * dt))
    return values


def _random_field(n: int, seed: int) -> list[complex]:
    rng = random.Random(seed)
    return [complex(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--beta2", type=float, default=-0.04)
    args = parser.parse_args()

    n = int(args.n)
    dt = float(args.dt)
    beta2 = float(args.beta2)
    scale = beta2 / 2.0

    omega = _omega_grid_unshifted(n, dt)
    field = _random_field(n, seed=42)
    common = dict(
        gamma=0.0,
        betas=[],
        alpha=0.0,
        propagation_distance=0.2,
        starting_step_size=1e-3,
        max_step_size=5e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        pulse_period=float(n) * dt,
        delta_time=dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
    )

    def dispersion_fn(w):
        return math.exp((1j * scale) * (w**2))

    api = NLolib()
    opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)

    prep_ms_string: list[float] = []
    prep_ms_callable: list[float] = []
    prop_ms_string: list[float] = []
    prop_ms_callable: list[float] = []

    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        cfg_string = prepare_sim_config(
            n,
            runtime=RuntimeOperators(
                dispersion_expr="exp(i*c0*w*w)",
                constants=[scale],
            ),
            **common,
        )
        prep_ms_string.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        cfg_callable = prepare_sim_config(
            n,
            runtime=RuntimeOperators(
                dispersion_fn=dispersion_fn,
            ),
            **common,
        )
        prep_ms_callable.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        api.propagate(cfg_string, field, 2, opts)
        prop_ms_string.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        api.propagate(cfg_callable, field, 2, opts)
        prop_ms_callable.append((time.perf_counter() - t0) * 1e3)

    print(f"n={n} runs={int(args.runs)}")
    print(f"prepare_ms_string_mean={mean(prep_ms_string):.3f}")
    print(f"prepare_ms_callable_mean={mean(prep_ms_callable):.3f}")
    print(f"propagate_ms_string_mean={mean(prop_ms_string):.3f}")
    print(f"propagate_ms_callable_mean={mean(prop_ms_callable):.3f}")


if __name__ == "__main__":
    main()
