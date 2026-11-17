"""
Assessment harness for the GPU JIT/CUDA branch evaluation.

Runs one workload case on one build of nlolib and appends a JSON row with
host init/solve timings (library-internal timers) and, on main's
instrumented build, Vulkan timestamp phase times. The final field of the
first timed run is saved for cross-build correctness checks.

--tree selects the repository whose python/ package is imported and --lib the
nlolib shared library to load. Feature toggles are passed through the
environment by the caller (NLO_OPERATOR_JIT, NLO_DISABLE_VK_FUSED_STEP,
NLO_GPU_TIMESTAMPS, ...).

The script uses the bindings' shared public API (docs/binding_api_vocabulary.md),
so it drives trees that have that API. The branch snapshots of the original
assessment predate it; docs/gpu-jit-cuda-assessment.md records those results.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PHASES = ("other", "fft", "linear", "nonlinear", "rk_algebra", "error_norm")


def _centered(n: int, d: float) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * d


def build_case(nlo, case: str, size: str, mode: str, refine: int = 1):
    """Return (pulse, linear_operator, nonlinear_operator, options, meta) for one workload.

    The workloads mirror the binding examples. `options` holds the propagate
    keywords the case fixes; the caller adds the record and execution options.
    """
    adaptive = mode == "adaptive"
    r = float(refine)

    def steps(z, h, max_step, min_step, tolerance):
        return {
            "propagation_distance": z,
            "starting_step_size": h,
            "max_step_size": max_step if adaptive else h,
            "min_step_size": min_step if adaptive else h,
            "error_tolerance": tolerance,
        }

    if case in ("kerr1d", "expr1d"):
        n = int(size)
        dt = 0.01
        t = _centered(n, dt)
        field = np.exp(-((t / 0.2) ** 2)).astype(np.complex128)
        h = 1.0e-3 / r
        # spm.py: gamma = 80, plus weak GVD so the linear step is not the identity.
        half_beta2, gamma = 0.5 * -0.01, 80.0
        linear = lambda A, w: 1j * half_beta2 * (w * w)  # noqa: E731
        if case == "kerr1d":
            nonlinear = lambda A, I: 1j * gamma * A * I  # noqa: E731
        else:
            # Arbitrary runtime expression: saturable cubic-quintic response.
            quintic, saturation = 20.0, 0.5
            nonlinear = lambda A, I: 1j * A * (gamma * I - quintic * I * I) / (1.0 + saturation * I)  # noqa: E731
        pulse = nlo.PulseSpec(samples=field.tolist(), delta_time=dt)
        return pulse, linear, nonlinear, steps(0.2, h, 4.0 * h, 1.0e-6, 1.0e-6), {"n": n}
    if case == "raman1d":
        n = int(size)
        dt = 0.02
        beta2, gamma, width = -0.01, 1.40, 0.08
        z = 0.70
        h = z / 200.0 / r
        t = _centered(n, dt)
        p0 = abs(beta2) / (gamma * width * width)
        field = (np.sqrt(p0) / np.cosh(t / width)).astype(np.complex128)
        half_beta2 = 0.5 * beta2
        options = steps(z, h, z / 80.0, z / 20000.0, 2.0e-6)
        options.update(
            nonlinear_model="kerr_raman",
            nonlinear_gamma=gamma,
            raman_fraction=0.18,
            raman_tau1=0.0522,
            raman_tau2=0.0320,
            shock_omega0=0.0,
        )
        pulse = nlo.PulseSpec(samples=field.tolist(), delta_time=dt)
        # The built-in model replaces the nonlinear operator, which is still a
        # required argument.
        return pulse, (lambda A, w: 1j * half_beta2 * (w * w)), (lambda A, I: 0.0 * A), options, {"n": n}
    if case in ("grin2p1", "tensor3d"):
        nxy = int(size)
        nt = 512
        potential = None
        if case == "grin2p1":
            # high_order_grin_soliton.py physics.
            dt, dx = 0.02, 0.03
            beta2, beta_t, gamma, i_sat, grin = -0.08, -0.08, 1.5, 4.0, 1.5e-3
            t = _centered(nt, dt)
            x = _centered(nxy, dx)
            width_t = 0.30
            p0 = abs(beta2) / (gamma * width_t * width_t)
            temporal = np.sqrt(p0) / np.cosh(t / width_t)
            xx, yy = np.meshgrid(x, x)
            spatial = np.exp(-((xx / 0.55) ** 2) - ((yy / 0.75) ** 2))
            potential = (grin * (xx * xx + yy * yy)).astype(np.complex128).reshape(-1).tolist()
            nonlinear = lambda A, I, V: 1j * A * (gamma * (I / (1.0 + I / i_sat)) + V)  # noqa: E731
            z = 0.04
            h = 2.0e-3 / r
        else:
            # tensor_dispersion_3d.py physics (linear only).
            dt, dx = 0.04, 0.15
            beta2, beta_t = 0.08, -0.20
            t = _centered(nt, dt)
            x = _centered(nxy, dx)
            temporal = np.exp(-((t / 0.24) ** 2))
            xx, yy = np.meshgrid(x, x)
            spatial = np.exp(-((xx / 0.60) ** 2) - ((yy / 0.70) ** 2))
            nonlinear = lambda A, I: 0.0 * A  # noqa: E731
            z = 0.08
            h = 8.0e-3 / r
        half_beta2 = 0.5 * beta2
        linear = lambda A, wt, kx, ky: 1j * (half_beta2 * (wt * wt) + beta_t * (kx * kx + ky * ky))  # noqa: E731
        # Layout XYT, t fastest: index = (y*nx + x)*nt + t.
        field = (spatial[:, :, None] * temporal[None, None, :]).astype(np.complex128).reshape(-1)
        pulse = nlo.PulseSpec(
            samples=field.tolist(),
            delta_time=dt,
            tensor_nt=nt,
            tensor_nx=nxy,
            tensor_ny=nxy,
            delta_x=dx,
            delta_y=dx,
            potential_grid=potential,
        )
        return pulse, linear, nonlinear, steps(z, h, 4.0 * h, 1.0e-6, 1.0e-6), {"nt": nt, "nx": nxy, "ny": nxy}
    raise ValueError(f"unknown case {case}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=str(Path(__file__).resolve().parents[2]),
                    help="repo root whose python/ package to import (default: this repository)")
    ap.add_argument("--lib", required=True, help="nlolib shared library to load")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--backend", default="vulkan", choices=("vulkan", "cpu"))
    ap.add_argument("--case", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--mode", default="fixed", choices=("fixed", "adaptive"))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", required=True, help="JSONL results file")
    ap.add_argument("--field-dir", default="", help="save first timed final field here")
    ap.add_argument("--refine", type=int, default=1, help="divide the fixed step by this factor (reference runs)")
    args = ap.parse_args()

    os.environ["NLOLIB_LIBRARY"] = str(Path(args.lib).resolve())
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(args.lib).resolve().parent))
    sys.path.insert(0, str(Path(args.tree) / "python"))
    import nlolib as nlo  # noqa: E402

    nlo.load(str(Path(args.lib).resolve()))
    nlo.set_log_level("error")
    # The phase timers have no binding wrapper; read them from the same library.
    lib = ctypes.CDLL(nlo.loaded_library_path())
    read_phases = lib.nlolib_perf_profile_read_gpu_phases
    read_phases.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    try:
        lib.nlolib_perf_profile_set_enabled(1)
    except AttributeError:
        pass

    def reset():
        for fn in ("nlolib_perf_profile_reset", "nlolib_assess_reset_host_slots"):
            if hasattr(lib, fn):
                getattr(lib, fn)()

    def read():
        ph = (ctypes.c_double * 6)()
        cnt = (ctypes.c_uint64 * 6)()
        host = (ctypes.c_double * 4)()
        read_phases(ctypes.byref(ph), ctypes.byref(cnt), ctypes.byref(host))
        return list(ph), list(cnt), list(host)

    exec_options = nlo.ExecutionOptions(
        backend_type=args.backend,
        fft_backend={"vulkan": "vkfft", "cpu": "fftw"}[args.backend],
    )

    pulse, linear, nonlinear, options, shape = build_case(nlo, args.case, args.size, args.mode, args.refine)
    field = pulse.samples
    nlo.set_progress_options(enabled=False)
    rows = []
    final_field = None
    step_meta = None
    for i in range(args.warmup + args.runs):
        reset()
        t0 = time.perf_counter()
        result = nlo.propagate(
            pulse,
            linear,
            nonlinear,
            num_records=2,
            exec_options=exec_options,
            capture_step_history=(args.mode == "adaptive"),
            **options,
        )
        wall_ms = (time.perf_counter() - t0) * 1e3
        ph, cnt, host = read()
        rows.append({"wall_ms": wall_ms, "init_ms": host[0], "solve_ms": host[1], "submits": host[2],
                     "ts_drops": host[3], "phase_ms": ph, "phase_intervals": cnt})
        if i == args.warmup:
            final_field = np.asarray(result.records, dtype=np.complex128).reshape(2, -1)[-1]
            history = result.step_history
            if history is not None:
                step_meta = {"count": len(history.step_size), "step_size": list(history.step_size)}

    timed = rows[args.warmup:]
    solve = [r["solve_ms"] for r in timed]
    q = statistics.quantiles(solve, n=4) if len(solve) >= 2 else [solve[0]] * 3
    out = {
        "variant": args.variant, "backend": args.backend, "case": args.case, "size": args.size,
        "mode": args.mode, "env": {k: v for k, v in os.environ.items() if k.startswith("NLO_") and k != "NLOLIB_LIBRARY"},
        "runs": len(timed), "solve_ms_median": statistics.median(solve), "solve_ms_q1": q[0], "solve_ms_q3": q[2],
        "init_ms_cold": rows[0]["init_ms"], "init_ms_median": statistics.median(r["init_ms"] for r in timed),
        "wall_ms_median": statistics.median(r["wall_ms"] for r in timed),
        "submits": timed[0]["submits"], "ts_drops": timed[0]["ts_drops"],
        "phase_ms_median": {p: statistics.median(r["phase_ms"][k] for r in timed) for k, p in enumerate(PHASES)},
        "phase_intervals": dict(zip(PHASES, timed[0]["phase_intervals"])),
        "steps": (step_meta or {}).get("count"),
        "energy_in": float(np.sum(np.abs(np.asarray(field)) ** 2)),
        "energy_out": float(np.sum(np.abs(final_field) ** 2)),
        "shape": shape,
    }
    if args.field_dir:
        Path(args.field_dir).mkdir(parents=True, exist_ok=True)
        stem = f"{args.variant}__{args.case}__{args.size}__{args.mode}"
        np.save(Path(args.field_dir) / f"{stem}.npy", final_field)
        if step_meta:
            np.save(Path(args.field_dir) / f"{stem}__steps.npy", np.asarray(step_meta["step_size"]))
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({k: out[k] for k in ("variant", "case", "size", "mode", "solve_ms_median", "init_ms_cold", "steps")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
