"""
Assessment harness for the GPU JIT/CUDA branch evaluation.

Runs one workload case on one build of nlolib and appends a JSON row with
host init/solve timings (library-internal timers) and, on main's
instrumented build, Vulkan timestamp phase times. The final field of the
first timed run is saved for cross-build correctness checks.

The same script drives main and the branch snapshot: --tree selects the
python package (and its API spelling), --lib the nlolib.dll to load.
Feature toggles are passed through the environment by the caller
(NLO_OPERATOR_JIT, NLO_DISABLE_VK_FUSED_STEP, NLO_GPU_TIMESTAMPS, ...).
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


def _const(nlo, *names):
    for name in names:
        if hasattr(nlo, name):
            return getattr(nlo, name)
    raise AttributeError(names)


def _centered(n: int, d: float) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * d


def build_case(nlo, case: str, size: str, mode: str, refine: int = 1):
    """Return (cfg, field_list, meta) for one workload, mirroring main's examples."""
    adaptive = mode == "adaptive"
    r = float(refine)
    if case in ("kerr1d", "expr1d"):
        n = int(size)
        dt = 0.01
        t = _centered(n, dt)
        field = np.exp(-((t / 0.2) ** 2)).astype(np.complex128)
        omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
        z = 0.2
        h = 1.0e-3 / r
        if case == "kerr1d":
            # spm_rk4ip.py: gamma = 80, plus weak GVD so the linear step is not the identity.
            runtime = nlo.RuntimeOperators(
                linear_factor_expr="i*c0*(w*w)",
                linear_expr="exp(h*D)",
                nonlinear_expr="i*c1*A*I",
                constants=[0.5 * -0.01, 80.0],
            )
        else:
            # Arbitrary runtime expression: saturable cubic-quintic response.
            runtime = nlo.RuntimeOperators(
                linear_factor_expr="i*c0*(w*w)",
                linear_expr="exp(h*D)",
                nonlinear_expr="i*A*(c1*I - c2*I*I)/(1.0 + c3*I)",
                constants=[0.5 * -0.01, 80.0, 20.0, 0.5],
            )
        cfg = nlo.prepare_sim_config(
            n,
            propagation_distance=z,
            starting_step_size=h,
            max_step_size=(4.0 * h) if adaptive else h,
            min_step_size=(1.0e-6) if adaptive else h,
            error_tolerance=1.0e-6,
            pulse_period=n * dt,
            delta_time=dt,
            frequency_grid=omega.astype(np.complex128).tolist(),
            runtime=runtime,
        )
        return cfg, field.tolist(), {"n": n}
    if case == "raman1d":
        n = int(size)
        dt = 0.02
        beta2, gamma, width = -0.01, 1.40, 0.08
        z = 0.70
        h = z / 200.0 / r
        t = _centered(n, dt)
        p0 = abs(beta2) / (gamma * width * width)
        field = (np.sqrt(p0) / np.cosh(t / width)).astype(np.complex128)
        omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
        runtime = nlo.RuntimeOperators(
            linear_factor_expr="i*c0*(w*w)",
            linear_expr="exp(h*D)",
            nonlinear_model=_const(nlo, "NLO_NONLINEAR_MODEL_KERR_RAMAN", "NONLINEAR_MODEL_KERR_RAMAN"),
            nonlinear_gamma=gamma,
            raman_fraction=0.18,
            raman_tau1=0.0522,
            raman_tau2=0.0320,
            shock_omega0=0.0,
            constants=[0.5 * beta2],
        )
        cfg = nlo.prepare_sim_config(
            n,
            propagation_distance=z,
            starting_step_size=h,
            max_step_size=(z / 80.0) if adaptive else h,
            min_step_size=(z / 20000.0) if adaptive else h,
            error_tolerance=2.0e-6,
            pulse_period=n * dt,
            delta_time=dt,
            frequency_grid=omega.astype(np.complex128).tolist(),
            runtime=runtime,
        )
        return cfg, field.tolist(), {"n": n}
    if case in ("grin2p1", "tensor3d"):
        nxy = int(size)
        nt = 512
        layout = _const(nlo, "NLO_TENSOR_LAYOUT_XYT_T_FAST", "TENSOR_LAYOUT_XYT_T_FAST")
        if case == "grin2p1":
            # high_order_grin_soliton_rk4ip.py parameters.
            dt, dx = 0.02, 0.03
            beta2, beta_t, gamma, i_sat, grin = -0.08, -0.08, 1.5, 4.0, 1.5e-3
            t = _centered(nt, dt)
            x = _centered(nxy, dx)
            width_t = 0.30
            p0 = abs(beta2) / (gamma * width_t * width_t)
            temporal = np.sqrt(p0) / np.cosh(t / width_t)
            xx, yy = np.meshgrid(x, x)
            spatial = np.exp(-((xx / 0.55) ** 2) - ((yy / 0.75) ** 2))
            potential = (grin * (xx * xx + yy * yy)).astype(np.complex128).reshape(-1)
            runtime = nlo.RuntimeOperators(
                linear_factor_expr="i*(c0*(wt*wt) + c1*(kx*kx + ky*ky))",
                linear_expr="exp(h*D)",
                nonlinear_expr="i*A*(c2*(I/(1.0 + I/c3)) + V)",
                constants=[0.5 * beta2, beta_t, gamma, i_sat],
            )
            z = 0.04
            h = 2.0e-3 / r
            extra = {"potential_grid": potential.tolist()}
        else:
            # tensor_dispersion_3d_rk4ip.py parameters (linear only).
            dt, dx = 0.04, 0.15
            beta2, beta_t = 0.08, -0.20
            t = _centered(nt, dt)
            x = _centered(nxy, dx)
            temporal = np.exp(-((t / 0.24) ** 2))
            xx, yy = np.meshgrid(x, x)
            spatial = np.exp(-((xx / 0.60) ** 2) - ((yy / 0.70) ** 2))
            runtime = nlo.RuntimeOperators(
                linear_factor_expr="i*(c0*(wt*wt) + c1*(kx*kx + ky*ky))",
                linear_expr="exp(h*D)",
                nonlinear_expr="0.0*A",
                constants=[0.5 * beta2, beta_t],
            )
            z = 0.08
            h = 8.0e-3 / r
            extra = {}
        # Layout XYT, t fastest: index = (y*nx + x)*nt + t.
        field = (spatial[:, :, None] * temporal[None, None, :]).astype(np.complex128).reshape(-1)
        cfg = nlo.prepare_sim_config(
            nt * nxy * nxy,
            propagation_distance=z,
            starting_step_size=h,
            max_step_size=(4.0 * h) if adaptive else h,
            min_step_size=(1.0e-6) if adaptive else h,
            error_tolerance=1.0e-6,
            pulse_period=float(nt) * dt,
            delta_time=dt,
            tensor_nt=nt,
            tensor_nx=nxy,
            tensor_ny=nxy,
            tensor_layout=int(layout),
            delta_x=dx,
            delta_y=dx,
            frequency_grid=(2.0 * np.pi * np.fft.fftfreq(nt, d=dt)).astype(np.complex128).tolist(),
            runtime=runtime,
            **extra,
        )
        return cfg, field.tolist(), {"nt": nt, "nx": nxy, "ny": nxy}
    raise ValueError(f"unknown case {case}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="repo root whose python/ package to import")
    ap.add_argument("--lib", required=True, help="nlolib.dll to load")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--backend", default="vulkan", choices=("vulkan", "cuda", "cpu"))
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
    os.add_dll_directory(str(Path(args.lib).resolve().parent))
    sys.path.insert(0, str(Path(args.tree) / "python"))
    import nlolib as nlo  # noqa: E402

    try:
        nlo.set_log_level(_const(nlo, "NLOLIB_LOG_LEVEL_ERROR"))
    except Exception:
        pass
    lib = nlo.load()
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

    backend = _const(nlo, {"vulkan": "NLO_VECTOR_BACKEND_VULKAN", "cuda": "NLO_VECTOR_BACKEND_CUDA",
                           "cpu": "NLO_VECTOR_BACKEND_CPU"}[args.backend])
    fft = _const(nlo, {"vulkan": "NLO_FFT_BACKEND_VKFFT", "cuda": "NLO_FFT_BACKEND_CUFFT",
                       "cpu": "NLO_FFT_BACKEND_FFTW"}[args.backend])
    exec_options = nlo.default_execution_options(backend_type=backend, fft_backend=fft)

    cfg, field, shape = build_case(nlo, args.case, args.size, args.mode, args.refine)
    api = nlo.NLolib()
    api.set_progress_options(enabled=False)
    rows = []
    final_field = None
    step_meta = None
    for i in range(args.warmup + args.runs):
        reset()
        t0 = time.perf_counter()
        kw = {}
        if args.mode == "adaptive":
            kw = {"capture_step_history": True, "step_history_capacity": 200000}
        result = api.propagate(cfg, field, 2, exec_options, **kw)
        wall_ms = (time.perf_counter() - t0) * 1e3
        ph, cnt, host = read()
        rows.append({"wall_ms": wall_ms, "init_ms": host[0], "solve_ms": host[1], "submits": host[2],
                     "ts_drops": host[3], "phase_ms": ph, "phase_intervals": cnt})
        if i == args.warmup:
            final_field = np.asarray(result.records, dtype=np.complex128).reshape(2, -1)[-1]
            sh = result.meta.get("step_history")
            if sh:
                step_meta = {"count": len(sh.get("step_size", [])), "step_size": sh.get("step_size", [])}

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
