"""
Two-mode linear beating analytical validation.

Implements a 2x2 linear coupler using explicit ND runtime configuration and
compares numerical mode exchange against the closed-form cosine/sine solution.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.storage import ExampleRunDB


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT / "python"
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))

import nlolib_ctypes as nlo


def _load_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _configure_runtime_logging(api: nlo.NLolib) -> None:
    try:
        api.set_log_level(nlo.NLOLIB_LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        api.set_progress_options(enabled=False, milestone_percent=5, emit_on_step_adjust=False)
    except Exception:
        pass


def _save_mode_power_plot(
    output_path: Path,
    z_axis: np.ndarray,
    mode1_num: np.ndarray,
    mode2_num: np.ndarray,
    mode1_ref: np.ndarray,
    mode2_ref: np.ndarray,
) -> Path | None:
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping two-mode power plot.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(z_axis, mode1_ref, lw=2.0, color="tab:blue", label="|A1|^2 analytical")
    ax.plot(z_axis, mode2_ref, lw=2.0, color="tab:orange", label="|A2|^2 analytical")
    ax.plot(z_axis, mode1_num, "--", lw=1.7, color="tab:blue", label="|A1|^2 numerical")
    ax.plot(z_axis, mode2_num, "--", lw=1.7, color="tab:orange", label="|A2|^2 numerical")
    ax.set_xlabel("Propagation distance z")
    ax.set_ylabel("Mode power")
    ax.set_title("Two-Mode Linear Beating: Power Exchange")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_error_plot(
    output_path: Path,
    z_axis: np.ndarray,
    error_curve: np.ndarray,
) -> Path | None:
    plt = _load_plt()
    if plt is None:
        print("matplotlib not available; skipping two-mode error plot.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(z_axis, error_curve, lw=1.8, color="tab:red")
    ax.set_xlabel("Propagation distance z")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Two-Mode Linear Beating: Error Over Propagation")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> float:
    api = nlo.NLolib()
    _configure_runtime_logging(api)

    parser = build_example_parser(
        example_slug="two_mode_linear_beating",
        description="Two-mode linear beating with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "two_mode_linear_beating_rk4ip"
    case_key = "default"

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        meta = loaded.meta
        kappa = float(meta["kappa"])
        records = np.asarray(loaded.records, dtype=np.complex128).reshape(-1, 2)
        z_axis = np.asarray(loaded.z_axis, dtype=np.float64)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        kappa = 2.0
        z_final = 1.0
        num_records = 200
        sim_cfg = nlo.prepare_sim_config(
            2,
            propagation_distance=z_final,
            starting_step_size=1e-3,
            max_step_size=1e-3,
            min_step_size=1e-3,
            error_tolerance=1e-6,
            pulse_period=2.0,
            delta_time=1.0,
            time_nt=2,
            frequency_grid=[complex(1.0, 0.0), complex(-1.0, 0.0)],
            spatial_nx=1,
            spatial_ny=1,
            delta_x=1.0,
            delta_y=1.0,
            runtime=nlo.RuntimeOperators(
                dispersion_factor_expr="i*c0*w",
                nonlinear_expr="0",
                constants=[kappa],
            ),
        )
        exec_options = nlo.default_execution_options(
            backend_type=nlo.NLO_VECTOR_BACKEND_AUTO,
            fft_backend=nlo.NLO_FFT_BACKEND_AUTO,
        )
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=32,
        )
        result = api.propagate(
            sim_cfg,
            [complex(1.0, 0.0), complex(0.0, 0.0)],
            num_records,
            exec_options,
            sqlite_path=storage_kwargs["sqlite_path"],
            run_id=storage_kwargs["run_id"],
            chunk_records=storage_kwargs["chunk_records"],
            sqlite_max_bytes=storage_kwargs["sqlite_max_bytes"],
            log_final_output_field_to_db=storage_kwargs["log_final_output_field_to_db"],
        )
        records = np.asarray(result.records, dtype=np.complex128).reshape(num_records, 2)
        z_axis = np.asarray(result.z_axis, dtype=np.float64)
        db.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            run_id=str(result.meta["storage_result"]["run_id"]),
            meta={"kappa": float(kappa)},
        )

    a_ref = np.empty_like(records)
    for i, z in enumerate(z_axis):
        theta = kappa * float(z)
        a_ref[i, 0] = math.cos(theta)
        a_ref[i, 1] = 1j * math.sin(theta)

    ref_norm = np.linalg.norm(a_ref, axis=1)
    denom = np.maximum(ref_norm, 1e-15)
    error_curve = np.linalg.norm(records - a_ref, axis=1) / denom

    mode1_num = np.abs(records[:, 0]) ** 2
    mode2_num = np.abs(records[:, 1]) ** 2
    mode1_ref = np.abs(a_ref[:, 0]) ** 2
    mode2_ref = np.abs(a_ref[:, 1]) ** 2

    max_complex_error = float(np.max(np.abs(records - a_ref)))
    max_power_exchange_error = float(
        max(
            np.max(np.abs(mode1_num - mode1_ref)),
            np.max(np.abs(mode2_num - mode2_ref)),
        )
    )
    total_power = mode1_num + mode2_num
    power_drift = float(np.max(np.abs(total_power - total_power[0])) / max(abs(total_power[0]), 1e-15))

    output_dir = Path(__file__).resolve().parent / "output" / "two_mode_linear_beating"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    p1 = _save_mode_power_plot(
        output_dir / "mode_power_exchange.png",
        z_axis,
        mode1_num,
        mode2_num,
        mode1_ref,
        mode2_ref,
    )
    if p1 is not None:
        saved.append(p1)

    p2 = _save_error_plot(output_dir / "error_over_propagation.png", z_axis, error_curve)
    if p2 is not None:
        saved.append(p2)

    print(f"two-mode linear beating summary (run_group={run_group}):")
    print(f"  max complex-field abs error  = {max_complex_error:.6e}")
    print(f"  max power-exchange abs error = {max_power_exchange_error:.6e}")
    print(f"  relative total-power drift   = {power_drift:.6e}")
    if saved:
        print("saved plots:")
        for path in saved:
            print(f"  {path}")

    return max_complex_error


if __name__ == "__main__":
    main()
