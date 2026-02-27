"""
Analytical GRIN validation sweep example.

Runs exact phase-only GRIN scenarios across several GRIN strengths and
summarizes analytical-vs-numerical error trends.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.cli import build_example_parser
from backend.runner import NloExampleRunner, SimulationOptions
from backend.storage import ExampleRunDB
from grin_fiber_xy_rk4ip import run_phase_validation


def _save_summary_plot(
    output_path: Path,
    x_values: list[float],
    y_values: list[float],
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping sweep summary plot.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(x_values, y_values, marker="o", lw=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = build_example_parser(
        example_slug="grin_fiber_analytic_sweep",
        description="Analytical GRIN sweep with DB-backed run/replot.",
    )
    args = parser.parse_args()
    db = ExampleRunDB(args.db_path)
    example_name = "grin_fiber_analytic_validations"

    runner = NloExampleRunner()
    exec_options = SimulationOptions(backend="auto", fft_backend="auto", device_heap_fraction=0.70)
    output_root = Path(__file__).resolve().parent / "output" / "grin_fiber_analytic_sweep"
    output_root.mkdir(parents=True, exist_ok=True)

    final_errors: list[float] = []
    power_drifts: list[float] = []
    grin_strengths: list[float] = []

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        cases = db.list_cases(example_name=example_name, run_group=run_group)
        if not cases:
            raise RuntimeError(f"no stored cases found in run_group '{run_group}'.")
        for case in cases:
            g = float(case.meta.get("grin_gx", 0.0))
            final_error = float(case.meta.get("final_error", float("nan")))
            power_drift = float(case.meta.get("power_drift", float("nan")))
            if not (g > 0.0):
                continue
            grin_strengths.append(g)
            final_errors.append(final_error)
            power_drifts.append(power_drift)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        grin_strengths = [0.5e-4, 1.0e-4, 2.0e-4, 3.0e-4]
        for g in grin_strengths:
            scenario_name = f"grin_phase_strength_{g:.1e}".replace("+", "")
            _, final_error, power_drift = run_phase_validation(
                runner,
                scenario_name=scenario_name,
                nx=256,
                ny=256,
                dx=0.7,
                dy=0.7,
                w0=8.0,
                grin_gx=g,
                grin_gy=g,
                x_offset=0.0,
                y_offset=0.0,
                propagation_distance=0.25,
                num_records=8,
                exec_options=exec_options,
                output_root=output_root,
                storage_db=db,
                storage_example_name=example_name,
                run_group=run_group,
            )
            final_errors.append(final_error)
            power_drifts.append(power_drift)

    order = sorted(range(len(grin_strengths)), key=lambda i: grin_strengths[i])
    grin_strengths = [grin_strengths[i] for i in order]
    final_errors = [final_errors[i] for i in order]
    power_drifts = [power_drifts[i] for i in order]

    p1 = _save_summary_plot(
        output_root / "summary_final_error_vs_grin_strength.png",
        grin_strengths,
        final_errors,
        x_label="GRIN coefficient g",
        y_label="Final relative L2 error",
        title="GRIN phase-only analytical validation: error vs GRIN strength",
    )
    p2 = _save_summary_plot(
        output_root / "summary_power_drift_vs_grin_strength.png",
        grin_strengths,
        power_drifts,
        x_label="GRIN coefficient g",
        y_label="Relative power drift",
        title="GRIN phase-only analytical validation: power drift vs GRIN strength",
    )

    print(f"analytical GRIN sweep completed (run_group={run_group}).")
    for g, err, drift in zip(grin_strengths, final_errors, power_drifts):
        print(f"  g={g:.3e}: final_error={err:.6e}, power_drift={drift:.6e}")
    if p1 is not None:
        print(f"  {p1}")
    if p2 is not None:
        print(f"  {p2}")


if __name__ == "__main__":
    main()
