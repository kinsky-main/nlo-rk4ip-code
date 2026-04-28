"""
Equal-point nlolib CPU vs GPU runtime benchmark.

This keeps the tensor scaling benchmark focused on nlolib only, using the
same tensor problem construction as the mixed MMTools comparison example.
"""

from __future__ import annotations

import argparse

from backend.app_base import ExampleAppBase
from tensor_backend_scaling_rk4ip import (
    BenchmarkRow,
    _NLOLIB_RUNTIME_PLOT_SPEC,
    _benchmark_nlolib_rows,
    _case_from_scale,
    _parse_int_csv,
    _plot_runtime,
    _print_summary,
    _write_csv,
    RuntimePlotSpec,
)


def _run(args: argparse.Namespace) -> float:
    from backend.runner import NloExampleRunner

    runner = NloExampleRunner()
    api = runner.nlo.NLolib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = [_case_from_scale(scale) for scale in args.scales]
    rows: list[BenchmarkRow] = _benchmark_nlolib_rows(args, runner, api, cases)

    csv_path = args.output_dir / "tensor_backend_scaling_nlolib_results.csv"
    _write_csv(rows, csv_path)
    plot_path = _plot_runtime(
        rows,
        args.output_dir,
        plot_spec=RuntimePlotSpec(
            series=_NLOLIB_RUNTIME_PLOT_SPEC.series,
            save_path="tensor_backend_scaling_nlolib_runtime.png",
        ),
    )

    print(f"saved csv: {csv_path}")
    _print_summary(rows)
    if plot_path is not None:
        print(f"saved plot: {plot_path}")
    return 0.0


class TensorBackendScalingNlolibApp(ExampleAppBase):
    example_slug = "tensor_backend_scaling_nlolib"
    description = "Equal-point nlolib CPU vs GPU tensor runtime benchmark."

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--scales",
            type=_parse_int_csv,
            default=[16, 32, 48, 64, 80, 96, 112, 128],
            help="Comma-separated default tensor scales with nt=2*scale and nx=ny=scale.",
        )
        parser.add_argument(
            "--num-records",
            type=int,
            default=1,
            help="Recorded snapshots per run. Use 1 to benchmark final-only output.",
        )
        parser.add_argument("--warmup", type=int, default=7, help="Warmup runs per benchmark point.")
        parser.add_argument("--runs", type=int, default=5, help="Measured runs per benchmark point.")

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return TensorBackendScalingNlolibApp.from_cli().run()


if __name__ == "__main__":
    main()
