"""
Equal-point nlolib CPU vs GPU runtime benchmark.

This keeps the tensor scaling benchmark focused on nlolib only, using the
same tensor problem construction as the mixed MMTools comparison example.
"""

from __future__ import annotations

import argparse
import numpy as np

from backend.app_base import ExampleAppBase
from backend.storage import ExampleRunDB
from tensor_backend_scaling_rk4ip import (
    BenchmarkRow,
    _NLOLIB_RUNTIME_PLOT_SPEC,
    _benchmark_nlolib_rows,
    _case_from_scale,
    _load_replot_benchmark_rows,
    _parse_int_csv,
    _plot_runtime,
    _print_summary,
    _save_benchmark_rows,
    _write_csv,
    RuntimePlotSpec,
    RuntimePlotSeries,
)


def _series_for_backend(backend: str) -> tuple[RuntimePlotSeries, ...]:
    backend_key = backend.upper()
    return tuple(
        series
        for series in _NLOLIB_RUNTIME_PLOT_SPEC.series
        if series.backend.upper() == backend_key
    )


def _benchmark_nlolib_rows_with_backend_scales(
    args: argparse.Namespace,
    runner,
    api,
) -> list[BenchmarkRow]:
    cpu_cases = [_case_from_scale(scale) for scale in args.scales]
    gpu_scales = getattr(args, "gpu_scales", None)
    gpu_cases = [
        _case_from_scale(scale)
        for scale in (args.scales if gpu_scales is None else gpu_scales)
    ]

    rows: list[BenchmarkRow] = []
    rows.extend(
        _benchmark_nlolib_rows(
            args,
            runner,
            api,
            cpu_cases,
            series=_series_for_backend("CPU"),
        )
    )
    rows.extend(
        _benchmark_nlolib_rows(
            args,
            runner,
            api,
            gpu_cases,
            series=_series_for_backend("GPU"),
        )
    )
    return rows


def _run(args: argparse.Namespace) -> float:
    db = ExampleRunDB(args.db_path)
    example_name = "tensor_backend_scaling_nlolib"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tensor_backend_scaling_nlolib_results.csv"

    if args.replot:
        rows: list[BenchmarkRow] = _load_replot_benchmark_rows(
            db,
            example_name=example_name,
            run_group=args.run_group,
            fallback_csv=csv_path,
        )
    else:
        from backend.runner import NloExampleRunner

        run_group = db.begin_group(example_name, args.run_group)
        runner = NloExampleRunner()
        api = runner.nlo.NLolib()

        rows = _benchmark_nlolib_rows_with_backend_scales(args, runner, api)
        _save_benchmark_rows(
            db,
            example_name=example_name,
            run_group=run_group,
            rows=rows,
        )

    _write_csv(rows, csv_path)
    plot_path = _plot_runtime(
        rows,
        args.output_dir,
        plot_spec=RuntimePlotSpec(
            series=_NLOLIB_RUNTIME_PLOT_SPEC.series,
            save_path="tensor_backend_scaling_nlolib_runtime.png",
            fit_skip_initial_points=_NLOLIB_RUNTIME_PLOT_SPEC.fit_skip_initial_points,
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
            default=np.geomspace(8, 32, num=4, dtype=int).tolist(),
            help=(
                "Comma-separated CPU tensor scales with nt=2*scale and nx=ny=scale. "
                "Also used for GPU unless --gpu-scales is set."
            ),
        )
        parser.add_argument(
            "--gpu-scales",
            type=_parse_int_csv,
            default=np.geomspace(8, 480, num=20, dtype=int).tolist(),
            help="Comma-separated GPU tensor scales. Defaults to --scales when omitted.",
        )
        parser.add_argument(
            "--num-records",
            type=int,
            default=1,
            help="Recorded snapshots per run. Use 1 to benchmark final-only output.",
        )
        parser.add_argument("--warmup", type=int, default=4, help="Warmup runs per benchmark point.")
        parser.add_argument("--runs", type=int, default=4, help="Measured runs per benchmark point.")

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return TensorBackendScalingNlolibApp.from_cli().run()


if __name__ == "__main__":
    main()
