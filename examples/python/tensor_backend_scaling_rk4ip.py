"""
Tensor-grid backend timing example with two plots:
- field-size scaling at one recorded snapshot
- record-count scaling at one moderate field size
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from backend.app_base import ExampleAppBase
from backend.plotting import plot_two_curve_comparison
from backend.runner import NloExampleRunner, centered_time_grid


@dataclass(frozen=True)
class BenchmarkRow:
    plot: str
    backend: str
    scale: int
    total_samples: int
    num_records: int
    status: str
    message: str
    elapsed_ms: float | None


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if len(values) <= 0:
        raise argparse.ArgumentTypeError("expected at least one integer.")
    return values


def _centered_spatial_grid(num_samples: int, delta: float) -> np.ndarray:
    return (np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)) * float(delta)


def _tensor_shape(scale: int) -> tuple[int, int, int, int]:
    resolved = int(scale)
    if resolved <= 0:
        raise ValueError("scale must be positive.")
    nt = 2 * resolved
    nx = resolved
    ny = resolved
    return nt, nx, ny, nt * nx * ny


def _build_case(runner: NloExampleRunner, scale: int, *, step_size: float):
    nlo = runner.nlo
    nt, nx, ny, total_samples = _tensor_shape(scale)
    dt = 0.04
    dx = 0.15
    dy = 0.15

    t_axis = centered_time_grid(nt, dt)
    x_axis = _centered_spatial_grid(nx, dx)
    y_axis = _centered_spatial_grid(ny, dy)
    omega = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)

    temporal = np.exp(-((t_axis / 0.24) ** 2))
    xx, yy = np.meshgrid(x_axis, y_axis)
    transverse = np.exp(-((xx / 0.60) ** 2) - ((yy / 0.70) ** 2))
    field0 = np.transpose(
        (temporal[:, None, None] * transverse[None, :, :]).astype(np.complex128),
        (2, 1, 0),
    ).reshape(-1)

    runtime = nlo.RuntimeOperators(
        linear_factor_expr="i*(c0*wt*wt + c1*(kx*kx + ky*ky))",
        nonlinear_expr="0",
        constants=[0.04, -0.20],
    )
    config = nlo.prepare_sim_config(
        total_samples,
        propagation_distance=0.20,
        starting_step_size=step_size,
        max_step_size=step_size,
        min_step_size=step_size,
        error_tolerance=1.0e-9,
        pulse_period=float(nt) * dt,
        delta_time=dt,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        tensor_layout=nlo.TENSOR_LAYOUT_XYT_T_FAST,
        frequency_grid=[complex(value, 0.0) for value in omega],
        delta_x=dx,
        delta_y=dy,
        runtime=runtime,
    )
    return config, field0, total_samples


def _run_case(
    api,
    runner: NloExampleRunner,
    *,
    plot_key: str,
    backend: str,
    scale: int,
    num_records: int,
    warmup: int,
    runs: int,
    step_size: float,
) -> BenchmarkRow:
    nlo = runner.nlo
    config, field0, nt, nx, ny, total_samples = _build_case(runner, scale)
    backend_type = nlo.VECTOR_BACKEND_CPU if backend == "cpu" else nlo.VECTOR_BACKEND_VULKAN
    exec_options = nlo.default_execution_options(backend_type)

    timings_ms: list[float] = []
    try:
        for run_idx in range(int(warmup) + int(runs)):
            start = time.perf_counter()
            api.propagate(
                config,
                field0,
                int(num_records),
                exec_options=exec_options,
                return_records=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if run_idx >= int(warmup):
                timings_ms.append(float(elapsed_ms))
    except RuntimeError as exc:
        return BenchmarkRow(
            plot=plot_key,
            backend=backend,
            scale=scale,
            total_samples=total_samples,
            num_records=int(num_records),
            status="error",
            message=str(exc),
            elapsed_ms=None,
        )

    return BenchmarkRow(
        plot=plot_key,
        backend=backend,
        scale=scale,
        total_samples=total_samples,
        num_records=int(num_records),
        status="ok",
        message="",
        elapsed_ms=float(np.mean(timings_ms)) if len(timings_ms) > 0 else None,
    )


def _write_csv(rows: list[BenchmarkRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["plot", "backend", "scale", "total_samples", "num_records", "status", "message", "elapsed_ms"])
        for row in rows:
            writer.writerow([
                row.plot,
                row.backend,
                row.scale,
                row.total_samples,
                row.num_records,
                row.status,
                row.message,
                row.elapsed_ms,
            ])


def _plot_series(rows: list[BenchmarkRow], *, plot_key: str, x_attr: str, output_path: Path, x_label: str) -> Path | None:
    filtered = [row for row in rows if row.plot == plot_key and row.status == "ok"]
    cpu_rows = [row for row in filtered if row.backend == "cpu"]
    gpu_rows = [row for row in filtered if row.backend == "gpu"]
    cpu_map = {float(getattr(row, x_attr)): float(row.elapsed_ms) for row in cpu_rows if row.elapsed_ms is not None}
    gpu_map = {float(getattr(row, x_attr)): float(row.elapsed_ms) for row in gpu_rows if row.elapsed_ms is not None}
    common = sorted(set(cpu_map) & set(gpu_map))
    if len(common) <= 0:
        return None
    return plot_two_curve_comparison(
        np.asarray(common, dtype=np.float64),
        np.asarray([cpu_map[value] for value in common], dtype=np.float64),
        np.asarray([gpu_map[value] for value in common], dtype=np.float64),
        output_path,
        label_a="CPU mean runtime",
        label_b="GPU mean runtime",
        x_label=x_label,
        y_label="Mean runtime (ms)",
    )


def _run(args: argparse.Namespace) -> float:
    runner = NloExampleRunner()
    api = runner.nlo.NLolib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []
    for scale in args.field_scales:
        rows.append(
            _run_case(
                api,
                runner,
                plot_key="field_scale",
                backend="cpu",
                scale=scale,
                num_records=1,
                warmup=args.warmup,
                runs=args.runs,
                step_size=0.02,
            )
        )
        rows.append(
            _run_case(
                api,
                runner,
                plot_key="field_scale",
                backend="gpu",
                scale=scale,
                num_records=1,
                warmup=args.warmup,
                runs=args.runs,
                step_size=0.02,
            )
        )

    max_record_count = max(args.record_counts)
    record_step_size = 0.02 if max_record_count <= 1 else 0.20 / float(max_record_count - 1)
    for count in args.record_counts:
        rows.append(
            _run_case(
                api,
                runner,
                plot_key="record_scale",
                backend="cpu",
                scale=args.record_scale,
                num_records=count,
                warmup=args.warmup,
                runs=args.runs,
                step_size=record_step_size,
            )
        )
        rows.append(
            _run_case(
                api,
                runner,
                plot_key="record_scale",
                backend="gpu",
                scale=args.record_scale,
                num_records=count,
                warmup=args.warmup,
                runs=args.runs,
                step_size=record_step_size,
            )
        )

    csv_path = args.output_dir / "tensor_backend_scaling_results.csv"
    _write_csv(rows, csv_path)
    field_plot = _plot_series(
        rows,
        plot_key="field_scale",
        x_attr="total_samples",
        output_path=args.output_dir / "tensor_backend_scaling_field_runtime.png",
        x_label="Total tensor samples",
    )
    record_plot = _plot_series(
        rows,
        plot_key="record_scale",
        x_attr="num_records",
        output_path=args.output_dir / "tensor_backend_scaling_record_runtime.png",
        x_label="Recorded snapshots",
    )

    print(f"saved csv: {csv_path}")
    for row in rows:
        print(
            f"{row.plot:>12s} {row.backend:>3s} scale={row.scale:<4d} "
            f"records={row.num_records:<4d} status={row.status:<5s} elapsed_ms={row.elapsed_ms} {row.message}"
        )
    if field_plot is not None:
        print(f"saved plot: {field_plot}")
    if record_plot is not None:
        print(f"saved plot: {record_plot}")
    return 0.0


class TensorBackendScalingApp(ExampleAppBase):
    example_slug = "tensor_backend_scaling"
    description = "Plot CPU/GPU tensor-grid timings for field-size scaling and record-count scaling."

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--field-scales",
            type=_parse_int_csv,
            default=[16, 32, 48, 64, 80, 96, 112, 128],
            help="Comma-separated tensor scales for the field-size scaling plot.",
        )
        parser.add_argument(
            "--record-scale",
            type=int,
            default=64,
            help="Tensor scale used for the record-count scaling plot.",
        )
        parser.add_argument(
            "--record-counts",
            type=_parse_int_csv,
            default=[1, 8, 16, 32, 64, 128],
            help="Comma-separated record counts for the record-scaling plot.",
        )
        parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per benchmark point.")
        parser.add_argument("--runs", type=int, default=3, help="Measured runs per benchmark point.")

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return TensorBackendScalingApp.from_cli().run()


if __name__ == "__main__":
    main()
