"""
Simple tensor-grid backend scaling benchmark with timing plots.

Choose scale lists that correspond to your machine's three regions:
- ``--gpu-fit-scales``: fits in GPU memory
- ``--host-fit-scales``: fits in system memory but not GPU memory
- ``--spill-records`` with ``--spill-scale``: output volume exceeds system memory
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from backend.app_base import ExampleAppBase
from backend.plotting import plot_summary_curve, plot_two_curve_comparison
from backend.runner import NloExampleRunner, centered_time_grid


@dataclass(frozen=True)
class BenchmarkRow:
    region: str
    backend: str
    scale: int
    nt: int
    nx: int
    ny: int
    total_samples: int
    num_records: int
    status: str
    message: str
    elapsed_ms: float | None
    records_spilled: int
    chunks_written: int
    db_size_bytes: int


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if len(values) <= 0:
        raise argparse.ArgumentTypeError("expected at least one integer.")
    return values


def _centered_spatial_grid(num_samples: int, delta: float) -> np.ndarray:
    return (np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)) * float(delta)


def _gaussian_tensor_field(
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> np.ndarray:
    temporal = np.exp(-((t_axis / 0.24) ** 2))
    xx, yy = np.meshgrid(x_axis, y_axis)
    transverse = np.exp(-((xx / 0.60) ** 2) - ((yy / 0.70) ** 2))
    return (temporal[:, None, None] * transverse[None, :, :]).astype(np.complex128)


def _flatten_tfast(field_tyx: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(field_tyx, dtype=np.complex128), (2, 1, 0)).reshape(-1)


def _tensor_shape(scale: int) -> tuple[int, int, int, int]:
    resolved = int(scale)
    if resolved <= 0:
        raise ValueError("scale must be positive.")
    nt = 2 * resolved
    nx = resolved
    ny = resolved
    return nt, nx, ny, nt * nx * ny


def _build_case(runner: NloExampleRunner, scale: int):
    nlo = runner.nlo
    nt, nx, ny, total_samples = _tensor_shape(scale)
    dt = 0.04
    dx = 0.15
    dy = 0.15

    t_axis = centered_time_grid(nt, dt)
    x_axis = _centered_spatial_grid(nx, dx)
    y_axis = _centered_spatial_grid(ny, dy)
    omega = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
    field0 = _flatten_tfast(_gaussian_tensor_field(t_axis, x_axis, y_axis))
    runtime = nlo.RuntimeOperators(
        linear_factor_expr="i*(c0*wt*wt + c1*(kx*kx + ky*ky))",
        nonlinear_expr="0",
        constants=[0.04, -0.20],
    )
    config = nlo.prepare_sim_config(
        total_samples,
        propagation_distance=0.20,
        starting_step_size=0.02,
        max_step_size=0.02,
        min_step_size=0.02,
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
    return config, field0, nt, nx, ny, total_samples


def _run_case(
    api,
    runner: NloExampleRunner,
    *,
    region: str,
    backend: str,
    scale: int,
    num_records: int,
    warmup: int,
    runs: int,
    sqlite_path: str | None,
) -> BenchmarkRow:
    nlo = runner.nlo
    config, field0, nt, nx, ny, total_samples = _build_case(runner, scale)
    backend_type = nlo.VECTOR_BACKEND_CPU if backend == "cpu" else nlo.VECTOR_BACKEND_VULKAN
    exec_options = nlo.default_execution_options(backend_type)
    storage_enabled = sqlite_path is not None
    if storage_enabled:
        exec_options.record_ring_target = 1
    return_records = not storage_enabled
    run_token = time.time_ns()

    timings_ms: list[float] = []
    last_meta: dict[str, object] = {}
    try:
        for run_idx in range(int(warmup) + int(runs)):
            start = time.perf_counter()
            result = api.propagate(
                config,
                field0,
                int(num_records),
                exec_options=exec_options,
                sqlite_path=sqlite_path,
                run_id=f"tensor-backend-scaling-{backend}-s{scale}-n{num_records}-{run_token}-{run_idx}",
                chunk_records=(1 if storage_enabled else 0),
                return_records=return_records,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            last_meta = dict(result.meta)
            if run_idx >= int(warmup):
                timings_ms.append(float(elapsed_ms))
    except RuntimeError as exc:
        return BenchmarkRow(
            region=region,
            backend=backend,
            scale=scale,
            nt=nt,
            nx=nx,
            ny=ny,
            total_samples=total_samples,
            num_records=int(num_records),
            status="error",
            message=str(exc),
            elapsed_ms=None,
            records_spilled=0,
            chunks_written=0,
            db_size_bytes=0,
        )

    storage = last_meta.get("storage_result", {})
    return BenchmarkRow(
        region=region,
        backend=backend,
        scale=scale,
        nt=nt,
        nx=nx,
        ny=ny,
        total_samples=total_samples,
        num_records=int(num_records),
        status="ok",
        message="",
        elapsed_ms=float(np.mean(timings_ms)) if len(timings_ms) > 0 else None,
        records_spilled=int(storage.get("records_spilled", 0)) if isinstance(storage, dict) else 0,
        chunks_written=int(storage.get("chunks_written", 0)) if isinstance(storage, dict) else 0,
        db_size_bytes=int(storage.get("db_size_bytes", 0)) if isinstance(storage, dict) else 0,
    )


def _write_csv(rows: list[BenchmarkRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "region",
            "backend",
            "scale",
            "nt",
            "nx",
            "ny",
            "total_samples",
            "num_records",
            "status",
            "message",
            "elapsed_ms",
            "records_spilled",
            "chunks_written",
            "db_size_bytes",
        ])
        for row in rows:
            writer.writerow([
                row.region,
                row.backend,
                row.scale,
                row.nt,
                row.nx,
                row.ny,
                row.total_samples,
                row.num_records,
                row.status,
                row.message,
                row.elapsed_ms,
                row.records_spilled,
                row.chunks_written,
                row.db_size_bytes,
            ])


def _plot_fit_region(rows: list[BenchmarkRow], output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    fit_rows = [row for row in rows if row.region == "gpu_fit" and row.status == "ok"]
    cpu_rows = [row for row in fit_rows if row.backend == "cpu"]
    gpu_rows = [row for row in fit_rows if row.backend == "gpu"]

    if len(cpu_rows) > 0 and len(gpu_rows) > 0:
        cpu_map = {row.total_samples: float(row.elapsed_ms) for row in cpu_rows if row.elapsed_ms is not None}
        gpu_map = {row.total_samples: float(row.elapsed_ms) for row in gpu_rows if row.elapsed_ms is not None}
        x_values = sorted(set(cpu_map) & set(gpu_map))
        if len(x_values) > 0:
            saved = plot_two_curve_comparison(
                np.asarray(x_values, dtype=np.float64),
                np.asarray([cpu_map[x] for x in x_values], dtype=np.float64),
                np.asarray([gpu_map[x] for x in x_values], dtype=np.float64),
                output_dir / "tensor_backend_scaling_fit_runtime.png",
                label_a="CPU mean runtime",
                label_b="GPU mean runtime",
                x_label="Total tensor samples",
                y_label="Mean runtime (ms)",
            )
            if saved is not None:
                saved_paths.append(saved)

    host_only_rows = [
        row for row in rows
        if row.region == "host_fit_only" and row.backend == "cpu" and row.status == "ok"
    ]
    if len(host_only_rows) > 0:
        host_only_rows.sort(key=lambda row: row.total_samples)
        saved = plot_summary_curve(
            [row.total_samples for row in host_only_rows],
            [float(row.elapsed_ms) for row in host_only_rows if row.elapsed_ms is not None],
            output_dir / "tensor_backend_scaling_host_only_runtime.png",
            x_label="Total tensor samples",
            y_label="CPU mean runtime (ms)",
        )
        if saved is not None:
            saved_paths.append(saved)

    return saved_paths


def _plot_spill_region(rows: list[BenchmarkRow], output_dir: Path) -> Path | None:
    spill_rows = [row for row in rows if row.region == "output_spill" and row.status == "ok"]
    cpu_rows = {row.num_records: row for row in spill_rows if row.backend == "cpu"}
    gpu_rows = {row.num_records: row for row in spill_rows if row.backend == "gpu"}
    common = sorted(set(cpu_rows) & set(gpu_rows))
    if len(common) <= 0:
        return None
    return plot_two_curve_comparison(
        np.asarray(common, dtype=np.float64),
        np.asarray([float(cpu_rows[count].elapsed_ms) for count in common], dtype=np.float64),
        np.asarray([float(gpu_rows[count].elapsed_ms) for count in common], dtype=np.float64),
        output_dir / "tensor_backend_scaling_output_spill_runtime.png",
        label_a="CPU mean runtime",
        label_b="GPU mean runtime",
        x_label="Recorded snapshots",
        y_label="Mean runtime (ms)",
    )


def _run(args: argparse.Namespace) -> float:
    runner = NloExampleRunner()
    api = runner.nlo.NLolib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []
    for scale in args.gpu_fit_scales:
        rows.append(_run_case(api, runner, region="gpu_fit", backend="cpu", scale=scale, num_records=1, warmup=args.warmup, runs=args.runs, sqlite_path=None))
        rows.append(_run_case(api, runner, region="gpu_fit", backend="gpu", scale=scale, num_records=1, warmup=args.warmup, runs=args.runs, sqlite_path=None))
    for scale in args.host_fit_scales:
        rows.append(_run_case(api, runner, region="host_fit_only", backend="cpu", scale=scale, num_records=1, warmup=args.warmup, runs=args.runs, sqlite_path=None))
    spill_scale = args.spill_scale if args.spill_scale > 0 else args.gpu_fit_scales[-1]
    for count in args.spill_records:
        rows.append(_run_case(api, runner, region="output_spill", backend="cpu", scale=spill_scale, num_records=count, warmup=args.warmup, runs=args.runs, sqlite_path=str(args.db_path)))
        rows.append(_run_case(api, runner, region="output_spill", backend="gpu", scale=spill_scale, num_records=count, warmup=args.warmup, runs=args.runs, sqlite_path=str(args.db_path)))

    csv_path = args.output_dir / "tensor_backend_scaling_results.csv"
    _write_csv(rows, csv_path)
    saved_paths = _plot_fit_region(rows, args.output_dir)
    spill_path = _plot_spill_region(rows, args.output_dir)
    if spill_path is not None:
        saved_paths.append(spill_path)

    print(f"saved csv: {csv_path}")
    for row in rows:
        print(
            f"{row.region:>12s} {row.backend:>3s} scale={row.scale:<4d} "
            f"records={row.num_records:<6d} status={row.status:<7s} elapsed_ms={row.elapsed_ms} {row.message}"
        )
    for path in saved_paths:
        print(f"saved plot: {path}")
    return 0.0


class TensorBackendScalingApp(ExampleAppBase):
    example_slug = "tensor_backend_scaling"
    description = "Plot CPU/GPU tensor-grid backend timings for three user-chosen scaling regions."

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--gpu-fit-scales",
            type=_parse_int_csv,
            default=[8, 16, 32],
            help="Comma-separated tensor scales that fit in GPU memory.",
        )
        parser.add_argument(
            "--host-fit-scales",
            type=_parse_int_csv,
            default=[48, 64],
            help="Comma-separated tensor scales that fit host memory but not GPU memory.",
        )
        parser.add_argument(
            "--spill-scale",
            type=int,
            default=0,
            help="Tensor scale used for output-spill runs. Defaults to the largest GPU-fit scale.",
        )
        parser.add_argument(
            "--spill-records",
            type=_parse_int_csv,
            default=[128, 256, 512],
            help="Comma-separated record counts used for output-spill timing runs.",
        )
        parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per benchmark point.")
        parser.add_argument("--runs", type=int, default=3, help="Measured runs per benchmark point.")

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return TensorBackendScalingApp.from_cli().run()


if __name__ == "__main__":
    main()
