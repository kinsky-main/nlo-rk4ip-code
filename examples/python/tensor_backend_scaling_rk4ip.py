"""
Equal-point nlolib CPU/GPU vs MMTools GPU runtime benchmark.

nlolib tensor cases use nt * nx * ny points. MMTools rows are treated as the
equal-point counterpart when they satisfy nt * mode_count with
mode_count = nx * ny.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit
from backend.app_base import ExampleAppBase
from backend.plotting import plt
from backend.runner import NloExampleRunner, centered_time_grid


@dataclass(frozen=True)
class TensorCase:
    nt: int
    nx: int
    ny: int

    @property
    def mode_count(self) -> int:
        return int(self.nx * self.ny)

    @property
    def total_points(self) -> int:
        return int(self.nt * self.nx * self.ny)


@dataclass(frozen=True)
class BenchmarkRow:
    solver: str
    backend: str
    nt: int
    nx: int
    ny: int
    mode_count: int
    total_points: int
    runtime_seconds: float | None
    runtime_seconds_std: float | None
    throughput_points_per_second: float | None
    status: str
    message: str


@dataclass(frozen=True)
class RuntimePlotSeries:
    solver: str
    backend: str
    marker: str = "o"
    linestyle: str = "-"
    fit_linestyle: str = "--"

    @property
    def label(self) -> str:
        return f"{self.solver} {self.backend}"


@dataclass(frozen=True)
class RuntimePlotSpec:
    series: tuple[RuntimePlotSeries, ...]
    save_path: str


_TIME_WINDOW = 2.56
_X_WINDOW = 4.80
_Y_WINDOW = 4.80
_DEFAULT_MMTOOLS_SUMMARY_CSV = Path(
    "examples/matlab/output/mmtools_tensor_scaling/mmtools_tensor_scaling_results.csv"
)
_MMTOOLS_REQUIRED_COLUMNS = (
    "solver",
    "backend",
    "nt",
    "nx",
    "ny",
    "mode_count",
    "total_points",
    "runtime_seconds",
    "status",
)
_MIXED_RUNTIME_PLOT_SPEC = RuntimePlotSpec(
    series=(
        RuntimePlotSeries("nlolib", "GPU"),
        RuntimePlotSeries("MMTools", "GPU"),
    ),
    save_path="tensor_backend_scaling_runtime.png",
)
_NLOLIB_RUNTIME_PLOT_SPEC = RuntimePlotSpec(
    series=(
        RuntimePlotSeries("nlolib", "CPU"),
        RuntimePlotSeries("nlolib", "GPU"),
    ),
    save_path="tensor_backend_scaling_runtime_nlolib_only.png",
)
_FIT_SAMPLE_COUNT = 256


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if len(values) <= 0:
        raise argparse.ArgumentTypeError("expected at least one integer.")
    return values


def _centered_spatial_grid(num_samples: int, delta: float) -> np.ndarray:
    return (
        np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)
    ) * float(delta)


def _gaussian_tensor_field(
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> np.ndarray:
    temporal_width = max(_TIME_WINDOW / 10.0, np.finfo(np.float64).eps)
    spatial_width = max(0.18 * min(_X_WINDOW, _Y_WINDOW), np.finfo(np.float64).eps)
    temporal = np.exp(-((t_axis / temporal_width) ** 2)) * np.exp(-1.0j * 2.8 * t_axis)
    xx, yy = np.meshgrid(x_axis, y_axis)
    radial = (xx * xx + yy * yy) / (spatial_width * spatial_width)
    angular = 0.75 + 0.18 * (xx / spatial_width) - 0.12j * (yy / spatial_width)
    transverse = np.exp(-radial) * angular
    return (temporal[:, None, None] * transverse[None, :, :]).astype(np.complex128)


def _flatten_tfast(field_tyx: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(field_tyx, dtype=np.complex128), (2, 1, 0)).reshape(
        -1
    )


def _case_from_scale(scale: int) -> TensorCase:
    resolved = int(scale)
    if resolved <= 0:
        raise ValueError("scale must be positive.")
    return TensorCase(nt=2 * resolved, nx=resolved, ny=resolved)


def _build_case(runner: NloExampleRunner, case: TensorCase):
    nlo = runner.nlo
    dt = _TIME_WINDOW / float(case.nt)
    dx = _X_WINDOW / float(case.nx)
    dy = _Y_WINDOW / float(case.ny)
    t_axis = centered_time_grid(case.nt, dt)
    x_axis = _centered_spatial_grid(case.nx, dx)
    y_axis = _centered_spatial_grid(case.ny, dy)
    omega = 2.0 * np.pi * np.fft.fftfreq(case.nt, d=dt)
    field0 = _flatten_tfast(_gaussian_tensor_field(t_axis, x_axis, y_axis))
    runtime = nlo.RuntimeOperators(
        linear_factor_expr="i*(c0*wt*wt + c1*(kx*kx + ky*ky))",
        linear_expr="exp(h*D)",
        nonlinear_expr="i*A*(c2*I)",
        constants=[0.04, -0.20, 0.015],
    )
    config = nlo.prepare_sim_config(
        case.total_points,
        propagation_distance=0.20,
        starting_step_size=0.0001,
        max_step_size=0.02,
        min_step_size=0.000001,
        error_tolerance=1.0e-9,
        pulse_period=float(case.nt) * dt,
        delta_time=dt,
        tensor_nt=case.nt,
        tensor_nx=case.nx,
        tensor_ny=case.ny,
        tensor_layout=nlo.TENSOR_LAYOUT_XYT_T_FAST,
        frequency_grid=[complex(value, 0.0) for value in omega],
        delta_x=dx,
        delta_y=dy,
        runtime=runtime,
    )
    return config, field0


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) <= 0:
        return None, None
    data = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    return mean, std


def _throughput(total_points: int, runtime_seconds: float | None) -> float | None:
    if runtime_seconds is None or runtime_seconds <= 0.0:
        return None
    return float(total_points / runtime_seconds)


def _benchmark_nlolib_case(
    api,
    runner: NloExampleRunner,
    *,
    case: TensorCase,
    num_records: int,
    warmup: int,
    runs: int,
    backend_type: int,
    backend_label: str,
) -> BenchmarkRow:
    nlo = runner.nlo
    config, field0 = _build_case(runner, case)
    exec_options = nlo.default_execution_options(backend_type)
    samples: list[float] = []

    try:
        for run_idx in range(int(warmup) + int(runs)):
            start = time.perf_counter()
            api.propagate(config, field0, int(num_records), exec_options=exec_options)
            runtime_seconds = time.perf_counter() - start
            if run_idx < int(warmup):
                continue
            samples.append(float(runtime_seconds))
    except RuntimeError as exc:
        return BenchmarkRow(
            solver="nlolib",
            backend=backend_label,
            nt=case.nt,
            nx=case.nx,
            ny=case.ny,
            mode_count=case.mode_count,
            total_points=case.total_points,
            runtime_seconds=None,
            runtime_seconds_std=None,
            throughput_points_per_second=None,
            status="error",
            message=str(exc),
        )

    mean_seconds, std_seconds = _mean_std(samples)
    return BenchmarkRow(
        solver="nlolib",
        backend=backend_label,
        nt=case.nt,
        nx=case.nx,
        ny=case.ny,
        mode_count=case.mode_count,
        total_points=case.total_points,
        runtime_seconds=mean_seconds,
        runtime_seconds_std=std_seconds,
        throughput_points_per_second=_throughput(case.total_points, mean_seconds),
        status="ok",
        message="",
    )


def _write_csv(rows: list[BenchmarkRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = (
        list(asdict(rows[0]).keys())
        if rows
        else list(BenchmarkRow.__dataclass_fields__.keys())
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _csv_float(row: dict[str, str], name: str) -> float | None:
    raw_value = row.get(name, "")
    raw = "" if raw_value is None else raw_value.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _csv_int(row: dict[str, str], name: str) -> int | None:
    value = _csv_float(row, name)
    return None if value is None else int(round(value))


def _require_csv_int(row: dict[str, str], name: str, *, positive: bool = False) -> int:
    value = _csv_int(row, name)
    if value is None or (positive and value <= 0):
        raise ValueError(f"MMTools CSV has invalid '{name}' value: {row.get(name)!r}")
    return int(value)


def _require_csv_float(
    row: dict[str, str], name: str, *, positive: bool = False
) -> float:
    value = _csv_float(row, name)
    if value is None or (positive and value <= 0.0):
        raise ValueError(f"MMTools CSV has invalid '{name}' value: {row.get(name)!r}")
    return float(value)


def _read_mmtools_rows(summary_csv: Path | None) -> list[BenchmarkRow]:
    if summary_csv is None or not summary_csv.is_file():
        return []

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in _MMTOOLS_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                "MMTools CSV is missing required columns " f"({', '.join(missing)})."
            )

        rows: list[BenchmarkRow] = []
        for row in reader:
            solver = row.get("solver", "").strip().lower()
            if solver != "mmtools":
                continue
            status = row.get("status", "ok").strip().lower()
            if status not in {"", "ok"}:
                rows.append(
                    BenchmarkRow(
                        solver="MMTools",
                        backend=row.get("backend", "GPU").strip() or "GPU",
                        nt=_require_csv_int(row, "nt", positive=True),
                        nx=_require_csv_int(row, "nx", positive=True),
                        ny=_require_csv_int(row, "ny", positive=True),
                        mode_count=_require_csv_int(row, "mode_count", positive=True),
                        total_points=_require_csv_int(
                            row, "total_points", positive=True
                        ),
                        runtime_seconds=_csv_float(row, "runtime_seconds"),
                        runtime_seconds_std=_csv_float(row, "runtime_seconds_std"),
                        throughput_points_per_second=_csv_float(
                            row, "throughput_points_per_second"
                        ),
                        status="error",
                        message=row.get("message", ""),
                    )
                )
                continue

            nt = _require_csv_int(row, "nt", positive=True)
            nx = _require_csv_int(row, "nx", positive=True)
            ny = _require_csv_int(row, "ny", positive=True)
            mode_count = _require_csv_int(row, "mode_count", positive=True)
            total_points = _require_csv_int(row, "total_points", positive=True)

            runtime_seconds = _require_csv_float(row, "runtime_seconds", positive=True)
            runtime_seconds_std = _csv_float(row, "runtime_seconds_std")
            throughput = _csv_float(row, "throughput_points_per_second")
            if throughput is None:
                throughput = _throughput(total_points, runtime_seconds)
            rows.append(
                BenchmarkRow(
                    solver="MMTools",
                    backend=row.get("backend", "GPU").strip() or "GPU",
                    nt=nt,
                    nx=nx,
                    ny=ny,
                    mode_count=mode_count,
                    total_points=total_points,
                    runtime_seconds=runtime_seconds,
                    runtime_seconds_std=runtime_seconds_std,
                    throughput_points_per_second=throughput,
                    status="ok",
                    message=row.get("message", ""),
                )
            )

    return sorted(rows, key=lambda item: item.total_points)


def _resolve_cases(
    args: argparse.Namespace,
    mmtools_rows: list[BenchmarkRow],
    mmtools_summary_csv: Path | None,
) -> list[TensorCase]:
    if len(mmtools_rows) > 0:
        print(
            f"using {len(mmtools_rows)} equal-point MMTools rows from: {mmtools_summary_csv}"
        )
        return [
            TensorCase(nt=row.nt, nx=row.nx, ny=row.ny)
            for row in mmtools_rows
            if row.status == "ok"
        ]
    return [_case_from_scale(scale) for scale in args.scales]


def _runtime_plot_rows(rows: Sequence[BenchmarkRow]) -> list[BenchmarkRow]:
    return sorted(
        [
            row
            for row in rows
            if row.status == "ok" and row.runtime_seconds is not None
        ],
        key=lambda row: (row.total_points, row.solver.lower(), row.backend.upper()),
    )


def _series_rows(
    rows: Sequence[BenchmarkRow], series: RuntimePlotSeries
) -> list[BenchmarkRow]:
    solver_key = series.solver.lower()
    backend_key = series.backend.upper()
    return [
        row
        for row in _runtime_plot_rows(rows)
        if row.solver.lower() == solver_key and row.backend.upper() == backend_key
    ]


def _series_xy(rows: Sequence[BenchmarkRow]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row.total_points for row in rows], dtype=np.float64),
        np.asarray([float(row.runtime_seconds) for row in rows], dtype=np.float64),
    )


def _growth_order_label(order: float) -> str:
    return r"$O(N^{" + f"{order:.2f}" + r"})$"


def _fit_runtime_series(
    rows: Sequence[BenchmarkRow],
) -> tuple[np.ndarray, np.ndarray, str] | None:
    x_values, y_values = _series_xy(rows)
    if (
        x_values.size < 2
        or np.any(x_values <= 0.0)
        or np.any(y_values <= 0.0)
    ):
        return None

    popt, pcov = curve_fit(lambda x, a, b, c: a * np.power(x, b) + c, x_values, y_values)
    order, intercept = float(popt[1]), float(popt[0])
    x_fit = np.linspace(
        float(np.min(x_values)),
        float(np.max(x_values)),
        _FIT_SAMPLE_COUNT,
        dtype=np.float64,
    )
    y_fit = intercept * np.power(x_fit, order) + float(popt[2])
    return x_fit, y_fit, _growth_order_label(float(order))


def _plot_runtime_series(ax, rows: Sequence[BenchmarkRow], series: RuntimePlotSeries) -> bool:
    solver_rows = _series_rows(rows, series)
    if len(solver_rows) <= 0:
        return False

    x_values, y_values = _series_xy(solver_rows)
    (line,) = ax.plot(
        x_values,
        y_values,
        marker=series.marker,
        linestyle="none",
    )
    fit = _fit_runtime_series(solver_rows)
    if fit is not None:
        x_fit, y_fit, growth_order = fit
        ax.plot(
            x_fit,
            y_fit,
            linestyle=series.fit_linestyle,
            linewidth=1.4,
            color=line.get_color(),
            label=f"{series.label} {growth_order}",
        )
    return True


def _plot_runtime(
    rows: list[BenchmarkRow],
    output_dir: Path,
    *,
    plot_spec: RuntimePlotSpec = _MIXED_RUNTIME_PLOT_SPEC,
) -> Path | None:
    if len(_runtime_plot_rows(rows)) <= 0:
        return None

    fig, ax = plt.subplots()
    any_series = False
    for series in plot_spec.series:
        any_series = _plot_runtime_series(ax, rows, series) or any_series

    if not any_series:
        plt.close(fig)
        return None

    ax.set_xlabel("State vector size")
    ax.set_ylabel("Runtime (s)")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / plot_spec.save_path
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _benchmark_nlolib_rows(
    args: argparse.Namespace,
    runner: NloExampleRunner,
    api,
    cases: Sequence[TensorCase],
    *,
    series: Sequence[RuntimePlotSeries] = _NLOLIB_RUNTIME_PLOT_SPEC.series,
) -> list[BenchmarkRow]:
    backend_type_by_label = {
        "CPU": runner.nlo.VECTOR_BACKEND_CPU,
        "GPU": runner.nlo.VECTOR_BACKEND_VULKAN,
    }
    rows: list[BenchmarkRow] = []
    for plot_series in series:
        backend_type = backend_type_by_label.get(plot_series.backend.upper())
        if backend_type is None or plot_series.solver.lower() != "nlolib":
            continue
        rows.extend(
            _benchmark_nlolib_case(
                api,
                runner,
                case=case,
                num_records=args.num_records,
                warmup=args.warmup,
                runs=args.runs,
                backend_type=backend_type,
                backend_label=plot_series.backend.upper(),
            )
            for case in cases
        )
    return rows


def _print_summary(rows: list[BenchmarkRow]) -> None:
    runtime_rows = [row for row in rows if row.runtime_seconds is not None]
    keyed_rows = {
        (row.solver.lower(), row.backend.upper(), row.total_points): row
        for row in runtime_rows
    }

    for row in sorted(
        rows,
        key=lambda item: (item.total_points, item.solver.lower(), item.backend.upper()),
    ):
        print(
            f"{row.solver:>7s} {row.backend:>3s} total_points={row.total_points:<8d} nt={row.nt:<4d} "
            f"nx={row.nx:<4d} ny={row.ny:<4d} mode_count={row.mode_count:<6d} "
            f"runtime_s={row.runtime_seconds} throughput={row.throughput_points_per_second} "
            f"status={row.status} {row.message}"
        )

    total_points_values = sorted({row.total_points for row in rows})
    for total_points in total_points_values:
        cpu_row = keyed_rows.get(("nlolib", "CPU", total_points))
        gpu_row = keyed_rows.get(("nlolib", "GPU", total_points))
        mmtools_row = keyed_rows.get(("mmtools", "GPU", total_points))

        if (
            cpu_row is not None
            and gpu_row is not None
            and cpu_row.runtime_seconds is not None
            and gpu_row.runtime_seconds is not None
            and gpu_row.runtime_seconds > 0.0
        ):
            cpu_vs_gpu = cpu_row.runtime_seconds / gpu_row.runtime_seconds
            print(
                f"compare total_points={total_points:<8d} "
                f"nlolib_cpu_runtime_s={cpu_row.runtime_seconds:.6f} "
                f"nlolib_gpu_runtime_s={gpu_row.runtime_seconds:.6f} "
                f"cpu_vs_gpu_runtime={cpu_vs_gpu:.3f}x"
            )

        if (
            gpu_row is None
            or mmtools_row is None
            or gpu_row.runtime_seconds is None
            or mmtools_row.runtime_seconds is None
            or mmtools_row.runtime_seconds <= 0.0
        ):
            continue
        runtime_ratio = gpu_row.runtime_seconds / mmtools_row.runtime_seconds
        print(
            f"compare total_points={total_points:<8d} "
            f"nlolib_gpu_runtime_s={gpu_row.runtime_seconds:.6f} "
            f"mmtools_runtime_s={mmtools_row.runtime_seconds:.6f} "
            f"nlolib_gpu_vs_mmtools_runtime={runtime_ratio:.3f}x"
        )


def _run(args: argparse.Namespace) -> float:
    runner = NloExampleRunner()
    api = runner.nlo.NLolib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mmtools_summary_csv = args.mmtools_summary_csv
    if mmtools_summary_csv is None and _DEFAULT_MMTOOLS_SUMMARY_CSV.is_file():
        mmtools_summary_csv = _DEFAULT_MMTOOLS_SUMMARY_CSV

    mmtools_rows = _read_mmtools_rows(mmtools_summary_csv)
    cases = _resolve_cases(args, mmtools_rows, mmtools_summary_csv)
    nlolib_rows = _benchmark_nlolib_rows(args, runner, api, cases)
    rows = nlolib_rows + mmtools_rows

    csv_path = args.output_dir / "tensor_backend_scaling_results.csv"
    _write_csv(rows, csv_path)
    plot_path = _plot_runtime(rows, args.output_dir, plot_spec=_MIXED_RUNTIME_PLOT_SPEC)
    plot_path_2 = _plot_runtime(
        rows,
        args.output_dir,
        plot_spec=_NLOLIB_RUNTIME_PLOT_SPEC,
    )

    print(f"saved csv: {csv_path}")
    _print_summary(rows)
    if plot_path is not None:
        print(f"saved plot: {plot_path}")
    if plot_path_2 is not None:
        print(f"saved plot: {plot_path_2}")
    return 0.0


class TensorBackendScalingApp(ExampleAppBase):
    example_slug = "tensor_backend_scaling"
    description = "Equal-point nlolib CPU/GPU and MMTools GPU runtime benchmark."

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--scales",
            type=_parse_int_csv,
            default=[3, 4, 5],
            help="Comma-separated default tensor scales with nt=2*scale and nx=ny=scale.",
        )
        parser.add_argument(
            "--num-records",
            type=int,
            default=1,
            help="Recorded snapshots per run. Use 1 to benchmark final-only output.",
        )
        parser.add_argument(
            "--warmup", type=int, default=0, help="Warmup runs per benchmark point."
        )
        parser.add_argument(
            "--runs", type=int, default=1, help="Measured runs per benchmark point."
        )
        parser.add_argument(
            "--mmtools-summary-csv",
            type=Path,
            default=None,
            help="Optional MMTools equal-point CSV used for comparison.",
        )

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return TensorBackendScalingApp.from_cli().run()


if __name__ == "__main__":
    main()
