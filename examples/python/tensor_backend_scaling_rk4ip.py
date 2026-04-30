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
from backend.app_base import ExampleAppBase
from backend.plotting import plt
from backend.runner import NloExampleRunner, centered_time_grid
from backend.storage import ExampleRunDB


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
    fit_skip_initial_points: int = 0


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
    fit_skip_initial_points=1,
)
_NLOLIB_RUNTIME_PLOT_SPEC = RuntimePlotSpec(
    series=(
        RuntimePlotSeries("nlolib", "CPU"),
        RuntimePlotSeries("nlolib", "GPU"),
    ),
    save_path="tensor_backend_scaling_runtime_nlolib_only.png",
    fit_skip_initial_points=1,
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


def _equivalent_total_points(nt: int, nx: int, ny: int) -> int:
    return int(nt * nx * ny)


def _case_key(row: BenchmarkRow) -> tuple[int, int, int]:
    return int(row.nt), int(row.nx), int(row.ny)


def _row_storage_case_key(row: BenchmarkRow) -> str:
    solver = "".join(ch if ch.isalnum() else "_" for ch in row.solver.lower()).strip("_")
    backend = "".join(ch if ch.isalnum() else "_" for ch in row.backend.upper()).strip("_")
    return f"{solver}_{backend}_nt{int(row.nt)}_nx{int(row.nx)}_ny{int(row.ny)}"


def _metadata_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _metadata_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"stored benchmark row has invalid integer value: {value!r}") from exc


def _row_from_metadata(meta: dict[str, object], *, case_key: str) -> BenchmarkRow:
    payload = meta.get("row")
    if not isinstance(payload, dict):
        raise RuntimeError(f"stored benchmark case '{case_key}' is missing row metadata.")

    required = BenchmarkRow.__dataclass_fields__.keys()
    missing = [name for name in required if name not in payload]
    if missing:
        raise RuntimeError(
            f"stored benchmark case '{case_key}' is missing fields: {', '.join(missing)}"
        )

    return BenchmarkRow(
        solver=str(payload["solver"]),
        backend=str(payload["backend"]),
        nt=_metadata_int(payload["nt"]),
        nx=_metadata_int(payload["nx"]),
        ny=_metadata_int(payload["ny"]),
        mode_count=_metadata_int(payload["mode_count"]),
        total_points=_metadata_int(payload["total_points"]),
        runtime_seconds=_metadata_float(payload["runtime_seconds"]),
        runtime_seconds_std=_metadata_float(payload["runtime_seconds_std"]),
        throughput_points_per_second=_metadata_float(
            payload["throughput_points_per_second"]
        ),
        status=str(payload["status"]),
        message=str(payload["message"]),
    )


def _save_benchmark_rows(
    db: ExampleRunDB,
    *,
    example_name: str,
    run_group: str,
    rows: Sequence[BenchmarkRow],
) -> None:
    db.begin_group(example_name, run_group)
    for row in rows:
        case_key = _row_storage_case_key(row)
        db.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            run_id=db.make_run_id(example_name, run_group, case_key),
            meta={
                "kind": "tensor_backend_scaling_benchmark_row",
                "schema": 1,
                "row": asdict(row),
            },
        )


def _load_benchmark_rows(
    db: ExampleRunDB,
    *,
    example_name: str,
    run_group: str,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for case in db.list_cases(example_name=example_name, run_group=run_group):
        if case.meta.get("kind") != "tensor_backend_scaling_benchmark_row":
            continue
        rows.append(_row_from_metadata(case.meta, case_key=case.case_key))

    if not rows:
        raise RuntimeError(
            f"run_group '{run_group}' has no stored tensor backend scaling rows."
        )

    return sorted(
        rows,
        key=lambda row: (row.total_points, row.solver.lower(), row.backend.upper()),
    )


def _read_benchmark_rows_csv(csv_path: Path) -> list[BenchmarkRow]:
    if not csv_path.is_file():
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = set(BenchmarkRow.__dataclass_fields__.keys())
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"benchmark CSV is missing required columns ({', '.join(missing)})."
            )

        rows = [
            BenchmarkRow(
                solver=str(row["solver"]),
                backend=str(row["backend"]),
                nt=_metadata_int(row["nt"]),
                nx=_metadata_int(row["nx"]),
                ny=_metadata_int(row["ny"]),
                mode_count=_metadata_int(row["mode_count"]),
                total_points=_metadata_int(row["total_points"]),
                runtime_seconds=_metadata_float(row["runtime_seconds"]),
                runtime_seconds_std=_metadata_float(row["runtime_seconds_std"]),
                throughput_points_per_second=_metadata_float(
                    row["throughput_points_per_second"]
                ),
                status=str(row["status"]),
                message=str(row["message"]),
            )
            for row in reader
        ]

    return sorted(
        rows,
        key=lambda row: (row.total_points, row.solver.lower(), row.backend.upper()),
    )


def _load_replot_benchmark_rows(
    db: ExampleRunDB,
    *,
    example_name: str,
    run_group: str | None,
    fallback_csv: Path | None,
) -> list[BenchmarkRow]:
    try:
        resolved_run_group = db.resolve_replot_group(example_name, run_group)
    except RuntimeError:
        if run_group is not None or fallback_csv is None:
            raise
        rows = _read_benchmark_rows_csv(fallback_csv)
        if rows:
            return rows
        raise

    return _load_benchmark_rows(
        db,
        example_name=example_name,
        run_group=resolved_run_group,
    )


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
            _require_csv_int(row, "total_points", positive=True)
            total_points = _equivalent_total_points(nt, nx, ny)

            runtime_seconds = _csv_float(row, "wall_seconds")
            runtime_seconds_std = _csv_float(row, "wall_seconds_std")
            if runtime_seconds is None:
                runtime_seconds = _require_csv_float(
                    row, "runtime_seconds", positive=True
                )
                runtime_seconds_std = _csv_float(row, "runtime_seconds_std")
            elif runtime_seconds <= 0.0:
                raise ValueError(
                    f"MMTools CSV has invalid 'wall_seconds' value: {row.get('wall_seconds')!r}"
                )
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
        [row for row in rows if row.status == "ok" and row.runtime_seconds is not None],
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


def _series_yerr(rows: Sequence[BenchmarkRow]) -> np.ndarray:
    return np.asarray(
        [
            (
                0.0
                if row.runtime_seconds_std is None
                or not np.isfinite(row.runtime_seconds_std)
                else max(float(row.runtime_seconds_std), 0.0)
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def _growth_order_label(order: float) -> str:
    return r"$O(N^{" + f"{round(order, 1)}" + r"})$"


def _fit_loglog_slope(
    x_values: np.ndarray,
    y_values: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    valid_mask = (
        np.asarray(fit_mask, dtype=bool)
        & np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 0.0)
        & (y_values > 0.0)
    )
    if int(np.count_nonzero(valid_mask)) < 2:
        return None

    log_x = np.asarray(np.log(x_values[valid_mask]), dtype=np.float64)
    log_y = np.asarray(np.log(y_values[valid_mask]), dtype=np.float64)
    order, log_scale = np.polyfit(log_x, log_y, deg=1)
    return float(order), float(log_scale), valid_mask


def _fit_runtime_series(
    rows: Sequence[BenchmarkRow],
    *,
    skip_initial_points: int = 0,
) -> tuple[np.ndarray, np.ndarray, str, float] | None:
    x_values, y_values = _series_xy(rows)
    if x_values.size <= int(skip_initial_points):
        return None

    fit_mask = np.ones(x_values.shape, dtype=bool)
    fit_mask[: max(0, int(skip_initial_points))] = False
    fitted = _fit_loglog_slope(x_values, y_values, fit_mask)
    if fitted is None:
        return None
    order, log_scale, valid_mask = fitted

    fit_x_values = x_values[valid_mask]
    x_fit = np.linspace(
        float(np.min(fit_x_values)),
        float(np.max(fit_x_values)),
        _FIT_SAMPLE_COUNT,
        dtype=np.float64,
    )
    y_fit = np.exp(log_scale) * np.power(x_fit, order)
    return x_fit, y_fit, _growth_order_label(float(order)), 0.0


def _plot_runtime_series(
    ax,
    rows: Sequence[BenchmarkRow],
    series: RuntimePlotSeries,
    *,
    fit_skip_initial_points: int = 0,
) -> tuple[bool, tuple[np.ndarray | None, np.ndarray | None, float | None]] | None:
    solver_rows = _series_rows(rows, series)
    if len(solver_rows) <= 0:
        return False, (None, None, None)

    x_values, y_values = _series_xy(solver_rows)
    y_error = _series_yerr(solver_rows)
    container = ax.errorbar(
        x_values / 1e3,
        y_values,
        yerr=y_error,
        marker=series.marker,
        linestyle="none",
        capsize=3.0,
        elinewidth=1.0,
    )
    line = container.lines[0]
    fit = _fit_runtime_series(
        solver_rows,
        skip_initial_points=fit_skip_initial_points,
    )
    if fit is not None:
        x_fit, y_fit, growth_order, error = fit
        ax.plot(
            x_fit / 1e3,
            y_fit,
            linestyle=series.fit_linestyle,
            linewidth=1.4,
            color=line.get_color(),
            label=f"{series.label} {growth_order}",
        )
    else:
        x_fit, y_fit, growth_order, error = None, None, None, None
    return True, (x_fit, y_fit, error)


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
    fits = []
    for series in plot_spec.series:
        series_plotted, fit = _plot_runtime_series(
            ax,
            rows,
            series,
            fit_skip_initial_points=plot_spec.fit_skip_initial_points,
        ) or (False, None)
        any_series = any_series or series_plotted
        fits.append(fit)

    if not any_series:
        plt.close(fig)
        return None

    ax.set_xlabel(r"State vector size ($10^3$ points)")
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / plot_spec.save_path
    fig.savefig(output_path)
    plt.close(fig)
    
    if len(fits) != 0 and all(fit is not None for fit in fits):
        # Print Intercepts of all pairs of fitted curves
        for num, fit_a in enumerate(fits):
            series = plot_spec.series[num]
            
            if len(fits) <= num + 1:
                break
            
            for other_idx, fit_b in enumerate(fits[num + 1:], start=num + 1):
                if fit_a is None or fit_b is None:
                    continue
                series_b = plot_spec.series[other_idx]
                x_fit_a, y_fit_a, error_a = fit_a
                x_fit_b, y_fit_b, error_b = fit_b
                if (
                    x_fit_a is None
                    or y_fit_a is None
                    or x_fit_b is None
                    or y_fit_b is None
                ):
                    continue

                overlap_min = max(float(np.min(x_fit_a)), float(np.min(x_fit_b)))
                overlap_max = min(float(np.max(x_fit_a)), float(np.max(x_fit_b)))
                if overlap_max <= overlap_min:
                    print(
                        f"Could not compute intercept of {series.label} curve and {series_b.label} curve."
                    )
                    continue

                sample_count = max(_FIT_SAMPLE_COUNT * 4, 2)
                x_common = np.linspace(
                    overlap_min,
                    overlap_max,
                    sample_count,
                    dtype=np.float64,
                )
                y_common_a = np.interp(x_common, x_fit_a, y_fit_a)
                y_common_b = np.interp(x_common, x_fit_b, y_fit_b)
                delta = y_common_a - y_common_b
                sign_changes = np.where(np.signbit(delta[:-1]) != np.signbit(delta[1:]))[0]

                x_intercept = None
                if sign_changes.size > 0:
                    idx = int(sign_changes[0])
                    x0 = float(x_common[idx])
                    x1 = float(x_common[idx + 1])
                    d0 = float(delta[idx])
                    d1 = float(delta[idx + 1])
                    if d1 != d0:
                        x_intercept = x0 - d0 * (x1 - x0) / (d1 - d0)
                    else:
                        x_intercept = x0
                else:
                    idx = int(np.argmin(np.abs(delta)))
                    if np.isclose(delta[idx], 0.0, atol=1.0e-9, rtol=0.0):
                        x_intercept = float(x_common[idx])

                if x_intercept is not None:
                    err_a = 0.0 if error_a is None else float(error_a)
                    err_b = 0.0 if error_b is None else float(error_b)
                    print(
                        f"Intercept of {series.label} curve and {series_b.label} curve: {x_intercept:.3f}"
                        + r" ± "
                        + f"{np.sqrt(err_a**2 + err_b**2):.3f}"
                    )
                else:
                    print(
                        f"Could not compute intercept of {series.label} curve and {series_b.label} curve."
                    )
        return output_path
    else:
        print("Could not compute fitted curves for intercept analysis.")
        return output_path


def _benchmark_nlolib_rows(
    args: argparse.Namespace,
    runner: NloExampleRunner,
    api,
    cases: Sequence[TensorCase],
    *,
    series: Sequence[RuntimePlotSeries] = _NLOLIB_RUNTIME_PLOT_SPEC.series,
    run_cpu: bool = True,
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
        if backend_type == runner.nlo.VECTOR_BACKEND_CPU and not run_cpu:
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
        (row.solver.lower(), row.backend.upper(), _case_key(row)): row
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

    case_keys = sorted(
        {_case_key(row) for row in rows},
        key=lambda item: _equivalent_total_points(*item),
    )
    for case_key in case_keys:
        cpu_row = keyed_rows.get(("nlolib", "CPU", case_key))
        gpu_row = keyed_rows.get(("nlolib", "GPU", case_key))
        mmtools_row = keyed_rows.get(("mmtools", "GPU", case_key))
        total_points = None
        for candidate in (gpu_row, cpu_row, mmtools_row):
            if candidate is not None:
                total_points = candidate.total_points
                break
        if total_points is None:
            total_points = _equivalent_total_points(*case_key)

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
    db = ExampleRunDB(args.db_path)
    example_name = "tensor_backend_scaling"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tensor_backend_scaling_results.csv"

    if args.replot:
        rows = _load_replot_benchmark_rows(
            db,
            example_name=example_name,
            run_group=args.run_group,
            fallback_csv=csv_path,
        )
    else:
        run_group = db.begin_group(example_name, args.run_group)
        runner = NloExampleRunner()
        api = runner.nlo.NLolib()

        mmtools_summary_csv = args.mmtools_summary_csv
        if mmtools_summary_csv is None and _DEFAULT_MMTOOLS_SUMMARY_CSV.is_file():
            mmtools_summary_csv = _DEFAULT_MMTOOLS_SUMMARY_CSV

        mmtools_rows = _read_mmtools_rows(mmtools_summary_csv)
        cases = _resolve_cases(args, mmtools_rows, mmtools_summary_csv)
        nlolib_rows = _benchmark_nlolib_rows(args, runner, api, cases, run_cpu=False)
        rows = nlolib_rows + mmtools_rows
        _save_benchmark_rows(
            db,
            example_name=example_name,
            run_group=run_group,
            rows=rows,
        )

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
