from __future__ import annotations

import importlib.util
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    module,
    solver: str,
    backend: str,
    total_points: int,
    runtime: float | None,
    *,
    runtime_std: float | None = None,
    status: str = "ok",
):
    scale = int(round((total_points / 2.0) ** (1.0 / 3.0)))
    return module.BenchmarkRow(
        solver=solver,
        backend=backend,
        nt=2 * scale,
        nx=scale,
        ny=scale,
        mode_count=scale * scale,
        total_points=total_points,
        runtime_seconds=runtime,
        runtime_seconds_std=runtime_std if runtime is not None else None,
        throughput_points_per_second=None,
        status=status,
        message="",
    )


def test_runtime_plot_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples" / "python"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    benchmark = _load_module(
        "tensor_backend_scaling_rk4ip",
        examples_dir / "tensor_backend_scaling_rk4ip.py",
    )
    nlolib_benchmark = _load_module(
        "tensor_backend_scaling_nlolib_rk4ip",
        examples_dir / "tensor_backend_scaling_nlolib_rk4ip.py",
    )

    rows = [
        _row(benchmark, "nlolib", "GPU", 512, 0.60, runtime_std=0.06),
        _row(benchmark, "nlolib", "GPU", 128, 0.30, runtime_std=0.03),
        _row(benchmark, "nlolib", "GPU", 256, None, status="error"),
        _row(benchmark, "MMTools", "GPU", 128, 0.25, runtime_std=0.02),
        _row(benchmark, "MMTools", "GPU", 512, 0.45, runtime_std=0.04),
        _row(benchmark, "MMTools", "GPU", 256, 0.31, runtime_std=0.01),
        _row(benchmark, "nlolib", "CPU", 128, 0.80, runtime_std=0.08),
        _row(benchmark, "nlolib", "CPU", 256, 1.10, runtime_std=0.10),
    ]

    gpu_rows = benchmark._series_rows(
        rows,
        benchmark.RuntimePlotSeries("nlolib", "GPU"),
    )
    assert [row.total_points for row in gpu_rows] == [128, 512]
    assert all(row.status == "ok" for row in gpu_rows)
    assert np.allclose(benchmark._series_yerr(gpu_rows), np.asarray([0.03, 0.06]))
    assert benchmark._equivalent_total_points(4, 8, 8) == 256
    assert benchmark._case_key(gpu_rows[0]) == (8, 4, 4)

    assert benchmark._fit_runtime_series(gpu_rows[:1]) is None
    fit = benchmark._fit_runtime_series(
        [
            _row(benchmark, "MMTools", "GPU", 64, 0.10),
            _row(benchmark, "MMTools", "GPU", 128, 0.16),
            _row(benchmark, "MMTools", "GPU", 256, 0.29),
            _row(benchmark, "MMTools", "GPU", 512, 0.55),
        ]
    )
    assert fit is not None
    x_fit, y_fit, growth_order, _ = fit
    assert x_fit.shape == y_fit.shape
    assert x_fit.size == benchmark._FIT_SAMPLE_COUNT
    assert np.all(np.isfinite(y_fit))
    assert growth_order.startswith("$O(N^{")
    assert benchmark._growth_order_label(1.25) == "$O(N^{1.2})$"
    skipped_fit = benchmark._fit_runtime_series(
        [
            _row(benchmark, "MMTools", "GPU", 64, 9.99),
            _row(benchmark, "MMTools", "GPU", 128, 0.16),
            _row(benchmark, "MMTools", "GPU", 256, 0.32),
            _row(benchmark, "MMTools", "GPU", 512, 0.64),
        ],
        skip_initial_points=1,
    )
    assert skipped_fit is not None
    skipped_x_fit, skipped_y_fit, skipped_order, _ = skipped_fit
    assert np.isclose(float(skipped_x_fit[0]), 128.0)
    assert np.isclose(float(skipped_x_fit[-1]), 512.0)
    assert skipped_order == "$O(N^{1.0})$"
    assert np.isclose(float(skipped_y_fit[0]), 0.16)
    assert np.isclose(float(skipped_y_fit[-1]), 0.64)

    mixed_labels = [series.label for series in benchmark._MIXED_RUNTIME_PLOT_SPEC.series]
    nlolib_labels = [series.label for series in benchmark._NLOLIB_RUNTIME_PLOT_SPEC.series]
    assert mixed_labels == ["nlolib GPU", "MMTools GPU"]
    assert nlolib_labels == ["nlolib CPU", "nlolib GPU"]
    assert benchmark._MIXED_RUNTIME_PLOT_SPEC.fit_skip_initial_points == 1
    assert benchmark._NLOLIB_RUNTIME_PLOT_SPEC.fit_skip_initial_points == 1

    fig, ax = benchmark.plt.subplots()
    assert benchmark._plot_runtime_series(ax, rows, benchmark._MIXED_RUNTIME_PLOT_SPEC.series[0])[0] is True
    assert benchmark._plot_runtime_series(ax, rows, benchmark._MIXED_RUNTIME_PLOT_SPEC.series[1])[0] is True
    _, legend_labels = ax.get_legend_handles_labels()
    benchmark.plt.close(fig)
    assert "nlolib GPU" in legend_labels
    assert "MMTools GPU" in legend_labels
    growth_labels = [
        label for label in legend_labels if label not in {"nlolib GPU", "MMTools GPU"}
    ]
    assert len(growth_labels) == 2
    assert any(label.startswith("nlolib GPU $O(N^{") for label in growth_labels)
    assert any(label.startswith("MMTools GPU $O(N^{") for label in growth_labels)

    temp_dir = repo_root / "build" / "test-artifacts" / "tensor-backend-scaling-cleanup"
    temp_dir.mkdir(parents=True, exist_ok=True)

    mixed_path = benchmark._plot_runtime(
        rows,
        temp_dir,
        plot_spec=benchmark._MIXED_RUNTIME_PLOT_SPEC,
    )
    nlolib_path = benchmark._plot_runtime(
        rows,
        temp_dir,
        plot_spec=benchmark._NLOLIB_RUNTIME_PLOT_SPEC,
    )
    gpu_only_path = benchmark._plot_runtime(
        [_row(benchmark, "nlolib", "GPU", 128, 0.30)],
        temp_dir,
        plot_spec=benchmark._MIXED_RUNTIME_PLOT_SPEC,
    )

    assert mixed_path is not None
    assert mixed_path.name == "tensor_backend_scaling_runtime.png"
    assert mixed_path.is_file()
    assert nlolib_path is not None
    assert nlolib_path.name == "tensor_backend_scaling_runtime_nlolib_only.png"
    assert nlolib_path.is_file()
    assert gpu_only_path is not None
    assert gpu_only_path.name == "tensor_backend_scaling_runtime.png"

    with tempfile.TemporaryDirectory() as temp_dir_name:
        csv_path = Path(temp_dir_name) / "mmtools.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "solver,backend,nt,nx,ny,mode_count,total_points,runtime_seconds,runtime_seconds_std,throughput_points_per_second,setup_seconds,setup_seconds_std,wall_seconds,wall_seconds_std,status,message",
                    "MMTools,GPU,4096,8,8,8,32768,0.5,0.1,,1.5,0.2,2.0,0.3,ok,",
                ]
            ),
            encoding="utf-8",
        )
        mmtools_rows = benchmark._read_mmtools_rows(csv_path)
        assert len(mmtools_rows) == 1
        assert mmtools_rows[0].total_points == benchmark._equivalent_total_points(4096, 8, 8)
        assert mmtools_rows[0].runtime_seconds == 2.0
        assert mmtools_rows[0].runtime_seconds_std == 0.3
        assert np.isclose(
            float(mmtools_rows[0].throughput_points_per_second),
            mmtools_rows[0].total_points / float(mmtools_rows[0].runtime_seconds),
        )

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        db = benchmark.ExampleRunDB(temp_dir / "tensor_backend_scaling.sqlite3")
        stored_rows = [
            _row(benchmark, "nlolib", "GPU", 128, 0.30, runtime_std=0.03),
            _row(benchmark, "MMTools", "GPU", 128, 0.25, runtime_std=0.02),
            _row(benchmark, "nlolib", "CPU", 128, None, status="error"),
        ]
        run_group = db.begin_group("tensor_backend_scaling", "stored")
        benchmark._save_benchmark_rows(
            db,
            example_name="tensor_backend_scaling",
            run_group=run_group,
            rows=stored_rows,
        )

        resolved = db.resolve_replot_group("tensor_backend_scaling", None)
        assert resolved == "stored"
        reloaded_rows = benchmark._load_benchmark_rows(
            db,
            example_name="tensor_backend_scaling",
            run_group=resolved,
        )
        assert [(row.solver, row.backend, row.status) for row in reloaded_rows] == [
            ("MMTools", "GPU", "ok"),
            ("nlolib", "CPU", "error"),
            ("nlolib", "GPU", "ok"),
        ]
        assert reloaded_rows[1].runtime_seconds is None
        assert reloaded_rows[2].runtime_seconds == 0.30

        output_dir = temp_dir / "out"
        benchmark.NloExampleRunner = lambda: (_ for _ in ()).throw(
            AssertionError("replot must not construct a solver runner")
        )
        benchmark._run(
            argparse.Namespace(
                db_path=db.db_path,
                output_dir=output_dir,
                report_dir=temp_dir / "report",
                save_plots="none",
                replot=True,
                run_group=None,
                scales=[3],
                num_records=1,
                warmup=0,
                runs=1,
                mmtools_summary_csv=None,
            )
        )
        assert (output_dir / "tensor_backend_scaling_results.csv").is_file()
        assert (output_dir / "tensor_backend_scaling_runtime.png").is_file()

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        output_dir = temp_dir / "out"
        output_dir.mkdir()
        benchmark._write_csv(
            [
                _row(benchmark, "nlolib", "GPU", 128, 0.30, runtime_std=0.03),
                _row(benchmark, "MMTools", "GPU", 128, 0.25, runtime_std=0.02),
            ],
            output_dir / "tensor_backend_scaling_results.csv",
        )
        benchmark._run(
            argparse.Namespace(
                db_path=temp_dir / "empty.sqlite3",
                output_dir=output_dir,
                report_dir=temp_dir / "report",
                save_plots="none",
                replot=True,
                run_group=None,
                scales=[3],
                num_records=1,
                warmup=0,
                runs=1,
                mmtools_summary_csv=None,
            )
        )
        assert (output_dir / "tensor_backend_scaling_runtime.png").is_file()

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        db = benchmark.ExampleRunDB(temp_dir / "tensor_backend_scaling_nlolib.sqlite3")
        rows = [
            _row(benchmark, "nlolib", "CPU", 128, 0.40, runtime_std=0.04),
            _row(benchmark, "nlolib", "GPU", 128, 0.20, runtime_std=0.02),
        ]
        run_group = db.begin_group("tensor_backend_scaling_nlolib", "stored")
        benchmark._save_benchmark_rows(
            db,
            example_name="tensor_backend_scaling_nlolib",
            run_group=run_group,
            rows=rows,
        )

        nlolib_benchmark._benchmark_nlolib_rows = lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("replot must not benchmark nlolib rows")
        )
        output_dir = temp_dir / "out"
        nlolib_benchmark._run(
            argparse.Namespace(
                db_path=db.db_path,
                output_dir=output_dir,
                report_dir=temp_dir / "report",
                save_plots="none",
                replot=True,
                run_group=None,
                scales=[3],
                num_records=1,
                warmup=0,
                runs=1,
            )
        )
        assert (output_dir / "tensor_backend_scaling_nlolib_results.csv").is_file()
        assert (output_dir / "tensor_backend_scaling_nlolib_runtime.png").is_file()


def main() -> None:
    test_runtime_plot_helpers()
    print("test_python_tensor_backend_scaling_cleanup: tensor scaling helpers validated.")


if __name__ == "__main__":
    main()
