"""OOP implementation for GRIN transverse-phase examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from backend.plotting import (
    plot_3d_intensity_scatter_propagation,
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from backend.runner import NloExampleRunner, SimulationOptions
from backend.storage import ExampleRunDB

from .models import PlotArtifact, ValidationReport
from .validation import PlotImageValidator, write_report


def flatten_xy_tfast(field_xy: np.ndarray) -> np.ndarray:
    return np.asarray(field_xy, dtype=np.complex128).T.reshape(-1)


def unflatten_xy_tfast(flat_tfast: np.ndarray, ny: int, nx: int) -> np.ndarray:
    return np.asarray(flat_tfast, dtype=np.complex128).reshape(nx, ny).T


def relative_l2_error_curve(records: np.ndarray, reference_records: np.ndarray) -> np.ndarray:
    if records.shape != reference_records.shape:
        raise ValueError("records and reference_records must have the same shape.")
    out = np.empty(records.shape[0], dtype=np.float64)
    for i in range(records.shape[0]):
        ref = np.asarray(reference_records[i], dtype=np.complex128)
        ref_norm = float(np.linalg.norm(ref))
        safe_ref_norm = ref_norm if ref_norm > 0.0 else 1.0
        out[i] = float(np.linalg.norm(np.asarray(records[i], dtype=np.complex128) - ref) / safe_ref_norm)
    return out


@dataclass(frozen=True)
class FiberScenario:
    scenario_name: str
    nx: int
    ny: int
    dx: float
    dy: float
    w0: float
    grin_gx: float
    grin_gy: float
    x_offset: float
    y_offset: float
    propagation_distance: float
    num_records: int


def run_phase_validation(
    runner: NloExampleRunner,
    *,
    scenario_name: str,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    w0: float,
    grin_gx: float,
    grin_gy: float,
    x_offset: float,
    y_offset: float,
    propagation_distance: float,
    num_records: int,
    exec_options: SimulationOptions,
    output_root: Path,
    storage_db: ExampleRunDB | None = None,
    storage_example_name: str | None = None,
    run_group: str | None = None,
) -> tuple[list[Path], float, float]:
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    phase_unit = (grin_gx * (xx * xx)) + (grin_gy * (yy * yy))
    field0 = np.exp(-(((xx - x_offset) ** 2 + (yy - y_offset) ** 2) / (w0 * w0))).astype(np.complex128)
    field0_tfast = flatten_xy_tfast(field0)
    nxy = nx * ny

    storage_kwargs: dict[str, object] = {}
    if storage_db is not None:
        if storage_example_name is None or run_group is None:
            raise ValueError("storage_example_name and run_group are required when storage_db is provided.")
        storage_kwargs = storage_db.storage_kwargs(
            example_name=storage_example_name,
            run_group=run_group,
            case_key=scenario_name,
            chunk_records=4,
        )

    nlo = runner.nlo
    gx = float(grin_gx)
    gy = float(grin_gy)
    runtime = nlo.RuntimeOperators(
        linear_factor_fn=lambda A, wt, kx, ky: 0.0,
        linear_fn=lambda A, D, h: np.exp(h * D),
        potential_fn=lambda x, y: (gx * (x * x)) + (gy * (y * y)),
        nonlinear_fn=lambda A, I, V: (1.0j * A) * V,
    )
    nt = 1
    z_records, records_flat = runner.propagate_tensor3d_records(
        field0_tfast=field0_tfast,
        nt=nt,
        nx=nx,
        ny=ny,
        num_records=num_records,
        propagation_distance=propagation_distance,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=5e-5,
        error_tolerance=1e-7,
        delta_time=1.0,
        pulse_period=1.0,
        frequency_grid=np.asarray([0.0 + 0.0j] * nt, dtype=np.complex128),
        delta_x=dx,
        delta_y=dy,
        runtime=runtime,
        exec_options=exec_options,
        **storage_kwargs,
    )
    records = np.asarray(
        [unflatten_xy_tfast(row, int(ny), int(nx)) for row in np.asarray(records_flat, dtype=np.complex128)],
        dtype=np.complex128,
    )
    solver_run_id = ""
    if storage_db is not None:
        solver_run_id = storage_db.save_case_from_solver_meta(
            example_name=storage_example_name,
            run_group=run_group,
            case_key=scenario_name,
            solver_meta=runner.last_meta,
            meta={},
        )

    analytical_records = np.empty_like(records, dtype=np.complex128)
    for i, z in enumerate(z_records):
        analytical_records[i] = field0 * np.exp((1.0j) * phase_unit * float(z))

    full_error = relative_l2_error_curve(
        np.asarray(records, dtype=np.complex128).reshape(num_records, nxy),
        np.asarray(analytical_records, dtype=np.complex128).reshape(num_records, nxy),
    )
    center_row = ny // 2
    profile_records = np.asarray(records[:, center_row, :], dtype=np.complex128)
    analytical_profile_records = np.asarray(analytical_records[:, center_row, :], dtype=np.complex128)
    profile_error = relative_l2_error_curve(profile_records, analytical_profile_records)

    out_dir = output_root / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    p1 = plot_intensity_colormap_vs_propagation(
        x,
        z_records,
        np.abs(profile_records) ** 2,
        out_dir / "centerline_intensity_colormap.png",
        x_label="Transverse coordinate x",
        y_label="Propagation distance z",
        title=f"{scenario_name}: center-line intensity vs propagation",
        colorbar_label="Normalized center-line intensity",
    )
    if p1 is not None:
        saved_paths.append(p1)
    p2 = plot_final_re_im_comparison(
        x,
        analytical_profile_records[-1],
        profile_records[-1],
        out_dir / "final_re_im_profile_comparison.png",
        x_label="Transverse coordinate x",
        title=f"{scenario_name}: final Re/Im profile (analytical vs numerical)",
        reference_label="Analytical final",
        final_label="Numerical final",
    )
    if p2 is not None:
        saved_paths.append(p2)
    p3 = plot_final_intensity_comparison(
        x,
        analytical_profile_records[-1],
        profile_records[-1],
        out_dir / "final_intensity_profile_comparison.png",
        x_label="Transverse coordinate x",
        title=f"{scenario_name}: final intensity profile (analytical vs numerical)",
        reference_label="Analytical final",
        final_label="Numerical final",
    )
    if p3 is not None:
        saved_paths.append(p3)
    p4 = plot_total_error_over_propagation(
        z_records,
        full_error,
        out_dir / "full_field_relative_error_over_propagation.png",
        title=f"{scenario_name}: full-field error over propagation",
        y_label="Relative L2 error (full field)",
    )
    if p4 is not None:
        saved_paths.append(p4)

    peak = float(max(np.max(np.abs(records) ** 2), np.max(np.abs(analytical_records) ** 2)))
    p5 = plot_3d_intensity_scatter_propagation(
        x,
        y,
        z_records,
        np.asarray(records, dtype=np.complex128),
        out_dir / "propagation_3d_numerical_scatter.png",
        intensity_cutoff=0.03,
        xy_stride=6,
        z_stride=1,
        min_marker_size=1.2,
        max_marker_size=30.0,
        alpha_min=0.04,
        alpha_max=0.92,
        dpi=360,
        normalization_peak=peak,
        title=f"{scenario_name}: numerical propagation (3D scatter)",
    )
    if p5 is not None:
        saved_paths.append(p5)
    p6 = plot_3d_intensity_scatter_propagation(
        x,
        y,
        z_records,
        np.asarray(analytical_records, dtype=np.complex128),
        out_dir / "propagation_3d_analytical_scatter.png",
        intensity_cutoff=0.03,
        xy_stride=6,
        z_stride=1,
        min_marker_size=1.2,
        max_marker_size=30.0,
        alpha_min=0.04,
        alpha_max=0.92,
        dpi=360,
        normalization_peak=peak,
        title=f"{scenario_name}: analytical propagation (3D scatter)",
    )
    if p6 is not None:
        saved_paths.append(p6)

    in_power = float(np.sum(np.abs(records[0]) ** 2))
    out_power = float(np.sum(np.abs(records[-1]) ** 2))
    power_drift = abs(out_power - in_power) / max(in_power, 1e-12)
    final_error = float(full_error[-1])
    final_profile_error = float(profile_error[-1])
    if storage_db is not None and solver_run_id:
        storage_db.save_case(
            example_name=storage_example_name,
            run_group=run_group,
            case_key=scenario_name,
            run_id=solver_run_id,
            meta={
                "scenario_name": scenario_name,
                "nx": int(nx),
                "ny": int(ny),
                "dx": float(dx),
                "dy": float(dy),
                "w0": float(w0),
                "grin_gx": float(grin_gx),
                "grin_gy": float(grin_gy),
                "x_offset": float(x_offset),
                "y_offset": float(y_offset),
                "propagation_distance": float(propagation_distance),
                "num_records": int(num_records),
                "final_error": float(final_error),
                "power_drift": float(power_drift),
                "final_profile_error": float(final_profile_error),
            },
        )
    print(
        f"{scenario_name}: records={num_records}, grid=({ny},{nx}), "
        f"final_full_error={final_error:.6e}, final_profile_error={final_profile_error:.6e}, "
        f"power_drift={power_drift:.6e}"
    )
    return saved_paths, final_error, power_drift


class GrinFiberApp:
    """Runs GRIN fiber phase validations and diffraction proof checks."""

    def __init__(self, args: Any):
        self.args = args
        self.runner = NloExampleRunner()
        self.db = ExampleRunDB(args.db_path)
        self.exec_options = SimulationOptions(backend="auto", fft_backend="auto", device_heap_fraction=0.70)
        self.example_name = "grin_fiber_xy_rk4ip"
        self.report_path: Path | None = None

    @staticmethod
    def _default_scenarios() -> list[FiberScenario]:
        return [
            FiberScenario(
                scenario_name="grin_phase_validation_symmetric",
                nx=384,
                ny=384,
                dx=0.6,
                dy=0.6,
                w0=7.5,
                grin_gx=2.0e-4,
                grin_gy=2.0e-4,
                x_offset=0.0,
                y_offset=0.0,
                propagation_distance=0.25,
                num_records=8,
            ),
            FiberScenario(
                scenario_name="grin_phase_validation_astigmatic_offset",
                nx=384,
                ny=320,
                dx=0.6,
                dy=0.7,
                w0=8.0,
                grin_gx=3.0e-4,
                grin_gy=1.2e-4,
                x_offset=2.5,
                y_offset=-1.8,
                propagation_distance=0.25,
                num_records=8,
            ),
        ]

    def _proof_check(
        self,
        *,
        output_root: Path,
        max_final_error: float = 5.0e-2,
    ) -> tuple[Path | None, float]:
        nlo = self.runner.nlo
        nx = 128
        ny = 128
        dx = 0.7
        dy = 0.7
        w0 = 9.0
        diffraction_coeff = -0.02
        propagation_distance = 0.20
        num_records = 6
        nxy = int(nx) * int(ny)

        x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
        y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
        xx, yy = np.meshgrid(x, y, indexing="xy")
        field0 = np.exp(-((xx * xx + yy * yy) / (w0 * w0))).astype(np.complex128)
        field0_tfast = flatten_xy_tfast(field0)
        kx = (2.0 * np.pi) * np.fft.fftfreq(nx, d=dx)
        ky = (2.0 * np.pi) * np.fft.fftfreq(ny, d=dy)
        k2 = (kx[np.newaxis, :] ** 2) + (ky[:, np.newaxis] ** 2)
        coeff = float(diffraction_coeff)
        runtime = nlo.RuntimeOperators(
            linear_factor_fn=lambda A, wt, kx, ky: (1.0j * coeff) * ((kx * kx) + (ky * ky)),
            linear_fn=lambda A, D, h: np.exp(h * D),
            nonlinear_fn=lambda A, I: 0.0,
        )
        z_records, records_flat = self.runner.propagate_tensor3d_records(
            field0_tfast=field0_tfast,
            nt=1,
            nx=nx,
            ny=ny,
            num_records=num_records,
            propagation_distance=float(propagation_distance),
            starting_step_size=1.0e-3,
            max_step_size=2.0e-3,
            min_step_size=5.0e-5,
            error_tolerance=1.0e-7,
            pulse_period=1.0,
            delta_time=1.0,
            frequency_grid=np.asarray([0.0 + 0.0j], dtype=np.complex128),
            delta_x=float(dx),
            delta_y=float(dy),
            runtime=runtime,
            exec_options=self.exec_options,
        )
        tensor_records = np.asarray(
            [unflatten_xy_tfast(row, int(ny), int(nx)) for row in np.asarray(records_flat, dtype=np.complex128)],
            dtype=np.complex128,
        )

        fft0 = np.fft.fft2(field0)
        reference_records = np.empty_like(tensor_records, dtype=np.complex128)
        for i, z in enumerate(z_records):
            reference_records[i] = np.fft.ifft2(fft0 * np.exp((1.0j) * float(diffraction_coeff) * k2 * float(z)))

        error_curve = relative_l2_error_curve(
            tensor_records.reshape(int(num_records), nxy),
            reference_records.reshape(int(num_records), nxy),
        )
        final_error = float(error_curve[-1])
        out_dir = output_root / "diffraction_operator_proof_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = plot_total_error_over_propagation(
            z_records,
            error_curve,
            out_dir / "tensor_diffraction_operator_error_vs_fft2_reference.png",
            title="GRIN diffraction proof check: tensor operator vs FFT2 reference",
            y_label="Relative L2 error (tensor vs FFT2 reference)",
        )
        print(
            "diffraction_operator_proof_check: "
            f"records={num_records}, grid=({ny},{nx}), final_error={final_error:.6e}"
        )
        if final_error > float(max_final_error):
            raise RuntimeError(
                "diffraction operator proof check failed: "
                f"final tensor-vs-FFT2 error {final_error:.6e} exceeds {max_final_error:.6e}"
            )
        return saved, final_error

    def run(self) -> tuple[str, list[Path], ValidationReport]:
        if self.args.replot:
            run_group = self.db.resolve_replot_group(self.example_name, self.args.run_group)
        else:
            run_group = self.db.begin_group(self.example_name, self.args.run_group)

        output_root = self.args.output_dir / run_group
        output_root.mkdir(parents=True, exist_ok=True)
        all_saved: list[Path] = []
        report = ValidationReport(example_name=self.example_name, run_group=run_group)
        report.metadata["output_root"] = str(output_root)
        report.metadata["mode"] = "replot" if self.args.replot else "run"

        if self.args.replot:
            cases = self.db.list_cases(example_name=self.example_name, run_group=run_group)
            if not cases:
                raise RuntimeError(f"no stored cases found in run_group '{run_group}'.")
            for case in cases:
                loaded = self.db.load_case(example_name=self.example_name, run_group=run_group, case_key=case.case_key)
                meta = loaded.meta
                scenario_name = str(meta["scenario_name"])
                nx = int(meta["nx"])
                ny = int(meta["ny"])
                dx = float(meta["dx"])
                dy = float(meta["dy"])
                w0 = float(meta["w0"])
                grin_gx = float(meta["grin_gx"])
                grin_gy = float(meta["grin_gy"])
                x_offset = float(meta["x_offset"])
                y_offset = float(meta["y_offset"])
                x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * dx
                y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * dy
                xx, yy = np.meshgrid(x, y, indexing="xy")
                phase_unit = (grin_gx * (xx * xx)) + (grin_gy * (yy * yy))
                field0 = np.exp(-(((xx - x_offset) ** 2 + (yy - y_offset) ** 2) / (w0 * w0))).astype(np.complex128)
                records = np.asarray(loaded.records, dtype=np.complex128).reshape(loaded.records.shape[0], ny, nx)
                analytical_records = np.empty_like(records, dtype=np.complex128)
                for i, z in enumerate(loaded.z_axis):
                    analytical_records[i] = field0 * np.exp((1.0j) * phase_unit * float(z))

                full_error = relative_l2_error_curve(
                    np.asarray(records, dtype=np.complex128).reshape(records.shape[0], nx * ny),
                    np.asarray(analytical_records, dtype=np.complex128).reshape(records.shape[0], nx * ny),
                )
                center_row = ny // 2
                profile_records = np.asarray(records[:, center_row, :], dtype=np.complex128)
                analytical_profile_records = np.asarray(analytical_records[:, center_row, :], dtype=np.complex128)
                out_dir = output_root / scenario_name
                out_dir.mkdir(parents=True, exist_ok=True)

                peak = float(max(np.max(np.abs(records) ** 2), np.max(np.abs(analytical_records) ** 2)))
                paths: list[Path] = []
                for maybe_path in (
                    plot_intensity_colormap_vs_propagation(
                        x,
                        loaded.z_axis,
                        np.abs(profile_records) ** 2,
                        out_dir / "centerline_intensity_colormap.png",
                        x_label="Transverse coordinate x",
                        y_label="Propagation distance z",
                        title=f"{scenario_name}: center-line intensity vs propagation",
                        colorbar_label="Normalized center-line intensity",
                    ),
                    plot_final_re_im_comparison(
                        x,
                        analytical_profile_records[-1],
                        profile_records[-1],
                        out_dir / "final_re_im_profile_comparison.png",
                        x_label="Transverse coordinate x",
                        title=f"{scenario_name}: final Re/Im profile (analytical vs numerical)",
                        reference_label="Analytical final",
                        final_label="Numerical final",
                    ),
                    plot_final_intensity_comparison(
                        x,
                        analytical_profile_records[-1],
                        profile_records[-1],
                        out_dir / "final_intensity_profile_comparison.png",
                        x_label="Transverse coordinate x",
                        title=f"{scenario_name}: final intensity profile (analytical vs numerical)",
                        reference_label="Analytical final",
                        final_label="Numerical final",
                    ),
                    plot_total_error_over_propagation(
                        loaded.z_axis,
                        full_error,
                        out_dir / "full_field_relative_error_over_propagation.png",
                        title=f"{scenario_name}: full-field error over propagation",
                        y_label="Relative L2 error (full field)",
                    ),
                    plot_3d_intensity_scatter_propagation(
                        x,
                        y,
                        loaded.z_axis,
                        np.asarray(records, dtype=np.complex128),
                        out_dir / "propagation_3d_numerical_scatter.png",
                        intensity_cutoff=0.03,
                        xy_stride=6,
                        z_stride=1,
                        min_marker_size=1.2,
                        max_marker_size=30.0,
                        alpha_min=0.04,
                        alpha_max=0.92,
                        dpi=360,
                        normalization_peak=peak,
                        title=f"{scenario_name}: numerical propagation (3D scatter)",
                    ),
                    plot_3d_intensity_scatter_propagation(
                        x,
                        y,
                        loaded.z_axis,
                        np.asarray(analytical_records, dtype=np.complex128),
                        out_dir / "propagation_3d_analytical_scatter.png",
                        intensity_cutoff=0.03,
                        xy_stride=6,
                        z_stride=1,
                        min_marker_size=1.2,
                        max_marker_size=30.0,
                        alpha_min=0.04,
                        alpha_max=0.92,
                        dpi=360,
                        normalization_peak=peak,
                        title=f"{scenario_name}: analytical propagation (3D scatter)",
                    ),
                ):
                    if maybe_path is not None:
                        paths.append(maybe_path)
                all_saved.extend(paths)
                final_error = float(full_error[-1])
                power_drift = abs(float(np.sum(np.abs(records[-1]) ** 2) - np.sum(np.abs(records[0]) ** 2))) / max(
                    float(np.sum(np.abs(records[0]) ** 2)),
                    1e-12,
                )
                report.add_threshold(name=f"{scenario_name}:final_error", value=final_error, threshold=1e-3)
                report.add_threshold(name=f"{scenario_name}:power_drift", value=power_drift, threshold=1e-3)
        else:
            for scenario in self._default_scenarios():
                saved_paths, final_error, power_drift = run_phase_validation(
                    self.runner,
                    scenario_name=scenario.scenario_name,
                    nx=scenario.nx,
                    ny=scenario.ny,
                    dx=scenario.dx,
                    dy=scenario.dy,
                    w0=scenario.w0,
                    grin_gx=scenario.grin_gx,
                    grin_gy=scenario.grin_gy,
                    x_offset=scenario.x_offset,
                    y_offset=scenario.y_offset,
                    propagation_distance=scenario.propagation_distance,
                    num_records=scenario.num_records,
                    exec_options=self.exec_options,
                    output_root=output_root,
                    storage_db=self.db,
                    storage_example_name=self.example_name,
                    run_group=run_group,
                )
                all_saved.extend(saved_paths)
                report.add_threshold(name=f"{scenario.scenario_name}:final_error", value=final_error, threshold=1e-3)
                report.add_threshold(name=f"{scenario.scenario_name}:power_drift", value=power_drift, threshold=1e-3)
            proof_saved, proof_error = self._proof_check(output_root=output_root)
            if proof_saved is not None:
                all_saved.append(proof_saved)
            report.add_threshold(name="diffraction_proof:final_error", value=proof_error, threshold=5e-2)

        if bool(getattr(self.args, "plot_validate", True)):
            validator = PlotImageValidator()
            artifacts = [PlotArtifact(key=Path(p).stem, path=Path(p), allow_uniform=False) for p in all_saved]
            validator.validate_artifacts(report, artifacts)

        report_name = (
            Path(self.args.plot_validation_report)
            if getattr(self.args, "plot_validation_report", None) is not None
            else (output_root / "plot_validation_report.json")
        )
        self.report_path = write_report(report, report_name)

        print(f"grin_fiber_xy run_group={run_group}")
        if all_saved:
            print("saved plots:")
            for path in all_saved:
                print(f"  {path}")
        print(f"plot validation report: {self.report_path}")

        has_failures = report.fail_count() > 0
        has_warnings = report.warn_count() > 0
        if has_failures or (has_warnings and bool(getattr(self.args, "fail_on_plot_warning", False))):
            raise RuntimeError(
                "plot validation failed"
                + (f": fails={report.fail_count()}" if has_failures else "")
                + (f", warns={report.warn_count()}" if has_warnings else "")
            )
        return run_group, all_saved, report
