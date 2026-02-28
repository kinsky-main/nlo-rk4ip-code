"""
GRIN transverse phase validations with analytical references.

This script now runs two analytical GRIN checks:
1) Symmetric parabolic GRIN phase accumulation.
2) Astigmatic (gx != gy) phase accumulation with offset input beam.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from backend.cli import build_example_parser
from backend.plotting import (
    plot_3d_intensity_scatter_propagation,
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from backend.runner import NloExampleRunner, SimulationOptions
from backend.storage import ExampleRunDB


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
    field0_flat = field0.reshape(-1)
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

    z_records, records = runner.propagate_flattened_xy_records(
        field0_flat=field0_flat,
        nx=nx,
        ny=ny,
        num_records=num_records,
        propagation_distance=propagation_distance,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=5e-5,
        error_tolerance=1e-7,
        delta_x=dx,
        delta_y=dy,
        potential_grid=phase_unit.astype(np.complex128).reshape(-1),
        gamma=0.0,
        alpha=0.0,
        exec_options=exec_options,
        **storage_kwargs,
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

    p5 = plot_3d_intensity_scatter_propagation(
        x,
        y,
        z_records,
        np.asarray(records, dtype=np.complex128),
        out_dir / "propagation_3d_numerical_scatter.png",
        intensity_cutoff=0.08,
        xy_stride=16,
        min_marker_size=2.0,
        max_marker_size=34.0,
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
        intensity_cutoff=0.08,
        xy_stride=16,
        min_marker_size=2.0,
        max_marker_size=34.0,
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


def main() -> None:
    parser = build_example_parser(
        example_slug="grin_fiber_xy",
        description="GRIN transverse phase validations with DB-backed run/replot.",
    )
    args = parser.parse_args()

    runner = NloExampleRunner()
    exec_options = SimulationOptions(backend="auto", fft_backend="auto", device_heap_fraction=0.70)
    output_root = Path(__file__).resolve().parent / "output" / "grin_fiber_xy"
    db = ExampleRunDB(args.db_path)
    example_name = "grin_fiber_xy_rk4ip"

    scenarios = [
        {
            "scenario_name": "grin_phase_validation_symmetric",
            "nx": 384,
            "ny": 384,
            "dx": 0.6,
            "dy": 0.6,
            "w0": 7.5,
            "grin_gx": 2.0e-4,
            "grin_gy": 2.0e-4,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "propagation_distance": 0.25,
            "num_records": 8,
        },
        {
            "scenario_name": "grin_phase_validation_astigmatic_offset",
            "nx": 384,
            "ny": 320,
            "dx": 0.6,
            "dy": 0.7,
            "w0": 8.0,
            "grin_gx": 3.0e-4,
            "grin_gy": 1.2e-4,
            "x_offset": 2.5,
            "y_offset": -1.8,
            "propagation_distance": 0.25,
            "num_records": 8,
        },
    ]

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        cases = db.list_cases(example_name=example_name, run_group=run_group)
        if not cases:
            raise RuntimeError(f"no stored cases found in run_group '{run_group}'.")
        all_saved: list[Path] = []
        for case in cases:
            loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case.case_key)
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
            profile_error = relative_l2_error_curve(profile_records, analytical_profile_records)
            out_dir = output_root / scenario_name
            out_dir.mkdir(parents=True, exist_ok=True)
            saved_paths: list[Path] = []
            p1 = plot_intensity_colormap_vs_propagation(
                x,
                loaded.z_axis,
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
                loaded.z_axis,
                full_error,
                out_dir / "full_field_relative_error_over_propagation.png",
                title=f"{scenario_name}: full-field error over propagation",
                y_label="Relative L2 error (full field)",
            )
            if p4 is not None:
                saved_paths.append(p4)
            p5 = plot_3d_intensity_scatter_propagation(
                x,
                y,
                loaded.z_axis,
                np.asarray(records, dtype=np.complex128),
                out_dir / "propagation_3d_numerical_scatter.png",
                intensity_cutoff=0.08,
                xy_stride=16,
                min_marker_size=2.0,
                max_marker_size=34.0,
                title=f"{scenario_name}: numerical propagation (3D scatter)",
            )
            if p5 is not None:
                saved_paths.append(p5)
            p6 = plot_3d_intensity_scatter_propagation(
                x,
                y,
                loaded.z_axis,
                np.asarray(analytical_records, dtype=np.complex128),
                out_dir / "propagation_3d_analytical_scatter.png",
                intensity_cutoff=0.08,
                xy_stride=16,
                min_marker_size=2.0,
                max_marker_size=34.0,
                title=f"{scenario_name}: analytical propagation (3D scatter)",
            )
            if p6 is not None:
                saved_paths.append(p6)

            in_power = float(np.sum(np.abs(records[0]) ** 2))
            out_power = float(np.sum(np.abs(records[-1]) ** 2))
            power_drift = abs(out_power - in_power) / max(in_power, 1e-12)
            final_error = float(full_error[-1])
            final_profile_error = float(profile_error[-1])
            print(
                f"{scenario_name}: records={records.shape[0]}, grid=({ny},{nx}), "
                f"final_full_error={final_error:.6e}, final_profile_error={final_profile_error:.6e}, "
                f"power_drift={power_drift:.6e}"
            )
            all_saved.extend(saved_paths)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        all_saved = []
        for scenario in scenarios:
            saved_paths, _, _ = run_phase_validation(
                runner,
                exec_options=exec_options,
                output_root=output_root,
                storage_db=db,
                storage_example_name=example_name,
                run_group=run_group,
                **scenario,
            )
            all_saved.extend(saved_paths)

    print(f"grin_fiber_xy run_group={run_group}")

    if all_saved:
        print("saved plots:")
        for path in all_saved:
            print(f"  {path}")


if __name__ == "__main__":
    main()
