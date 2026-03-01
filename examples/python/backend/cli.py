"""Shared command-line parsing helpers for Python examples."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_save_plots_arg(raw_value: str | None) -> set[str] | None:
    if raw_value is None:
        return None

    text = raw_value.strip()
    if text == "":
        return None

    lowered = text.lower()
    if lowered in {"all", "*"}:
        return None
    if lowered in {"none", "off"}:
        return set()

    selected: set[str] = set()
    for token in text.split(","):
        stripped = token.strip()
        if stripped == "":
            continue
        stem = Path(stripped).stem.strip().lower()
        if stem in {"all", "*"}:
            return None
        if stem in {"none", "off"}:
            continue
        selected.add(stem)
    return selected


def _configure_plot_saving_from_args(args: argparse.Namespace, selected_plot_keys: set[str] | None) -> None:
    try:
        from .plotting import configure_plot_saving
    except Exception:
        return

    configure_plot_saving(
        primary_output_dir=getattr(args, "output_dir", None),
        report_dir=getattr(args, "report_dir", None),
        selected_plot_keys=selected_plot_keys,
    )


class _ExampleArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        selected_plot_keys = _parse_save_plots_arg(getattr(parsed, "save_plots", None))
        _configure_plot_saving_from_args(parsed, selected_plot_keys)
        return parsed


def build_example_parser(
    *,
    example_slug: str,
    description: str,
) -> argparse.ArgumentParser:
    parser = _ExampleArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / example_slug,
        help="Primary directory for example output plots.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("C:/Users/Wenzel/Final Year Project/report/figs"),
        help="Optional additional directory where selected plots are mirrored.",
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        default="linear_drift_intensity_propagation_map,linear_drift_final_intensity_comparison,soliton_total_error_over_propagation,soliton_final_re_im_comparison,soliton_wavelength_intensity_colormap,error_vs_fixed_step_size,spm_time_intensity_propagation,spm_frequency_intensity_propagation",
        help=(
            "Comma-separated plot keys to save (output filename stems, without extension). "
            "Use 'all' (default) to save every plot or 'none' to disable plot writes."
        ),
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Replot from an existing DB run group instead of running the solver.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / example_slug / "runs.sqlite3",
        help="SQLite path used for solver run storage and replot reads.",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help="Optional run-group ID. Defaults to generated ID (run mode) or latest (replot mode).",
    )
    return parser
