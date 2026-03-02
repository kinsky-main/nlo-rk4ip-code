"""
GRIN potential + temporal soliton analytical validation (tensor 3D API).

Thin CLI wrapper over the OOP implementation in ``grin.soliton``.
"""

from __future__ import annotations

from backend.cli import build_example_parser
from grin.soliton import GrinSolitonApp


def main() -> None:
    parser = build_example_parser(
        example_slug="grin_soliton_potential",
        description="GRIN potential + temporal soliton analytical validation (tensor 3D API).",
    )
    parser.add_argument(
        "--validation-case",
        type=str,
        default="both",
        choices=["phase_only", "diffraction", "both"],
        help="Choose which validation branch to execute.",
    )
    parser.add_argument(
        "--plot-validate",
        dest="plot_validate",
        action="store_true",
        default=True,
        help="Enable plot/data validation gate.",
    )
    parser.add_argument(
        "--no-plot-validate",
        dest="plot_validate",
        action="store_false",
        help="Disable plot/data validation gate.",
    )
    parser.add_argument(
        "--plot-validation-report",
        type=str,
        default=None,
        help="Optional explicit JSON path for plot validation report.",
    )
    parser.add_argument(
        "--wavelength-mass",
        type=float,
        default=0.999,
        help="Cumulative spectral mass used to select wavelength plotting support.",
    )
    parser.add_argument(
        "--fail-on-plot-warning",
        action="store_true",
        default=False,
        help="Treat plot validation warnings as hard failures.",
    )

    args = parser.parse_args()
    app = GrinSolitonApp(args)
    app.run()


if __name__ == "__main__":
    main()
