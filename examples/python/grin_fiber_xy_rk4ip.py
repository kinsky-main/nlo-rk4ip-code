"""
GRIN transverse phase validations with analytical references.

Thin CLI wrapper over the OOP implementation in ``grin.fiber``.
"""

from __future__ import annotations

from backend.cli import build_example_parser
from grin.fiber import GrinFiberApp, run_phase_validation


def main() -> None:
    parser = build_example_parser(
        example_slug="grin_fiber_xy",
        description="GRIN transverse phase validations with DB-backed run/replot.",
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
        "--fail-on-plot-warning",
        action="store_true",
        default=False,
        help="Treat plot validation warnings as hard failures.",
    )

    args = parser.parse_args()
    app = GrinFiberApp(args)
    app.run()


if __name__ == "__main__":
    main()
