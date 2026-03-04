"""
Full coupled GRIN example (3+1D tensor):
dispersion + diffraction + Kerr nonlinearity + GRIN potential.
"""

from __future__ import annotations

from backend.app_base import ExampleAppBase
from grin.coupled import FullCoupledGrinApp


class GrinFullCoupledEntryApp(ExampleAppBase):
    example_slug = "grin_full_coupled"
    description = (
        "Full coupled GRIN example: dispersion + diffraction + Kerr nonlinearity "
        "with a GRIN potential, plus a linear baseline comparison."
    )

    def run(self) -> None:
        FullCoupledGrinApp(self.args).run()


def main() -> None:
    GrinFullCoupledEntryApp.from_cli().run()


if __name__ == "__main__":
    main()
