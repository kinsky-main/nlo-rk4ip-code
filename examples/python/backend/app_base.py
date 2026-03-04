"""Shared base class for Python example entrypoints."""

from __future__ import annotations

import argparse
from typing import Any

from .cli import build_example_parser


class ExampleAppBase:
    """Minimal shared CLI lifecycle for example app entrypoints."""

    example_slug: str = ""
    description: str = ""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Hook for per-example CLI options."""

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        if not cls.example_slug:
            raise ValueError(f"{cls.__name__}.example_slug must be defined.")
        if not cls.description:
            raise ValueError(f"{cls.__name__}.description must be defined.")
        parser = build_example_parser(
            example_slug=cls.example_slug,
            description=cls.description,
        )
        cls.configure_parser(parser)
        return parser

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "ExampleAppBase":
        parser = cls.build_parser()
        args = parser.parse_args(argv)
        return cls(args)

    def run(self) -> Any:
        raise NotImplementedError
