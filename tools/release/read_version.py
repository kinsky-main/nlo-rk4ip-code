#!/usr/bin/env python3
"""Print the project version from CMakeLists.txt.

The same version is parsed by matlab/package_mltbx.m when it names the toolbox
bundle, so the release workflow uses this to check a tag against the package it
is about to publish.
"""

import pathlib
import re
import sys

PATTERN = r"project\([^)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)"


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    cmake = root / "CMakeLists.txt"
    match = re.search(PATTERN, cmake.read_text(encoding="utf-8"))
    if not match:
        print(f"Could not parse a project version from {cmake}", file=sys.stderr)
        return 1
    print(match.group(1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
