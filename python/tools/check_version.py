#!/usr/bin/env python3
"""Print the Python package version, and check a release tag against it.

``pyproject.toml`` is the source of truth for the package version, which is
released independently of the core library (``CORE_REF`` pins the core
revision the wheels are built from). Tags are bare versions with no "v"
prefix; the wheel filename carries the version, so a tag that disagreed with
it would publish something other than what it names.

    python tools/check_version.py            # prints the version
    python tools/check_version.py --tag 1.21.0
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATTERN = r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag to compare against, e.g. 1.21.0")
    args = parser.parse_args()

    path = ROOT / "pyproject.toml"
    match = re.search(PATTERN, path.read_text(encoding="utf-8"))
    if not match:
        print(f"::error::could not parse a version from {path}", file=sys.stderr)
        return 1
    version = match.group(1)
    if args.tag and args.tag != version:
        print(
            f"::error::tag {args.tag} does not match the version {version} in "
            "pyproject.toml; bump pyproject.toml before tagging",
            file=sys.stderr,
        )
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
