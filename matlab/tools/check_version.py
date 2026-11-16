#!/usr/bin/env python3
"""Print the MATLAB toolbox version, and check a release tag against it.

The ``VERSION`` file is the source of truth for the toolbox version, which is
released independently of the core library (``CORE_REF`` pins the core
revision the toolbox is built from). ``package_mltbx.m`` reads the same file.
Tags are bare versions with no "v" prefix, because File Exchange reads the tag
as the release version.

    python tools/check_version.py            # prints the version
    python tools/check_version.py --tag 1.21.0
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag to compare against, e.g. 1.21.0")
    args = parser.parse_args()

    path = ROOT / "VERSION"
    version = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        print(f"::error::expected MAJOR.MINOR.PATCH in {path}, found {version!r}", file=sys.stderr)
        return 1
    if args.tag and args.tag != version:
        print(
            f"::error::tag {args.tag} does not match the version {version} in "
            "VERSION; bump VERSION before tagging",
            file=sys.stderr,
        )
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
