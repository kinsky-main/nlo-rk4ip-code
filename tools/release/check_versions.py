#!/usr/bin/env python3
"""Check, or sync, the packaged versions against the CMake project version.

``CMakeLists.txt`` is the single source of truth. ``pyproject.toml`` and
``julia/Project.toml`` hold copies, because a wheel and a Julia package each
have to carry a literal version of their own.

Those copies drift by design during development: the ``pre-commit`` hook that
``INSTALL_GIT_HOOKS`` installs (generated from ``cmake/pre-commit.in``) bumps
the project MINOR version on every commit to ``main``, so the CMake version
advances once per commit while the copies stay where the last release left
them. So this is a release-time tool, not a pre-commit invariant:

    --write   rewrite the copies from the CMake version (what a release build
              does before it packages anything)
    default   verify the copies already match, and fail if they do not

With ``--tag NAME`` the tag is checked too. Tags are bare versions with no "v"
prefix, because File Exchange reads the tag as the release version (see
.github/workflows/release-mltbx.yml).
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

CMAKE_PATTERN = r"project\([^)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)"
COPY_PATTERN = r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"'
COPIES = ("pyproject.toml", "julia/Project.toml")


def read(relative: str, pattern: str) -> str:
    path = ROOT / relative
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if not match:
        print(f"Could not parse a version from {path}", file=sys.stderr)
        raise SystemExit(1)
    return match.group(1)


def write(relative: str, version: str) -> bool:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(COPY_PATTERN, f'version = "{version}"', text, count=1)
    if count != 1:
        print(f"Could not rewrite the version in {path}", file=sys.stderr)
        raise SystemExit(1)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag to compare against, e.g. 1.16.0")
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the copies from the CMake version instead of verifying them",
    )
    args = parser.parse_args()

    expected = read("CMakeLists.txt", CMAKE_PATTERN)
    problems = []

    if args.tag and args.tag != expected:
        problems.append(
            f"tag {args.tag} does not match project version {expected}; "
            "the tag is read as the release version, so they must agree"
        )

    for relative in COPIES:
        if args.write:
            if write(relative, expected):
                print(f"{relative}: set to {expected}", file=sys.stderr)
        else:
            found = read(relative, COPY_PATTERN)
            if found != expected:
                problems.append(
                    f"{relative} has {found}, expected {expected} from CMakeLists.txt"
                )

    if problems:
        for problem in problems:
            print(f"::error::{problem}", file=sys.stderr)
        return 1

    print(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
