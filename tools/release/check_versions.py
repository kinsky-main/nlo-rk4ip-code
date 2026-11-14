#!/usr/bin/env python3
"""Check, sync, or bump the project version and its packaged copies.

``CMakeLists.txt`` is the single source of truth. ``pyproject.toml`` and
``julia/Project.toml`` hold copies, because a wheel and a Julia package each
have to carry a literal version of their own, and ``sbom.cdx.json`` stamps it
as well.

They must agree at every commit, so that a release tag can be cut from any
commit and the published artifacts carry the version their tag names. That
invariant is enforced in three places: the ``test_version_copies_match`` and
``test_sbom_pins_match`` CTest cases, the pre-commit hook
``INSTALL_GIT_HOOKS`` installs, and the ``verify`` job of each release
workflow.

    default        verify the two literal copies agree, and fail if they do
                   not (``test_sbom_pins_match`` covers the SBOM)
    --write        rewrite the copies, and the SBOM, from the CMake version
    --bump KIND    advance the CMake version (major, minor or patch) and
                   rewrite the copies to match

Nothing advances the version automatically. ``--bump`` is the deliberate step a
release takes; until this is run again, every commit keeps naming the released
version, which is what makes the equality check above possible.

With ``--tag NAME`` the tag is checked too. Tags are bare versions with no "v"
prefix, because File Exchange reads the tag as the release version (see
.github/workflows/release-mltbx.yml).
"""

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

CMAKE_PATTERN = r"project\([^)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)"
COPY_PATTERN = r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"'
COPIES = ("pyproject.toml", "julia/Project.toml")

BUMP_KINDS = ("major", "minor", "patch")


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


def rewrite_sbom() -> None:
    """Regenerate sbom.cdx.json, which stamps the project version three times.

    It is a copy like the other two, but a structured one: the version appears
    as a bom-ref, a version field and a purl. tools/write_sbom.py already owns
    that layout, and test_sbom_pins_match already verifies it, so delegate
    rather than reimplement the JSON surgery here.
    """
    script = ROOT / "tools" / "write_sbom.py"
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode != 0:
        print(f"::error::{script} failed", file=sys.stderr)
        raise SystemExit(1)


def bump(kind: str) -> str:
    """Advance the CMake project version and return the new value."""
    path = ROOT / "CMakeLists.txt"
    text = path.read_text(encoding="utf-8")
    current = read("CMakeLists.txt", CMAKE_PATTERN)
    major, minor, patch = (int(part) for part in current.split("."))

    if kind == "major":
        major, minor, patch = major + 1, 0, 0
    elif kind == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    new_version = f"{major}.{minor}.{patch}"

    # Rewrite the version inside the project() call only; the file also holds
    # cmake_minimum_required(VERSION ...), which must not move.
    updated, count = re.subn(
        r"(project\([^)]*VERSION\s+)[0-9]+\.[0-9]+\.[0-9]+",
        rf"\g<1>{new_version}",
        text,
        count=1,
    )
    if count != 1:
        print(f"Could not rewrite the project version in {path}", file=sys.stderr)
        raise SystemExit(1)

    path.write_text(updated, encoding="utf-8")
    print(f"CMakeLists.txt: {current} -> {new_version}", file=sys.stderr)
    return new_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag to compare against, e.g. 1.16.0")
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the copies from the CMake version instead of verifying them",
    )
    parser.add_argument(
        "--bump",
        choices=BUMP_KINDS,
        help="advance the CMake project version, then rewrite the copies to match",
    )
    args = parser.parse_args()

    if args.bump:
        if args.tag:
            parser.error("--bump sets the version; --tag checks an existing one")
        bump(args.bump)
        args.write = True

    expected = read("CMakeLists.txt", CMAKE_PATTERN)
    problems = []

    if args.tag and args.tag != expected:
        problems.append(
            f"tag {args.tag} does not match project version {expected}; "
            "the tag is read as the release version, so they must agree"
        )

    if args.write:
        rewrite_sbom()

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
