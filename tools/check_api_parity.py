#!/usr/bin/env python3
"""Check that the Python, Julia and MATLAB bindings agree on the API constants.

Every binding re-declares the C API's enums and limits in its own source, so a
value changed in src/nlolib.h, or a name added to one binding only, drifts
silently -- there is no compile-time link between the four copies. This is the
static guard for that failure mode, in the same spirit as
cmake/check_matlab_binding_types.cmake.

tests/api_parity.json holds the canonical names and values. For each binding
this script reads the flat table of literal definitions and checks that every
canonical name is defined with the canonical value, and -- for Python and Julia,
which have explicit export lists -- that it is actually exported rather than
merely defined.

Scans source text only: no interpreter, no MATLAB installation, nothing built.

  python tools/check_api_parity.py [--manifest PATH] [--root PATH]
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Python: `NAME = 123` in the binding module, plus a quoted entry in __all__.
# The NLO_* aliases assign from another name rather than a literal, and are not
# part of the canonical set, so a non-literal right-hand side is skipped.
PYTHON_DEFINE = re.compile(r'(?m)^([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$')

# Julia: `const NAME = Cint(0)` / `const NAME = 64`, plus `export NAME`.
JULIA_DEFINE = re.compile(r'(?m)^const\s+([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$')
JULIA_EXPORT = re.compile(r'(?m)^export\s+([A-Za-z_][A-Za-z0-9_!]*)\s*$')

# MATLAB: `NAME = int32(0)` inside the properties (Constant) block. Everything
# in that block is public, so there is no separate export list to check.
MATLAB_DEFINE = re.compile(r'(?m)^\s{8}([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$')

# Shared shapes of an integer literal across the three languages: a bare
# integer, a cast such as int32(0) / Cint(0) / uint64(64), or the uint64 maximum
# spelled as a saturation call.
LITERAL = re.compile(r'^(?:[A-Za-z_][A-Za-z0-9_]*\(\s*)?(-?\d+)\s*\)?$')
UINT64_MAX = 2**64 - 1
UINT64_MAX_CALLS = ("intmax('uint64')", 'intmax("uint64")', "typemax(UInt64)")


def parse_value(text):
    """Return the integer a definition's right-hand side denotes, or None."""
    text = text.split('#')[0].split('%')[0].strip()
    if text in UINT64_MAX_CALLS:
        return UINT64_MAX
    if text == "ctypes.c_size_t(-1).value":
        return UINT64_MAX
    match = LITERAL.match(text)
    return int(match.group(1)) if match else None


def read(relative):
    path = ROOT / relative
    if not path.exists():
        raise SystemExit(f"::error::missing source file {path}")
    return path.read_text(encoding="utf-8")


def collect(pattern, text):
    found = {}
    for name, raw in pattern.findall(text):
        value = parse_value(raw)
        if value is not None:
            found[name] = value
    return found


def bindings():
    """Return {language: (defined_values, exported_names_or_None)}."""
    python_text = read("python/nlolib/_binding.py")
    python_init = read("python/nlolib/__init__.py")
    python_all = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', python_init))

    julia_text = read("julia/src/types.jl")
    julia_exports = set(JULIA_EXPORT.findall(read("julia/src/NLOLib.jl")))

    matlab_text = read("matlab/+nlolib/constants.m")

    return {
        "python": (collect(PYTHON_DEFINE, python_text), python_all),
        "julia": (collect(JULIA_DEFINE, julia_text), julia_exports),
        "matlab": (collect(MATLAB_DEFINE, matlab_text), None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="tests/api_parity.json")
    args = parser.parse_args()

    manifest = json.loads(read(args.manifest))
    canonical = manifest["constants"]
    gaps = {k: v for k, v in manifest["known_gaps"].items() if not k.startswith("_")}

    problems = []
    checked = 0

    for language, (defined, exported) in bindings().items():
        allowed = gaps.get(language, {})
        allowed = {k: v for k, v in allowed.items() if not k.startswith("_")}

        for name, want in canonical.items():
            excused = name in allowed
            if name not in defined:
                if not excused:
                    problems.append(f"{language}: {name} is not defined")
                continue
            if defined[name] != want:
                problems.append(
                    f"{language}: {name} is {defined[name]}, canonical value is {want}"
                )
            elif exported is not None and name not in exported:
                if not excused:
                    problems.append(
                        f"{language}: {name} is defined but not exported"
                    )
            elif excused:
                problems.append(
                    f"{language}: {name} is listed as a known gap but is present; "
                    f"remove it from known_gaps in {args.manifest}"
                )
            checked += 1

        for name in allowed:
            if name not in canonical:
                problems.append(
                    f"{language}: known_gaps names {name}, which is not in the "
                    f"canonical set"
                )

    if problems:
        for problem in sorted(problems):
            print(f"::error::{problem}", file=sys.stderr)
        print(
            f"\n{len(problems)} parity problem(s) across the bindings. "
            f"Canonical names live in {args.manifest}.",
            file=sys.stderr,
        )
        return 1

    print(f"api parity: {len(canonical)} constants agree across python, julia, matlab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
