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

tests/api_surface.json is the second manifest: the whole public surface the
three bindings are meant to share -- functions and their parameters, types and
their fields, string enum values, presets, error identifiers, names that must
not come back, and the shared examples. The bindings are compared with it the
same text-only way. That comparison reports and does not fail (``--surface
report``, the default) until the bindings have been moved to the shared
surface; ``--surface enforce`` makes a difference fail, ``--surface off`` skips
it. docs/binding_api_vocabulary.md describes the surface in prose.

Each binding lives in its own repository, mounted here as the python/, julia/
and matlab/ submodules. All three are required by default, so a checkout that
forgot ``git submodule update --init`` fails instead of passing vacuously. A
binding repository checks itself alone against the core revision it pins:

  python tools/check_api_parity.py
  python core/tools/check_api_parity.py --bindings python --python-dir .
"""

import argparse
import ast
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


def read(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(
            f"::error::missing source file {path}; is the binding submodule "
            "checked out (git submodule update --init)?"
        )
    return path.read_text(encoding="utf-8")


def collect(pattern, text):
    found = {}
    for name, raw in pattern.findall(text):
        value = parse_value(raw)
        if value is not None:
            found[name] = value
    return found


def python_binding(root):
    init = read(root / "nlolib" / "__init__.py")
    exported = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', init))
    return collect(PYTHON_DEFINE, read(root / "nlolib" / "_binding.py")), exported


def julia_binding(root):
    exported = set(JULIA_EXPORT.findall(read(root / "src" / "NLOLib.jl")))
    return collect(JULIA_DEFINE, read(root / "src" / "types.jl")), exported


def matlab_binding(root):
    return collect(MATLAB_DEFINE, read(root / "+nlolib" / "constants.m")), None


# {language: reader}, each returning (defined_values, exported_names_or_None).
BINDINGS = {"python": python_binding, "julia": julia_binding, "matlab": matlab_binding}


# --------------------------------------------------------------------------
# Surface check: the bindings against tests/api_surface.json.
#
# Each reader returns the differences by category. One difference is one thing
# a binding has to change, so the counts are a work list.
# --------------------------------------------------------------------------

NUMBER = re.compile(r'(?<![A-Za-z_.\d])-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?')
BLOCK_BEGIN = "nlolib-example-parameters: begin"
BLOCK_END = "nlolib-example-parameters: end"
BLOCK_LINE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;#%]+?)\s*;?\s*(?:[#%].*)?$')
EXAMPLE_SUFFIX = {"python": ".py", "julia": ".jl", "matlab": ".m"}
QUOTE = "[\"']"


class Differences:
    def __init__(self):
        self.by_category = {}

    def add(self, category, message):
        self.by_category.setdefault(category, []).append(message)

    def total(self):
        return sum(len(v) for v in self.by_category.values())


def public(mapping):
    """Drop the _comment keys a manifest section carries."""
    return {k: v for k, v in mapping.items() if not k.startswith("_")}


def word(name):
    return re.compile(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])')


def gaps_for(manifest, language):
    return set(public(manifest["known_gaps"].get(language, {})))


def expected_params(manifest, language, function):
    """Ordered parameter specs of a function, minus this binding's known gaps."""
    gaps = gaps_for(manifest, language)
    return [
        p for p in manifest["functions"][function]["params"]
        if f"{function}.{p['name']}" not in gaps
    ]


def same_value(found, want):
    if want is None or found is None:
        return found is want
    if isinstance(want, bool) or isinstance(found, bool):
        return found is want
    if isinstance(want, (int, float)) and isinstance(found, (int, float)):
        return abs(float(found) - float(want)) <= 1e-12 * max(1.0, abs(float(want)))
    return found == want


def compare_names(diffs, category, label, found, want, what="parameter"):
    """Report names missing from, extra in, or out of order in `found`."""
    for name in want:
        if name not in found:
            diffs.add(category, f"{label}: {what} {name} is missing")
    for name in found:
        if name not in want:
            diffs.add(category, f"{label}: {what} {name} is not in the manifest")
    common_found = [n for n in found if n in want]
    common_want = [n for n in want if n in found]
    if common_found != common_want:
        diffs.add(
            category,
            f"{label}: {what} order is {', '.join(common_found)}; "
            f"expected {', '.join(common_want)}",
        )


def check_removed(diffs, manifest, language, exported, sources):
    removed = manifest["removed"]
    if exported is not None:
        for name in removed["names"]["all"] + removed["names"][language]:
            if name in exported:
                diffs.add("removed", f"{name} is still exported")
    for name in removed["words"]["all"] + removed["words"][language]:
        pattern = word(name)
        hits = sorted(path for path, text in sources.items() if pattern.search(text))
        if hits:
            diffs.add("removed", f"{name} still occurs in {', '.join(hits)}")


def check_enums(diffs, manifest, sources):
    """Every enum string must occur as a quoted literal in the package source."""
    joined = "\n".join(sources.values())
    for enum, spec in public(manifest["enums"]).items():
        missing = [
            value for value in spec["values"]
            if f'"{value}"' not in joined and f"'{value}'" not in joined
        ]
        if missing:
            diffs.add(
                "enums",
                f"{enum} ({', '.join(spec['used_by'])}): no string literal for "
                f"{', '.join(missing)}",
            )


def check_presets(diffs, manifest, sources):
    """Each preset's five numbers must follow its quoted name in the source."""
    joined = "\n".join(sources.values())
    for preset, spec in public(manifest["presets"]).items():
        want = [float(v) for v in spec.values()]
        best_missing = None
        for match in re.finditer(QUOTE + re.escape(preset) + QUOTE, joined):
            window = joined[match.end():match.end() + 700]
            numbers = [float(n) for n in NUMBER.findall(window)]
            missing = [w for w in want if not any(same_value(n, w) for n in numbers)]
            if best_missing is None or len(missing) < len(best_missing):
                best_missing = missing
        if best_missing is None:
            diffs.add("presets", f"preset {preset} is not defined")
        elif best_missing:
            diffs.add(
                "presets",
                f"preset {preset}: no table with "
                f"{', '.join(format(v, 'g') for v in best_missing)}",
            )


def read_parameter_block(text):
    """Return {name: number-or-text} of an example's parameter block, or None."""
    if BLOCK_BEGIN not in text or BLOCK_END not in text:
        return None
    body = text.split(BLOCK_BEGIN, 1)[1].split(BLOCK_END, 1)[0]
    found = {}
    for line in body.splitlines()[1:]:
        match = BLOCK_LINE.match(line)
        if not match:
            continue
        raw = match.group(2).strip()
        try:
            found[match.group(1)] = float(raw.replace("_", ""))
        except ValueError:
            found[match.group(1)] = raw
    return found


def check_examples(diffs, manifest, language, root):
    examples = manifest["examples"]
    suffix = EXAMPLE_SUFFIX[language]
    directory = pathlib.Path(root) / "examples"
    present = {p.stem: p for p in directory.glob(f"*{suffix}")} if directory.is_dir() else {}
    shared = public(examples["shared"])
    on_demand = set(examples["on_demand"].get(language, []))
    dropped = public(examples["dropped"])

    for name, spec in shared.items():
        if name not in present:
            diffs.add("examples", f"examples/{name}{suffix} is missing")
            continue
        block = read_parameter_block(present[name].read_text(encoding="utf-8"))
        if block is None:
            diffs.add("examples", f"examples/{name}{suffix} has no parameter block")
            continue
        for key, want in spec["params"].items():
            if key not in block:
                diffs.add("examples", f"{name}: parameter {key} is missing")
            elif not same_value(block[key], want):
                diffs.add("examples", f"{name}: {key} is {block[key]}, expected {want}")
        for key in block:
            if key not in spec["params"]:
                diffs.add("examples", f"{name}: parameter {key} is not in the manifest")

    for stem in sorted(present):
        if stem in shared or stem in on_demand:
            continue
        covered = dropped.get(stem)
        reason = f"covered by {covered}" if covered else "not in the shared set"
        diffs.add("examples", f"examples/{stem}{suffix} should go: {reason}")


# -- Python ----------------------------------------------------------------

def constant_value(node):
    """Value of a literal or of plain arithmetic on literals; ValueError otherwise."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left, right = constant_value(node.left), constant_value(node.right)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (left, right)):
            raise ValueError("not numeric")
        if isinstance(node.op, ast.Mult):
            return left * right
        return left + right if isinstance(node.op, ast.Add) else left - right
    return ast.literal_eval(node)


def python_literal(node):
    """(True, value) for a literal default, (False, source text) otherwise."""
    try:
        return True, constant_value(node)
    except (ValueError, SyntaxError):
        return False, ast.unparse(node)


def python_signature(node):
    """[(name, default node or None)] plus whether it takes *args/**kwargs."""
    args = node.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    params = [(a.arg, d) for a, d in zip(positional, defaults)]
    params += [(a.arg, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    return params, args.vararg is not None or args.kwarg is not None


def python_surface(root, manifest):
    diffs = Differences()
    package = pathlib.Path(root) / "nlolib"
    sources = {p.name: read(p) for p in sorted(package.glob("*.py"))}
    trees = {name: ast.parse(text) for name, text in sources.items()}

    exported, origin = [], {}
    for node in trees["__init__.py"].body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exported = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                origin[alias.asname or alias.name] = (node.module + ".py", alias.name)

    def find(name, kind):
        module, real = origin.get(name, (None, name))
        order = ([module] if module in trees else []) + [m for m in trees if m != module]
        for candidate in order:
            for node in trees[candidate].body:
                if isinstance(node, kind) and node.name == real:
                    return node
        return None

    errors = manifest["errors"]["classes"]["python"]
    types = {
        spec["names"]["python"]: spec
        for spec in public(manifest["types"]).values() if spec["names"]["python"]
    }
    want_exports = list(manifest["functions"]) + list(types) + list(errors)
    removed = set(manifest["removed"]["names"]["all"] + manifest["removed"]["names"]["python"])
    for name in want_exports:
        if name not in exported:
            diffs.add("exports", f"{name} is not in __all__")
    for name in exported:
        if name not in want_exports and name not in removed:
            diffs.add("exports", f"{name} is in __all__ but not in the manifest")

    for function in manifest["functions"]:
        node = find(function, ast.FunctionDef)
        if node is None:
            diffs.add("functions", f"{function} is not defined as a module function")
            continue
        params, variadic = python_signature(node)
        want = expected_params(manifest, "python", function)
        if variadic:
            diffs.add("functions", f"{function}: takes *args/**kwargs; name every option")
        compare_names(diffs, "functions", function, [p[0] for p in params], [p["name"] for p in want])
        found = dict(params)
        for spec in want:
            if spec["name"] not in found:
                continue
            default = found[spec["name"]]
            if spec.get("required"):
                if default is not None:
                    diffs.add("functions", f"{function}: {spec['name']} must be required")
                continue
            if default is None:
                diffs.add("functions", f"{function}: {spec['name']} needs default {spec['default']!r}")
                continue
            literal, value = python_literal(default)
            if not literal or not same_value(value, spec["default"]):
                diffs.add(
                    "functions",
                    f"{function}: {spec['name']} defaults to {value!r}, expected {spec['default']!r}",
                )

    for name, spec in types.items():
        node = find(name, ast.ClassDef)
        if node is None:
            diffs.add("types", f"{name} is not defined")
            continue
        fields = [
            (item.target.id, item.value) for item in node.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        ]
        compare_names(
            diffs, "types", name, [f[0] for f in fields],
            [f["name"] for f in spec["fields"]], what="field",
        )
        found = dict(fields)
        for field in spec["fields"]:
            if field["name"] not in found or "default" not in field:
                continue
            default = found[field["name"]]
            literal, value = python_literal(default) if default is not None else (False, "no default")
            if not literal or not same_value(value, field["default"]):
                diffs.add(
                    "types",
                    f"{name}.{field['name']} defaults to {value!r}, expected {field['default']!r}",
                )

    for name, spec in errors.items():
        node = find(name, ast.ClassDef)
        if node is None:
            diffs.add("errors", f"{name} is not defined")
        elif spec["base"] not in [ast.unparse(base) for base in node.bases]:
            diffs.add("errors", f"{name} must derive from {spec['base']}")

    check_removed(diffs, manifest, "python", set(exported), sources)
    check_enums(diffs, manifest, sources)
    check_presets(diffs, manifest, sources)
    check_examples(diffs, manifest, "python", root)
    return diffs


# -- Julia -----------------------------------------------------------------

JULIA_NAME = r'[A-Za-z_][A-Za-z0-9_!]*'
JULIA_STRUCT = re.compile(
    r'(?ms)^(?:Base\.@kwdef\s+|@kwdef\s+)?(?:mutable\s+)?struct\s+(' + JULIA_NAME
    + r')(?:\{[^}]*\})?(?:\s*<:\s*([A-Za-z_.]+))?\s*\n(.*?)^end\b'
)
JULIA_FIELD = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:::[^=\n]*)?(?:=\s*(.+?))?\s*$')


def balanced(text, start):
    """Text between the parenthesis at `start` and its match."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] in "([{":
            depth += 1
        elif text[index] in ")]}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return text[start + 1:]


def split_top(text, separator):
    """Split at `separator` outside brackets and string literals."""
    parts, depth, current, quoted = [], 0, [], False
    for char in text:
        if quoted:
            quoted = char != '"'
        elif char == '"':
            quoted = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts]


def julia_argument(text):
    """(name, default text or None, is_splat) of one Julia argument."""
    text = text.strip()
    splat = text.endswith("...")
    name_part, _, default = text.rstrip(".").partition("=")
    name = name_part.split("::")[0].strip()
    return name, (default.strip() or None), splat


def julia_methods(source, function):
    """Every method of `function` as (positional, keyword) argument lists."""
    source = re.sub(r'(?m)#.*$', '', source)
    methods = []
    pattern = re.compile(
        r'(?m)^(function\s+)?(?:[A-Za-z_][A-Za-z0-9_.]*\.)?' + re.escape(function) + r'\('
    )
    for match in pattern.finditer(source):
        signature = balanced(source, match.end() - 1)
        after = source[match.end() + len(signature) + 1:]
        if not match.group(1) and not re.match(r'[^\n=]*=(?!=)', after):
            continue
        halves = split_top(signature, ";")
        positional = [julia_argument(a) for a in split_top(halves[0], ",") if a]
        keyword = (
            [julia_argument(a) for a in split_top(halves[1], ",") if a]
            if len(halves) > 1 else []
        )
        methods.append((positional, keyword))
    return methods


def julia_literal(text):
    if text is None:
        return False, None
    if text == "nothing":
        return True, None
    if text in ("true", "false"):
        return True, text == "true"
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return True, text[1:-1]
    try:
        return True, float(text.replace("_", ""))
    except ValueError:
        pass
    try:
        return True, constant_value(ast.parse(text, mode="eval").body)
    except (ValueError, SyntaxError):
        return False, text


def julia_surface(root, manifest):
    diffs = Differences()
    src = pathlib.Path(root) / "src"
    sources = {p.name: read(p) for p in sorted(src.glob("*.jl"))}
    joined = "\n".join(sources.values())
    exported = JULIA_EXPORT.findall(sources["NLOLib.jl"])
    structs = {m.group(1): (m.group(2), m.group(3)) for m in JULIA_STRUCT.finditer(joined)}

    errors = manifest["errors"]["classes"]["julia"]
    types = {
        spec["names"]["julia"]: spec
        for spec in public(manifest["types"]).values() if spec["names"]["julia"]
    }
    want_exports = list(manifest["functions"]) + list(types) + list(errors)
    removed = set(manifest["removed"]["names"]["all"] + manifest["removed"]["names"]["julia"])
    for name in want_exports:
        if name not in exported:
            diffs.add("exports", f"{name} is not exported")
    for name in exported:
        if name not in want_exports and name not in removed:
            diffs.add("exports", f"{name} is exported but not in the manifest")

    for function in manifest["functions"]:
        methods = julia_methods(joined, function)
        if not methods:
            diffs.add("functions", f"{function} is not defined")
            continue
        want = expected_params(manifest, "julia", function)
        want_names = [p["name"] for p in want]

        def overlap(method):
            names = [a[0] for a in method[0] + method[1]]
            return len(set(names) & set(want_names)) - 0.01 * len(names)

        positional, keyword = max(methods, key=overlap)
        if len(methods) > 1:
            diffs.add("functions", f"{function}: {len(methods)} methods defined, expected one")
        if any(a[2] for a in positional + keyword):
            diffs.add("functions", f"{function}: takes a splat; name every option")
        named = [a for a in positional + keyword if a[0] and not a[2]]
        compare_names(diffs, "functions", function, [a[0] for a in named], want_names)
        is_keyword = {a[0]: a in keyword for a in named}
        defaults = {a[0]: a[1] for a in named}
        for spec in want:
            name = spec["name"]
            if name not in defaults:
                continue
            if is_keyword[name] != (spec["kind"] == "option"):
                where = "a keyword" if spec["kind"] == "option" else "positional"
                diffs.add("functions", f"{function}: {name} must be {where}")
            if spec.get("required"):
                if defaults[name] is not None:
                    diffs.add("functions", f"{function}: {name} must be required")
                continue
            literal, value = julia_literal(defaults[name])
            if not literal or not same_value(value, spec["default"]):
                diffs.add(
                    "functions",
                    f"{function}: {name} "
                    + (f"defaults to {defaults[name]}" if defaults[name] else "has no default")
                    + f", expected {spec['default']!r}",
                )

    for name, spec in types.items():
        if name not in structs:
            diffs.add("types", f"{name} is not defined")
            continue
        fields = []
        for line in structs[name][1].splitlines():
            match = JULIA_FIELD.match(re.sub(r'#.*', '', line))
            if match and match.group(1) not in ("function", "end", "new"):
                fields.append((match.group(1), match.group(2)))
        compare_names(
            diffs, "types", name, [f[0] for f in fields],
            [f["name"] for f in spec["fields"]], what="field",
        )
        found = dict(fields)
        for field in spec["fields"]:
            if field["name"] not in found or "default" not in field:
                continue
            default = found[field["name"]]
            literal, value = julia_literal(default)
            if not literal or not same_value(value, field["default"]):
                diffs.add(
                    "types",
                    f"{name}.{field['name']} "
                    + (f"defaults to {default}" if default else "has no default")
                    + f", expected {field['default']!r}",
                )

    for name, spec in errors.items():
        if name not in structs:
            diffs.add("errors", f"{name} is not defined")
        elif structs[name][0] != spec["base"]:
            diffs.add("errors", f"{name} must be a subtype of {spec['base']}")

    check_removed(diffs, manifest, "julia", set(exported), sources)
    check_enums(diffs, manifest, sources)
    check_presets(diffs, manifest, sources)
    check_examples(diffs, manifest, "julia", root)
    return diffs


# -- MATLAB ----------------------------------------------------------------

MATLAB_ERROR_ID = re.compile(QUOTE + r'(nlolib:[A-Za-z][A-Za-z0-9_]*)' + QUOTE)


def matlab_surface(root, manifest):
    diffs = Differences()
    root = pathlib.Path(root)
    package = root / "+nlolib"
    if not package.is_dir():
        raise SystemExit(f"::error::missing source directory {package}")
    sources = {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(package.rglob("*.m"))
    }
    native = [p.read_text(encoding="utf-8") for p in sorted(root.glob("mex/*.c"))]
    joined = "\n".join(sources.values())

    extras = manifest["binding_extras"]["matlab"]
    want_files = list(manifest["functions"]) + list(extras["functions"])
    present = sorted(p.stem for p in package.glob("*.m") if p.stem != "Contents")
    removed = set(manifest["removed"]["names"]["all"] + manifest["removed"]["names"]["matlab"])
    for name in want_files:
        if name not in present:
            diffs.add("exports", f"+nlolib/{name}.m is missing")
    for name in present:
        if name not in want_files:
            why = "removed" if name in removed else "not in the manifest"
            diffs.add("exports", f"+nlolib/{name}.m is public but {why}")

    signatures = public(json.loads(read(root / "resources" / "functionSignatures.json")))
    allowed_entries = {f"nlolib.{name}" for name in want_files} | set(extras["top_level"])
    for entry in signatures:
        if entry not in allowed_entries:
            diffs.add("functions", f"functionSignatures.json: {entry} is not in the manifest")
    gaps = gaps_for(manifest, "matlab")
    for function in manifest["functions"]:
        entry = signatures.get(f"nlolib.{function}")
        if entry is None:
            diffs.add("functions", f"functionSignatures.json has no nlolib.{function}")
            continue
        found = [i["name"] for i in entry.get("inputs", [])]
        for gap in sorted(gaps):
            if gap.startswith(function + ".") and gap.split(".", 1)[1] in found:
                diffs.add(
                    "functions",
                    f"{gap} is listed as a known gap but is present; remove it from known_gaps",
                )
        if function == "propagate":
            found = [n for n in found if n not in extras["propagate_options"]]
        want = [p["name"] for p in expected_params(manifest, "matlab", function)]
        compare_names(diffs, "functions", function, found, want)

    for name, spec in public(manifest["types"]).items():
        for field in spec["fields"] + spec.get("meta", []):
            if not word(field["name"]).search(joined):
                diffs.add("types", f"{name}: field {field['name']} does not occur in +nlolib")

    allowed = set(manifest["errors"]["matlab_ids"])
    used = set()
    for text in list(sources.values()) + native:
        used.update(MATLAB_ERROR_ID.findall(text))
    for identifier in sorted(used - allowed):
        diffs.add("errors", f"{identifier} is not one of the shared identifiers")
    for identifier in sorted(allowed - used):
        diffs.add("errors", f"{identifier} is never raised")

    check_removed(diffs, manifest, "matlab", None, sources)
    check_enums(diffs, manifest, sources)
    check_presets(diffs, manifest, sources)
    check_examples(diffs, manifest, "matlab", root)
    return diffs


SURFACES = {"python": python_surface, "julia": julia_surface, "matlab": matlab_surface}


def report_surface(manifest, selected, directories, summary_only):
    """Print the differences per binding and return how many there are."""
    total = 0
    for language in selected:
        diffs = SURFACES[language](directories[language], manifest)
        total += diffs.total()
        counts = ", ".join(f"{k} {len(v)}" for k, v in diffs.by_category.items())
        print(
            f"api surface: {language}: {diffs.total()} difference(s)"
            + (f" ({counts})" if counts else "")
        )
        if summary_only:
            continue
        for category, messages in diffs.by_category.items():
            for message in messages:
                print(f"  {language}: {category}: {message}")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "tests" / "api_parity.json"))
    parser.add_argument(
        "--bindings",
        default=",".join(BINDINGS),
        help="comma-separated bindings to check (default: all three)",
    )
    for language in BINDINGS:
        parser.add_argument(
            f"--{language}-dir",
            default=str(ROOT / language),
            help=f"checkout of the {language} binding repository",
        )
    parser.add_argument(
        "--surface",
        choices=("off", "report", "enforce"),
        default="report",
        help="compare the bindings with the surface manifest: report prints the "
        "differences and does not fail (default), enforce fails on any",
    )
    parser.add_argument(
        "--surface-manifest", default=str(ROOT / "tests" / "api_surface.json")
    )
    parser.add_argument(
        "--surface-summary",
        action="store_true",
        help="print only the difference counts of the surface comparison",
    )
    args = parser.parse_args()

    selected = [name for name in args.bindings.split(",") if name]
    unknown = sorted(set(selected) - set(BINDINGS))
    if unknown or not selected:
        parser.error(f"--bindings takes any of {', '.join(BINDINGS)}")

    manifest = json.loads(read(args.manifest))
    canonical = manifest["constants"]
    gaps = {k: v for k, v in manifest["known_gaps"].items() if not k.startswith("_")}

    problems = []
    checked = 0

    for language in selected:
        defined, exported = BINDINGS[language](
            pathlib.Path(getattr(args, f"{language}_dir"))
        )
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

    print(f"api parity: {len(canonical)} constants agree across {', '.join(selected)}")

    if args.surface != "off":
        directories = {
            language: pathlib.Path(getattr(args, f"{language}_dir")) for language in selected
        }
        differences = report_surface(
            json.loads(read(args.surface_manifest)), selected, directories, args.surface_summary
        )
        if differences and args.surface == "enforce":
            print(
                f"\n{differences} surface difference(s). The target surface lives in "
                f"{args.surface_manifest}.",
                file=sys.stderr,
            )
            return 1
        if differences:
            print(
                f"api surface: {differences} difference(s) reported, not enforced "
                "(--surface enforce makes them fail)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
