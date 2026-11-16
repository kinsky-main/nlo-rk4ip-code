"""Derive main's NLO_-prefix rename map (commit 2669906) and apply it to a tree.

derive: python rename_map.py derive <repo> <out.json>
apply:  python rename_map.py apply <map.json> <tree_root>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

TOKEN = re.compile(r"\b(?:nlo_|NLO_)[A-Za-z0-9_]+\b")
EXTS = {".c", ".h", ".cu", ".in", ".txt", ".cmake", ".py", ".jl", ".comp", ".glslinc", ".m", ".md"}


def tokens_at(repo, rev):
    files = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", rev],
                           capture_output=True, text=True, check=True).stdout.split()
    toks = set()
    text_all = []
    for f in files:
        if Path(f).suffix not in EXTS and not f.endswith("CMakeLists.txt"):
            continue
        out = subprocess.run(["git", "-C", repo, "show", f"{rev}:{f}"], capture_output=True)
        t = out.stdout.decode("utf-8", "replace")
        text_all.append(t)
        toks.update(TOKEN.findall(t))
    words = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", "\n".join(text_all)))
    return toks, words


def derive(repo, out):
    before, _ = tokens_at(repo, "2669906^")
    after_toks, after_words = tokens_at(repo, "2669906")
    mapping = {}
    for tok in sorted(before - after_toks):
        stripped = tok[4:]
        if stripped in after_words:
            mapping[tok] = stripped
    Path(out).write_text(json.dumps(mapping, indent=0), encoding="utf-8")
    print(f"{len(mapping)} renames")


def apply(map_path, root):
    mapping = json.loads(Path(map_path).read_text(encoding="utf-8"))
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r")\b")
    n = 0
    for p in Path(root).rglob("*"):
        if ".git" in p.parts or not p.is_file():
            continue
        if p.suffix not in EXTS and p.name != "CMakeLists.txt":
            continue
        if any(part.startswith("build") or part.startswith("dist") for part in p.relative_to(root).parts):
            continue
        s = p.read_bytes().decode("utf-8", "surrogateescape")
        t = pat.sub(lambda m: mapping[m.group(1)], s)
        if t != s:
            p.write_bytes(t.encode("utf-8", "surrogateescape"))
            n += 1
    print(f"rewrote {n} files")


if __name__ == "__main__":
    if sys.argv[1] == "derive":
        derive(sys.argv[2], sys.argv[3])
    else:
        apply(sys.argv[2], sys.argv[3])
