from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "python"


def _iter_top_level_example_files() -> list[Path]:
    files = sorted(EXAMPLES_DIR.glob("*.py"))
    out: list[Path] = []
    for path in files:
        if path.name.startswith("_"):
            continue
        out.append(path)
    return out


def _is_example_app_base_subclass(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "ExampleAppBase":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "ExampleAppBase":
            return True
    return False


def test_entrypoints_define_oop_app_and_main() -> None:
    example_files = _iter_top_level_example_files()
    if len(example_files) <= 0:
        raise AssertionError("no example entrypoint files found")

    missing_main: list[str] = []
    missing_app_class: list[str] = []
    for path in example_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        has_main = any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
        has_app = any(
            isinstance(node, ast.ClassDef) and _is_example_app_base_subclass(node)
            for node in tree.body
        )
        if not has_main:
            missing_main.append(path.name)
        if not has_app:
            missing_app_class.append(path.name)

    if missing_main:
        raise AssertionError(f"example files missing main(): {missing_main}")
    if missing_app_class:
        raise AssertionError(f"example files missing ExampleAppBase subclass: {missing_app_class}")


def main() -> None:
    test_entrypoints_define_oop_app_and_main()
    print("test_python_example_oop_structure: all checks passed.")


if __name__ == "__main__":
    main()
