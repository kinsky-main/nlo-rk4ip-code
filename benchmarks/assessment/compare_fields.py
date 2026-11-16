"""Compare saved final fields of each variant against a reference variant.

Usage: compare_fields.py <field_dir> [reference_variant]
Prints relative L2 error of the field and of the intensity, per case.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    field_dir = Path(sys.argv[1])
    ref = sys.argv[2] if len(sys.argv) > 2 else "main"
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in sorted(field_dir.glob("*.npy")):
        if f.stem.endswith("__steps"):
            continue
        variant, rest = f.stem.split("__", 1)
        groups[rest][variant] = f
    print("case,variant,rel_l2_field,rel_l2_intensity,steps,ref_steps,max_step_rel_diff")
    for case, variants in groups.items():
        if ref not in variants:
            continue
        a_ref = np.load(variants[ref])
        steps_ref_path = variants[ref].with_name(variants[ref].stem + "__steps.npy")
        steps_ref = np.load(steps_ref_path) if steps_ref_path.exists() else None
        for variant, path in variants.items():
            if variant == ref:
                continue
            a = np.load(path)
            e_field = float(np.linalg.norm(a - a_ref) / np.linalg.norm(a_ref))
            i, i_ref = np.abs(a) ** 2, np.abs(a_ref) ** 2
            e_int = float(np.linalg.norm(i - i_ref) / np.linalg.norm(i_ref))
            steps_path = path.with_name(path.stem + "__steps.npy")
            s_cnt = s_ref_cnt = diff = ""
            if steps_ref is not None and steps_path.exists():
                s = np.load(steps_path)
                s_cnt, s_ref_cnt = str(s.size), str(steps_ref.size)
                k = min(s.size, steps_ref.size)
                if k:
                    diff = f"{float(np.max(np.abs(s[:k] - steps_ref[:k]) / np.abs(steps_ref[:k]))):.3e}"
            print(f"{case},{variant},{e_field:.3e},{e_int:.3e},{s_cnt},{s_ref_cnt},{diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
