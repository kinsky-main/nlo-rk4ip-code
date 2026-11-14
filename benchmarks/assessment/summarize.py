"""Summarise assessment JSONL rows into markdown tables.

Usage: summarize.py <rows.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict

ORDER = ["main", "snap_interp", "snap_fused", "snap_jit", "snap_jit_fused", "cuda", "cuda_nofuse", "nvrtc_off", "nvrtc_jit"]


def main() -> int:
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    cases: "OrderedDict[tuple[str, str], dict[str, dict]]" = OrderedDict()
    for r in rows:
        cases.setdefault((r["case"], r["size"]), {})[r["variant"]] = r

    variants = [v for v in ORDER if any(v in d for d in cases.values())]
    print("### Solve time, median ms (IQR) - speedup vs main\n")
    print("| Case | " + " | ".join(variants) + " |")
    print("|---" * (len(variants) + 1) + "|")
    for (c, s), d in cases.items():
        base = d.get("main", {}).get("solve_ms_median")
        cells = []
        for v in variants:
            r = d.get(v)
            if r is None:
                cells.append("-")
                continue
            m = r["solve_ms_median"]
            iqr = r["solve_ms_q3"] - r["solve_ms_q1"]
            sp = f" x{base / m:.2f}" if base and v != "main" else ""
            cells.append(f"{m:.0f} ({iqr:.0f}){sp}")
        print(f"| {c} {s} | " + " | ".join(cells) + " |")

    print("\n### Init time, ms: cold (first call in process) / median of timed calls\n")
    print("| Case | " + " | ".join(variants) + " |")
    print("|---" * (len(variants) + 1) + "|")
    for (c, s), d in cases.items():
        cells = []
        for v in variants:
            r = d.get(v)
            cells.append("-" if r is None else f"{r['init_ms_cold']:.0f} / {r['init_ms_median']:.0f}")
        print(f"| {c} {s} | " + " | ".join(cells) + " |")

    print("\n### main GPU phase breakdown (Vulkan timestamps, % of solve wall time)\n")
    print("| Case | solve ms | FFT | linear | nonlinear | RK algebra | error norm | GPU idle / host | f_op | submits |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (c, s), d in cases.items():
        r = d.get("main_ts")
        if r is None:
            continue
        p = r["phase_ms_median"]
        t = r["solve_ms_median"]
        g = sum(p.values())
        fop = p["linear"] + p["nonlinear"] + p["rk_algebra"]
        pct = lambda x: f"{100.0 * x / t:.0f} %"  # noqa: E731
        print(f"| {c} {s} | {t:.0f} | {pct(p['fft'])} | {pct(p['linear'])} | {pct(p['nonlinear'])} | "
              f"{pct(p['rk_algebra'])} | {pct(p['error_norm'])} | {pct(t - g)} | {pct(fop)} | {r['submits']:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
