#!/usr/bin/env python3
"""G3 dev evaluation and frozen selection (test sealed during selection)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
G3 = REPO / "code/repro/out/table5_g3"
REF = REPO / "code/repro/out/table5_ablation/runs"
OUT = REPO / "LOGS/final_five_tables/table5"
SEEDS = [211, 212, 213]
FOLDS = {"mcp": ["fold0", "fold1"], "decomp": ["fold0"]}
CONFIGS = ["G3_p25", "G3_p50", "G3_p25_raw", "G3_p50_raw"]
REFS = {"K0": "full", "structure_only": "structure_only", "text_only": "text_only",
        "chrono_only": "chrono_only"}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(s); ss = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def dev_scores(path: Path):
    d = json.loads(path.read_text())
    rows = d.get("scores", {}).get("val")
    if rows is None:
        return None
    return [(r["label"], r["score"]) for r in rows]


def seed_means(kind, label):
    out = {}
    for seed in SEEDS:
        cells = []
        for cell in FOLDS[kind]:
            variant = label if label in CONFIGS else REFS[label]
            p = (G3 / "dev" / f"{label}_{kind}_{cell}_s{seed}.json") if label in CONFIGS else (
                REF / f"{kind}_{cell}_{variant}_s{seed}.json")
            if not p.exists() and kind == "mrt" and variant == "full":
                p = (REPO / "code/repro/out/unified_gnn/runs" /
                     f"mrt_{cell.replace('fold_', '')}_s{seed}.json")
            if not p.exists():
                continue
            rows = dev_scores(p)
            if not rows:
                continue
            y = [r[0] for r in rows]; s = [r[1] for r in rows]
            a = auroc(y, s)
            npos = sum(y)
            cells.append((a, npos * (len(y) - npos)))
        if cells:
            out[seed] = float(np.average([a for a, _ in cells],
                                         weights=[w for _, w in cells]))
    return out


def main() -> int:
    res = {"experiment_id": "table5-g3-20260919",
           "dev": {"mcp": {}, "decomp": {}},
           "selection": {}}
    for kind in ("mcp", "decomp"):
        for label in list(REFS) + CONFIGS:
            vals = seed_means(kind, label)
            if vals:
                key = label
                res["dev"][kind][key] = {
                    "per_seed": {str(k): round(v, 4) for k, v in vals.items()},
                    "mean": float(np.mean(list(vals.values())))}
    k0_mcp = res["dev"]["mcp"].get("K0", {}).get("mean")
    k0_dec = res["dev"]["decomp"].get("K0", {}).get("mean")
    struct_mcp = res["dev"]["mcp"].get("structure_only", {}).get("mean")
    for label in CONFIGS:
        m = res["dev"]["mcp"].get(label, {}).get("mean")
        d = res["dev"]["decomp"].get(label, {}).get("mean")
        if m is None or d is None or k0_mcp is None or k0_dec is None:
            res["selection"][label] = {"eligible": False, "reason": "incomplete"}
            continue
        eligible = (d >= k0_dec - 0.005
                    and m > k0_mcp
                    and (struct_mcp is None or m >= struct_mcp - 0.005))
        res["selection"][label] = {
            "mcp_dev": m, "decomp_dev": d,
            "delta_mcp_vs_K0": m - k0_mcp, "delta_decomp_vs_K0": d - k0_dec,
            "delta_mcp_vs_structure": (m - struct_mcp) if struct_mcp else None,
            "eligible": bool(eligible),
            "reason": ("ok" if eligible else
                       "fails: " + ", ".join(
                           [x for x, ok in (("decomp_regression", d >= k0_dec - 0.005),
                                            ("mcp_not_better_than_K0", m > k0_mcp),
                                            ("mcp_below_structure", (struct_mcp is None or
                                                                     m >= struct_mcp - 0.005)))
                            if not ok])),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g3_dev_results.json").write_text(json.dumps(res, indent=1))

    md = ["# G3 comprehensive candidate — dev results (test sealed)", "",
          "| Candidate | MCP dev | Δ vs K0 | Δ vs structure | Decomp dev | Δ vs K0 | eligible |",
          "|---|---|---|---|---|---|---|"]
    for label in CONFIGS:
        s = res["selection"].get(label, {})
        if "mcp_dev" not in s:
            md.append(f"| {label} | -- | -- | -- | -- | -- | no |")
            continue
        md.append(f"| {label} | {s['mcp_dev']:.4f} | {s['delta_mcp_vs_K0']:+.4f} | "
                  f"{s['delta_mcp_vs_structure']:+.4f} | {s['decomp_dev']:.4f} | "
                  f"{s['delta_decomp_vs_K0']:+.4f} | {s['eligible']} |")
    md += ["", "| Reference | MCP dev | Decomp dev |", "|---|---|---|"]
    for name in ("K0", "structure_only", "text_only", "chrono_only"):
        md.append(f"| {name} | {res['dev']['mcp'].get(name, {}).get('mean', '--')} | "
                  f"{res['dev']['decomp'].get(name, {}).get('mean', '--')} |")
    (OUT / "g3_dev_results.md").write_text("\n".join(md) + "\n")
    print("wrote g3_dev_results.json/md")
    print(json.dumps(res["selection"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
