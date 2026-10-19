#!/usr/bin/env python3
"""Phase C keyword-candidate dev evaluation (frozen rule; test sealed).

K candidates: K0=full (reference), K1=no_keyword_features (reference),
K2=K2_kwnorm, K3_25=K3_kwdrop25, K3_50=K3_kwdrop50.
Dev = MCP fold0/1 dev, Decomp fold0 dev, MRT calibration per fold; seeds 211-213.

Writes LOGS/final_five_tables/table5/keyword_dev_results.{json,md}.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
KW = REPO / "code/repro/out/table5_keyword"
REF = REPO / "code/repro/out/table5_ablation/runs"
ARCH = REPO / "code/repro/out/unified_gnn/runs"
OUT = REPO / "LOGS/final_five_tables/table5"
SEEDS = [211, 212, 213]
FOLDS = {"mcp": ["fold0", "fold1"], "decomp": ["fold0"],
         "mrt": [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                                       "bank_transfer_2", "expenditure_accounting",
                                       "spam_filter_update")]}
CAND_LABELS = {"K2": "K2_kwnorm", "K3_25": "K3_kwdrop25", "K3_50": "K3_kwdrop50"}
REF_VARIANTS = {"K0": "full", "K1": "no_keyword_features",
                "structure_only": "structure_only", "text_only": "text_only",
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


def dev_auroc(path: Path):
    d = json.loads(path.read_text())
    rows = d.get("scores", {}).get("val")
    if rows is None:
        return None
    y = [r["label"] for r in rows]; s = [r["score"] for r in rows]
    return auroc(y, s), len(y), sum(y)


def candidate_path(ds, label, cell, seed):
    # prefer GPU run, then local CPU dev
    p = KW / "runs" / f"{label}_{ds}_{cell}_s{seed}.json"
    if p.exists():
        return p
    if ds == "mcp":
        q = KW / "dev" / f"{label}_{cell}_s{seed}.json"
        if q.exists():
            return q
    return None


def ref_path(ds, variant, cell, seed):
    p = REF / f"{ds}_{cell}_{variant}_s{seed}.json"
    if p.exists():
        return p
    if ds == "mrt" and variant == "full":
        q = ARCH / f"mrt_{cell.replace('fold_', '')}_s{seed}.json"
        if q.exists():
            return q
    return None


def seed_means(fn) -> dict:
    out = {}
    for seed in SEEDS:
        vals = []
        for cell in FOLDS[fn["ds"]]:
            p = fn["path"](cell, seed)
            if p is None:
                continue
            r = dev_auroc(p)
            if r is None:
                continue
            a, n, npos = r
            vals.append((a, npos * (n - npos)))
        if vals:
            out[seed] = float(np.average([a for a, _ in vals],
                                         weights=[w for _, w in vals]))
    return out


def main() -> int:
    result = {"experiment_id": "table5-keyword-20260918",
              "dev_protocol": {"folds": FOLDS, "seeds": SEEDS,
                               "aggregation": "fold-weighted by n_pos*n_neg within "
                                              "seed, mean over seeds",
                               "test_read": False},
              "datasets": {}}
    coverage = {}
    for ds in ("mcp", "decomp", "mrt"):
        result["datasets"][ds] = {}
        for label, variant in REF_VARIANTS.items():
            vals = seed_means({"ds": ds, "path": lambda c, s, v=variant: ref_path(ds, v, c, s)})
            if vals:
                result["datasets"][ds][label] = {
                    "per_seed": {str(k): round(v, 4) for k, v in vals.items()},
                    "mean": float(np.mean(list(vals.values()))),
                    "source": "replication (GPU)",
                }
        for label, variant in CAND_LABELS.items():
            vals = seed_means({"ds": ds, "path": lambda c, s, v=variant: candidate_path(ds, v, c, s)})
            if vals:
                result["datasets"][ds][label] = {
                    "per_seed": {str(k): round(v, 4) for k, v in vals.items()},
                    "mean": float(np.mean(list(vals.values()))),
                    "source": "keyword experiment",
                }
        coverage[ds] = sorted(result["datasets"][ds])

    # selection rule evaluation
    base = {ds: result["datasets"][ds].get("K0", {}).get("mean") for ds in FOLDS}
    struct = {ds: result["datasets"][ds].get("structure_only", {}).get("mean") for ds in FOLDS}
    eligible = {}
    for label in ("K1", "K2", "K3_25", "K3_50"):
        regressions, gains = [], []
        ok = True
        for ds in FOLDS:
            v = result["datasets"][ds].get(label, {}).get("mean")
            b = base.get(ds)
            if v is None or b is None:
                ok = False
                continue
            if v < b - 0.01:
                ok = False
                regressions.append((ds, v - b))
            gains.append(v - b)
        means = [result["datasets"][ds][label]["mean"] for ds in FOLDS
                 if label in result["datasets"][ds]]
        beats_struct = all(
            result["datasets"][ds][label]["mean"] >= struct[ds] - 0.005
            for ds in ("mcp", "mrt")
            if struct.get(ds) is not None and label in result["datasets"][ds])
        eligible[label] = {
            "no_regression": not regressions,
            "regressions": regressions,
            "mean_delta_vs_K0": float(np.mean(gains)) if gains else None,
            "beats_structure": beats_struct,
            "eligible": bool(ok and gains and np.mean(gains) > 0 and beats_struct),
        }
    result["selection_rule"] = eligible

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "keyword_dev_results.json").write_text(json.dumps(result, indent=1, default=str))

    md = ["# Keyword candidates — dev results (test sealed)", "",
          "Dev = MCP fold0/1, Decomp fold0, MRT calibration; seeds 211/212/213; "
          "fold-weighted within seed. K0/K1 and matched references come from the "
          "frozen GPU replication; K2/K3 are new runs.", ""]
    for ds, rows in result["datasets"].items():
        md += [f"## {ds}", "",
               "| Method | per-seed | mean | vs K0 (mean) |", "|---|---|---|---|"]
        b = rows.get("K0", {}).get("mean")
        for label, d in rows.items():
            delta = f"{d['mean'] - b:+.4f}" if b is not None else "--"
            md.append(f"| {label} | {d['per_seed']} | {d['mean']:.4f} | {delta} |")
        md.append("")
    md += ["## Frozen selection rule", "",
           "| Candidate | no regression >0.01 | mean delta vs K0 | matches/beats structure on MCP+MRT | eligible |",
           "|---|---|---|---|---|"]
    for label, v in eligible.items():
        delta = ("--" if v["mean_delta_vs_K0"] is None
                 else f"{v['mean_delta_vs_K0']:+.4f}")
        md.append(f"| {label} | {v['no_regression']} | {delta} | "
                  f"{v['beats_structure']} | {v['eligible']} |")
    md += ["", f"coverage: {coverage}"]
    (OUT / "keyword_dev_results.md").write_text("\n".join(md) + "\n")
    print("wrote keyword_dev_results.json/md")
    print(json.dumps(result["datasets"], indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
