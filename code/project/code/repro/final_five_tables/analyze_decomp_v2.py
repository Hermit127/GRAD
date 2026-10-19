#!/usr/bin/env python3
"""Decomp v2 one-shot confirmation: frozen methods on never-converted data."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
V2 = REPO / "code/repro/out/table5_decomp_v2"
OUT = REPO / "LOGS/final_five_tables/table5"
SEEDS = [211, 212, 213]
METHODS = ["K0_full", "G3_p25", "structure_only", "S3_reg"]

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

def pr_auc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))

def group_of(tid: str) -> str:
    return tid.split("_mix")[0].rsplit("_r", 1)[0]

def load(method, seed):
    p = V2 / "runs" / f"{method}_s{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None

def main() -> int:
    graphs = [json.loads(l) for l in (V2 / "test.jsonl").read_text().splitlines() if l.strip()]
    res = {"data": {"n_graphs": len(graphs), "sha256_manifest": json.loads(
        (V2 / "manifest.json").read_text())["sha256"]}, "methods": {}, "clustered": {}}
    rows_by_method = {}
    for method in METHODS:
        per_seed = {}
        for seed in SEEDS:
            d = load(method, seed)
            if d is None:
                continue
            rows = d["scores"]["test"]
            assert len(rows) == len(graphs)
            y = [r["label"] for r in rows]; s = [r["score"] for r in rows]
            if "test" in d and isinstance(d["test"], dict):   # ablate2 trainer
                tpr = d["test"]["metrics_at_threshold"].get("recall")
                fpr = d["test"]["metrics_at_threshold"].get("fpr")
            else:                                             # frozen trainer
                tpr = d["threshold_reports"]["fpr_10"]["test"]["recall"]
                fpr = d["threshold_reports"]["fpr_10"]["test"]["fpr"]
            per_seed[seed] = {"auroc": auroc(y, s), "pr_auc": pr_auc(y, s),
                              "dev_tpr": tpr, "dev_fpr": fpr}
            rows_by_method.setdefault(method, {})[seed] = rows
        if per_seed:
            res["methods"][method] = {
                "per_seed": {str(k): {kk: round(vv, 4) for kk, vv in v.items()}
                             for k, v in per_seed.items()},
                "auroc_mean": float(np.mean([v["auroc"] for v in per_seed.values()])),
                "auroc_std": float(np.std([v["auroc"] for v in per_seed.values()])),
                "pr_auc_mean": float(np.mean([v["pr_auc"] for v in per_seed.values()])),
                "dev_tpr_mean": float(np.mean([v["dev_tpr"] for v in per_seed.values()])),
                "dev_fpr_mean": float(np.mean([v["dev_fpr"] for v in per_seed.values()])),
            }
    # paired comparisons with group-clustered bootstrap (334 base-id groups)
    def clustered(a, b):
        per_seed = {}
        for seed in SEEDS:
            ra, rb = rows_by_method.get(a, {}).get(seed), rows_by_method.get(b, {}).get(seed)
            if ra is None or rb is None:
                continue
            y = np.array([r["label"] for r in ra])
            sa = np.array([r["score"] for r in ra]); sb = np.array([r["score"] for r in rb])
            gid = np.array([group_of(g["task_id"]) for g in graphs])
            groups = defaultdict(list)
            for i, t in enumerate(gid):
                groups[t].append(i)
            gl = sorted(groups)
            rng = np.random.default_rng(seed)
            deltas = []
            for _ in range(10000):
                pick = rng.choice(len(gl), len(gl), replace=True)
                idx = [i for gi in pick for i in groups[gl[gi]]]
                yy = y[idx]
                if yy.sum() in (0, len(yy)):
                    continue
                deltas.append(auroc(yy, sa[idx]) - auroc(yy, sb[idx]))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            per_seed[seed] = {"delta": float(auroc(y, sa) - auroc(y, sb)),
                              "ci95": [float(lo), float(hi)], "n_groups": len(gl)}
        return per_seed
    for a, b in (("G3_p25", "K0_full"), ("G3_p25", "structure_only"),
                 ("structure_only", "K0_full"), ("S3_reg", "structure_only"),
                 ("S3_reg", "G3_p25"), ("S3_reg", "K0_full")):
        res["clustered"][f"{a}_minus_{b}"] = clustered(a, b)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decomp_v2_confirmation.json").write_text(json.dumps(res, indent=1))
    md = ["# Decomp v2 one-shot confirmation (never-converted data)", "",
          f"n_graphs={res['data']['n_graphs']}; corpus sha256 "
          f"`{res['data']['sha256_manifest'][:16]}…`; methods frozen, test read once.", "",
          "| Method | AUROC (mean±std over 3 seeds) | PR-AUC | dev TPR@FPR10 | dev FPR@FPR10 | per-seed AUROC |",
          "|---|---|---|---|---|---|"]
    for m in METHODS:
        r = res["methods"].get(m)
        if not r:
            md.append(f"| {m} | -- | -- | -- | -- | -- |")
            continue
        md.append(f"| {m} | {r['auroc_mean']:.4f}±{r['auroc_std']:.4f} | "
                  f"{r['pr_auc_mean']:.4f} | {r['dev_tpr_mean']:.3f} | "
                  f"{r['dev_fpr_mean']:.3f} | "
                  f"{[round(v['auroc'],4) for v in r['per_seed'].values()]} |")
    md += ["", "## Paired differences (group-clustered by base id, 334 groups)", "",
           "| Pair | seed | ΔAUROC | 95% CI |", "|---|---|---|---|"]
    for k, v in res["clustered"].items():
        for s, d in v.items():
            md.append(f"| {k} | {s} | {d['delta']:+.4f} | "
                      f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    (OUT / "decomp_v2_confirmation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
