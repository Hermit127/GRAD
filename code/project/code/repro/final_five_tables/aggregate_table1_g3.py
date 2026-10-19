#!/usr/bin/env python3
"""Re-aggregate Table 1 (detection) with G3_p25, from raw results only.

Reads (no training, no API):
  - old batch: code/repro/out/unified_gnn/runs/*.json          (57-run sweep)
  - paired batch: code/repro/out/table5_ablation/runs/*.json   (K0 full per dataset)
  - G3 batch: code/repro/out/table5_g3/confirm/G3_p25_*.json
  - old main table cells/provenance: LOGS/final_five_tables/raw_results/table1.json

Aggregation rule (identical to the archived unified aggregator, explicitly):
  per run: weight = n_pos * n_neg;
  per seed: weighted mean over folds;
  across seeds: mean and SAMPLE SD (n-1);
  operating point: dev `threshold_reports.fpr_10` threshold (or the ablate2
  trainer's val dev_selected_threshold) applied to test scores, confusion counts
  pooled over runs before recomputing TPR/FPR.

Outputs: LOGS/final_five_tables/table1_g3/{table1_g3.json,table1_g3.tex,
         table1_g3_alignment.md, missing_runs.md}
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OLD = REPO / "code/repro/out/unified_gnn/runs"
PAIRED = REPO / "code/repro/out/table5_ablation/runs"
G3 = REPO / "code/repro/out/table5_g3/confirm"
OLD_TABLE = REPO / "LOGS/final_five_tables/raw_results/table1.json"
OUT = REPO / "LOGS/final_five_tables/table1_g3"
SEEDS = (211, 212, 213)
BENCHES = {"Decomp": "decomp", "MCP": "mcp", "MRT": "mrt",
           "MG": "mg", "MINJA": "minja", "InjecAgent": "injecagent",
           "ASB": "asb", "AgentHarm": "agentharm"}
DEV_THRESHOLD_RULE = "dev FPR<=10% threshold from the run's threshold_reports.fpr_10"


def auroc(y, s):
    import numpy as np
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


def run_metrics(d: dict) -> dict:
    """AUROC from raw test scores + dev-selected operating point when available."""
    rows = (d.get("scores") or {}).get("test", [])
    y = [r["label"] for r in rows]; s = [r["score"] for r in rows]
    npos = sum(y); nneg = len(y) - npos
    out = {"auroc": auroc(y, s), "n_pos": npos, "n_neg": nneg,
           "weight": npos * nneg, "n": len(y), "threshold": None, "confusion": None,
           "auroc_trainer_reported": (d.get("test_graph") or {}).get("auroc")}
    thr = None
    if "threshold_reports" in d:                       # frozen trainer
        rep = (d.get("threshold_reports") or {}).get("fpr_10") or {}
        thr = rep.get("threshold")
    elif isinstance(d.get("val"), dict):               # ablate2 trainer
        thr = d["val"].get("dev_selected_threshold")
    if thr is not None:
        tp = sum(1 for r in rows if r["score"] >= thr and r["label"] == 1)
        fp = sum(1 for r in rows if r["score"] >= thr and r["label"] == 0)
        fn = sum(1 for r in rows if r["score"] < thr and r["label"] == 1)
        tn = sum(1 for r in rows if r["score"] < thr and r["label"] == 0)
        out.update({"threshold": float(thr),
                    "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}})
    return out


def collect(folder: Path, pattern: str, ident) -> dict:
    """{(bench, fold, seed): metrics}."""
    out = {}
    for p in sorted(folder.glob(pattern)):
        d = json.loads(p.read_text())
        key = ident(p.name, d)
        if key is None:
            continue
        out[key] = {**run_metrics(d), "file": str(p.relative_to(REPO))}
    return out


def agg_cells(runs: dict, bench: str) -> dict | None:
    per_seed = defaultdict(list)
    for (b, fold, seed), m in runs.items():
        if b == bench and seed in SEEDS:
            per_seed[seed].append(m)
    if not per_seed:
        return None
    seed_means = {s: sum(m["auroc"] * m["weight"] for m in ms) /
                  sum(m["weight"] for m in ms) for s, ms in per_seed.items()}
    vals = list(seed_means.values())
    seed_means_trainer = {}
    for s_, ms in per_seed.items():
        reps = [m["auroc_trainer_reported"] for m in ms
                if m.get("auroc_trainer_reported") is not None]
        if reps:
            seed_means_trainer[s_] = sum(
                m["auroc_trainer_reported"] * m["weight"] for m in ms
                if m.get("auroc_trainer_reported") is not None) / sum(
                m["weight"] for m in ms if m.get("auroc_trainer_reported") is not None)
    counts = [m["confusion"] for ms in per_seed.values() for m in ms
              if m.get("confusion")]
    pooled = None
    if counts:
        tp = sum(c["tp"] for c in counts); fp = sum(c["fp"] for c in counts)
        tn = sum(c["tn"] for c in counts); fn = sum(c["fn"] for c in counts)
        pooled = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                  "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)}
    return {
        "auroc_mean": st.mean(vals),
        "auroc_sd_sample": st.stdev(vals) if len(vals) > 1 else 0.0,
        "auroc_mean_trainer_reported": (st.mean(seed_means_trainer.values())
                                        if len(seed_means_trainer) == len(vals) else None),
        "per_seed": {str(k): round(v, 6) for k, v in seed_means.items()},
        "n_seeds": len(vals),
        "confusion_pooled": pooled,
        "threshold_rule": DEV_THRESHOLD_RULE,
        "units": {f"{b}_f{fold}_s{seed}": {"auroc": round(m["auroc"], 6),
                                           "threshold": m["threshold"],
                                           "file": m["file"]}
                  for (b, fold, seed), m in sorted(runs.items())
                  if b == bench and seed in SEEDS},
        "aggregation": "fold-weighted by n_pos*n_neg within seed; mean/SAMPLE SD "
                       "(n-1) across seeds 211-213 (archived convention)",
    }


def main() -> int:
    # old batch identity: names are decomp_f0_s211 / mcp_f0_s211 / mrt_<task>_s211
    def old_ident(name, d):
        stem = name.replace(".json", "")
        parts = stem.rsplit("_s", 1)
        if len(parts) != 2:
            return None
        seed = int(parts[1])
        base = parts[0]
        if base.startswith("decomp_f"):
            return ("decomp", base.replace("decomp_f", ""), seed)
        if base.startswith("mcp_f"):
            return ("mcp", base.replace("mcp_f", ""), seed)
        if base.startswith("mrt_"):
            return ("mrt", base.replace("mrt_", ""), seed)
        if base.startswith("mg_f"):
            return ("mg", base.replace("mg_f", ""), seed)
        if base.startswith("minja_f"):
            return ("minja", base.replace("minja_f", ""), seed)
        return None

    # paired batch names: <bench>_fold<N>_full_s<seed> / mrt_fold_<task>_full_s<seed>
    def paired_ident(name, d):
        stem = name.replace(".json", "")
        parts = stem.rsplit("_s", 1)
        if len(parts) != 2 or not stem.endswith("_full_s" + parts[1]):
            return None
        seed = int(parts[1])
        base = stem.rsplit("_full_s", 1)[0]
        for b in ("decomp", "mcp", "mrt"):
            if base.startswith(b + "_"):
                fold = base[len(b) + 1:]
                if fold.startswith("fold") and b != "mrt":
                    fold = fold.replace("fold", "")
                return (b, fold, seed)
        return None

    # G3 batch names: G3_p25_<bench>_fold<N|fold_task>_s<seed>
    def g3_ident(name, d):
        stem = name.replace(".json", "")
        if not stem.startswith("G3_p25_"):
            return None
        parts = stem.rsplit("_s", 1)
        seed = int(parts[1])
        base = parts[0][len("G3_p25_"):]
        for b in ("decomp", "mcp", "mrt"):
            if base.startswith(b + "_"):
                fold = base[len(b) + 1:]
                if b != "mrt" and fold.startswith("fold"):
                    fold = fold.replace("fold", "")
                return (b, fold, seed)
        return None

    old_runs = collect(OLD, "*.json", old_ident)
    old_runs = {(b, f, s): m for (b, f, s), m in old_runs.items() if s != 42}
    paired_runs = collect(PAIRED, "*.json", paired_ident)
    g3_runs = collect(G3, "G3_p25_*.json", g3_ident)

    old_cells = {col: agg_cells(old_runs, b) for col, b in BENCHES.items()}
    paired_cells = {col: agg_cells(paired_runs, b) for col, b in BENCHES.items()}
    g3_cells = {col: agg_cells(g3_runs, b) for col, b in BENCHES.items()}

    # sanity: reproduce the archived values
    archived = json.loads(OLD_TABLE.read_text())
    repro = {}
    for col in ("Decomp", "MCP", "MRT"):
        ar = archived["rows"]["GRAD"]["cells"][col]
        want = ar["auroc"]["mean"]
        got = (old_cells[col] or {}).get("auroc_mean")
        repro[col] = {"archived": want, "recomputed": got,
                      "delta": None if got is None else got - want}

    out = {
        "table": "table1_g3_detection",
        "batches": {
            "old_archived": {"path": "code/repro/out/unified_gnn/runs",
                             "note": "57-run original sweep; source of the old Table 1 GRAD row"},
            "paired_K0": {"path": "code/repro/out/table5_ablation/runs",
                          "note": "same-protocol re-run batch, pairs with G3"},
            "g3": {"path": "code/repro/out/table5_g3/confirm",
                   "note": "frozen G3_p25 runs (test scores + dev threshold)"},
        },
        "reproduction_check": repro,
        "cells": {"old_archived": old_cells, "paired_K0": paired_cells, "g3": g3_cells},
        "baselines_unchanged": {k: archived["rows"][k]["cells"]
                                for k in archived["rows"] if k != "GRAD"},
        "protocol": {
            "aggregation": "fold-weighted n_pos*n_neg within seed; mean/sample SD "
                           "(n-1) across seeds 211-213",
            "operating_point": DEV_THRESHOLD_RULE + "; counts pooled over runs; "
                               "no test-selected thresholds",
            "splits": "code/repro/out/unified_gnn/fix2_splits (unchanged since "
                      "2026-09-15 15:27; sha256 in table1_g3_alignment.md)",
            "preprocessing": "train_gnn_v2.py / train_gnn_ablate.py; only diff in git "
                             "history between batches is the --out-dir plumbing",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table1_g3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(repro, indent=1))
    print("old:", {c: (old_cells[c] or {}).get("auroc_mean") for c in BENCHES})
    print("paired:", {c: (paired_cells[c] or {}).get("auroc_mean") for c in BENCHES})
    print("g3:", {c: (g3_cells[c] or {}).get("auroc_mean") for c in BENCHES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
