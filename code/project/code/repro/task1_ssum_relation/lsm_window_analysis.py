#!/usr/bin/env python3
"""Zero-API LSM window analysis on Decomp + MCP (protocol A re-aggregation).

Recomputes stored Sequential Monitor per-prefix probabilities under the two
window conventions (local prefix-1 vs official prefix-2) with identical
aggregation rules, and emits per-task trajectory scores. No model training and
no API calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protocol_a import (align_rows, graph_index, ranking_metrics,
                        trajectory_scores)


def load_graphs(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def op_points(labels, scores):
    out = {}
    for thr in (0.5, 0.9):
        pred = [s >= thr for s in scores]
        pos = sum(labels)
        neg = len(labels) - pos
        tp = sum(1 for y, p in zip(labels, pred) if y == 1 and p)
        fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p)
        out[f"at_{thr}"] = {"tp": tp, "fp": fp, "pos": pos, "neg": neg,
                            "tpr": tp / max(pos, 1), "fpr": fp / max(neg, 1)}
    return out


def analyse(name, graph_path, lsm_path, out):
    graphs = load_graphs(graph_path)
    labels, deadlines = graph_index(graphs)
    lsm = json.load(open(lsm_path))
    probs = lsm["per_task_probs"]
    report = {"corpus": name, "graph_path": graph_path, "lsm_path": lsm_path,
              "n_graphs": len(graphs), "n_lsm_tasks": len(probs),
              "overlap": len(set(labels) & set(probs)), "windows": {}}
    for tag, mi in (("local_prefix1", 0), ("official_prefix2", 1)):
        l, s, tids = align_rows(probs, labels, deadlines, min_index=mi)
        report["windows"][tag] = {
            "ranking": ranking_metrics(l, s),
            "operating_points": op_points(l, s),
        }
        report.setdefault("per_task", {})[tag] = {
            tid: score for tid, score in zip(tids, s)}
    report["delta_auroc_official_minus_local"] = (
        report["windows"]["official_prefix2"]["ranking"]["auroc"] -
        report["windows"]["local_prefix1"]["ranking"]["auroc"])
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--decomp-graphs", default="code/repro/out/unified_gnn/fix2_splits/decomp/fold0/test.jsonl")
    ap.add_argument("--decomp-lsm", default="LOGS/lsm_decomp_fix2_cumulative_20260916.json")
    ap.add_argument("--mcp", action="store_true")
    args = ap.parse_args()
    reports = [analyse("decomp", args.decomp_graphs, args.decomp_lsm, args.out)]
    if args.mcp:
        mcp_graphs = []
        for fold in range(5):
            for split in ("train", "dev", "test"):
                p = Path(f"code/repro/out/unified_gnn/fix2_splits/mcp/fold{fold}/{split}.jsonl")
                if p.exists():
                    mcp_graphs.extend(load_graphs(p))
        tmp = Path(args.out).parent / "_mcp_all_graphs.jsonl"
        with open(tmp, "w") as f:
            for g in mcp_graphs:
                f.write(json.dumps(g) + "\n")
        reports.append(analyse("mcp_all232", str(tmp),
                               "LOGS/lsm_infomatch_mcp_cumulative_20260917.json",
                               args.out))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"reports": reports}, f, indent=1, sort_keys=True, default=float)
    for r in reports:
        print(r["corpus"], {
            tag: (round(w["ranking"]["auroc"], 4),
                  w["operating_points"]["at_0.5"]["tpr"],
                  w["operating_points"]["at_0.5"]["fpr"])
            for tag, w in r["windows"].items()},
            "delta=%.4f" % r["delta_auroc_official_minus_local"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
