#!/usr/bin/env python3
"""W1 fair comparison aggregation: AUROC, paired cluster bootstrap, TPR/FPR.

Reads whatever training runs exist under LOGS/w1_fair_comparison/runs/ and the
monitor reuse summary, then writes predictions + results.md + paper_tables.tex.
Partial runs are tolerated; re-run after the queue advances.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "LOGS/w1_fair_comparison"
TEST = OUT / "data/obs1/test.jsonl"
ARMS = ("GRAD", "TXT", "TEMP")
BOOT = 5000
RNG = np.random.default_rng(20260925)


def load_run(dirp: Path) -> dict | None:
    cands = sorted(dirp.glob("gnn_v2_results_*.json")) + sorted(dirp.glob("phase2_*.json"))
    if not cands:
        return None
    return json.loads(cands[-1].read_text())


def extract(run: dict, arm: str) -> dict:
    scores = [r["score"] for r in run["scores"]["test"]]
    labels = [int(r["label"]) for r in run["scores"]["test"]]
    if arm == "GRAD":
        thr = float(run["test"]["dev_selected_threshold"])
        params = int(run["candidate_meta"]["params"])
    else:
        thr = float(run["threshold_reports"]["fpr_10"]["threshold"])
        params = int(run["ablation"]["model_params"])
    return {"scores": np.array(scores, dtype=float), "labels": np.array(labels),
            "threshold": thr, "params": params}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


def cluster_bootstrap_delta(y, sa, sb, groups, n=BOOT, rng=None) -> tuple[float, float, float]:
    rng = rng or RNG
    gids = sorted(set(groups.tolist()))
    idx_by_g = {g: np.where(groups == g)[0] for g in gids}
    deltas = []
    for _ in range(n):
        pick = rng.choice(len(gids), size=len(gids), replace=True)
        idx = np.concatenate([idx_by_g[gids[i]] for i in pick])
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        deltas.append(auroc(yy, sa[idx]) - auroc(yy, sb[idx]))
    d = auroc(y, sa) - auroc(y, sb)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(OUT / "runs"))
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)

    rows = [json.loads(l) for l in open(TEST)]
    n_test = len(rows)
    y_ref = np.array([int(r["label"]) for r in rows])
    groups = np.array([str(r["group"]) for r in rows])
    task_ids = [r["task_id"] for r in rows]

    per_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    for dirp in sorted(runs_dir.glob(f"*_s*")):
        if not dirp.is_dir():
            continue
        arm, seed = dirp.name.rsplit("_s", 1)
        if arm not in ARMS or not seed.isdigit():
            continue
        run = load_run(dirp)
        if run is None:
            continue
        e = extract(run, arm)
        if len(e["scores"]) != n_test:
            print(f"skip {dirp.name}: n={len(e['scores'])} != {n_test}")
            continue
        if not np.array_equal(e["labels"], y_ref):
            print(f"skip {dirp.name}: label order mismatch")
            continue
        e["dir"] = dirp.name
        per_arm[arm][int(seed)] = e

    OUT.joinpath("predictions").mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "predictions/per_example_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "row", "task_id", "group", "label", "score", "threshold"])
        for arm in ARMS:
            for seed, e in sorted(per_arm[arm].items()):
                for i, (s, l) in enumerate(zip(e["scores"], e["labels"])):
                    w.writerow([arm, seed, i, task_ids[i], groups[i], int(l),
                                f"{s:.8f}", f"{e['threshold']:.8f}"])

    summary: dict = {"n_test": n_test, "n_pos": int(y_ref.sum()),
                     "n_groups": len(set(groups.tolist())), "arms": {}}
    for arm in ARMS:
        seeds = sorted(per_arm[arm])
        if not seeds:
            continue
        aucs = [auroc(y_ref, per_arm[arm][s]["scores"]) for s in seeds]
        summary["arms"][arm] = {
            "seeds": seeds,
            "auroc_per_seed": aucs,
            "auroc_mean": float(np.mean(aucs)),
            "auroc_std": float(np.std(aucs)),
            "params": per_arm[arm][seeds[0]]["params"],
        }
        tprs, fprs, thrs = [], [], []
        for s in seeds:
            e = per_arm[arm][s]
            pred = (e["scores"] >= e["threshold"]).astype(int)
            tp = int(((y_ref == 1) & (pred == 1)).sum())
            fn = int(((y_ref == 1) & (pred == 0)).sum())
            fp = int(((y_ref == 0) & (pred == 1)).sum())
            tn = int(((y_ref == 0) & (pred == 0)).sum())
            tprs.append(tp / max(tp + fn, 1))
            fprs.append(fp / max(fp + tn, 1))
            thrs.append(e["threshold"])
        summary["arms"][arm]["tpr_at_dev_threshold"] = tprs
        summary["arms"][arm]["fpr_at_dev_threshold"] = fprs
        summary["arms"][arm]["dev_threshold"] = thrs
        if len(seeds) == 3:
            mean_scores = np.mean([per_arm[arm][s]["scores"] for s in seeds], axis=0)
            per_arm[arm]["mean"] = {"scores": mean_scores, "labels": y_ref,
                                    "threshold": None, "params": None}

    pairs = [("GRAD", "TXT"), ("GRAD", "TEMP"), ("TEMP", "TXT")]
    summary["paired"] = {}
    for a, b in pairs:
        if a not in per_arm or b not in per_arm:
            continue
        common = sorted(set(per_arm[a]) & set(per_arm[b]) - {"mean"})
        entry = {"per_seed": []}
        for s in common:
            d, lo, hi = cluster_bootstrap_delta(y_ref, per_arm[a][s]["scores"],
                                                per_arm[b][s]["scores"], groups)
            entry["per_seed"].append({"seed": s, "delta": d, "ci_lo": lo, "ci_hi": hi,
                                      "excludes_zero": bool(lo > 0 or hi < 0)})
        if "mean" in per_arm[a] and "mean" in per_arm[b]:
            d, lo, hi = cluster_bootstrap_delta(y_ref, per_arm[a]["mean"]["scores"],
                                                per_arm[b]["mean"]["scores"], groups)
            entry["seed_mean"] = {"delta": d, "ci_lo": lo, "ci_hi": hi,
                                  "excludes_zero": bool(lo > 0 or hi < 0)}
        if entry["per_seed"]:
            entry["delta_mean"] = float(np.mean([e["delta"] for e in entry["per_seed"]]))
            entry["seeds_ci_excl_0"] = sum(1 for e in entry["per_seed"] if e["excludes_zero"])
        summary["paired"][f"{a}-{b}"] = entry

    lens = np.array([json.loads(l)["num_nodes"] for l in open(TEST)])
    summary["length_only_auroc"] = auroc(y_ref, lens.astype(float))

    mon_path = OUT / "predictions/monitor_summary.json"
    if mon_path.exists():
        summary["monitor"] = json.loads(mon_path.read_text())
        mon_rows = [json.loads(l) for l in open(OUT / "predictions/monitor_test.jsonl")]
        assert [r["task_id"] for r in mon_rows] == task_ids
        summary["monitor"]["tp_at_decision"] = sum(
            1 for r in mon_rows if r["label"] == 1 and r["detected"])
        summary["monitor"]["fn"] = sum(1 for r in mon_rows if r["label"] == 1 and not r["detected"])

    (OUT / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
