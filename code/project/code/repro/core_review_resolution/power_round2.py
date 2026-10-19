#!/usr/bin/env python3
"""A-line round 2: pre-scoring power estimate for the PRIMARY endpoint.

Primary endpoint = seed-mean AUROC difference GRAD - Temporal, paired by base
case (same resample for every arm and seed). Estimated ONLY from W1 test data
(58 base cases), before any round-2 generation or scoring:

  outer: draw `n_groups` base cases with replacement from the 58 W1 groups;
         compute the seed-mean delta on the simulated set;
  inner: cluster-bootstrap the drawn cases (300 draws) for a 95% percentile CI;
  power: fraction of outer draws whose inner CI excludes 0.

Also reports the mean CI half-width (MDE scale) and the observed W1 delta.

Usage:
  python power_round2.py --out LOGS/core_review_resolution/power_round2.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PRED = REPO / "LOGS/w1_final_confirmation/per_example_predictions.csv"
SEEDS = ("211", "212", "213")
ARMS = ("GRAD", "TEMP")


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    b = np.concatenate([[0], np.flatnonzero(np.diff(ss)) + 1, [len(ss)]])
    for a, bb in zip(b[:-1], b[1:]):
        if bb - a > 1:
            ranks[order[a:bb]] = (a + bb - 1) / 2 + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def seed_mean_delta(y, idx, scores):
    d = []
    for seed in SEEDS:
        d.append(auroc(y[idx], scores[("GRAD", seed)][idx])
                 - auroc(y[idx], scores[("TEMP", seed)][idx]))
    return float(np.mean(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "LOGS/core_review_resolution/power_round2.json"))
    ap.add_argument("--groups", type=int, default=60)
    ap.add_argument("--outer", type=int, default=500)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(PRED)) if r["set"] == "w4_old_test"]
    task_ids = sorted(set(r["task_id"] for r in rows))
    tid_idx = {t: i for i, t in enumerate(task_ids)}
    labels = {}
    scores = {}
    for r in rows:
        tid = r["task_id"]
        labels[tid] = int(r["label"])
        scores[(r["arm"], r["seed"])] = scores.get((r["arm"], r["seed"]), {})
        scores[(r["arm"], r["seed"])][tid] = float(r["score"])
    y_all = np.array([labels[t] for t in task_ids])
    S = {k: np.array([v[t] for t in task_ids]) for k, v in scores.items()}

    meta = [json.loads(l) for l in
            open(REPO / "LOGS/w1_fair_comparison/data/obs1/test.w1meta.jsonl")]
    group_of = {m["task_id"]: str(m["group"]) for m in meta}
    gids = sorted(set(group_of[t] for t in task_ids))
    idx_by_g = {g: np.array([tid_idx[t] for t in task_ids if group_of[t] == g]) for g in gids}

    rng = np.random.default_rng(args.seed)
    obs = seed_mean_delta(y_all, np.arange(len(y_all)), S)
    power_hits, half_widths, deltas = 0, [], []
    for _ in range(args.outer):
        pick = rng.choice(len(gids), size=args.groups, replace=True)
        groups_picked = [gids[i] for i in pick]
        idx = np.concatenate([idx_by_g[g] for g in groups_picked])
        d = seed_mean_delta(y_all, idx, S)
        deltas.append(d)
        inner = []
        for _ in range(args.inner):
            pick2 = rng.choice(len(groups_picked), size=len(groups_picked), replace=True)
            idx2 = np.concatenate([idx_by_g[groups_picked[i]] for i in pick2])
            if len(set(y_all[idx2].tolist())) < 2:
                continue
            inner.append(seed_mean_delta(y_all, idx2, S))
        arr = np.array(inner)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        half_widths.append((hi - lo) / 2)
        if lo > 0:
            power_hits += 1

    out = {
        "endpoint": "seed-mean AUROC difference GRAD - TEMP (cluster bootstrap by base case)",
        "explicitly_not": "score-ensemble endpoint; ensemble effects are not used here",
        "source": "W1 test only (58 base cases): LOGS/w1_final_confirmation/per_example_predictions.csv",
        "observed_w1_seed_mean_delta": obs,
        "planned_groups": args.groups,
        "simulation": {"outer": args.outer, "inner": args.inner, "seed": args.seed},
        "power_estimate": power_hits / args.outer,
        "mean_ci_half_width": float(np.mean(half_widths)),
        "mean_simulated_delta": float(np.mean(deltas)),
        "undersample_rule": "if valid base cases < 40 after the fixed exclusion rules, the round is reported as underpowered; no regeneration or top-up",
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
