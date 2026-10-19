#!/usr/bin/env python3
"""A-line round 2 statistics: frozen models x frozen thresholds x new cases.

Endpoints (frozen in a_prereg.md):
  PRIMARY   seed-mean AUROC difference GRAD - TEMP, paired by base case,
            cluster bootstrap over base cases (same resample for all arms and
            seeds), 95% percentile CI + two-sided p.
  SECONDARY seed-mean AUROC difference GRAD - TXT (same machinery); per-seed
            deltas; per-arm per-seed AUROC; frozen-threshold operating points
            (target FPR <= 5% / 10%) pooled across seeds with counts.
  The score-ensemble endpoint is NOT used.

Usage:
  python a_stats.py --scores-dir LOGS/core_review_resolution/scores_round2 \
      --data-dir LOGS/core_review_resolution/data/round2/confirm \
      --thresholds LOGS/w1_final_confirmation/frozen_thresholds.json \
      --out LOGS/core_review_resolution/a_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
ARMS = ("GRAD", "TXT", "TEMP")
SEEDS = (211, 212, 213)
PRIMARY = ("GRAD", "TEMP")
SECONDARY = ("GRAD", "TXT")
TARGETS = {"dev_fpr05": 0.05, "dev_fpr10": 0.10}


def auroc(y, s) -> float:
    if len(set(np.asarray(y).tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def load_all(scores_dir: Path, data_dir: Path):
    graphs = [json.loads(l) for l in open(data_dir / "graphs.jsonl")]
    meta = [json.loads(l) for l in open(data_dir / "meta.jsonl")]
    gmap = {m["task_id"]: str(m["group"]) for m in meta}
    ref_ids = [g["task_id"] for g in graphs]
    y = np.array([int(g["label"]) for g in graphs])
    groups = np.array([gmap[t] for t in ref_ids])
    S = {}
    for arm in ARMS:
        for seed in SEEDS:
            p = scores_dir / f"{arm}_s{seed}.jsonl"
            rows = [json.loads(l) for l in open(p)]
            ids = [r["task_id"] for r in rows]
            assert ids == ref_ids, f"row order mismatch: {p}"
            S[(arm, seed)] = np.array([float(r["score"]) for r in rows])
    return ref_ids, y, groups, S


def seed_mean(y, idx, S, a1, a2) -> float:
    d = [auroc(y[idx], S[(a1, s)][idx]) - auroc(y[idx], S[(a2, s)][idx]) for s in SEEDS]
    return float(np.mean(d))


def bootstrap(y, groups, S, a1, a2, rng, boot):
    gids = sorted(set(groups.tolist()))
    idx_by_g = {g: np.where(groups == g)[0] for g in gids}
    vals = []
    for _ in range(boot):
        pick = rng.choice(len(gids), size=len(gids), replace=True)
        idx = np.concatenate([idx_by_g[gids[i]] for i in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(seed_mean(y, idx, S, a1, a2))
    return np.array(vals)


def summarize(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(vals <= 0)), float(np.mean(vals >= 0)))
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": min(1.0, max(p, 1.0 / (len(vals) + 1))),
            "n_draws": int(len(vals))}


def operating_point(y, s, threshold):
    pred = s >= threshold
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    return {"threshold": float(threshold), "tp": tp, "fp": fp,
            "tn": n_neg - fp, "fn": n_pos - tp,
            "tpr": tp / max(n_pos, 1), "fpr": fp / max(n_neg, 1),
            "n_pos": n_pos, "n_neg": n_neg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    def R(p):
        p = Path(p)
        return p if p.is_absolute() else REPO / p

    scores_dir, data_dir = R(args.scores_dir), R(args.data_dir)
    _, y, groups, S = load_all(scores_dir, data_dir)
    thaw = json.loads(R(args.thresholds).read_text())
    rng = np.random.default_rng(args.seed)

    result = {
        "protocol": "a_round2_frozen_new_generated_case_validation",
        "data": {"n": int(len(y)), "n_pos": int(y.sum()),
                 "n_neg": int((y == 0).sum()), "n_groups": int(len(set(groups)))},
        "absolute_auroc": {},
        "primary_grad_minus_temp": {},
        "secondary_grad_minus_txt": {},
        "operating_points": {},
        "ensemble_endpoint_used": False,
    }
    for arm in ARMS:
        per_seed = {str(s): auroc(y, S[(arm, s)]) for s in SEEDS}
        result["absolute_auroc"][arm] = {
            "per_seed": per_seed, "seed_mean": float(np.mean(list(per_seed.values())))}

    for name, (a1, a2) in (("primary_grad_minus_temp", PRIMARY),
                           ("secondary_grad_minus_txt", SECONDARY)):
        obs = seed_mean(y, np.arange(len(y)), S, a1, a2)
        vals = bootstrap(y, groups, S, a1, a2, rng, args.boot)
        per_seed = {str(s): auroc(y, S[(a1, s)]) - auroc(y, S[(a2, s)]) for s in SEEDS}
        result[name] = {"observed_seed_mean_delta": obs,
                        "per_seed_delta": per_seed,
                        "bootstrap": summarize(vals)}

    for arm in ARMS:
        result["operating_points"][arm] = {}
        for tname, target in TARGETS.items():
            agg = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "per_seed": {}}
            for seed in SEEDS:
                key = f"{arm}|{seed}"
                if key not in thaw:
                    continue
                thr = thaw[key][tname]["threshold"]
                m = operating_point(y, S[(arm, seed)], thr)
                agg["per_seed"][str(seed)] = m
                for k in ("tp", "fp", "tn", "fn"):
                    agg[k] += m[k]
            n_pos, n_neg = agg["tp"] + agg["fn"], agg["tn"] + agg["fp"]
            agg.update(n_pos=n_pos, n_neg=n_neg,
                       tpr=agg["tp"] / max(n_pos, 1),
                       fpr=agg["fp"] / max(n_neg, 1), target_fpr=target)
            result["operating_points"][arm][tname] = agg

    out = R(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in
                      ("absolute_auroc", "primary_grad_minus_temp",
                       "secondary_grad_minus_txt")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
