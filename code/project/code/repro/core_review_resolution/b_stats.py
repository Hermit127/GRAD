#!/usr/bin/env python3
"""B-line statistics: E/Z/R attribution on the frozen W1 obs1 test split.

Pre-registered endpoints (LOGS/core_review_resolution/b_prereg.md):
  PRIMARY   seed-mean AUROC difference, paired by base case:
              D1 = mean_s AUROC_s(E) - mean_s AUROC_s(Z)
              D2 = mean_s AUROC_s(E) - mean_s AUROC_s(R)
            cluster bootstrap over base cases (same resample for every arm and
            seed); two-sided bootstrap p-values with Holm correction across the
            two primary tests; 95% percentile CIs reported for all endpoints.
  SECONDARY per-seed deltas (same bootstrap), dev AUROC per arm/seed, per-arm
            test AUROC per seed.

Usage:
  python b_stats.py --scores-dir LOGS/core_review_resolution/out/b_scores \
      --out LOGS/core_review_resolution/b_stats.json [--seed 20260926]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
SEEDS = (211, 212, 213)
ARMS = ("E", "Z", "R")
PRIMARY = (("E", "Z"), ("E", "R"))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def load_group_map(split: str) -> dict[str, str]:
    meta = REPO / f"LOGS/w1_fair_comparison/data/obs1/{split}.w1meta.jsonl"
    out = {}
    for l in open(meta):
        r = json.loads(l)
        out[r["task_id"]] = str(r["group"])
    return out


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def arm_seed_matrix(scores_dir: Path, split: str):
    """Returns (task_ids, labels, groups, scores[arm][seed] aligned arrays)."""
    ref = None
    scores, labels, groups = {}, None, None
    for arm in ARMS:
        for seed in SEEDS:
            p = scores_dir / f"{arm}_{split}_s{seed}.jsonl"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            rows = load_rows(p)
            ids = [r["task_id"] for r in rows]
            if ref is None:
                ref = ids
                labels = np.array([int(r["label"]) for r in rows])
            elif ids != ref:
                raise SystemExit(f"row order mismatch in {p}")
            scores[(arm, seed)] = np.array([float(r["score"]) for r in rows])
    gmap = load_group_map(split)
    groups = np.array([gmap[i] for i in ref])
    return ref, labels, groups, scores


def cluster_bootstrap(labels, groups, scores, rng, boot=5000):
    gids = sorted(set(groups.tolist()))
    idx_by_g = {g: np.where(groups == g)[0] for g in gids}
    draws = []
    for _ in range(boot):
        pick = rng.choice(len(gids), size=len(gids), replace=True)
        idx = np.concatenate([idx_by_g[gids[i]] for i in pick])
        y = labels[idx]
        if len(set(y.tolist())) < 2:
            continue
        per_seed = {}
        for (arm, seed), s in scores.items():
            per_seed.setdefault(arm, []).append(auroc(y, s[idx]))
        entry = {}
        for a1, a2 in PRIMARY:
            d_s = [per_seed[a1][k] - per_seed[a2][k] for k in range(len(SEEDS))]
            entry[f"{a1}-{a2}"] = float(np.mean(d_s))
            for k, seed in enumerate(SEEDS):
                entry[f"{a1}-{a2}|s{seed}"] = float(d_s[k])
        draws.append(entry)
    return draws


def summarize(vals):
    arr = np.array(vals)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
    return {"mean": float(arr.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": min(1.0, max(p, 1.0 / (len(arr) + 1))),
            "n_draws": int(len(arr))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, p * (m - i)))
        out[k] = adj
        prev = adj
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--e-parity", default="")
    args = ap.parse_args()

    sd = Path(args.scores_dir)
    if not sd.is_absolute():
        sd = REPO / sd
    rng = np.random.default_rng(args.seed)

    _, test_y, test_g, test_s = arm_seed_matrix(sd, "test")
    result = {
        "protocol": "crr-b-attribution-v1",
        "test": {"n": int(len(test_y)), "n_pos": int(test_y.sum()),
                 "n_neg": int((test_y == 0).sum()), "n_groups": int(len(set(test_g)))},
        "arms": {},
        "bootstrap": {"draws": args.boot, "seed": args.seed,
                      "unit": "base case (obs1 group)",
                      "endpoint": "seed-mean AUROC difference"},
    }

    # per-arm per-seed AUROC + dev descriptive
    for arm in ARMS:
        result["arms"][arm] = {"test_auroc_per_seed": {}, "dev_auroc_per_seed": {}}
        for seed in SEEDS:
            result["arms"][arm]["test_auroc_per_seed"][str(seed)] = auroc(
                test_y, test_s[(arm, seed)])
        try:
            _, dev_y, _, dev_s = arm_seed_matrix(sd, "dev")
            for seed in SEEDS:
                result["arms"][arm]["dev_auroc_per_seed"][str(seed)] = auroc(
                    dev_y, dev_s[(arm, seed)])
        except SystemExit:
            pass
        vals = [result["arms"][arm]["test_auroc_per_seed"][str(s)] for s in SEEDS]
        result["arms"][arm]["test_auroc_seed_mean"] = float(np.mean(vals))

    # observed deltas
    obs = {}
    for a1, a2 in PRIMARY:
        d_s = [auroc(test_y, test_s[(a1, s)]) - auroc(test_y, test_s[(a2, s)])
               for s in SEEDS]
        obs[f"{a1}-{a2}"] = {"seed_mean_delta": float(np.mean(d_s)),
                             "per_seed_delta": {str(s): float(v) for s, v in zip(SEEDS, d_s)}}
    result["observed"] = obs

    # bootstrap
    draws = cluster_bootstrap(test_y, test_g, test_s, rng, args.boot)
    p_raw = {}
    for a1, a2 in PRIMARY:
        key = f"{a1}-{a2}"
        result["bootstrap"][key] = summarize([d[key] for d in draws])
        p_raw[key] = result["bootstrap"][key]["p_two_sided"]
        for seed in SEEDS:
            result["bootstrap"][f"{key}|s{seed}"] = summarize(
                [d[f"{key}|s{seed}"] for d in draws])
    result["bootstrap"]["holm_adjusted_p"] = holm(p_raw)
    result["bootstrap"]["multiplicity_plan"] = (
        "Holm-Bonferroni over the two primary endpoints D1=E-Z, D2=E-R; "
        "per-seed endpoints are secondary/descriptive")

    # E parity check vs frozen W1 GRAD scores (if requested)
    if args.e_parity:
        frozen = {}
        for seed in SEEDS:
            d = REPO / f"LOGS/w1_fair_comparison/runs_srv/GRAD_s{seed}"
            js = sorted(d.glob("phase2_*.json"))
            run = json.loads(js[-1].read_text())
            frozen[seed] = np.array([r["score"] for r in run["scores"]["test"]])
        parity = {}
        for seed in SEEDS:
            diff = np.abs(frozen[seed] - test_s[("E", seed)])
            parity[str(seed)] = {"max_abs_diff": float(diff.max()),
                                 "n_bitwise_equal": int((diff == 0).sum()),
                                 "n": int(len(diff))}
        result["e_parity_vs_frozen_w1"] = parity

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["observed"], indent=1))
    print(json.dumps({k: v for k, v in result["bootstrap"].items()
                      if not k.endswith(tuple(f"|s{s}" for s in SEEDS))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
