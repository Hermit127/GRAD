#!/usr/bin/env python3
"""W4/W5 final statistics under frozen models and frozen dev thresholds.

Endpoints are kept strictly separate:
  E1 auroc_seed_mean        : mean_s AUROC_s(arm); paired delta of the means
  E2 auroc_ensemble_scores  : AUROC of per-example seed-averaged scores
Operating points use frozen dev thresholds (target FPR <= 5% / 10%) selected
with the frozen tie rule; test/confirm labels are never used for selection.

Usage:
  python final_stats.py --target test   --runs-dir LOGS/w1_fair_comparison/runs_srv
  python final_stats.py --target confirm --scores-dir LOGS/w1_final_confirmation/scores \
      --meta LOGS/w1_final_confirmation/data/confirm_synth/meta.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
W1 = REPO / "LOGS/w1_fair_comparison"
OUT = REPO / "LOGS/w1_final_confirmation"
ARMS = ("GRAD", "TXT", "TEMP")
SEEDS = (211, 212, 213)
TARGETS = (0.05, 0.10)
BOOT = 5000
RNG = np.random.default_rng(20260926)


def threshold_for_target_fpr(labels, scores, target) -> tuple[float, dict]:
    """Frozen trainer rule: descending candidates, FPR <= target, max recall."""
    candidates = sorted(set(float(s) for s in scores), reverse=True)
    candidates = [float("inf")] + candidates + [float("-inf")]
    best_t, best = candidates[0], None
    for t in candidates:
        pred = [1 if s >= t else 0 for s in scores]
        fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 0)
        n_neg = sum(1 for y in labels if y == 0)
        n_pos = sum(1 for y in labels if y == 1)
        fpr = fp / max(n_neg, 1)
        if fpr <= target + 1e-12:
            m = {"threshold": t, "tp": n_pos - fn, "fp": fp,
                 "tn": n_neg - fp, "fn": fn,
                 "tpr": (n_pos - fn) / max(n_pos, 1), "fpr": fpr}
            if best is None or (m["tpr"], m["fp"]) > (best["tpr"], -best["fp"]):
                best, best_t = m, t
    return best_t, best


def operating_point(labels, scores, threshold) -> dict:
    pred = [1 if s >= threshold else 0 for s in scores]
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    tp = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "tn": n_neg - fp,
            "fn": n_pos - tp, "tpr": tp / max(n_pos, 1), "fpr": fp / max(n_neg, 1),
            "n_pos": n_pos, "n_neg": n_neg}


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def bootstrap(y, groups, fn):
    gids = sorted(set(groups.tolist()))
    idx_by_g = {g: np.where(groups == g)[0] for g in gids}
    vals = []
    for _ in range(BOOT):
        pick = RNG.choice(len(gids), size=len(gids), replace=True)
        idx = np.concatenate([idx_by_g[gids[i]] for i in pick])
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        vals.append(fn(idx, yy))
    return vals


def ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def load_w1_run(runs_dir: Path, arm: str, seed: int) -> dict:
    d = runs_dir / f"{arm}_s{seed}"
    cands = sorted(d.glob("gnn_v2_results_*.json")) + sorted(d.glob("phase2_*.json"))
    run = json.loads(cands[-1].read_text())
    dev = run["scores"]["val"]
    tst = run["scores"]["test"]
    return {"dev_scores": np.array([r["score"] for r in dev]),
            "dev_labels": np.array([int(r["label"]) for r in dev]),
            "test_scores": np.array([r["score"] for r in tst]),
            "test_labels": np.array([int(r["label"]) for r in tst]),
            "json": str(cands[-1])}


def load_scores_dir(scores_dir: Path, arm: str, seed: int):
    p = scores_dir / f"{arm}_s{seed}.jsonl"
    rows = [json.loads(l) for l in open(p)]
    return (np.array([r["score"] for r in rows]), np.array([int(r["label"]) for r in rows]),
            [r["task_id"] for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["test", "confirm"], required=True)
    ap.add_argument("--runs-dir", default=str(W1 / "runs_srv"))
    ap.add_argument("--scores-dir", default=str(OUT / "scores"))
    ap.add_argument("--meta", default=str(OUT / "data/confirm_synth/meta.jsonl"))
    ap.add_argument("--thresholds", default=str(OUT / "frozen_thresholds.json"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag if args.tag else ("w4_old_test" if args.target == "test" else "w5_confirm")
    OUT.mkdir(parents=True, exist_ok=True)
    thr_path = Path(args.thresholds)
    assert thr_path.exists(), f"missing frozen thresholds: {thr_path} (run freeze.py first)"
    frozen_thresholds = {tuple(k.split("|")): v for k, v in json.loads(thr_path.read_text()).items()}
    frozen_thresholds = {(k[0], int(k[1])): v for k, v in frozen_thresholds.items()}

    if args.target == "test":
        meta = [json.loads(l) for l in open(W1 / "data/obs1/test.w1meta.jsonl")]
        labels = np.array([m["label"] for m in meta])
        groups = np.array([str(m["group"]) for m in meta])
        task_ids = [m["task_id"] for m in meta]
    else:
        meta = [json.loads(l) for l in open(args.meta)]
        labels = np.array([m["label"] for m in meta])
        groups = np.array([str(m["group"]) for m in meta])
        task_ids = [m["task_id"] for m in meta]

    runs: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            if args.target == "test":
                run = load_w1_run(Path(args.runs_dir), arm, seed)
                assert np.array_equal(run["test_labels"], labels), f"{arm} s{seed} label order"
                runs[arm][seed] = run
            else:
                sc, lb, tids = load_scores_dir(Path(args.scores_dir), arm, seed)
                assert np.array_equal(lb, labels), f"{arm} s{seed} label order"
                assert tids == task_ids, f"{arm} s{seed} task order"
                runs[arm][seed] = {"test_scores": sc, "test_labels": lb}

    result: dict = {
        "target": args.target, "tag": tag,
        "n_rows": len(labels), "n_pos": int(labels.sum()), "n_neg": int((labels == 0).sum()),
        "n_groups": len(set(groups.tolist())),
        "fpr_resolution": 1.0 / max(int((labels == 0).sum()), 1),
        "note_1pct_fpr": "not reported: resolution is 1/n_neg and the support is insufficient",
        "threshold_rule": "frozen trainer rule on dev only; tie = score >= threshold; "
                          "descending unique candidates, FPR <= target, max recall",
        "endpoints": {
            "auroc_seed_mean": {"definition": "mean over seeds of per-seed AUROC; paired delta "
                                              "recomputed inside each cluster-bootstrap draw"},
            "auroc_ensemble_scores": {"definition": "AUROC of per-example mean score across seeds"},
        },
        "operating_points": {}, "auroc": {}, "paired": {},
    }

    if args.target == "test":
        dev = {}
        for arm in ARMS:
            for seed in SEEDS:
                dev[(arm, seed)] = runs[arm][seed]
        fp = W1 / "stats.json"
        if fp.exists():
            result["reused_w1_stats"] = str(fp.relative_to(REPO))

    for arm in ARMS:
        per_seed_auroc = []
        for seed in SEEDS:
            per_seed_auroc.append(auroc(labels, runs[arm][seed]["test_scores"]))
        result["auroc"][arm] = {"per_seed": per_seed_auroc,
                                "seed_mean": float(np.mean(per_seed_auroc)),
                                "seed_std": float(np.std(per_seed_auroc))}

    for target_fpr in TARGETS:
        key = f"dev_fpr{int(target_fpr*100):02d}"
        result["operating_points"][key] = {"target_dev_fpr": target_fpr, "arms": {}}
        for arm in ARMS:
            rows = []
            for seed in SEEDS:
                thr = frozen_thresholds[(arm, seed)][key]["threshold"]
                devm = frozen_thresholds[(arm, seed)][key]["dev_metrics"]
                if args.target == "test":
                    ds, dl = runs[arm][seed]["dev_scores"], runs[arm][seed]["dev_labels"]
                    assert abs(thr - threshold_for_target_fpr(dl, ds, target_fpr)[0]) < 1e-9, \
                        f"frozen threshold mismatch {arm} s{seed} {key}"
                m = operating_point(labels, runs[arm][seed]["test_scores"], thr)
                rows.append({"seed": seed, "dev_threshold": thr, "dev_selected": devm, **m})
            agg = {k: float(np.mean([r[k] for r in rows]))
                   for k in ("tpr", "fpr", "tp", "fp", "tn", "fn")}
            result["operating_points"][key]["arms"][arm] = {"per_seed": rows, "seed_mean": agg}

    pairs = [("GRAD", "TXT"), ("GRAD", "TEMP")]
    for a, b in pairs:
        entry = {}
        for target_fpr in TARGETS:
            key = f"dev_fpr{int(target_fpr*100):02d}"
            per_seed = []
            for seed in SEEDS:
                ta = result["operating_points"][key]["arms"][a]["per_seed"][SEEDS.index(seed)]
                tb = result["operating_points"][key]["arms"][b]["per_seed"][SEEDS.index(seed)]
                vals = bootstrap(labels, groups, lambda idx, yy, ta=ta, tb=tb, seed=seed: (
                    operating_point(yy, runs[a][seed]["test_scores"][idx], ta["dev_threshold"])["tpr"]
                    - operating_point(yy, runs[b][seed]["test_scores"][idx], tb["dev_threshold"])["tpr"]))
                lo, hi = ci(vals)
                per_seed.append({"seed": seed,
                                 "delta_tpr": ta["tpr"] - tb["tpr"],
                                 "ci95": [lo, hi],
                                 "delta_fpr": ta["fpr"] - tb["fpr"],
                                 "ci_excludes_zero": bool(lo > 0 or hi < 0)})
            entry[key] = {"per_seed": per_seed,
                          "mean_delta_tpr": float(np.mean([r["delta_tpr"] for r in per_seed])),
                          "seeds_ci_excl_0_tpr": sum(1 for r in per_seed if r["ci_excludes_zero"])}
        result["paired"][f"{a}-{b}"] = entry

    for a, b in pairs:
        ent = {}
        vals = bootstrap(labels, groups, lambda idx, yy, a=a, b=b: (
            np.mean([auroc(yy, runs[a][s]["test_scores"][idx]) for s in SEEDS])
            - np.mean([auroc(yy, runs[b][s]["test_scores"][idx]) for s in SEEDS])))
        lo, hi = ci(vals)
        d = (float(np.mean([auroc(labels, runs[a][s]["test_scores"]) for s in SEEDS]))
             - float(np.mean([auroc(labels, runs[b][s]["test_scores"]) for s in SEEDS])))
        ent["auroc_seed_mean"] = {"delta": d, "ci95": [lo, hi],
                                  "excludes_zero": bool(lo > 0 or hi < 0)}
        ens_a = np.mean([runs[a][s]["test_scores"] for s in SEEDS], axis=0)
        ens_b = np.mean([runs[b][s]["test_scores"] for s in SEEDS], axis=0)
        d2 = auroc(labels, ens_a) - auroc(labels, ens_b)
        vals2 = bootstrap(labels, groups, lambda idx, yy, ens_a=ens_a, ens_b=ens_b: (
            auroc(yy, ens_a[idx]) - auroc(yy, ens_b[idx])))
        lo2, hi2 = ci(vals2)
        ent["auroc_ensemble_scores"] = {"delta": d2, "ci95": [lo2, hi2],
                                        "excludes_zero": bool(lo2 > 0 or hi2 < 0),
                                        "auroc": {"a": auroc(labels, ens_a),
                                                  "b": auroc(labels, ens_b)}}
        result["paired"][f"{a}-{b}"]["auroc_endpoints"] = ent

    op = result["operating_points"]["dev_fpr10"]["arms"]
    pred_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            thr = None
            for r in op[arm]["per_seed"]:
                if r["seed"] == seed:
                    thr = r["dev_threshold"]
            for i, (s, y) in enumerate(zip(runs[arm][seed]["test_scores"], labels)):
                pred_rows.append([arm, seed, i, task_ids[i], groups[i], int(y),
                                  f"{s:.8f}", f"{thr:.8f}"])
    csv_path = OUT / f"per_example_predictions_{args.target}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "row", "task_id", "group", "label", "score", "dev_threshold_fpr10"])
        w.writerows(pred_rows)
    (OUT / f"stats_{args.target}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("target", "n_rows", "n_pos", "n_neg", "n_groups", "auroc",
                               "endpoints")}, indent=1))
    print("wrote", OUT / f"stats_{args.target}.json", "and", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
