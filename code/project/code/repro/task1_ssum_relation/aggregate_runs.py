#!/usr/bin/env python3
"""Merge per-worker Task 1 pilot runs into one aggregation table."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(root: Path):
    runs = {}
    for p in sorted(root.glob("*/runs/*.json")):
        with open(p) as f:
            runs[p.stem] = json.load(f)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="",
                    help="comma list of armA:armB paired seed comparisons")
    args = ap.parse_args()
    root = Path(args.root)
    runs = load_runs(root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_arm = defaultdict(list)
    for name, d in runs.items():
        arm = name.rsplit("_s", 1)[0]
        m = d["metrics"]
        rec = {
            "run": name, "seed": d["seed"],
            "auroc": m["test"]["auroc"], "pr_auc": m["test"]["pr_auc"],
            "tpr_at_dev_fpr": m["test"]["at_dev_threshold"]["tpr"],
            "test_fpr": m["test"]["at_dev_threshold"]["fpr"],
            "dev_auroc": m["dev"]["auroc"],
            "pair_order_acc": m["test"]["pair"]["order_acc"],
            "mean_margin": m["test"]["pair"]["mean_margin"],
            "margin_ci95": m["test"]["pair"]["mean_margin_ci95"],
        }
        for k in ("pair_read_target", "pair_param_source", "pair_auth_source"):
            if k in m["test"]:
                rec[f"{k}_acc"] = m["test"][k]["order_acc"]
        by_arm[arm].append(rec)
    summary = {"runs": {k: v for k, v in sorted(runs.items())},
               "per_arm": {a: rs for a, rs in sorted(by_arm.items())}}
    with open(out / "aggregate_merged.json", "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=float)
    lines = ["# Task 1 dev pilot — merged", "",
             "| arm | seeds | AUROC | PR-AUC | TPR@devFPR10 | test FPR | pair-order | margin | read_target | param_source | auth_source |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, rs in sorted(by_arm.items()):
        def ms(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if not vals:
                return "n/a"
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        lines.append(
            f"| {arm} | {len(rs)} | {ms('auroc')} | {ms('pr_auc')} | "
            f"{ms('tpr_at_dev_fpr')} | {ms('test_fpr')} | {ms('pair_order_acc')} | "
            f"{ms('mean_margin')} | {ms('pair_read_target_acc')} | "
            f"{ms('pair_param_source_acc')} | {ms('pair_auth_source_acc')} |")
    (out / "aggregate_merged.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    pert = {}
    for name, d in runs.items():
        if "perturbations" in d:
            pert[name] = d["perturbations"]
    if pert:
        with open(out / "perturbations.json", "w") as f:
            json.dump(pert, f, indent=1, sort_keys=True, default=float)
        print("\nFrozen-model perturbations:")
        for name, rows in pert.items():
            for kind, r in rows.items():
                print(f"  {name} {kind}: mean_abs_delta={r['mean_abs_delta']:.4f} "
                      f"flip_rate={r['score_flip_rate']:.3f} "
                      f"pair_acc={r['pair']['order_acc']}")

    paired = {}
    if args.pairs:
        for spec in args.pairs.split(","):
            if ":" not in spec:
                continue
            a, b = spec.split(":")
            ra = {r["seed"]: r for r in by_arm.get(a, [])}
            rb = {r["seed"]: r for r in by_arm.get(b, [])}
            seeds = sorted(set(ra) & set(rb))
            deltas = []
            for s in seeds:
                deltas.append({k: ra[s][k] - rb[s][k]
                               for k in ("auroc", "pair_order_acc", "mean_margin")})
            if not deltas:
                continue
            rec = {"seeds": seeds, "per_seed": deltas}
            for k in ("auroc", "pair_order_acc", "mean_margin"):
                vals = np.array([d[k] for d in deltas], dtype=float)
                rec[f"mean_delta_{k}"] = float(vals.mean())
                if len(vals) > 1:
                    rec[f"ci95_delta_{k}"] = [float(vals.mean() - 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))),
                                              float(vals.mean() + 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))]
                rec[f"sign_{k}"] = int((vals > 0).sum())
            paired[spec] = rec
        with open(out / "paired_seed_comparisons.json", "w") as f:
            json.dump(paired, f, indent=1, sort_keys=True, default=float)
        print("\nPaired seed comparisons (delta = A - B):")
        for spec, rec in paired.items():
            print(f"  {spec}: dAUROC={rec['mean_delta_auroc']:+.4f} "
                  f"(sign +{rec['sign_auroc']}/{len(rec['seeds'])}), "
                  f"dPairOrder={rec['mean_delta_pair_order_acc']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
