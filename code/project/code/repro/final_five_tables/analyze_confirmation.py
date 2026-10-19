#!/usr/bin/env python3
"""Phase 2 confirmation analysis (test is now read, once, after freeze).

Inputs:
  code/repro/out/table5_confirm/confirm/*.json        candidates + sealed refs
  code/repro/out/table5_ablation/runs/*.json          legacy refs (GPU replication)
  code/repro/out/table5_candidates/sealed_mcp/manifest.json
Outputs:
  LOGS/final_five_tables/table5/phase2_confirmation.{json,md}

Selection was frozen in phase2_dev_results.json; nothing here re-selects.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
CONF = REPO / "code/repro/out/table5_confirm/confirm"
LEGACY = REPO / "code/repro/out/table5_ablation/runs"
SEALED = REPO / "code/repro/out/table5_candidates/sealed_mcp"
OUT = REPO / "LOGS/final_five_tables/table5"
SEEDS = [211, 212, 213]

LEGACY_CELLS = {
    "mcp": [f"fold{k}" for k in range(5)],
    "decomp": ["fold0"],
    "mrt": [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                                  "bank_transfer_2", "expenditure_accounting",
                                  "spam_filter_update")],
}
REF_VARIANTS = ["full", "structure_only", "text_only", "chrono_only"]
CANDIDATES = ["B_gate", "A_moddrop_0.1"]


def scores_of(d: dict, split: str = "test"):
    rows = d.get("scores", {}).get(split, [])
    return [(int(r["label"]), float(r["score"])) for r in rows]


def metrics(rows):
    if not rows:
        return None
    y = [r[0] for r in rows]
    s = [r[1] for r in rows]
    if len(set(y)) < 2:
        return {"auroc": float("nan"), "pr_auc": float("nan"),
                "n_pos": sum(y), "n": len(y)}
    return {"auroc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "n_pos": sum(y), "n": len(y)}


def load_confirm(label, ds, cell, seed):
    p = CONF / f"{label}_{ds}_{cell}_s{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_legacy_ref(variant, ds, cell, seed):
    p = LEGACY / f"{ds}_{cell}_{variant}_s{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fold_weighted(values_by_cell, labels_by_cell):
    """values_by_cell: {(cell,seed): metrics}; fold-weight by n_pos*n_neg per seed."""
    per_seed = defaultdict(list)
    for (cell, seed), m in values_by_cell.items():
        w = m["n_pos"] * (m["n"] - m["n_pos"])
        per_seed[seed].append((m["auroc"], w))
    return {s: float(np.average([a for a, _ in items], weights=[w for _, w in items]))
            for s, items in per_seed.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "phase2_confirmation.json"))
    args = ap.parse_args()

    out = {"test_read": "once, after dev-only freeze",
           "legacy": {}, "sealed": {}, "gate_comparators": {}}

    # ---------------- legacy replication (5 folds / 1 fold x 3 seeds) ----------
    for ds, cells in LEGACY_CELLS.items():
        cand_vals = {}
        for label in CANDIDATES:
            vals = {}
            for cell in cells:
                for seed in SEEDS:
                    d = load_confirm(label, ds, cell, seed)
                    if d is None:
                        continue
                    m = metrics(scores_of(d))
                    if m:
                        vals[(cell, seed)] = m
            cand_vals[label] = fold_weighted(vals, None)
        ref_vals = {}
        for variant in REF_VARIANTS:
            vals = {}
            for cell in cells:
                for seed in SEEDS:
                    d = load_legacy_ref(variant, ds, cell, seed)
                    if d is None:
                        continue
                    test = d.get("scores", {}).get("test")
                    if test is None:
                        # archived runs store test scores under scores.test too
                        test = []
                    m = metrics([(int(r["label"]), float(r["score"])) for r in test])
                    if m:
                        vals[(cell, seed)] = m
            ref_vals[variant] = fold_weighted(vals, None)
        if not any(cand_vals.values()) and not any(ref_vals.values()):
            continue
        entry = {"cells": cells, "seed_level": {},
                 "note": "seed-level fold-weighted AUROC; test read once"}
        for label, vals in {**ref_vals, **cand_vals}.items():
            if not vals:
                continue
            base = ref_vals.get("full") or {}
            entry["seed_level"][label] = {
                "per_seed": {str(k): round(v, 4) for k, v in vals.items()},
                "mean": float(np.mean(list(vals.values()))),
                "std": float(np.std(list(vals.values()))),
                "delta_vs_full": {str(k): round(v - base.get(k, float("nan")), 4)
                                  for k, v in vals.items()} if base else None,
            }
        out["legacy"][ds] = entry

    # ---------------- sealed group-isolated split ------------------------------
    sealed_vals = defaultdict(dict)
    for label in CANDIDATES + REF_VARIANTS:
        for seed in SEEDS:
            d = load_confirm(label, "sealed", "sealed", seed)
            if d is None:
                continue
            m = metrics(scores_of(d))
            if m:
                sealed_vals[label][seed] = m
    entry = {"per_seed": {}, "note": "sealed group-isolated MCP split; groups disjoint"}
    for label, vals in sealed_vals.items():
        entry["per_seed"][label] = {str(k): round(v["auroc"], 4) for k, v in vals.items()}
        if vals:
            entry["per_seed"][label]["mean"] = float(np.mean(
                [v["auroc"] for v in vals.values()]))
    out["sealed"] = entry

    # ---------------- B_gate gate vs fixed-weight comparator -------------------
    comp = {"note": "w fitted on the matching dev split only, applied to test"}
    for tag, (ds, cell) in {"legacy_mcp": ("mcp", "fold0"), "sealed": ("sealed", "sealed")}.items():
        per_seed = {}
        for seed in SEEDS:
            if tag == "sealed":
                d = load_confirm("B_gate", "sealed", "sealed", seed)
            else:
                d = load_confirm("B_gate", ds, cell, seed)
            if d is None:
                continue
            val = d["scores"]["val"]
            test = d["scores"]["test"]
            if not val or not test or "s_s" not in val[0]:
                continue
            y = np.array([r["label"] for r in val])
            s_s = np.array([r["s_s"] for r in val])
            s_t = np.array([r["s_t"] for r in val])
            best_w, best_auc = 0.0, -1.0
            for w in np.arange(0, 1.0001, 0.05):
                auc = roc_auc_score(y, (1 - w) * s_s + w * s_t) if len(set(y)) > 1 else float("nan")
                if auc > best_auc:
                    best_auc, best_w = auc, float(w)
            yt = np.array([r["label"] for r in test])
            s_st = np.array([r["s_s"] for r in test])
            s_tt = np.array([r["s_t"] for r in test])
            per_seed[seed] = {
                "w": best_w,
                "dev_branch_auroc": {k: float(roc_auc_score(y, v)) if len(set(y)) > 1 else None
                                     for k, v in (("struct", s_s), ("semantic", s_t))},
                "test_fixed_fusion_auroc": float(roc_auc_score(
                    yt, (1 - best_w) * s_st + best_w * s_tt)),
            }
            # gate test AUROC is the run's own test auroc
            per_seed[seed]["test_gate_auroc"] = d.get("test", {}).get("auroc")
            if "alpha" in val[0]:
                per_seed[seed]["alpha_mean_val"] = float(np.mean([r["alpha"] for r in val]))
        comp[tag] = per_seed
    out["gate_comparators"] = comp

    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    md = ["# Phase 2 confirmation (test read once after freeze)", "",
          "Frozen selection: " + ", ".join(CANDIDATES) + ".", ""]
    for ds, entry in out["legacy"].items():
        md += [f"## Legacy replication — {ds}", "",
               "| Method | seed 211 | seed 212 | seed 213 | mean | vs full (per seed) |",
               "|---|---|---|---|---|---|"]
        for label, d in entry["seed_level"].items():
            dv = d.get("delta_vs_full") or {}
            dv_s = ", ".join(f"{k}:{v:+.3f}" for k, v in sorted(dv.items()))
            md.append(f"| {label} | {d['per_seed'].get('211', '--')} | "
                      f"{d['per_seed'].get('212', '--')} | {d['per_seed'].get('213', '--')} | "
                      f"{d['mean']:.4f} | {dv_s or '--'} |")
        md.append("")
    md += ["## Sealed group-isolated MCP split", "",
           "| Method | 211 | 212 | 213 | mean |", "|---|---|---|---|---|"]
    for label, d in out["sealed"]["per_seed"].items():
        md.append(f"| {label} | {d.get('211', '--')} | {d.get('212', '--')} | "
                  f"{d.get('213', '--')} | {d.get('mean', '--')} |")
    md += ["", "## B_gate: learned gate vs dev-fitted fixed-weight fusion", ""]
    for tag, val in out["gate_comparators"].items():
        if tag == "note":
            continue
        for seed, d in val.items():
            md.append(f"- {tag} seed {seed}: w={d['w']}, test gate AUROC={d['test_gate_auroc']}, "
                      f"test fixed-fusion AUROC={d['test_fixed_fusion_auroc']}")
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {args.out} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
