#!/usr/bin/env python3
"""Phase 2 dev evaluation and candidate selection (dev only; test sealed).

Reads:
  code/repro/out/table5_candidates/dev/*.json   (phase2 candidates + full_ref)
  code/repro/out/table5_ablation/runs_cpu/      (frozen references: full/structure/...)

Writes LOGS/final_five_tables/table5/phase2_dev_results.{json,md}.
Applies the preregistered criteria; selects at most two configs for confirmation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEV = REPO / "code/repro/out/table5_candidates/dev"
REF = REPO / "code/repro/out/table5_ablation/runs"          # GPU replication (primary)
REF_CPU = REPO / "code/repro/out/table5_ablation/runs_cpu"  # CPU replication (secondary)
OUT = REPO / "LOGS/final_five_tables/table5"
FOLDS = ["fold0", "fold1"]
SEEDS = [211, 212]
CONFIGS = ["full_ref", "A_norm_aux_0.1", "A_norm_aux_0.3", "A_moddrop_0.1",
           "B_gate", "C_raw_rel", "C_raw_rel_drop"]
REF_VARIANTS = ["full", "structure_only", "chrono_only", "text_only",
                "no_relation_types", "no_keyword_features"]


def auroc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores); ss = scores[order]
    ranks = np.empty(len(scores)); i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def fold_weighted(rows_by_cell):
    """rows_by_cell: {(fold, seed): [(label, score), ...]} -> per-seed values."""
    per_seed = defaultdict(list)
    for (fold, seed), rows in rows_by_cell.items():
        y = [r[0] for r in rows]; s = [r[1] for r in rows]
        npos = sum(y); nneg = len(y) - npos
        per_seed[seed].append((auroc(y, s), npos * nneg))
    out = {}
    for seed, items in per_seed.items():
        w = [w for _, w in items if w > 0]
        v = [a for a, _ in items if not math_isnan(a)]
        ww = [w for a, w in items if not math_isnan(a)]
        out[seed] = float(np.average(v, weights=ww)) if ww else float("nan")
    return out


def math_isnan(x):
    return x != x


def load_phase2() -> dict:
    out = defaultdict(dict)
    for p in DEV.glob("*.json"):
        d = json.loads(p.read_text())
        cfg = d["config"]
        label = cfg["candidate"]
        if label == "A_norm_aux":
            label = f"A_norm_aux_{cfg.get('lambda_aux', 0.1)}"
        elif label == "A_moddrop":
            label = f"A_moddrop_{cfg.get('lambda_aux', 0.1)}"
        fold = next((f for f in FOLDS if f"_{f}_" in p.name), None)
        if fold is None:
            continue
        seed = cfg["seed"]
        rows = [(r["label"], r["score"]) for r in d["scores"]["val"]]
        out[label][(fold, seed)] = rows
        out[label][("meta", fold, seed)] = {
            "params": d["candidate_meta"]["params"],
            "alpha_mean": d["candidate_meta"].get("alpha_mean"),
            "fusion": d.get("fusion_comparator"),
            "threshold": d["val"].get("dev_selected_threshold"),
            "recall": d["val"]["metrics_at_threshold"].get("recall"),
            "fpr": d["val"]["metrics_at_threshold"].get("fpr"),
        }
    return out


def load_ref(base: Path = REF) -> dict:
    out = defaultdict(dict)
    for variant in REF_VARIANTS:
        for fold in FOLDS:
            for seed in SEEDS:
                p = base / f"mcp_{fold}_{variant}_s{seed}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                rows = d["scores"]["val"]
                out[variant][(fold, seed)] = [(r["label"], r["score"]) for r in rows]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "phase2_dev_results.json"))
    args = ap.parse_args()

    p2 = load_phase2()
    ref = load_ref(REF)
    ref_values = {v: fold_weighted(dict(r)) for v, r in ref.items()}
    ref_cpu = load_ref(REF_CPU)
    ref_cpu_values = {v: fold_weighted(dict(r)) for v, r in ref_cpu.items()}

    results = {"dev_protocol": {"folds": FOLDS, "seeds": SEEDS,
                                "aggregation": "fold-weighted by n_pos*n_neg within seed; mean over 2 seeds",
                                "test_read": False,
                                "reference_hardware": "GPU replication runs (primary); "
                                                      "CPU replication reported as secondary"},
               "references": {v: {"per_seed": {str(k): round(x, 4) for k, x in vals.items()},
                                  "mean": float(np.mean(list(vals.values())))}
                              for v, vals in ref_values.items()},
               "references_cpu": {v: {"per_seed": {str(k): round(x, 4) for k, x in vals.items()},
                                      "mean": float(np.mean(list(vals.values())))}
                                  for v, vals in ref_cpu_values.items()},
               "candidates": {}}
    full_ref = None
    if "full_ref" in p2:
        full_ref = fold_weighted({k: v for k, v in p2["full_ref"].items()
                                  if isinstance(k, tuple) and len(k) == 2
                                  and k[0] in FOLDS})
        results["references"]["phase2_full_ref"] = {
            "per_seed": {str(k): round(x, 4) for k, x in full_ref.items()},
            "mean": float(np.mean(list(full_ref.values())))}

    for cfg in CONFIGS:
        data = p2.get(cfg)
        if not data:
            continue
        values = fold_weighted({k: v for k, v in data.items()
                                if isinstance(k, tuple) and len(k) == 2 and k[0] in FOLDS})
        if not values:
            continue
        delta_phase2 = {s: values[s] - full_ref[s] for s in values} if full_ref else {}
        delta_frozen = {s: values[s] - ref_values["full"].get(s, float("nan"))
                        for s in values}
        delta_struct = {s: values[s] - ref_values["structure_only"].get(s, float("nan"))
                        for s in values}
        meta_by_cell = {k[1:]: v for k, v in data.items()
                        if isinstance(k, tuple) and len(k) == 3 and k[0] == "meta"}
        survivors = {
            "beats_phase2_full_mean": bool(delta_phase2) and float(np.mean(list(delta_phase2.values()))) > 0,
            "same_direction_both_seeds": (all(v > 0 for v in delta_phase2.values())
                                          if delta_phase2 else False),
            "beats_frozen_full_mean": float(np.mean([v for v in delta_frozen.values()
                                                     if not math_isnan(v)] or [float("nan")])) > 0,
            "beats_structure_mean": float(np.mean([v for v in delta_struct.values()
                                                   if not math_isnan(v)] or [float("nan")])) > 0,
        }
        results["candidates"][cfg] = {
            "per_seed": {str(k): round(v, 4) for k, v in values.items()},
            "mean": float(np.mean(list(values.values()))),
            "std": float(np.std(list(values.values()))),
            "delta_vs_phase2_full": {str(k): round(v, 4) for k, v in delta_phase2.items()},
            "delta_vs_frozen_full": {str(k): round(v, 4) for k, v in delta_frozen.items()},
            "delta_vs_structure": {str(k): round(v, 4) for k, v in delta_struct.items()},
            "criteria": survivors,
            "meta_by_cell": {f"{f}_{s}": v for (f, s), v in meta_by_cell.items()},
        }

    OUT.mkdir(parents=True, exist_ok=True)
    # ---- automatic selection (preregistered rules) ----
    passing = []
    for cfg, d in results["candidates"].items():
        if cfg == "full_ref":
            continue
        crit = d["criteria"]
        if crit.get("beats_phase2_full_mean") and crit.get("same_direction_both_seeds"):
            passing.append((cfg, d["mean"]))
    passing.sort(key=lambda kv: -kv[1])
    selected = [c for c, _ in passing[:2]]
    results["selection"] = {
        "ranked_passing": passing,
        "selected_for_confirmation": selected,
        "rule": "dev mean AUROC > phase2 full_ref AND both dev seeds positive; "
                "top-2 by dev mean; test sealed",
    }
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))

    md = ["# Phase 2 dev results (test sealed)", "",
          "Aggregation: fold-weighted AUROC within seed, mean over seeds 211/212. "
          "`test_read: false`. References are the GPU replication runs (primary, same "
          "hardware as candidates); CPU replication is secondary.", ""]
    md += ["## References (dev, GPU replication)", "",
           "| Reference | seed 211 | seed 212 | mean |", "|---|---|---|---|"]
    for v, vals in results["references"].items():
        md.append(f"| {v} | {vals['per_seed'].get('211', '--')} | {vals['per_seed'].get('212', '--')} | "
                  f"{vals['mean']:.4f} |")
    md += ["", "## Candidates", "",
           "| Config | seed 211 | seed 212 | mean | vs phase2 full | vs GPU full | vs GPU structure |",
           "|---|---|---|---|---|---|---|"]
    for cfg, d in results["candidates"].items():
        md.append(f"| {cfg} | {d['per_seed'].get('211')} | {d['per_seed'].get('212')} | "
                  f"{d['mean']:.4f} | {np.mean(list(d['delta_vs_phase2_full'].values())):+.4f} | "
                  f"{np.mean(list(d['delta_vs_frozen_full'].values())):+.4f} | "
                  f"{np.mean(list(d['delta_vs_structure'].values())):+.4f} |")
    md += ["", f"**Selected for confirmation: {', '.join(selected) if selected else 'none'}**", "",
           "Caveat recorded in the preregistration: MCP dev has 4 cells (2 folds x 2 seeds); "
           "per-seed directions can disagree with the mean, so confirmation uses the full "
           "5-fold x 3-seed matrix plus Decomp/MRT and a sealed group-isolated split."]
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {args.out} and .md; selected: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
