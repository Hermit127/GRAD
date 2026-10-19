#!/usr/bin/env python3
"""Core ablation table: G3_p25 vs structure-only vs text-only, matched protocol.

PI requirement 2026-09-21: the three-row table is the core evidence for
"joint semantic+structure encoding"; text-only numbers must be matched to the
same protocol/seeds/splits, the G3 gaps need paired uncertainty, and the
regularized-structure result (MRT 89.47) must stay disclosed.

Aggregation matches the manuscript convention (`analyze_attribution.seed_means`):
per-cell AUROC, then weighted mean with weights n_pos*n_neg (cells are separate
tasks/folds with different score scales, so scores are never pooled).

Inputs (stored runs, test split, same harness/folds/seeds):
  code/repro/out/table5_g3/confirm/G3_p25_<ds>_<cell>_s<seed>.json
  code/repro/out/table5_ablation/runs/<ds>_<cell>_<structure_only|text_only>_s<seed>.json
  frozen splits referenced by each run's config.test
Outputs:
  LOGS/final_five_tables/table5/core_ablation_stats.{json,md}
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))
from analyze_attribution import auroc, group_of, CELLS, SEEDS  # noqa: E402

REF = REPO / "code/repro/out/table5_ablation/runs"
G3 = REPO / "code/repro/out/table5_g3/confirm"
OUT = REPO / "LOGS/final_five_tables/table5"
N_BOOT = 10000
METHODS = ("g3", "struct", "text")


def resolve_local(path_str: str) -> Path | None:
    p = str(path_str or "")
    for marker, base in (("/fix2_splits/", REPO / "code/repro/out/unified_gnn/fix2_splits"),
                         ("/mrt_cross_task_v1/", REPO / "results/mrt_cross_task_v1")):
        if marker in p:
            return base / p.split(marker, 1)[1]
    return None


def unit_rows(ds: str, cell: str, seed: int) -> dict | None:
    paths = {
        "g3": G3 / f"G3_p25_{ds}_{cell}_s{seed}.json",
        "struct": REF / f"{ds}_{cell}_structure_only_s{seed}.json",
        "text": REF / f"{ds}_{cell}_text_only_s{seed}.json",
    }
    data, split_path = {}, None
    for k, p in paths.items():
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        rows = d.get("scores", {}).get("test", [])
        if not rows:
            return None
        data[k] = rows
        if k == "g3":
            split_path = resolve_local((d.get("config") or {}).get("test", ""))
    if split_path is None or not split_path.exists():
        return None
    graphs = [json.loads(l) for l in split_path.read_text().splitlines() if l.strip()]
    if not all(len(data[k]) == len(graphs) for k in data):
        return None
    rows = {}
    for g, a, b, c in zip(graphs, data["g3"], data["struct"], data["text"]):
        assert int(g["label"]) == int(a["label"]) == int(b["label"]) == int(c["label"])
        tid = str(g.get("task_id"))
        if tid in rows:
            continue
        rows[tid] = {"tid": tid, "label": int(g["label"]), "group": group_of(ds, g),
                     "g3": float(a["score"]), "struct": float(b["score"]), "text": float(c["score"])}
    return rows or None


def arrays_of(rows: dict) -> dict:
    tids = sorted(rows)
    y = np.array([rows[t]["label"] for t in tids])
    scores = {k: np.array([rows[t][k] for t in tids], dtype=float) for k in METHODS}
    by_group = defaultdict(list)
    for i, t in enumerate(tids):
        by_group[rows[t]["group"]].append(i)
    w = float(y.sum() * (len(y) - y.sum()))
    return {"tids": tids, "y": y, "s": scores, "groups": [np.array(v) for v in by_group.values()],
            "weight": w}


def cell_auroc(a: dict, key: str) -> float:
    return float(auroc(a["y"], a["s"][key]))


def paired_cell_delta_ci(cells: list[dict], key_a: str, key_b: str, n_boot: int = N_BOOT) -> dict:
    point = _weighted_delta(cells, key_a, key_b)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_boot):
        sample = []
        for c in cells:
            idx = np.concatenate([c["groups"][g] for g in
                                  rng.integers(0, len(c["groups"]), len(c["groups"]))])
            y = c["y"][idx]
            if len(set(y.tolist())) < 2:
                continue
            sample.append({"y": y, "s": {k: c["s"][k][idx] for k in METHODS}, "weight": c["weight"]})
        if sample:
            draws.append(_weighted_delta(sample, key_a, key_b))
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"), float("nan"))
    return {"delta": point, "ci95_group_cluster": [float(lo), float(hi)],
            "n_cells": len(cells), "n_boot_valid": len(draws),
            "n_groups_per_cell": [len(c["groups"]) for c in cells]}


def _weighted_delta(cells: list[dict], key_a: str, key_b: str) -> float:
    ws = np.array([c["weight"] for c in cells], dtype=float)
    ds = np.array([cell_auroc(c, key_a) - cell_auroc(c, key_b) for c in cells])
    return float(np.sum(ws * ds) / np.sum(ws)) if ws.sum() else float("nan")


def weighted_mean_per_seed(per_cell: dict[str, dict], key: str, weights: dict[str, float]) -> float:
    ws = np.array([weights[c] for c in per_cell])
    vs = np.array([per_cell[c][key] for c in per_cell])
    return float(np.sum(ws * vs) / np.sum(ws))


def exact_sign_test(deltas: list[float]) -> dict:
    n = len(deltas)
    if n == 0:
        return {"n_clusters": 0}
    k = sum(d > 0 for d in deltas)
    tail = sum(comb(n, i) for i in range(n + 1) if i <= min(k, n - k))
    return {"n_clusters": n, "n_positive": k, "p_two_sided_exact": min(1.0, tail / 2 ** n)}


def main() -> None:
    res: dict = {"table": {}, "per_seed": {}, "aggregation": "per-cell AUROC, weight n_pos*n_neg"}
    for ds in ("decomp", "mcp", "mrt"):
        per_seed = []
        for seed in SEEDS:
            cells = []
            for cell in CELLS[ds]:
                u = unit_rows(ds, cell, seed)
                if u:
                    cells.append({"cell": cell, **arrays_of(u)})
            if not cells:
                continue
            weights = {c["cell"]: c["weight"] for c in cells}
            per_cell = {c["cell"]: {k: cell_auroc(c, k) for k in METHODS} for c in cells}
            entry = {
                "seed": seed,
                "cells": {c["cell"]: len(c["y"]) for c in cells},
                "n_graphs": int(sum(len(c["y"]) for c in cells)),
                "n_groups": int(sum(len(c["groups"]) for c in cells)),
                "auroc": {k: weighted_mean_per_seed(per_cell, k, weights) for k in METHODS},
                "per_cell": per_cell,
            }
            for name, a, b in (("g3_minus_struct", "g3", "struct"),
                               ("g3_minus_text", "g3", "text"),
                               ("struct_minus_text", "struct", "text")):
                entry[name] = paired_cell_delta_ci(cells, a, b)
            if ds == "mrt":
                for name, a, b in (("g3_minus_struct", "g3", "struct"),
                                   ("g3_minus_text", "g3", "text")):
                    cl = [{"cluster": c["cell"], "n": len(c["y"]),
                           "delta_auroc": float(cell_auroc(c, a) - cell_auroc(c, b))} for c in cells]
                    entry.setdefault("mrt_cluster_tests", {})[name] = {
                        "clusters": cl, **exact_sign_test([c["delta_auroc"] for c in cl])}
            per_seed.append(entry)
        if per_seed:
            res["per_seed"][ds] = per_seed
            res["table"][ds] = {
                "per_seed_mean": {k: float(np.mean([p["auroc"][k] for p in per_seed])) for k in METHODS},
                "seed_sd": {k: float(np.std([p["auroc"][k] for p in per_seed])) for k in METHODS},
                "n_graphs_per_seed": per_seed[0]["n_graphs"],
                "n_groups_per_seed": per_seed[0]["n_groups"],
            }
            for name in ("g3_minus_struct", "g3_minus_text", "struct_minus_text"):
                deltas = [p[name]["delta"] for p in per_seed]
                ci_excl = sum(1 for p in per_seed if p[name]["ci95_group_cluster"][0] > 0
                              or p[name]["ci95_group_cluster"][1] < 0)
                res["table"][ds][name] = {"per_seed_delta": deltas,
                                          "mean_delta": float(np.mean(deltas)),
                                          "n_seeds_ci_excl_0": ci_excl}
    (OUT / "core_ablation_stats.json").write_text(json.dumps(res, indent=2))
    write_md(res)
    print("wrote", OUT / "core_ablation_stats.json")


def write_md(res: dict) -> None:
    L = ["# Core ablation table (matched protocol): G3_p25 vs structure-only vs text-only", "",
         "PI 2026-09-21: same harness/folds/seeds/test split for all three rows. Aggregation follows",
         "the manuscript convention (per-cell AUROC, weighted by n_pos*n_neg); scores are never pooled",
         f"across folds. Gaps use paired group-clustered bootstrap ({N_BOOT} draws per seed within each",
         "cell, then weight-combined). MRT has 5 task cells -> exact sign test on per-task deltas.",
         "All numbers from stored runs; zero new training.", ""]
    L.append("## Table (AUROC x100, per-seed mean)")
    L.append("")
    L.append("| Method | Decomp | MCP | MRT |")
    L.append("|---|---|---|---|")
    for k, name in (("g3", "G3_p25"), ("struct", "Structure only"), ("text", "Text only")):
        vals = []
        for ds in ("decomp", "mcp", "mrt"):
            t = res["table"].get(ds)
            vals.append("n/a" if not t else f"{100 * t['per_seed_mean'][k]:.2f}")
        L.append(f"| {name} | " + " | ".join(vals) + " |")
    L.append("")
    L.append("Seed SD: " + "; ".join(
        f"{ds}: " + "/".join(f"{100 * v:.2f}" for v in t["seed_sd"].values())
        for ds, t in res["table"].items()))
    L.append("")
    L.append("## Paired gaps (AUROC x100; per-seed group-clustered 95% CI)")
    L.append("")
    L.append("| Gap | Decomp | MCP | MRT |")
    L.append("|---|---|---|---|")
    for name in ("g3_minus_struct", "g3_minus_text", "struct_minus_text"):
        row = []
        for ds in ("decomp", "mcp", "mrt"):
            t = res["table"].get(ds)
            if not t or name not in t:
                row.append("n/a")
                continue
            d = t[name]
            ci = [p[name]["ci95_group_cluster"] for p in res["per_seed"][ds]]
            note = ""
            if ds == "mrt" and all(p[name].get("n_groups_per_cell") == [1] * p[name]["n_cells"]
                                   for p in res["per_seed"][ds]):
                note = " (degenerate: 1 group per task cell -> use cluster sign test below)"
            row.append(f"{100 * d['mean_delta']:+.2f}; CI excl 0 in {d['n_seeds_ci_excl_0']}/3 seeds{note}; "
                       f"per-seed {', '.join(f'[{100 * c[0]:+.2f},{100 * c[1]:+.2f}]' for c in ci)}")
        L.append(f"| {name} | " + " | ".join(row) + " |")
    L.append("")
    L.append("## Per-seed AUROC x100 (weighted per-cell mean)")
    L.append("")
    L.append("| dataset | seed | graphs | G3 | structure | text |")
    L.append("|---|---|---|---|---|---|")
    for ds, ps in res["per_seed"].items():
        for p in ps:
            L.append(f"| {ds} | {p['seed']} | {p['n_graphs']} | {100 * p['auroc']['g3']:.2f} | "
                     f"{100 * p['auroc']['struct']:.2f} | {100 * p['auroc']['text']:.2f} |")
    L.append("")
    L.append("## Per-cell AUROC x100 (raw, for audit)")
    L.append("")
    L.append("| dataset | seed | cell | G3 | structure | text |")
    L.append("|---|---|---|---|---|---|")
    for ds, ps in res["per_seed"].items():
        for p in ps:
            for cell, vals in p["per_cell"].items():
                L.append(f"| {ds} | {p['seed']} | {cell} | {100 * vals['g3']:.2f} | "
                         f"{100 * vals['struct']:.2f} | {100 * vals['text']:.2f} |")
    L.append("")
    if any("mrt_cluster_tests" in p for p in res["per_seed"].get("mrt", [])):
        L.append("## MRT task-cluster tests (per-cell AUROC deltas)")
        L.append("")
        for p in res["per_seed"].get("mrt", []):
            for name, t in (p.get("mrt_cluster_tests") or {}).items():
                L.append(f"- seed {p['seed']} {name}: " +
                         ", ".join(f"{c['cluster']}(n={c['n']}) {100 * c['delta_auroc']:+.2f} "
                                   for c in t["clusters"]) +
                         f"; sign test {t['n_positive']}/{t['n_clusters']} positive, "
                         f"p={t['p_two_sided_exact']:.3f}")
        L.append("")
    L.append("## Disclosure (must stay in main text)")
    L.append("")
    L.append("- S3_reg (regularized structure-only) MRT = 89.47 > G3_p25 87.30; the MRT level is")
    L.append("  reproduced by structure-channel regularization (LN+dropout), not by joint modeling")
    L.append("  (EXP-2026W38-055). Component ablations -> appendix.")
    L.append("- Test split numbers above are the same legacy test split used in the manuscript table;")
    L.append("  the held-out group evaluation (reused) is a separate column and must not be merged.")
    (OUT / "core_ablation_stats.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
