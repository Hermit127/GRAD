#!/usr/bin/env python3
"""GRAD repair confirmation report (frozen-scheme replication; legacy test exposed).

PI 2026-09-21: finish the frozen four-arm confirmation, no mid-batch changes, report
all cells/seeds/failures. This batch uses the already-exposed legacy test split, so
it is labeled "frozen-scheme replication", not independent confirmation.

Inputs:
  code/repro/out/grad_repair/confirm/<ARM>_<ds>_<cell>_s<seed>.json
    ARM in {G3_512 (candidate), G3_192, TXT_512, STRUCT}
  legacy manuscript rows (same 100/25 protocol, test):
    code/repro/out/table5_g3/confirm/G3_p25_<ds>_<cell>_s<seed>.json
    code/repro/out/table5_ablation/runs/<ds>_<cell>_<structure_only|text_only>_s<seed>.json
Outputs:
  LOGS/final_five_tables/table5/grad_repair_confirm_report.{json,md}
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
from analyze_attribution import auroc, group_of  # noqa: E402

CONFIRM = REPO / "code/repro/out/grad_repair/confirm"
LEGACY_G3 = REPO / "code/repro/out/table5_g3/confirm"
LEGACY_REF = REPO / "code/repro/out/table5_ablation/runs"
OUT = REPO / "LOGS/final_five_tables/table5"
CELLS = [("mcp", f"fold{k}") for k in range(5)] + [("decomp", "fold0")] + [
    ("mrt", c) for c in ("fold_bank_transfer_2", "fold_api_key_calendar_agendas_2",
                         "fold_american_airlines_2", "fold_expenditure_accounting",
                         "fold_spam_filter_update")]
SEEDS = [211, 212, 213]
N_BOOT = 10000
CANDIDATE = "G3_512"
BASELINES = {
    "STRUCT_fresh": lambda ds, c, s: CONFIRM / f"STRUCT_{ds}_{c}_s{s}.json",
    "TXT512_fresh": lambda ds, c, s: CONFIRM / f"TXT_512_{ds}_{c}_s{s}.json",
    "G3_192_fresh": lambda ds, c, s: CONFIRM / f"G3_192_{ds}_{c}_s{s}.json",
    "structure_legacy": lambda ds, c, s: LEGACY_REF / f"{ds}_{c}_structure_only_s{s}.json",
    "text_legacy": lambda ds, c, s: LEGACY_REF / f"{ds}_{c}_text_only_s{s}.json",
    "G3_legacy": lambda ds, c, s: LEGACY_G3 / f"G3_p25_{ds}_{c}_s{s}.json",
}


def resolve_local(path_str: str) -> Path | None:
    p = str(path_str or "")
    for marker, base in (("/fix2_splits/", REPO / "code/repro/out/unified_gnn/fix2_splits"),
                         ("/mrt_cross_task_v1/", REPO / "results/mrt_cross_task_v1")):
        if marker in p:
            return base / p.split(marker, 1)[1]
    return None


_graph_cache: dict = {}


def graphs_of(ds: str, cell: str, cfg: dict) -> list[dict]:
    key = (ds, cell)
    if key not in _graph_cache:
        split = resolve_local((cfg or {}).get("test", ""))
        _graph_cache[key] = [json.loads(l) for l in split.read_text().splitlines() if l.strip()] \
            if split and split.exists() else []
    return _graph_cache[key]


def load_method(ds: str, cell: str, seed: int, path_fn) -> dict | None:
    p = path_fn(ds, cell, seed)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rows = d.get("scores", {}).get("test", [])
    graphs = graphs_of(ds, cell, d.get("config") or {})
    if not rows or len(rows) != len(graphs):
        return None
    out = {}
    for g, r in zip(graphs, rows):
        assert int(g["label"]) == int(r["label"])
        tid = str(g.get("task_id"))
        out[tid] = {"label": int(g["label"]), "group": group_of(ds, g), "score": float(r["score"])}
    return out


def weighted(vals, weights):
    vals, weights = np.asarray(vals, float), np.asarray(weights, float)
    return float(np.sum(vals * weights) / np.sum(weights))


def per_seed_report(method_path_fn, cand_path_fn) -> dict:
    """Per dataset: per-seed weighted AUROC, delta vs baseline, cluster-bootstrap CI."""
    res = {}
    for ds in ("decomp", "mcp", "mrt"):
        ds_cells = [c for d, c in CELLS if d == ds]
        per_seed = []
        for seed in SEEDS:
            cells_cand, cells_base = [], []
            for cell in ds_cells:
                gc, gb = load_method(ds, cell, seed, cand_path_fn), load_method(ds, cell, seed, method_path_fn)
                if gc is None or gb is None:
                    continue
                tids = sorted(set(gc) & set(gb))
                if len(tids) < 4:
                    continue
                y = np.array([gc[t]["label"] for t in tids])
                if len(set(y.tolist())) < 2:
                    continue
                sc = np.array([gc[t]["score"] for t in tids])
                sb = np.array([gb[t]["score"] for t in tids])
                groups = defaultdict(list)
                for i, t in enumerate(tids):
                    groups[gc[t]["group"]].append(i)
                cells_cand.append({"y": y, "sc": sc, "sb": sb,
                                   "groups": [np.array(v) for v in groups.values()],
                                   "w": float(y.sum() * (len(y) - y.sum()))})
            if not cells_cand:
                continue
            a_c = weighted([auroc(c["y"], c["sc"]) for c in cells_cand], [c["w"] for c in cells_cand])
            a_b = weighted([auroc(c["y"], c["sb"]) for c in cells_cand], [c["w"] for c in cells_cand])
            d0 = a_c - a_b
            rng = np.random.default_rng(0)
            draws = []
            for _ in range(N_BOOT):
                idxs = []
                for c in cells_cand:
                    take = rng.integers(0, len(c["groups"]), len(c["groups"]))
                    idx = np.concatenate([c["groups"][g] for g in take])
                    if len(set(c["y"][idx].tolist())) < 2:
                        continue
                    idxs.append((c, idx))
                if not idxs:
                    continue
                w = np.array([c["w"] for c, _ in idxs])
                da = np.array([auroc(c["y"][i], c["sc"][i]) - auroc(c["y"][i], c["sb"][i])
                               for c, i in idxs])
                draws.append(float(np.sum(w * da) / np.sum(w)))
            ci = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
            per_seed.append({"seed": seed, "auroc_candidate": a_c, "auroc_baseline": a_b,
                            "delta": d0, "ci95": [float(ci[0]), float(ci[1])],
                            "n_cells": len(cells_cand)})
        d = [p["delta"] for p in per_seed]
        excl = sum(1 for p in per_seed if p["ci95"][0] > 0 or p["ci95"][1] < 0)
        res[ds] = {"per_seed": per_seed,
                   "mean_delta": float(np.mean(d)) if d else None,
                   "n_seeds": len(per_seed),
                   "n_seeds_ci_excl_0": excl}
    return res


def mrt_task_sign(method_path_fn, cand_path_fn) -> dict:
    out = {}
    for seed in SEEDS:
        deltas = []
        for ds, cell in CELLS:
            if ds != "mrt":
                continue
            gc, gb = load_method(ds, cell, seed, cand_path_fn), load_method(ds, cell, seed, method_path_fn)
            if gc is None or gb is None:
                continue
            tids = sorted(set(gc) & set(gb))
            y = [gc[t]["label"] for t in tids]
            if len(set(y)) < 2:
                continue
            deltas.append({"cell": cell,
                           "delta": float(auroc(y, [gc[t]["score"] for t in tids])
                                          - auroc(y, [gb[t]["score"] for t in tids]))})
        n = len(deltas)
        k = sum(d["delta"] > 0 for d in deltas)
        tail = sum(comb(n, i) for i in range(n + 1) if i <= min(k, n - k))
        out[seed] = {"clusters": deltas, "n_positive": k, "n_clusters": n,
                     "p_two_sided_exact": min(1.0, tail / 2 ** n) if n else None}
    return out


def main() -> None:
    cand_path_fn = lambda ds, c, s: CONFIRM / f"{CANDIDATE}_{ds}_{c}_s{s}.json"
    n_cand = sum(1 for ds, c in CELLS for s in SEEDS if cand_path_fn(ds, c, s).exists())
    report: dict = {"label": "frozen-scheme replication (legacy test, exposed)",
                    "candidate": CANDIDATE, "candidate_runs": f"{n_cand}/33",
                    "comparisons": {}, "mrt_sign_tests": {}}
    for name, fn in BASELINES.items():
        report["comparisons"][name] = per_seed_report(fn, cand_path_fn)
        if name != "G3_192_fresh":
            report["mrt_sign_tests"][name] = mrt_task_sign(fn, cand_path_fn)
    # dataset-level AUROC table for all arms
    table = {}
    for name, fn in [("candidate", cand_path_fn)] + [(k, v) for k, v in BASELINES.items()]:
        table[name] = {}
        for ds in ("decomp", "mcp", "mrt"):
            ds_cells = [c for d, c in CELLS if d == ds]
            per_seed = []
            for seed in SEEDS:
                cells = []
                for cell in ds_cells:
                    g = load_method(ds, cell, seed, fn)
                    if g is None:
                        continue
                    tids = sorted(g)
                    y = [g[t]["label"] for t in tids]
                    if len(set(y)) < 2:
                        continue
                    cells.append({"y": y, "s": [g[t]["score"] for t in tids],
                                  "w": float(sum(y) * (len(y) - sum(y)))})
                if cells:
                    per_seed.append(weighted([auroc(c["y"], c["s"]) for c in cells],
                                             [c["w"] for c in cells]))
            table[name][ds] = {"per_seed": per_seed,
                               "mean": float(np.mean(per_seed)) if per_seed else None}
    report["table"] = table
    (OUT / "grad_repair_confirm_report.json").write_text(json.dumps(report, indent=2))
    write_md(report)
    print("wrote", OUT / "grad_repair_confirm_report.json")


def write_md(report: dict) -> None:
    L = ["# GRAD repair confirmation report", "",
         "Label: **frozen-scheme replication (legacy test, exposed)** — not independent confirmation.",
         f"Candidate: {report['candidate']} ({report['candidate_runs']} test runs). All cells/seeds below.", ""]
    L.append("## AUROC x100 (per-seed mean, test)")
    L.append("")
    L.append("| arm | Decomp | MCP | MRT |")
    L.append("|---|---|---|---|")
    for name, v in report["table"].items():
        row = []
        for ds in ("decomp", "mcp", "mrt"):
            m = v[ds]["mean"]
            ns = len(v[ds]["per_seed"])
            row.append("n/a" if m is None else f"{100 * m:.2f} (n={ns})")
        L.append(f"| {name} | " + " | ".join(row) + " |")
    L.append("")
    L.append("## Paired deltas candidate - baseline (AUROC x100, group-clustered CI)")
    L.append("")
    for name, c in report["comparisons"].items():
        L.append(f"### vs {name}")
        L.append("")
        L.append("| dataset | mean delta | seeds CI excl 0 | per-seed [CI] |")
        L.append("|---|---|---|---|")
        for ds, v in c.items():
            if v["mean_delta"] is None:
                L.append(f"| {ds} | n/a | n/a | n/a |")
                continue
            ci = ", ".join(f"[{100 * p['ci95'][0]:+.2f},{100 * p['ci95'][1]:+.2f}]" for p in v["per_seed"])
            L.append(f"| {ds} | {100 * v['mean_delta']:+.2f} | {v['n_seeds_ci_excl_0']}/{v['n_seeds']} | {ci} |")
        L.append("")
        for ds in ("mrt",):
            if name in report["mrt_sign_tests"]:
                L.append(f"MRT task sign tests: " + "; ".join(
                    f"s{s}: {t['n_positive']}/{t['n_clusters']} positive, p={t['p_two_sided_exact']:.3f}"
                    for s, t in report["mrt_sign_tests"][name].items()))
                L.append("")
    (OUT / "grad_repair_confirm_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
