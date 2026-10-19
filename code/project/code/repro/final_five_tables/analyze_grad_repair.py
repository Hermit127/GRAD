#!/usr/bin/env python3
"""GRAD repair pilot analysis + pre-registered decision rule (dev only).

Inputs:
  code/repro/out/grad_repair/pilot/<arm>_<ds>_<cell>_s<seed>.json
  code/repro/out/table5_ablation/runs/<ds>_<cell>_structure_only_s<seed>.json (reference)
Outputs:
  LOGS/final_five_tables/table5/grad_repair_pilot_report.{json,md}

Rule (grad_repair_prereg_20260921.md §5): a candidate advances only if
  1. MCP AUROC >= G3_192 + 1.0pp
  2. Decomp/MRT no cell below G3_192 by > 1.5pp
  3. MCP structure-correct/fused-wrong flip rate reduced >= 1/3 relative to G3_192
  4. (G4 only) s_s AUROC on MCP >= structure_only - 1.0pp
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))
from analyze_attribution import auroc  # noqa: E402

PILOT = REPO / "code/repro/out/grad_repair/pilot"
REF = REPO / "code/repro/out/table5_ablation/runs"
OUT = REPO / "LOGS/final_five_tables/table5"
ARMS = ["G3_192", "G3_512", "G4_192", "G4_512"]
CELLS = [("mcp", "fold0"), ("mcp", "fold1"), ("decomp", "fold0"),
         ("mrt", "fold_bank_transfer_2"), ("mrt", "fold_american_airlines_2")]
SEEDS = [211, 212]
MCP_KEYS = ["mcp_fold0", "mcp_fold1"]
PREF = {"G3_512": 0, "G4_192": 1, "G4_512": 2}


def resolve_local(path_str: str) -> Path | None:
    p = str(path_str or "")
    for marker, base in (("/fix2_splits/", REPO / "code/repro/out/unified_gnn/fix2_splits"),
                         ("/mrt_cross_task_v1/", REPO / "results/mrt_cross_task_v1")):
        if marker in p:
            return base / p.split(marker, 1)[1]
    return None


def margins(labels, scores) -> np.ndarray:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    out = np.full(len(scores), np.nan)
    for i, (y, s) in enumerate(zip(labels, scores)):
        opp = neg if y == 1 else pos
        if opp.size:
            above = float(np.mean(s > opp) + 0.5 * np.mean(s == opp))
            out[i] = above if y == 1 else 1.0 - above
    return out


def load_unit(arm: str, ds: str, cell: str, seed: int) -> dict | None:
    p = PILOT / f"{arm}_{ds}_{cell}_s{seed}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rows = d.get("scores", {}).get("val", [])
    split = resolve_local((d.get("config") or {}).get("val", ""))
    if split is None or not split.exists() or not rows:
        return None
    graphs = [json.loads(l) for l in split.read_text().splitlines() if l.strip()]
    if len(graphs) != len(rows):
        return None
    ref_p = REF / f"{ds}_{cell}_structure_only_s{seed}.json"
    ref_rows = json.loads(ref_p.read_text()).get("scores", {}).get("val", []) if ref_p.exists() else []
    recs = {}
    for i, (g, r) in enumerate(zip(graphs, rows)):
        tid = str(g.get("task_id"))
        rec = {"tid": tid, "label": int(g["label"]), "fused": float(r["score"]),
               "s_s": r.get("s_s"), "s_t": r.get("s_t"), "alpha": r.get("alpha")}
        if len(ref_rows) == len(graphs):
            assert int(ref_rows[i]["label"]) == int(g["label"])
            rec["struct"] = float(ref_rows[i]["score"])
        recs[tid] = rec
    tids = sorted(recs)
    y = [recs[t]["label"] for t in tids]
    recs_arr = [recs[t] for t in tids]
    for r, m in zip(recs_arr, margins(y, [r["fused"] for r in recs_arr])):
        r["m_fused"] = m
    has_struct = all(r.get("struct") is not None for r in recs_arr)
    has_ss = all(r.get("s_s") is not None for r in recs_arr)
    if has_struct:
        for r, m in zip(recs_arr, margins(y, [r["struct"] for r in recs_arr])):
            r["m_struct"] = m
    return {
        "arm": arm, "ds": ds, "cell": cell, "seed": seed, "recs": recs,
        "auroc": float(auroc(y, [r["fused"] for r in recs_arr])),
        "auroc_struct_ref": float(auroc(y, [r["struct"] for r in recs_arr])) if has_struct else None,
        "auroc_s_s": float(auroc(y, [r["s_s"] for r in recs_arr])) if has_ss else None,
    }


def flip_stats(recs: dict) -> dict:
    pos = [r for r in recs.values() if r["label"] == 1 and "m_struct" in r]
    neg = [r for r in recs.values() if r["label"] == 0 and "m_struct" in r]
    n_pairs = flips = 0
    for p in pos:
        for n in neg:
            if p["m_struct"] > 0.5:
                n_pairs += 1
                if p["m_fused"] <= 0.5:
                    flips += 1
    return {"struct_correct_pairs": n_pairs, "flips": flips,
            "flip_rate": flips / max(1, n_pairs)}


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    units = [u for u in (load_unit(a, ds, c, s) for a in ARMS for ds, c in CELLS for s in SEEDS) if u]
    per_cell: dict = {a: {} for a in ARMS}
    for a in ARMS:
        for ds, cell in CELLS:
            vals = [u["auroc"] for u in units if u["arm"] == a and u["ds"] == ds and u["cell"] == cell]
            if len(vals) == len(SEEDS):
                per_cell[a][f"{ds}_{cell}"] = float(np.mean(vals))

    mcp_gain, other_delta = {}, {}
    for a in ARMS:
        mcp = mean([per_cell[a].get(k) for k in MCP_KEYS])
        base = mean([per_cell["G3_192"].get(k) for k in MCP_KEYS])
        mcp_gain[a] = (mcp - base) if (mcp is not None and base is not None) else None
        other_delta[a] = {}
        for ds, cell in CELLS:
            if ds == "mcp":
                continue
            v, b = per_cell[a].get(f"{ds}_{cell}"), per_cell["G3_192"].get(f"{ds}_{cell}")
            if v is not None and b is not None:
                other_delta[a][f"{ds}_{cell}"] = v - b

    mcp_flip, other_flip, ss_auroc, struct_ref_mcp = {}, {}, {}, {}
    for a in ARMS:
        f_mcp, f_other = [], []
        for ds, cell in CELLS:
            for u in units:
                if u["arm"] != a or u["ds"] != ds or u["cell"] != cell:
                    continue
                fs = flip_stats(u["recs"])
                if fs["struct_correct_pairs"] == 0:
                    continue
                (f_mcp if ds == "mcp" else f_other).append(fs["flip_rate"])
        mcp_flip[a] = mean(f_mcp)
        other_flip[a] = mean(f_other)
        ss = [u["auroc_s_s"] for u in units if u["arm"] == a and u["ds"] == "mcp" and u["auroc_s_s"] is not None]
        ss_auroc[a] = mean(ss)
        struct_ref_mcp[a] = mean([u["auroc_struct_ref"] for u in units if u["arm"] == a and u["ds"] == "mcp"])

    baseline_flip = mcp_flip.get("G3_192")
    decisions = {}
    for a in ("G3_512", "G4_192", "G4_512"):
        ss_ok = (a == "G3_512")
        if not ss_ok and ss_auroc.get(a) is not None and struct_ref_mcp.get(a) is not None:
            ss_ok = ss_auroc[a] >= struct_ref_mcp[a] - 0.01
        checks = {
            "mcp_gain_ge_1pp": (mcp_gain.get(a) is not None) and mcp_gain[a] >= 0.01,
            "no_other_cell_regress_gt_1.5pp": bool(other_delta.get(a)) and all(v >= -0.015 for v in other_delta[a].values()),
            "flip_rate_reduced_third": (baseline_flip not in (None, 0.0) and mcp_flip.get(a) is not None
                                        and mcp_flip[a] <= baseline_flip * (2 / 3)),
            "s_s_not_weakened": ss_ok,
        }
        decisions[a] = {"checks": checks, "pass": all(checks.values())}
    passed = [a for a in decisions if decisions[a]["pass"]]
    if passed:
        passed.sort(key=lambda a: (-(mcp_gain[a] or -9), PREF.get(a, 9)))
        winner = passed[0]
    else:
        winner = None
    report = {"units_seen": len(units), "expected_units": len(ARMS) * len(CELLS) * len(SEEDS),
              "per_cell_auroc": per_cell, "mcp_gain_vs_G3_192": mcp_gain,
              "other_cell_delta_vs_G3_192": other_delta,
              "mcp_flip_rate": mcp_flip, "other_flip_rate": other_flip,
              "mcp_s_s_auroc": ss_auroc, "mcp_structure_ref_auroc": struct_ref_mcp,
              "decisions": decisions, "winner": winner}
    (OUT / "grad_repair_pilot_report.json").write_text(json.dumps(report, indent=2))
    write_md(report)
    print("winner:", winner)


def write_md(report: dict) -> None:
    L = ["# GRAD repair pilot report (dev only)", "",
         "Pre-registered rule: `grad_repair_prereg_20260921.md` §5; 2 seeds x 5 cells per arm.",
         f"Units: {report['units_seen']}/{report['expected_units']}.", ""]
    L.append("## Per-cell dev AUROC x100 (2-seed mean)")
    L.append("")
    L.append("| arm | mcp fold0 | mcp fold1 | decomp fold0 | mrt bank | mrt airlines |")
    L.append("|---|---|---|---|---|---|")
    keys = ["mcp_fold0", "mcp_fold1", "decomp_fold0",
            "mrt_fold_bank_transfer_2", "mrt_fold_american_airlines_2"]
    for a in ARMS:
        row = []
        for k in keys:
            v = report["per_cell_auroc"].get(a, {}).get(k)
            row.append("n/a" if v is None else f"{100 * v:.2f}")
        L.append(f"| {a} | " + " | ".join(row) + " |")
    L.append("")
    L.append("## Pre-registered checks")
    L.append("")
    L.append("| arm | MCP gain (pp) | worst other-cell delta (pp) | MCP flip rate | s_s AUROC | struct ref | pass |")
    L.append("|---|---|---|---|---|---|---|")
    for a in ("G3_512", "G4_192", "G4_512", "G3_192"):
        gain = report["mcp_gain_vs_G3_192"].get(a)
        od = report["other_cell_delta_vs_G3_192"].get(a) or {}
        worst = min(od.values()) * 100 if od else float("nan")
        fl = report["mcp_flip_rate"].get(a)
        ss = report["mcp_s_s_auroc"].get(a)
        sref = report["mcp_structure_ref_auroc"].get(a)
        d = report["decisions"].get(a, {})
        L.append("| " + a
                 + " | " + (f"{100 * gain:+.2f}" if gain is not None else "n/a")
                 + " | " + f"{worst:+.2f}"
                 + " | " + (f"{fl:.3f}" if fl is not None else "n/a")
                 + " | " + (f"{100 * ss:.2f}" if ss is not None else "n/a")
                 + " | " + (f"{100 * sref:.2f}" if sref is not None else "n/a")
                 + " | " + ("YES" if d.get("pass") else "no") + " |")
    L.append("")
    L.append(f"**Winner: {report['winner'] or 'none'}**")
    (OUT / "grad_repair_pilot_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
