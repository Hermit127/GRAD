#!/usr/bin/env python3
"""Aggregate the Panel B fixed-pool router comparison (CPU preview or GPU run).

Reads code/repro/out/panelb_router/runs/*/results.json (2 orders x 3 seeds),
writes LOGS/final_five_tables/table5/panelb_router.{json,md}.

Statistics: run-level mean±std; seed-level means (both orders averaged) for the
cluster-aware view; paired run-level differences vs all-expert and vs centroid.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "code/repro/out/panelb_router/runs"
OUT = REPO / "LOGS/final_five_tables/table5"
STRATEGIES = ["random", "all", "centroid", "learned_top1",
              "learned_top1_fallback", "learned_top2",
              "learned_top2_fallback", "oracle"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--hardware", default="Apple M2 CPU (preview)")
    ap.add_argument("--out", default=str(OUT / "panelb_router.json"))
    args = ap.parse_args()
    runs = {}
    for p in sorted(Path(args.runs).glob("*/results.json")):
        d = json.loads(p.read_text())
        key = p.parent.name
        runs[key] = d
    if not runs:
        print("no runs")
        return 1

    rows = defaultdict(list)
    actives = defaultdict(list)
    latencies = defaultdict(list)
    for key, d in runs.items():
        for name, s in d["summary"].items():
            rows[name].append(s["final_macro_acc"])
            if s.get("avg_active_experts") is not None:
                actives[name].append(s["avg_active_experts"])
            latencies[name].append(s.get("routing_ms_per_graph"))

    summary = {}
    for name in STRATEGIES:
        vals = rows.get(name, [])
        if not vals:
            continue
        by_seed = defaultdict(list)
        for key in runs:
            order, seed = key.rsplit("_s", 1)
            by_seed[seed].append(runs[key]["summary"][name]["final_macro_acc"])
        summary[name] = {
            "final_macro_acc_mean": float(np.mean(vals)),
            "final_macro_acc_std": float(np.std(vals)),
            "seed_level_mean": float(np.mean([np.mean(v) for v in by_seed.values()])),
            "seed_level_std": float(np.std([np.mean(v) for v in by_seed.values()])),
            "forgetting_mean": float(np.mean([runs[k]["summary"][name]["forgetting"]
                                              for k in runs])),
            "avg_active_experts": float(np.mean(actives[name])) if actives[name] else None,
            "routing_ms_per_graph": float(np.nanmean(latencies[name]))
            if latencies[name] else None,
        }

    pairwise = {}
    for a, b in (("learned_top1_fallback", "all"), ("learned_top2_fallback", "all"),
                 ("learned_top1", "all"), ("centroid", "all"),
                 ("learned_top1_fallback", "centroid"),
                 ("learned_top2_fallback", "centroid"),
                 ("learned_top1_fallback", "random")):
        da = [runs[k]["summary"][a]["final_macro_acc"] -
              runs[k]["summary"][b]["final_macro_acc"] for k in runs]
        df = [runs[k]["summary"][a]["forgetting"] -
              runs[k]["summary"][b]["forgetting"] for k in runs]
        # seed-level pairing (average the two orders within a seed)
        seeds = sorted({k.rsplit("_s", 1)[1] for k in runs})
        sa, sf = [], []
        for s in seeds:
            ks = [k for k in runs if k.endswith(f"_s{s}")]
            sa.append(np.mean([runs[k]["summary"][a]["final_macro_acc"] -
                               runs[k]["summary"][b]["final_macro_acc"] for k in ks]))
            sf.append(np.mean([runs[k]["summary"][a]["forgetting"] -
                               runs[k]["summary"][b]["forgetting"] for k in ks]))
        pairwise[f"{a}_minus_{b}"] = {
            "acc_delta_run_mean": float(np.mean(da)),
            "acc_delta_run_std": float(np.std(da)),
            "acc_delta_seed_mean": float(np.mean(sa)),
            "acc_delta_per_seed": [round(v, 4) for v in sa],
            "acc_wins": int(sum(1 for v in da if v > 0)), "n_runs": len(da),
            "forget_delta_run_mean": float(np.mean(df)),
            "forget_delta_per_seed": [round(v, 4) for v in sf],
            "note": "n=3 seeds clustered; no significance claim at this n",
        }

    pool_bytes = next(iter(runs.values()))["pool_storage_bytes"]
    out = {"runs": sorted(runs), "hardware": args.hardware,
           "experiment_id": "table5-phase2-panelb-router-20260918",
           "summary": summary, "pairwise": pairwise,
           "pool_storage_bytes": pool_bytes,
           "protocol": next(iter(runs.values()))["config"]}

    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    md = [f"# Panel B fixed-pool router comparison ({args.hardware})", "",
          f"Runs: {len(runs)} (orders x seeds). All strategies share the same frozen "
          "expert pool per run; no calibration anywhere (raw sigmoid @0.5).", "",
          "| Strategy | ACC (run mean±std) | ACC (seed level) | F_T mean | active experts | route ms/graph |",
          "|---|---|---|---|---|---|"]
    for name, s in summary.items():
        act = "--" if s["avg_active_experts"] is None else f"{s['avg_active_experts']:.2f}"
        lat = "--" if s["routing_ms_per_graph"] is None or np.isnan(
            s["routing_ms_per_graph"]) else f"{s['routing_ms_per_graph']:.2f}"
        md.append(f"| {name} | {s['final_macro_acc_mean']:.4f}±{s['final_macro_acc_std']:.4f} "
                  f"| {s['seed_level_mean']:.4f}±{s['seed_level_std']:.4f} "
                  f"| {s['forgetting_mean']:.4f} | {act} | {lat} |")
    md += ["", "## Paired differences (seed-level means of the 3 seeds)", ""]
    for k, v in pairwise.items():
        md.append(f"- {k}: ACC {v['acc_delta_seed_mean']:+.4f} "
                  f"(per seed {v['acc_delta_per_seed']}), "
                  f"F_T {v['forget_delta_run_mean']:+.4f}")
    md += ["", f"Pool storage: backbone {pool_bytes['backbone']} B + adapters "
               f"{pool_bytes['experts_adapters']} B + heads {pool_bytes['experts_heads']} B; "
               f"5-expert pool file {pool_bytes['pool_file']} B."]
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
