#!/usr/bin/env python3
"""Independent re-verification of Table 5 Panel B from raw per-graph scores.

Zero API, zero training. Reads the 40 archived REAL-CL p16 runs
(code/repro/out/real_cl/p16_40/) and:

  1. checks protocol identity across arms/runs (flow, seeds, order, budget,
     evaluated task sets);
  2. recomputes every acc_matrix cell from scores.jsonl (pred = score >= 0.5)
     and cross-checks against per_item.jsonl and results.json;
  3. recomputes F_T=(1/T)*sum_j[max_{t=j..T} a_tj - a_Tj] and compares with
     the archived results.json values;
  4. recomputes the Panel B aggregates with two uncertainty models:
       - 40-run iid bootstrap (the archived convention, kept for continuity);
       - 20-seed cluster bootstrap (seed is the independent unit; the two
         orders of the same seed are correlated and are resampled together).

Usage:
  python verify_table5_panelb.py --src code/repro/out/real_cl/p16_40 \
      --out-json LOGS/final_five_tables/raw_results/table5_panelb_verify.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata as _rankdata

DOMAINS = ["financial_analysis", "web_search", "location_navigation",
           "repository_management", "browser_automation"]
ARMS = ["naive", "specialists_random", "specialists_all", "specialists",
        "specialists_oracle"]


def load_run(rd: Path) -> dict:
    config = json.loads((rd / "config.json").read_text())
    results = json.loads((rd / "results.json").read_text())
    per_item = {}
    for line in (rd / "per_item.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            per_item[(r["arm"], r["stage"], r["domain"], r["split"])] = r
    scores = defaultdict(list)
    for line in (rd / "scores.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            scores[(r["arm"], r["stage"], r["domain"])].append(r)
    return {"dir": rd.name, "config": config, "results": results,
            "per_item": per_item, "scores": scores}


def recompute_acc(cell: list[dict]) -> float:
    correct = sum(1 for r in cell if (1 if r["score"] >= 0.5 else 0) == r["label"])
    return correct / len(cell)


def recompute_matrix(run: dict, arm: str) -> dict:
    """scores.jsonl stage is 1-based; results.json acc_matrix stage is 0-based."""
    matrix = {}
    for (a, stage, domain), cell in run["scores"].items():
        if a != arm:
            continue
        matrix[(stage - 1, domain)] = recompute_acc(cell)
    return matrix


def recompute_forgetting(matrix: dict, task_order: list[str]) -> float:
    n = len(task_order)
    forget = []
    for i, d in enumerate(task_order):
        vals = [matrix[(j, d)] for j in range(i, n) if (j, d) in matrix]
        if not vals:
            continue
        best = max(vals)
        final = matrix.get((n - 1, d))
        if final is not None:
            forget.append(best - final)
    return float(np.mean(forget)) if forget else float("nan")


def macro_acc(matrix: dict, task_order: list[str]) -> float:
    n = len(task_order)
    vals = [matrix[(n - 1, d)] for d in task_order if (n - 1, d) in matrix]
    return float(np.mean(vals))


def wilcoxon_p(x: list[float], y: list[float]) -> tuple[float | None, int]:
    """Two-sided Wilcoxon signed-rank (normal approx, as in the archive)."""
    d = [a - b for a, b in zip(x, y) if a is not None and b is not None]
    d = [v for v in d if v != 0]
    n = len(d)
    if n < 5:
        return None, n
    ranks = _rankdata([abs(v) for v in d])
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mu) / sigma if sigma else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return min(1.0, p), n


def iid_bootstrap_ci(vals: list[float], n_boot: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def cluster_bootstrap_ci(by_seed: dict[str, list[float]], n_boot: int = 10000,
                         seed: int = 0) -> list[float]:
    """Resample seeds (clusters) with replacement; keep all runs within a seed."""
    rng = np.random.default_rng(seed)
    seeds = sorted(by_seed)
    means = []
    for _ in range(n_boot):
        draw = rng.choice(len(seeds), size=len(seeds), replace=True)
        vals = [v for i in draw for v in by_seed[seeds[i]]]
        means.append(float(np.mean(vals)))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def seed_of(name: str) -> str:
    return name.split("_")[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="code/repro/out/real_cl/p16_40")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    src = Path(args.src)
    run_dirs = sorted(d for d in src.iterdir() if d.is_dir() and "PARTIAL" not in d.name)
    runs = [load_run(d) for d in run_dirs]
    assert len(runs) == 40, f"expected 40 runs, found {len(runs)}"

    checks: dict = {"n_runs": len(runs), "errors": []}
    seeds = sorted(seed_of(r["dir"]) for r in runs)
    checks["seeds"] = seeds
    checks["orders"] = sorted(set(r["config"]["order"] for r in runs))
    checks["runs_per_seed_order"] = all(
        sum(1 for r in runs if seed_of(r["dir"]) == s and r["config"]["order"] == o) == 1
        for s in sorted(set(seeds)) for o in ("forward", "reversed"))

    for key in ("updates", "batch", "buffer", "ewc_lambda", "joint_mult",
                "data_expanded", "report_test"):
        vals = {r["config"].get(key) for r in runs}
        if len(vals) != 1:
            checks["errors"].append(f"config mismatch {key}: {vals}")
        checks[key] = vals.pop() if len(vals) == 1 else sorted(map(str, vals))

    run_checks: list[dict] = []
    per_run: dict[str, dict] = {}
    for r in runs:
        name = r["dir"]
        task_order = [d for d, _ in sorted(
            {k.split("|")[1]: min(int(k.split("|")[0]), 99)
             for k in r["results"][ARMS[0]]["acc_matrix"]}.items(),
            key=lambda kv: kv[1])]
        assert task_order == r["config"]["order_list"], (
            f"{name}: derived order {task_order} != config order_list "
            f"{r['config']['order_list']}")
        per = {}
        for arm in ARMS:
            matrix = recompute_matrix(r, arm)
            if not matrix:
                checks["errors"].append(f"{name}: no scores for arm {arm}")
                continue
            archived = {tuple(k.split("|")): v for k, v in
                        r["results"][arm]["acc_matrix"].items()}
            mism = []
            for (stage, domain), v in matrix.items():
                key = (str(stage), domain)
                if key in archived and abs(archived[key] - v) > 5e-4:
                    mism.append((key, archived[key], v))
                pi = r["per_item"].get((arm, stage + 1, domain, "test"))
                if pi is not None and abs(pi["acc"] - v) > 1e-9:
                    mism.append((key, "per_item", pi["acc"], v))
            fog_re = recompute_forgetting(matrix, task_order)
            fog_ar = r["results"][arm]["forgetting"]
            if fog_ar is not None and abs(fog_ar - fog_re) > 1e-9:
                mism.append(("forgetting", fog_ar, fog_re))
            per[arm] = {
                "macro_acc_recomputed": macro_acc(matrix, task_order),
                "macro_acc_archived_last_row": (
                    st.mean([r["results"][arm]["acc_matrix"][f"{len(task_order)-1}|{d}"]
                             for d in task_order
                             if f"{len(task_order)-1}|{d}" in r["results"][arm]["acc_matrix"]])),
                "forgetting_recomputed": fog_re,
                "forgetting_archived": fog_ar,
                "router_hit": r["results"][arm].get("router_hit"),
                "mismatches": mism,
            }
        per_run[name] = {"seed": seed_of(name), "order": r["config"]["order"],
                         "task_order": task_order, "arms": per}
        run_checks.append({"run": name, "n_mismatches": sum(
            len(v["mismatches"]) for v in per.values())})

    checks["runs_with_zero_recompute_mismatch"] = sum(
        1 for c in run_checks if c["n_mismatches"] == 0)
    checks["max_recompute_mismatch"] = max(c["n_mismatches"] for c in run_checks)

    summary: dict = {}
    for arm in ARMS:
        acc40 = [per_run[n]["arms"][arm]["macro_acc_recomputed"] for n in per_run]
        fog40 = [per_run[n]["arms"][arm]["forgetting_recomputed"] for n in per_run]
        acc_by_seed = defaultdict(list)
        fog_by_seed = defaultdict(list)
        for n, rec in per_run.items():
            acc_by_seed[rec["seed"]].append(rec["arms"][arm]["macro_acc_recomputed"])
            fog_by_seed[rec["seed"]].append(rec["arms"][arm]["forgetting_recomputed"])
        seed_means_acc = [float(np.mean(v)) for v in acc_by_seed.values()]
        seed_means_fog = [float(np.mean(v)) for v in fog_by_seed.values()]
        summary[arm] = {
            "macro_acc": {
                "mean_40runs": float(np.mean(acc40)),
                "std_40runs": float(np.std(acc40)),
                "mean_20seeds": float(np.mean(seed_means_acc)),
                "std_20seeds": float(np.std(seed_means_acc)),
                "ci95_iid_40runs": iid_bootstrap_ci(acc40),
                "ci95_cluster_seed": cluster_bootstrap_ci(acc_by_seed),
            },
            "forgetting": {
                "mean_40runs": float(np.mean(fog40)),
                "std_40runs": float(np.std(fog40)),
                "mean_20seeds": float(np.mean(seed_means_fog)),
                "std_20seeds": float(np.std(seed_means_fog)),
                "ci95_iid_40runs": iid_bootstrap_ci(fog40),
                "ci95_cluster_seed": cluster_bootstrap_ci(fog_by_seed),
            },
            "router_hit": {
                "mean_40runs": float(np.mean([per_run[n]["arms"][arm]["router_hit"]
                                              for n in per_run
                                              if per_run[n]["arms"][arm]["router_hit"] is not None])),
            },
        }

    paired: dict = {}
    naive_acc = {n: per_run[n]["arms"]["naive"]["macro_acc_recomputed"] for n in per_run}
    naive_fog = {n: per_run[n]["arms"]["naive"]["forgetting_recomputed"] for n in per_run}
    for arm in ARMS:
        if arm == "naive":
            continue
        d_acc = [per_run[n]["arms"][arm]["macro_acc_recomputed"] - naive_acc[n]
                 for n in per_run]
        d_fog = [per_run[n]["arms"][arm]["forgetting_recomputed"] - naive_fog[n]
                 for n in per_run]
        d_acc_seed = defaultdict(list)
        d_fog_seed = defaultdict(list)
        for n, rec in per_run.items():
            d_acc_seed[rec["seed"]].append(
                rec["arms"][arm]["macro_acc_recomputed"] - naive_acc[n])
            d_fog_seed[rec["seed"]].append(
                rec["arms"][arm]["forgetting_recomputed"] - naive_fog[n])
        seed_acc = [float(np.mean(v)) for v in d_acc_seed.values()]
        seed_fog = [float(np.mean(v)) for v in d_fog_seed.values()]
        p_acc40, n_acc40 = wilcoxon_p(
            [per_run[n]["arms"][arm]["macro_acc_recomputed"] for n in per_run],
            [naive_acc[n] for n in per_run])
        p_fog40, n_fog40 = wilcoxon_p(
            [per_run[n]["arms"][arm]["forgetting_recomputed"] for n in per_run],
            [naive_fog[n] for n in per_run])
        p_acc_seed, _ = wilcoxon_p(seed_acc, [0.0] * len(seed_acc)) if len(seed_acc) >= 5 else (None, 0)
        p_fog_seed, _ = wilcoxon_p(seed_fog, [0.0] * len(seed_fog)) if len(seed_fog) >= 5 else (None, 0)
        paired[arm] = {
            "delta_macro_acc_40runs": {
                "mean": float(np.mean(d_acc)), "std": float(np.std(d_acc)),
                "wins": sum(1 for v in d_acc if v > 0),
                "ci95_iid": iid_bootstrap_ci(d_acc),
                "ci95_cluster_seed": cluster_bootstrap_ci(d_acc_seed),
                "p_wilcoxon_40runs": p_acc40, "n_pairs_40runs": n_acc40,
                "p_wilcoxon_20seedmeans": p_acc_seed,
            },
            "delta_forgetting_40runs": {
                "mean": float(np.mean(d_fog)), "std": float(np.std(d_fog)),
                "improved": sum(1 for v in d_fog if v < 0),
                "ci95_iid": iid_bootstrap_ci(d_fog),
                "ci95_cluster_seed": cluster_bootstrap_ci(d_fog_seed),
                "p_wilcoxon_40runs": p_fog40, "n_pairs_40runs": n_fog40,
                "p_wilcoxon_20seedmeans": p_fog_seed,
            },
        }

    def compare(x_by_run: dict[str, float], y_by_run: dict[str, float]) -> dict:
        delta = {n: x_by_run[n] - y_by_run[n] for n in x_by_run}
        by_seed = defaultdict(list)
        for n, v in delta.items():
            by_seed[seed_of(n)].append(v)
        seed_vals = [float(np.mean(v)) for v in by_seed.values()]
        p40, n40 = wilcoxon_p(list(delta.values()), [0.0] * len(delta))
        p20, _ = wilcoxon_p(seed_vals, [0.0] * len(seed_vals))
        return {
            "mean": float(np.mean(list(delta.values()))),
            "std": float(np.std(list(delta.values()))),
            "wins": sum(1 for v in delta.values() if v > 0),
            "n": len(delta),
            "ci95_iid": iid_bootstrap_ci(list(delta.values())),
            "ci95_cluster_seed": cluster_bootstrap_ci(by_seed),
            "p_wilcoxon_40runs": p40,
            "p_wilcoxon_20seedmeans": p20,
        }

    pairwise = {}
    for a, b in (("specialists", "specialists_all"),
                 ("specialists", "specialists_random"),
                 ("specialists_all", "specialists_random"),
                 ("specialists_oracle", "specialists")):
        pairwise[f"{a}_minus_{b}"] = {
            "macro_acc": compare(
                {n: per_run[n]["arms"][a]["macro_acc_recomputed"] for n in per_run},
                {n: per_run[n]["arms"][b]["macro_acc_recomputed"] for n in per_run}),
            "forgetting": compare(
                {n: per_run[n]["arms"][a]["forgetting_recomputed"] for n in per_run},
                {n: per_run[n]["arms"][b]["forgetting_recomputed"] for n in per_run}),
        }

    out = {
        "table": "table5_panelb_verification",
        "source": str(src),
        "method": "recomputed from per-graph scores.jsonl (pred = score >= 0.5)",
        "protocol_checks": checks,
        "summary_recomputed": summary,
        "paired_vs_naive_recomputed": paired,
        "pairwise_recomputed": pairwise,
        "per_run": per_run,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    print(f"runs with zero mismatch: {checks['runs_with_zero_recompute_mismatch']}/40")
    for arm in ARMS:
        s = summary[arm]
        print(f"{arm:22s} ACC {s['macro_acc']['mean_40runs']:.4f}"
              f"±{s['macro_acc']['std_40runs']:.4f} | "
              f"F_T {s['forgetting']['mean_40runs']:.4f}"
              f"±{s['forgetting']['std_40runs']:.4f} | "
              f"router {s['router_hit']['mean_40runs']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
