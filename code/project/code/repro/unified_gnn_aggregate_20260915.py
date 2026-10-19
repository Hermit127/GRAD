"""Aggregate unified CausalCharGNN sweep runs into main-table rows (v2, post-review).

Fixes vs v1 (see LOGS/unified_table_independent_review_20260915.md):
  1. identity read from each run's internal config (not filename);
  2. dev-calibrated test metrics from threshold_reports.fpr_10.test (threshold chosen on dev),
     plus separate test-ROC diagnostic from test_graph.fixed_fpr (NOT a deployment point);
  3. fold weighting by n_pos*n_neg within each seed (matches historical baseline protocol),
     seed-level mean/SD across seeds.

Usage:
  python code/repro/unified_gnn_aggregate_20260915.py --src <runs_dir> --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_gnn_repair_manifest_20260915 import identity_from_config  # noqa: E402

BENCHES = ("decomp", "mg", "minja", "mcp", "mrt")


def per_run_metrics(r: dict) -> dict:
    tg = r.get("test_graph", {}) or {}
    tr = ((r.get("threshold_reports") or {}).get("fpr_10") or {}).get("test") or {}
    ff = (tg.get("fixed_fpr") or {}).get("fpr_10") or {}
    labels = [int(s["label"]) for s in (r.get("scores") or {}).get("test", [])]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return {
        "auroc": tg.get("auroc"),
        "dev_tpr": tr.get("recall"),
        "dev_fpr": tr.get("fpr"),
        "diag_tpr": ff.get("recall"),
        "diag_fpr": ff.get("fpr"),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "weight": n_pos * n_neg,
    }


def wmean(vals: list[tuple[float, float]]) -> float | None:
    num = sum(v * w for v, w in vals if v is not None)
    den = sum(w for v, w in vals if v is not None)
    return num / den if den else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--exclude-seed", type=int, default=42)
    args = ap.parse_args()

    units: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    skipped = []
    for path in sorted(Path(args.src).glob("*.json")):
        r = json.loads(path.read_text())
        ident = identity_from_config(r.get("config", {}))
        if ident is None:
            skipped.append(path.name)
            continue
        bench, fold, seed = ident
        if seed == args.exclude_seed:
            continue
        m = per_run_metrics(r)
        m.update({"fold": fold, "seed": seed, "file": path.name})
        units[bench][seed].append(m)

    out = {"src": args.src, "protocol": {
        "auroc": "fold-weighted by n_pos*n_neg within seed; mean/SD across seeds",
        "dev_tpr/dev_fpr": "threshold from dev FPR<=10% (threshold_reports.fpr_10); evaluated on test",
        "diag_tpr/diag_fpr": "test-ROC diagnostic (test_graph.fixed_fpr); NOT a deployment point",
    }, "benchmarks": {}, "skipped": skipped}

    metrics = ("auroc", "dev_tpr", "dev_fpr", "diag_tpr", "diag_fpr")
    summary = {}
    for bench in BENCHES:
        per_seed = {s: rows for s, rows in units.get(bench, {}).items()}
        if not per_seed:
            summary[bench] = None
            continue
        seeds = sorted(per_seed)
        agg = {}
        for met in metrics:
            seed_vals = [wmean([(rr[met], rr["weight"]) for rr in per_seed[s]]) for s in seeds]
            seed_vals = [v for v in seed_vals if v is not None]
            agg[met] = {
                "mean": st.mean(seed_vals) if seed_vals else None,
                "seed_sd": st.stdev(seed_vals) if len(seed_vals) > 1 else 0.0,
                "n_seeds": len(seed_vals),
            }
        per_fold = []
        for s in seeds:
            for rr in per_seed[s]:
                per_fold.append({k: rr[k] for k in ("fold", "seed", "auroc", "dev_tpr", "dev_fpr",
                                                    "diag_tpr", "diag_fpr", "n_pos", "n_neg")})
        summary[bench] = {"agg": agg, "per_fold": per_fold,
                          "n_folds": len({rr["fold"] for s in seeds for rr in per_seed[s]})}
        out["benchmarks"][bench] = summary[bench]

    order = [("Decomp", "decomp"), ("MemoryGraft", "mg"), ("MINJA(fix2)", "minja"),
             ("MCP", "mcp"), ("MRT", "mrt")]
    lines = [
        "| 方法 | " + " | ".join(n for n, _ in order) + " |",
        "|---" * (len(order) + 1) + "|",
    ]

    def fmt(bench: str, met: str, decimals: int = 4) -> str:
        s = summary.get(bench)
        if not s:
            return "n/a"
        a = s["agg"][met]
        if a["mean"] is None:
            return "n/a"
        return f"{a['mean']:.{decimals}f}±{a['seed_sd']:.{decimals}f}"

    lines.append("| AUROC（加权） | " + " | ".join(fmt(b, "auroc") for _, b in order) + " |")
    lines.append("| dev 校准后 test TPR（真实 FPR） | " + " | ".join(
        f"{fmt(b, 'dev_tpr', 3)}（{fmt(b, 'dev_fpr', 3)}）" for _, b in order) + " |")
    lines.append("| test-ROC 诊断 TPR（诊断，非部署点） | " + " | ".join(
        fmt(b, "diag_tpr", 3) for _, b in order) + " |")

    meta = []
    for _, b in order:
        s = summary.get(b)
        meta.append(f"{b}: folds={s['n_folds']} seeds={s['agg']['auroc']['n_seeds']}" if s else f"{b}: none")
    md = ("# 统一 CausalCharGNN 聚合 v2（审查修正版，2026-09-15）\n\n" +
          "\n".join(lines) + "\n\n- " + "; ".join(meta) +
          "\n- AUROC 折内按 n_pos*n_neg 加权后按 seed 汇总；± 为 seed 级 SD（非纯随机 seed SD 之外的误差）。\n"
          "- dev 校准后 test TPR/FPR：阈值由 dev FPR≤10% 选点（threshold_reports.fpr_10），在 test 上评估。\n"
          "- test-ROC 诊断为 test_graph.fixed_fpr（在 test 上重选阈值），只作诊断，不得写部署操作点。\n" +
          (f"\n- skipped(unparsed): {skipped}\n" if skipped else ""))

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    Path(args.out_md).write_text(md)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
