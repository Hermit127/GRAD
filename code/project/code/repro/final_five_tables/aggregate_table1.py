#!/usr/bin/env python3
"""Raw-results aggregator for Table 1 (cross-dataset attack detection).

Rows : Sequential Monitor, Agent-Sentry, AgentTrust+RiskChain, NeuroTaint,
       AgentArmor, AEGIS, AuthGraph, GRAD
Cols : Decomp, MCP, MRT, MG, MINJA, InjecAgent, ASB, AgentHarm

Hard rules
----------
* Every number is extracted from an existing raw result file. Nothing is
  invented or estimated; cells that cannot be filled keep status
  ``pending`` / ``blocked`` / ``N/A`` with a reason and print a
  machine-readable line::

      WARNING {"row": ..., "column": ..., "status": ..., "reason": ...}

* Unit = trajectory / graph.  Where a source only emits node-level output the
  archived protocol maps "any node positive => trajectory positive"; all
  sources used by this table already carry trajectory-level confusion counts.
* TP/FP/TN/FN and denominators (n_pos/n_neg) are recorded for every filled cell.
* GRAD statistics are computed by REUSING
  ``code/repro/unified_gnn_aggregate_20260915.py`` (``per_run_metrics`` +
  ``wmean`` + seed-level mean/SD) and the identity parser of
  ``code/repro/unified_gnn_repair_manifest_20260915.py``; no statistics are
  re-implemented.
* Binary baselines (AgentTrust / AgentArmor / AEGIS) pool TP/FP/TN/FN across the
  evaluation units of a column and then recompute TPR/FPR -- the same pooling
  the fidelity table (``LOGS/baseline_fidelity_table_20260916.md``) uses.
* Deterministic and rerunnable: no timestamps, sorted iteration, no random.

Usage
-----
    python code/repro/final_five_tables/aggregate_table1.py
    python code/repro/final_five_tables/aggregate_table1.py --check

``--check`` asserts the archived reference numbers below, prints a PASS/FAIL
diff per method and exits non-zero on any mismatch.  Outputs are written even
when checks fail (raw extraction is independent of the archived numbers), but
the exit code tells the caller whether extraction and archive agree.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CODE_REPRO = REPO / "code/repro"
sys.path.insert(0, str(CODE_REPRO))

import unified_gnn_aggregate_20260915 as grad_agg  # noqa: E402
from unified_gnn_repair_manifest_20260915 import identity_from_config  # noqa: E402

RAW_DIR = REPO / "LOGS/final_five_tables/raw_results"
TABLE_JSON = RAW_DIR / "table1.json"
TABLE_MD = RAW_DIR / "table1.md"

COLUMNS = ("Decomp", "MCP", "MRT", "MG", "MINJA", "InjecAgent", "ASB", "AgentHarm")
ROWS = ("Sequential Monitor", "Agent-Sentry", "AgentTrust+RiskChain", "NeuroTaint",
        "AgentArmor", "AEGIS", "AuthGraph", "GRAD")
BENCH = {"Decomp": "decomp", "MCP": "mcp", "MRT": "mrt", "MG": "mg",
         "MINJA": "minja", "InjecAgent": "injecagent", "ASB": "asb",
         "AgentHarm": "agentharm"}
DIAGNOSTIC = {"GRAD": {"MG", "MINJA"}}   # archived protocol marks these diagnostic (dagger)
GRAD_SEEDS = (211, 212, 213)             # LOGS/main_table_unified_20260915.md protocol
GRAD_EXCLUDE_SEED = 42                   # default --exclude-seed of the reused aggregator
SENTRY_SEEDS = (211, 212, 213)

# ---------------------------------------------------------------------------
# raw sources (all read-only)
# ---------------------------------------------------------------------------
GRAD_RUNS = REPO / "code/repro/out/unified_gnn/runs"
SENTRY_FIX2 = REPO / "code/repro/out/static_big_table_v1/fix2_20260910/runs"
SENTRY_MRT = REPO / "LOGS/mrt_cross_task_eval_20260913.json"
FIX3 = REPO / "code/repro/out/baseline_fidelity_fix3_20260916"
TRUST_DIR = FIX3 / "f1_agenttrust_prod_20260916"
TRUST_X4 = FIX3 / "f1_agenttrust_prod_x4_20260916"
ARMOR_DIR = FIX3 / "f3_agentarmor_20260916"
ARMOR_X4 = FIX3 / "f3_agentarmor_x4_20260916"
AEGIS_DIR = FIX3 / "f4_aegis_official_20260916"
NT_V3 = FIX3 / "neurotaint_fixed/neurotaint_fixed_full_20260916_v3.json"
NT_X4 = FIX3 / "neurotaint_fixed/neurotaint_fixed_full_20260916_x4_20260916.json"
NT_V2 = RAW_DIR / "nt_op_point_v2.json"
LSM = {
    "decomp_fix2_cumulative": REPO / "LOGS/lsm_decomp_fix2_cumulative_20260916.json",
    "official_cumulative": REPO / "LOGS/lsm_official_test_cumulative_20260916.json",
    "official_separate": REPO / "LOGS/lsm_official_test_separate_20260916.json",
}

# ---------------------------------------------------------------------------
# archived reference values checked by --check (never used to fill cells)
# ---------------------------------------------------------------------------
REFERENCE = {
    "GRAD": {
        "auroc": {"Decomp": 0.9516, "MG": 0.9982, "MINJA": 0.9491,
                  "MCP": 0.6210, "MRT": 0.7400},
        "dev_tpr": {"Decomp": 0.908, "MG": 0.684, "MINJA": 0.798,
                    "MCP": 0.334, "MRT": 0.793},
        "dev_fpr": {"Decomp": 0.121, "MG": 0.002, "MINJA": 0.023,
                    "MCP": 0.133, "MRT": 0.653},
        "n_seeds": 3,
    },
    "AgentTrust+RiskChain": {
        "tpr": {"Decomp": 0.000, "MG": 0.300, "MINJA": 0.000, "MCP": 0.035,
                "MRT": 0.200, "InjecAgent": 0.000, "ASB": 0.000, "AgentHarm": 0.000},
        "fpr": {"Decomp": 0.000, "MG": 0.000, "MINJA": 0.000, "MCP": 0.000,
                "MRT": 0.120, "InjecAgent": 0.000, "ASB": 0.000, "AgentHarm": 0.000},
    },
    "AgentArmor": {
        "tpr": {"Decomp": 0.989, "MG": 1.000, "MINJA": 0.333, "MCP": 0.649,
                "MRT": 1.000, "InjecAgent": 0.000, "ASB": 0.000, "AgentHarm": 0.000},
        "fpr": {"Decomp": 0.914, "MG": 1.000, "MINJA": 0.470, "MCP": 0.695,
                "MRT": 1.000, "InjecAgent": 0.000, "ASB": 0.000, "AgentHarm": 0.000},
    },
    "AEGIS": {
        # pooled per column (pool folds then recompute), plus per-unit folds
        "pooled": {
            "Decomp": (40, 4, 54, 134), "MG": (558, 397, 3, 42),
            "MINJA": (0, 7, 108, 75), "MCP": (62, 64, 54, 52),
            "MRT": (50, 50, 0, 0), "InjecAgent": (260, 248, 124, 112),
            "ASB": (48, 50, 0, 2), "AgentHarm": (44, 44, 8, 8),
        },
        "units": {
            "decomp": (40, 4, 54, 134),
            "mg_f0": (120, 77, 3, 0), "mg_f1": (120, 80, 0, 0),
            "mg_f2": (78, 80, 0, 42), "mg_f3": (120, 80, 0, 0),
            "mg_f4": (120, 80, 0, 0),
            "minja_f0": (0, 2, 36, 27), "minja_f1": (0, 3, 33, 24),
            "minja_f2": (0, 2, 39, 24),
            "mcp_f0": (12, 14, 10, 11), "mcp_f1": (14, 11, 13, 9),
            "mcp_f2": (14, 14, 10, 9), "mcp_f3": (13, 13, 11, 10),
            "mcp_f4": (9, 12, 10, 13),
            "mrt_famerican_airlines_2": (10, 10, 0, 0),
            "mrt_fapi_key_calendar_agendas_2": (10, 10, 0, 0),
            "mrt_fbank_transfer_2": (10, 10, 0, 0),
            "mrt_fexpenditure_accounting": (10, 10, 0, 0),
            "mrt_fspam_filter_update": (10, 10, 0, 0),
            "injecagent": (260, 248, 124, 112), "asb": (48, 50, 0, 2),
            "agentharm": (44, 44, 8, 8),
        },
    },
    "NeuroTaint": {
        "auroc": {"Decomp": 0.7253, "MG": 0.7920, "MINJA": 0.4924,
                  "MCP": 0.5000, "MRT": 0.5000, "InjecAgent": 0.5484,
                  "ASB": 0.5000, "AgentHarm": 0.5000},
        # nt_op_point_v2: pooled test TPR/FPR at each fold's dev-FPR<=10%
        # threshold, strict > decision (ties at the threshold are negative)
        "pooled_tpr_fpr": {
            "Decomp": (0.259, 0.086), "MCP": (0.000, 0.000),
            "MRT": (0.000, 0.000), "MG": (0.595, 0.152),
            "MINJA": (0.067, 0.078), "InjecAgent": (0.645, 0.333),
            "ASB": (0.000, 0.000), "AgentHarm": (0.000, 0.000)},
        "pooled_confusion": {   # order: tp, fp, tn, fn
            "Decomp": (45, 5, 53, 129), "MCP": (0, 0, 118, 114),
            "MRT": (0, 0, 50, 50), "MG": (357, 61, 339, 243),
            "MINJA": (5, 9, 106, 70), "InjecAgent": (240, 124, 248, 132),
            "ASB": (0, 0, 50, 50), "AgentHarm": (0, 0, 52, 52)},
        "degenerate_columns": ("MCP", "MRT", "ASB", "AgentHarm"),
    },
    "Sequential Monitor": {
        "decomp_fix2_cumulative": {"auroc": 0.8103, "tp": 174, "fp": 21, "tn": 37, "fn": 0},
        "official_cumulative": {"auroc": 0.8239, "tp": 87, "fp": 30, "tn": 58, "fn": 1},
        "official_separate": {"auroc": 0.8523, "tp": 80, "fp": 14, "tn": 74, "fn": 8},
    },
    "Agent-Sentry": {
        "auroc": {"Decomp": 0.8444, "MG": 0.9138, "MINJA": 0.4168,
                  "MCP": 0.5312, "MRT": 0.519},
        # supplemental archived native-thetaH op point (paper_tables_v4_static.md)
        "supplemental_native_tpr_fpr": {
            "Decomp": (0.7414, 0.1552), "MG": (0.7650, 0.1325),
            "MINJA": (0.0533, 0.0957), "MCP": (0.1579, 0.0763),
        },
    },
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO))


def pool_confusion(counts: list[dict]) -> dict:
    """Sum TP/FP/TN/FN across units, then recompute TPR/FPR (fidelity-table pooling)."""
    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    tn = sum(c["tn"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


class Warnings:
    """Machine-readable warnings for cells that cannot be filled."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, row: str, column: str, status: str, reason: str,
            metric: str | None = None) -> None:
        item = {"row": row, "column": column, "status": status,
                "metric": metric, "reason": reason}
        self.items.append(item)
        print("WARNING " + json.dumps(item, ensure_ascii=False))


def cell(status: str, *, kind: str | None = None, auroc: dict | None = None,
         tpr: float | None = None, fpr: float | None = None,
         tpr_fpr_status: str | None = None, confusion: dict | None = None,
         n_pos: int | None = None, n_neg: int | None = None,
         threshold: dict | None = None, units: list | None = None,
         sources: list | None = None, diagnostic: bool = False,
         reason: str | None = None, **extra) -> dict:
    out = {
        "status": status,
        "kind": kind,
        "auroc": auroc,
        "tpr": tpr,
        "fpr": fpr,
        "tpr_fpr_status": tpr_fpr_status,
        "confusion": confusion,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": threshold,
        "diagnostic": diagnostic,
        "sources": sources or [],
        "units": units or [],
        "reason": reason,
    }
    out.update(extra)
    return out


def warn_cell(warn: Warnings, row: str, col: str, status: str, reason: str,
              metric: str | None = None) -> dict:
    warn.add(row, col, status, reason, metric)
    return cell(status, reason=reason)


# ===========================================================================
# ROW BUILDERS
# ===========================================================================
def build_grad(warn: Warnings) -> dict:
    """GRAD (unified CausalCharGNN): reuse unified_gnn_aggregate_20260915 logic."""
    by_bench: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    skipped: list[str] = []
    for path in sorted(GRAD_RUNS.glob("*.json")):
        run = read_json(path)
        ident = identity_from_config(run.get("config", {}))
        if ident is None:
            skipped.append(path.name)
            continue
        bench, fold, seed = ident
        if seed == GRAD_EXCLUDE_SEED:
            continue
        m = grad_agg.per_run_metrics(run)  # reused statistics
        tr = ((run.get("threshold_reports") or {}).get("fpr_10") or {}).get("test") or {}
        thr = tr.get("threshold")
        counts = None
        if thr is not None:
            scores = (run.get("scores") or {}).get("test", [])
            tp = sum(1 for e in scores if e["score"] >= thr and int(e["label"]) == 1)
            fp = sum(1 for e in scores if e["score"] >= thr and int(e["label"]) == 0)
            fn = sum(1 for e in scores if e["score"] < thr and int(e["label"]) == 1)
            tn = sum(1 for e in scores if e["score"] < thr and int(e["label"]) == 0)
            rec = tp / (tp + fn) if tp + fn else None
            fpr = fp / (fp + tn) if fp + tn else None
            rec_ref = tr.get("recall")
            fpr_ref = tr.get("fpr")
            if (rec is None or rec_ref is None or abs(rec - rec_ref) > 1e-9
                    or fpr is None or fpr_ref is None or abs(fpr - fpr_ref) > 1e-9):
                warn.add("GRAD", bench, "pending",
                         f"counts from raw scores disagree with threshold_reports ({path.name})",
                         metric="confusion")
                counts = None
            else:
                counts = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        m.update({"fold": fold, "seed": seed, "file": rel(path),
                  "threshold": thr, "confusion_dev_threshold": counts})
        by_bench[bench][seed].append(m)

    cells: dict[str, dict] = {}
    for col in COLUMNS:
        bench = BENCH[col]
        per_seed = by_bench.get(bench)
        if not per_seed:
            cells[col] = warn_cell(
                warn, "GRAD", col, "blocked",
                "unified CausalCharGNN backbone not evaluated on this corpus "
                "(legacy channel results are appendix-only)")
            continue
        seeds = sorted(per_seed)
        metrics = ("auroc", "dev_tpr", "dev_fpr", "diag_tpr", "diag_fpr")
        agg: dict[str, dict] = {}
        for met in metrics:
            seed_vals = [grad_agg.wmean([(rr[met], rr["weight"]) for rr in per_seed[s]])
                         for s in seeds]
            seed_vals = [v for v in seed_vals if v is not None]
            agg[met] = {
                "mean": st.mean(seed_vals) if seed_vals else None,
                "seed_sd": st.stdev(seed_vals) if len(seed_vals) > 1 else 0.0,
                "n_seeds": len(seed_vals),
            }
        units = []
        counts: list[dict] = []
        for s in seeds:
            for rr in per_seed[s]:
                units.append({
                    "unit": f"{bench}_f{rr['fold']}_s{rr['seed']}",
                    "fold": rr["fold"], "seed": rr["seed"],
                    "n_pos": rr["n_pos"], "n_neg": rr["n_neg"],
                    "auroc": rr["auroc"], "dev_tpr": rr["dev_tpr"],
                    "dev_fpr": rr["dev_fpr"], "weight": rr["weight"],
                    "threshold": rr["threshold"], "file": rr["file"],
                })
                if rr["confusion_dev_threshold"]:
                    counts.append(rr["confusion_dev_threshold"])
        pooled = pool_confusion(counts) if counts else None
        cells[col] = cell(
            "filled", kind="continuous_score",
            auroc={**agg["auroc"], "protocol":
                   "fold-weighted by n_pos*n_neg within seed; mean/SD across "
                   "seeds 211-213 (code/repro/unified_gnn_aggregate_20260915.py)"},
            tpr=agg["dev_tpr"]["mean"],
            fpr=agg["dev_fpr"]["mean"],
            tpr_fpr_status="dev_calibrated_test_at_dev_fpr10_threshold",
            confusion=pooled,
            n_pos=sum(rr["n_pos"] for s in seeds for rr in per_seed[s]),
            n_neg=sum(rr["n_neg"] for s in seeds for rr in per_seed[s]),
            threshold={"label": "threshold_reports.fpr_10 (dev FPR<=10%) per run",
                       "value": None},
            units=units,
            sources=sorted({rr["file"] for s in seeds for rr in per_seed[s]}),
            diagnostic=col in DIAGNOSTIC["GRAD"],
            dev_tpr=agg["dev_tpr"], dev_fpr=agg["dev_fpr"],
            diag_tpr=agg["diag_tpr"], diag_fpr=agg["diag_fpr"],
            skipped_unparsed=skipped,
        )
    return {"identity": "GRAD = unified CausalCharGNN (train_gnn_v2 backbone)",
            "aggregation": "reuse code/repro/unified_gnn_aggregate_20260915.py",
            "cells": cells}


def _load_binary_units(run_dirs: list[Path], subdir: str, key_pick) -> tuple[dict, list[str]]:
    by_bench: dict[str, list[dict]] = defaultdict(list)
    sources: list[str] = []
    for run_dir in run_dirs:
        for mpath in sorted((run_dir / subdir).glob("*/metrics.json")):
            m = read_json(mpath)
            conf = m.get("confusion") or {}
            key = key_pick(conf)
            if key is None or m.get("status") != "ok":
                continue
            c = conf[key]
            by_bench[m["benchmark"]].append({
                "unit": m["unit"], "tp": c["tp"], "fp": c["fp"], "tn": c["tn"],
                "fn": c["fn"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                "source": rel(mpath),
            })
            sources.append(rel(mpath))
    return by_bench, sources


def build_binary_row(row_name: str, run_dirs: list[Path], subdir: str, key_pick,
                     rule_label: str, warn: Warnings,
                     identity: str) -> dict:
    by_bench, _ = _load_binary_units(run_dirs, subdir, key_pick)
    cells: dict[str, dict] = {}
    for col in COLUMNS:
        units = by_bench.get(BENCH[col])
        if not units:
            cells[col] = warn_cell(warn, row_name, col, "pending",
                                   "no raw per-unit metrics located for this corpus")
            continue
        pooled = pool_confusion(units)
        cells[col] = cell(
            "filled", kind="binary_rule", tpr=pooled["tpr"], fpr=pooled["fpr"],
            tpr_fpr_status="native_rule_op_point",
            confusion=pooled,
            n_pos=sum(u["n_pos"] for u in units),
            n_neg=sum(u["n_neg"] for u in units),
            threshold={"label": rule_label, "value": None},
            units=units, sources=sorted({u["source"] for u in units}),
        )
    return {"identity": identity,
            "aggregation": "pool TP/FP/TN/FN across units per column, then TPR/FPR "
                           "(same as LOGS/baseline_fidelity_table_20260916.md)",
            "cells": cells}


def build_aegis(warn: Warnings) -> dict:
    def key_pick(conf):
        for k in sorted(conf):
            if k.endswith("_block"):
                return k
        return None

    by_bench, _ = _load_binary_units([AEGIS_DIR], "aegis_official", key_pick)
    cells: dict[str, dict] = {}
    for col in COLUMNS:
        units = by_bench.get(BENCH[col])
        if not units:
            cells[col] = warn_cell(warn, "AEGIS", col, "pending",
                                   "no raw per-unit metrics located for this corpus")
            continue
        pooled = pool_confusion(units)
        cells[col] = cell(
            "filled", kind="binary_rule", tpr=pooled["tpr"], fpr=pooled["fpr"],
            tpr_fpr_status="native_policy_op_point",
            confusion=pooled,
            n_pos=sum(u["n_pos"] for u in units),
            n_neg=sum(u["n_neg"] for u in units),
            threshold={"label": "official TS classifier + published policy schema; "
                                "any violation => block",
                       "value": None},
            units=units, sources=sorted({u["source"] for u in units}),
        )
    return {"identity": "AEGIS official TypeScript classifier rerun "
                        "(f4_aegis_official_20260916)",
            "aggregation": "pool TP/FP/TN/FN across units per column, then TPR/FPR "
                           "(same as LOGS/baseline_fidelity_table_20260916.md)",
            "cells": cells}


def build_neurotaint(warn: Warnings) -> dict:
    """NeuroTaint propagation component: AUROC from the fixed full-run JSONs,
    TPR/FPR from the v2 dev-calibrated operating points.

    Operating point (pre-registered in code/repro/final_five_tables/
    nt_op_point_v2.py): per fold, threshold = (floor(0.1*n_devneg)+1)-th
    largest dev-negative score; decision = score > threshold (strict; ties at
    the threshold are negative, so the realized dev FPR stays <= target under
    ties). The same threshold and rule are applied to test -- never tuned on
    test. Degenerate folds (all dev scores tie) keep the all-negative test
    result and are flagged, not blanked. The native score>0 point is reported
    alongside as the pre-determined fallback (the paper tier thresholds live
    inside the score).
    """
    v3 = read_json(NT_V3)
    x4 = read_json(NT_X4)
    op = read_json(NT_V2)
    sources = [rel(NT_V3), rel(NT_X4), rel(NT_V2)]
    cells: dict[str, dict] = {}
    for col in COLUMNS:
        bench = BENCH[col]
        info = (v3.get("datasets") or {}).get(bench) or (x4.get("datasets") or {}).get(bench)
        if not info:
            cells[col] = warn_cell(warn, "NeuroTaint", col, "pending",
                                   "dataset absent from the fixed full-result JSON files")
            continue
        block = (op.get("datasets") or {}).get(bench)
        if not block or not block.get("folds"):
            cells[col] = warn_cell(warn, "NeuroTaint", col, "pending",
                                   "operating-point calibration missing from "
                                   "raw_results/nt_op_point_v2.json for this corpus")
            continue
        primary = info["variants"]["primary"]
        pooled = block["pooled_at_dev_threshold"]
        native = block.get("pooled_test_native_score_gt_0") or {}
        units = []
        for f in block["folds"]:
            tm = f.get("test_at_dev_threshold") or {}
            dm = f.get("dev_at_threshold") or {}
            units.append({
                "unit": f"{bench}_f{f['fold']}",
                "fold": f["fold"],
                "n_dev": f["n_dev"], "n_dev_pos": f["n_dev_pos"],
                "n_dev_neg": f["n_dev_neg"], "n_test": f["n_test"],
                "n_pos": f["n_test_pos"], "n_neg": f["n_test_neg"],
                "dev_threshold": f["dev_threshold_fpr10"],
                "dev_tpr": dm.get("tpr"), "dev_fpr": dm.get("fpr"),
                "tpr": tm.get("tpr"), "fpr": tm.get("fpr"),
                "confusion": {"tp": tm.get("tp"), "fp": tm.get("fp"),
                              "tn": tm.get("tn"), "fn": tm.get("fn")},
                "native_tpr": (f.get("native_test") or {}).get("tpr"),
                "native_fpr": (f.get("native_test") or {}).get("fpr"),
                "degenerate_dev_calibration": f["degenerate_dev_calibration"],
                "dev_all_scores_tied": f["dev_all_scores_tied"],
                "source": rel(NT_V2),
            })
        degenerate = block.get("degenerate_dev_calibration")
        cells[col] = cell(
            "filled", kind="continuous_score",
            auroc={"mean": primary["auroc_mean"],
                   "per_fold": primary.get("auroc_per_fold"),
                   "n_folds": primary.get("n_folds"),
                   "seed_sd": None, "n_seeds": 1,
                   "protocol": "primary variant (char-LCS, restricted, no proxy); "
                               "auroc_mean as recorded in the raw JSON"},
            tpr=pooled["tpr"], fpr=pooled["fpr"],
            tpr_fpr_status="dev_calibrated_fpr10_strict_ties_negative",
            confusion={"tp": pooled["tp"], "fp": pooled["fp"],
                       "tn": pooled["tn"], "fn": pooled["fn"]},
            n_pos=pooled["n_pos"], n_neg=pooled["n_neg"],
            threshold={"label": "dev FPR<=10% threshold per fold (strict >; ties "
                                "at the threshold are negative; no test-set "
                                "tuning)",
                       "value": [f["dev_threshold_fpr10"] for f in block["folds"]]},
            units=units, sources=sources,
            raw_tpr_at_fpr10_mean=primary.get("tpr_at_fpr10_mean"),
            n_test_graphs=info.get("n_test_graphs"),
            causal_status=primary.get("causal_status"),
            native_op_point={
                "rule": "positive iff score > 0 (stage != none; paper tier "
                        "thresholds are applied inside the score)",
                "tpr": native.get("tpr"), "fpr": native.get("fpr"),
                "confusion": {"tp": native.get("tp"), "fp": native.get("fp"),
                              "tn": native.get("tn"), "fn": native.get("fn")},
            },
            degenerate_dev_calibration=degenerate,
            degeneracy_note=block.get("note"),
        )
    return {"identity": "NeuroTaint propagation component (fixed semantic taint; "
                        "causal counterfactual stage not run)",
            "aggregation": "AUROC per dataset straight from the full-result JSON; "
                           "TPR/FPR pooled across folds at each fold's dev-selected "
                           "threshold (nt_op_point_v2.py, strict > rule)",
            "cells": cells}


def _lsm_variant(entry: dict, path_key: str, src: Path) -> dict:
    metrics = entry["metrics"]
    at05 = metrics["at_0.5"]
    n_pos = metrics.get("n_harmful")
    n_neg = entry["n_tasks"] - n_pos if n_pos is not None else None
    fpr_at_05 = at05["fp"] / (at05["fp"] + at05["tn"]) if at05["fp"] + at05["tn"] else None
    return {
        "variant": path_key,
        "source": rel(src),
        "auroc": metrics["auroc"],
        "confusion_at_0.5": {"tp": at05["tp"], "fp": at05["fp"],
                             "tn": at05["tn"], "fn": at05["fn"]},
        "tpr_at_0.5": at05["recall"], "fpr_at_0.5": fpr_at_05,
        "n_tasks": entry["n_tasks"], "n_harmful": n_pos, "n_neg": n_neg,
        "eval_mode": entry["eval_mode"], "model": entry["model"],
        "errors": entry["errors"], "batch_calls": entry.get("batch_calls"),
        "implementation": entry["implementation"],
    }


def build_sequential_monitor(warn: Warnings) -> dict:
    loaded = {k: read_json(p) for k, p in LSM.items()}
    variants = {k: _lsm_variant(loaded[k], k, LSM[k]) for k in LSM}
    primary = variants["decomp_fix2_cumulative"]
    primary_cell = cell(
        "filled", kind="sequence_monitor",
        auroc={"mean": primary["auroc"], "seed_sd": None, "n_seeds": 1,
               "protocol": "session-level official metric (single run)"},
        tpr=primary["tpr_at_0.5"], fpr=primary["fpr_at_0.5"],
        tpr_fpr_status="at_0.5_recorded_threshold",
        confusion=primary["confusion_at_0.5"],
        n_pos=primary["n_harmful"], n_neg=primary["n_neg"],
        threshold={"label": "at_0.5", "value": 0.5},
        sources=[primary["source"]],
        variants={k: v for k, v in variants.items() if k != "decomp_fix2_cumulative"},
        model=primary["model"], eval_mode=primary["eval_mode"],
        harness_note="official prompts/data; DeepSeek adapter (disclosed); "
                     "decomp_fix2 corpus evaluated with the same monitor",
    )
    cells = {"Decomp": primary_cell}
    for col in COLUMNS:
        if col == "Decomp":
            continue
        cells[col] = warn_cell(
            warn, "Sequential Monitor", col, "N/A",
            "monitor requires a subtask sequence; corpus has no such interface")
    return {"identity": "Sequential Monitor (ICLR'26) -- session-level official monitor",
            "aggregation": "official test metrics; decomp column from the "
                           "decomp_fix2 cumulative run",
            "cells": cells}


def build_agent_sentry(warn: Warnings) -> dict:
    cells: dict[str, dict] = {}
    for col in ("Decomp", "MG", "MINJA", "MCP"):
        bench = BENCH[col]
        per_seed: dict[int, list[dict]] = defaultdict(list)
        sources: set[str] = set()
        for seed in SENTRY_SEEDS:
            path = SENTRY_FIX2 / f"{bench}_s{seed}" / "metrics.json"
            if not path.exists():
                warn.add("Agent-Sentry", col, "pending",
                         f"missing raw run {rel(path)}", metric="auroc")
                continue
            run = read_json(path)
            for u in run["units"]:
                a = (u.get("auroc") or {}).get("agent_sentry_l1")
                c = (u.get("confusion") or {}).get("agent_sentry_l1")
                if a is None or c is None:
                    continue
                per_seed[seed].append({
                    "unit": u["unit"], "fold": u["fold"],
                    "n_pos": u["n_test_pos"], "n_neg": u["n_test_neg"],
                    "auroc": a, "tp": c["tp"], "fp": c["fp"],
                    "tn": c["tn"], "fn": c["fn"], "threshold": c.get("threshold"),
                    "source": rel(path),
                })
                sources.add(rel(path))
        if not per_seed:
            cells[col] = warn_cell(warn, "Agent-Sentry", col, "pending",
                                   "no raw per-unit scores located")
            continue
        seed_means = []
        for seed in sorted(per_seed):
            rows = per_seed[seed]
            seed_means.append(grad_agg.wmean([(r["auroc"], r["n_pos"] * r["n_neg"])
                                              for r in rows]))
        auroc_mean = st.mean(seed_means)
        auroc_sd = st.stdev(seed_means) if len(seed_means) > 1 else 0.0
        # native op point: pool one deterministic fit (seed 211)
        units = per_seed[sorted(per_seed)[0]]
        pooled = pool_confusion(units)
        thresholds = sorted({r["threshold"] for r in units if r["threshold"] is not None})
        cells[col] = cell(
            "filled", kind="continuous_score",
            auroc={"mean": auroc_mean, "seed_sd": auroc_sd,
                   "n_seeds": len(seed_means),
                   "protocol": "ROC weighted by n_pos*n_neg within seed, mean/SD "
                               "across seeds 211-213 (deterministic fit, identical "
                               "across seeds)"},
            tpr=pooled["tpr"], fpr=pooled["fpr"],
            tpr_fpr_status="native_threshold",
            confusion=pooled,
            n_pos=sum(r["n_pos"] for r in units),
            n_neg=sum(r["n_neg"] for r in units),
            threshold={"label": "native thetaH", "value": thresholds[0] if len(thresholds) == 1 else thresholds},
            units=units, sources=sorted(sources),
            seeds=sorted(per_seed), deterministic_across_seeds=True,
        )
    # MRT: raw per-fold scores exist, no threshold recorded
    mrt = read_json(SENTRY_MRT)
    rows = []
    for key in sorted(mrt["per_fold"]):
        fold, seed = key.split("|")
        e = mrt["per_fold"][key]["agent_sentry_l1"]
        rows.append({"unit": f"mrt_f{fold}_s{seed}", "fold": fold,
                     "seed": int(seed.replace("s", "")), "auroc": e["auroc"],
                     "tp": e["tp"], "fp": e["fp"], "tn": e["tn"], "fn": e["fn"],
                     "tpr": e["tpr"], "fpr": e["fpr"]})
    seed_means = {}
    for r in rows:
        seed_means.setdefault(r["seed"], []).append(r["auroc"])
    auroc_mean = st.mean([st.mean(v) for _, v in sorted(seed_means.items())])
    fold_sd = st.stdev([r["auroc"] for r in rows])
    warn.add("Agent-Sentry", "MRT", "pending",
             "raw per-fold scores exist but no native/dev threshold is recorded in "
             "LOGS/mrt_cross_task_eval_20260913.json", metric="tpr_fpr")
    cells["MRT"] = cell(
        "partial", kind="continuous_score",
        auroc={"mean": auroc_mean, "seed_sd": 0.0, "fold_sd": fold_sd,
               "n_seeds": len(seed_means), "n_units": len(rows),
               "protocol": "mean across 15 fold-seed units (3 seeds x 5 folds; "
                           "same as LOGS/mrt_cross_task_eval_20260913.json aggregate)"},
        tpr=None, fpr=None,
        tpr_fpr_status="pending:no_native_or_dev_threshold_recorded",
        reason="TPR/FPR requires a documented threshold; the MRT raw file records "
               "per-fold tpr/fpr but no threshold",
        units=rows, sources=[rel(SENTRY_MRT)],
    )
    for col in ("InjecAgent", "ASB", "AgentHarm"):
        cells[col] = warn_cell(
            warn, "Agent-Sentry", col, "pending",
            "no raw per-unit result located (Agent-Sentry x4 not rerun; L1-only "
            "cannot stand in for the complete three-layer method)")
    return {"identity": "Agent-Sentry L1 local repro (arXiv:2603.22868; deterministic, "
                        "1 fit)",
            "aggregation": "AUROC fold-weighted by n_pos*n_neg within seed, mean/SD "
                           "across seeds; native thetaH op point for decomp/mg/minja/mcp",
            "cells": cells}


def build_authgraph(warn: Warnings) -> dict:
    cells = {}
    for col in COLUMNS:
        cells[col] = warn_cell(
            warn, "AuthGraph", col, "pending",
            "no evaluation run yet: deterministic core (S-set/IRG/dual-pointer) "
            "and LLM Planner/Builder/Checker are implemented + offline-verified "
            "(code/repro/authgraph/, tests 42/42, zero API) but the paid "
            "evaluation has not run (budget pending); marking N/A would require "
            "positive evidence of structural inapplicability, which this work "
            "has not established")
    return {"identity": "AuthGraph faithful reimplementation (IRG Builder + "
                        "clean-context Planner + dual-pointer three-layer Checker); "
                        "evaluation pending",
            "aggregation": "no comparable predictions yet; "
                           "LOGS/authgraph_local_* legacy template-authorization "
                           "variant stays appendix-only (mechanism comparison)",
            "cells": cells}


X4_RUNS = REPO / "code/repro/out/unified_gnn_x4/runs"
X4_AUDIT = REPO / "LOGS/final_five_tables/raw_results/x4_shortcut_audit.json"


def build_grad_x4(warn: Warnings) -> dict:
    """x4 columns for GRAD (InjecAgent/ASB/AgentHarm).

    All three x4 columns are flagged diagnostic: the shortcut audit shows
    single-token / 1-NN probes separate the corpora at or above the model
    (InjecAgent: text-1NN 1.0; ASB: token "please" 1.0 and text-1NN 1.0;
    AgentHarm: text-1NN 0.90 and it is a harmful-task setting, not indirect
    injection). The values are reported for completeness only.
    """
    audit = read_json(X4_AUDIT) if X4_AUDIT.exists() else {}
    out = {}
    for col, bench in (("InjecAgent", "injecagent"), ("ASB", "asb"),
                       ("AgentHarm", "agentharm")):
        per = []
        for path in sorted(X4_RUNS.glob(f"{bench}_s*.json")):
            run = read_json(path)
            tr = ((run.get("threshold_reports") or {}).get("fpr_10") or {}).get("test") or {}
            a = (run.get("test_graph") or {}).get("auroc")
            scores = (run.get("scores") or {}).get("test") or []
            thr = tr.get("threshold")
            counts = None
            if thr is not None and scores:
                tp = sum(1 for e in scores if e["score"] >= thr and int(e["label"]) == 1)
                fp = sum(1 for e in scores if e["score"] >= thr and int(e["label"]) == 0)
                fn = sum(1 for e in scores if e["score"] < thr and int(e["label"]) == 1)
                tn = sum(1 for e in scores if e["score"] < thr and int(e["label"]) == 0)
                counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "tpr": tp / (tp + fn) if tp + fn else None,
                          "fpr": fp / (fp + tn) if fp + tn else None}
            per.append({"seed": path.stem.rsplit("_s", 1)[1], "auroc": a,
                        "test": tr, "counts": counts, "source": rel(path)})
        if not per:
            out[col] = warn_cell(warn, "GRAD", col, "pending",
                                 "x4 training runs not located")
            continue
        aucs = [x["auroc"] for x in per if x["auroc"] is not None]
        mean = sum(aucs) / len(aucs) if aucs else None
        sd = (sum((a - mean) ** 2 for a in aucs) / len(aucs)) ** 0.5 if len(aucs) > 1 else None
        pooled = None
        cs = [x["counts"] for x in per if x["counts"]]
        if cs:
            tp = sum(c["tp"] for c in cs); fp = sum(c["fp"] for c in cs)
            fn = sum(c["fn"] for c in cs); tn = sum(c["tn"] for c in cs)
            pooled = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "tpr": tp / (tp + fn) if tp + fn else None,
                      "fpr": fp / (fp + tn) if fp + tn else None}
        probe = (audit.get(bench) or {}).get("probes") or {}
        reason = ("shortcut-confounded corpus: single-token/1-NN probes at or "
                  f"above the model ({probe}); diagnostic only, excluded from "
                  "discriminative claims" if probe else
                  "diagnostic only (x4 conversion, shortcut audit pending)")
        if col == "AgentHarm":
            reason = ("threat-model mismatch (harmful-task setting, not indirect "
                      "injection) + " + reason)
        out[col] = cell("filled", kind="continuous",
                        auroc={"mean": mean, "seed_sd": sd, "n_seeds": len(aucs),
                               "fold_sd": None, "n_units": len(per),
                               "protocol": "x4 local-CPU training (same config as "
                                           "fix2 sweep); seeds: " +
                                           ",".join(x["seed"] for x in per)},
                        tpr=(pooled or {}).get("tpr"), fpr=(pooled or {}).get("fpr"),
                        tpr_fpr_status="dev_calibrated_fpr10",
                        confusion=pooled,
                        n_pos=sum(c["tp"] + c["fn"] for c in cs) if cs else None,
                        n_neg=sum(c["fp"] + c["tn"] for c in cs) if cs else None,
                        threshold={"label": "dev fpr_10", "value": None},
                        units=per,
                        sources=[x["source"] for x in per],
                        diagnostic=True, reason=reason)
        out[col]["shortcut_audit"] = probe
    return out


def build_table(warn: Warnings) -> dict:
    rows = {
        "Sequential Monitor": build_sequential_monitor(warn),
        "Agent-Sentry": build_agent_sentry(warn),
        "AgentTrust+RiskChain": build_binary_row(
            "AgentTrust+RiskChain", [TRUST_DIR, TRUST_X4], "agenttrust_prod",
            lambda conf: next((k for k in sorted(conf) if k.endswith("_chain")), None),
            "native strict rule {block,review} (chain-inclusive predictions)", warn,
            "AgentTrust+RiskChain (official rules-only adapter; chain-inclusive; "
            "observable fields only: tool_name/server/record_type, not full arguments)"),
        "NeuroTaint": build_neurotaint(warn),
        "AgentArmor": build_binary_row(
            "AgentArmor", [ARMOR_DIR, ARMOR_X4], "agentarmor",
            lambda conf: next((k for k in sorted(conf) if k == "agentarmor"), None),
            "native mechanism rule (dual propagation + published Forbid rules)", warn,
            "AgentArmor (mechanism comparison; assumed registry values and temporal-over-approximated dependency edges)"),
        "AEGIS": build_aegis(warn),
        "AuthGraph": build_authgraph(warn),
        "GRAD": build_grad(warn),
    }
    gx4 = build_grad_x4(warn)
    for col in ("InjecAgent", "ASB", "AgentHarm"):
        if col in gx4:
            rows["GRAD"]["cells"][col] = gx4[col]
    displays = {
        "Sequential Monitor": "Sequential Monitor (official pipeline; DeepSeek monitor, disclosed)",
        "Agent-Sentry": "Agent-Sentry L1 (component; adapted features)",
        "AgentTrust+RiskChain": "AgentTrust+RiskChain (rules-only; chain-inclusive; observable fields only)",
        "NeuroTaint": "NeuroTaint (propagation component; causal stage not run)",
        "AgentArmor": "AgentArmor (mechanism comparison; assumed registry)",
        "AEGIS": "AEGIS (official TS classifier; published policy; text fallback)",
        "AuthGraph": "AuthGraph (implementation verified; evaluation pending)",
        "GRAD": "GRAD (unified CausalCharGNN)",
    }
    full_method_status = {
        "Agent-Sentry": ("full L1+L2+L3 method NOT verified: native L1 training "
                         "positives = 0 (blocked_data, unpublished Agent-Sentry "
                         "Bench); only the L1 component and the full-pipeline "
                         "util/ABR adaptation are reported"),
        "AgentArmor": ("full method NOT verified: registry values and dependency "
                       "edges are unpublished; mechanism comparison only"),
        "AgentTrust+RiskChain": ("rules-only configuration with observable fields "
                                 "only (paper reports a rules+LLM-judge variant "
                                 "that is not enabled here)"),
        "NeuroTaint": ("causal counterfactual stage not run (needs neutralized "
                       "full-context traces); propagation component only. "
                       "TPR/FPR are the pooled test rates at each fold's "
                       "dev-FPR<=10% threshold (strict > rule, ties negative); "
                       "folds whose dev scores all tie degenerate to the "
                       "all-negative prediction and are flagged, never tuned on "
                       "test"),
        "AuthGraph": ("deterministic core + LLM Planner/Builder/Checker "
                      "implemented and offline-verified (42/42 tests); no "
                      "evaluation run yet (budget pending); the legacy local "
                      "template-authorization variant is appendix-only"),
    }
    for key, row in rows.items():
        row["display"] = displays.get(key, key)
        if key in full_method_status:
            row["full_method_status"] = full_method_status[key]
    return {
        "table": "table1_cross_dataset_attack_detection",
        "displays": displays,
        "full_method_status": full_method_status,
        "generated_by": "code/repro/final_five_tables/aggregate_table1.py",
        "protocol": {
            "unit": "trajectory / graph (one graph = one decision unit)",
            "node_to_trajectory_rule": "when a source only emits node/action-level "
                "decisions: any node positive => trajectory positive (archived "
                "protocol; all sources used here already provide trajectory-level "
                "confusion counts)",
            "metrics": "AUROC for continuous scores; TPR/FPR from the recorded "
                "confusion at the native threshold, or the dev-calibrated threshold "
                "for GRAD; every filled cell records TP/FP/TN/FN and n_pos/n_neg",
            "grad_protocol": "reuse of code/repro/unified_gnn_aggregate_20260915.py: "
                "fold-weighted n_pos*n_neg within seed, mean/stdev across seeds "
                "211-213 (seed 42 excluded); dev TPR/FPR from "
                "threshold_reports.fpr_10 (dev FPR<=10%) evaluated on test",
            "pooling_for_binary_methods": "TP/FP/TN/FN summed across evaluation "
                "units of a column, then TPR=tp/(tp+fn), FPR=fp/(fp+tn) (same "
                "pooling as LOGS/baseline_fidelity_table_20260916.md)",
            "diagnostic_units": {"GRAD": ["MG", "MINJA"],
                                 "marker": "dagger in table1.md; diagnostic=true in table1.json"},
            "neurotaint_op_point": "raw_results/nt_op_point_v2.json; per fold "
                "t* = (floor(0.1*n_devneg)+1)-th largest dev-negative score, "
                "positive iff score > t* (ties negative); same rule on test; "
                "degenerate folds stay all-negative and are flagged",
            "columns": list(COLUMNS),
            "rows": list(ROWS),
        },
        "rows": rows,
        "warnings": warn.items,
    }


# ===========================================================================
# MARKDOWN RENDERER
# ===========================================================================
def _fmt_auroc(auroc: dict | None, status: str, deterministic: bool = False) -> str:
    if auroc and auroc.get("mean") is not None:
        s = f"{auroc['mean']:.4f}"
        if deterministic:
            s += " (det.)"
        elif auroc.get("seed_sd") and auroc.get("n_seeds", 0) > 1:
            s += f"±{auroc['seed_sd']:.4f}"
        elif auroc.get("fold_sd") is not None and auroc.get("n_units", 0) > 1:
            s += f"±{auroc['fold_sd']:.4f} (fold SD)"
        return s
    if status == "N/A":
        return "N/A"
    if status == "filled" and auroc is None:
        return "n/a (binary)"
    return "\u2014"


def _fmt_tpr_fpr(tpr, fpr, status: str) -> str:
    if tpr is not None and fpr is not None:
        return f"{tpr:.3f}/{fpr:.3f}"
    if status == "N/A":
        return "N/A"
    if status in ("pending", "blocked"):
        return status
    if tpr is None and status == "partial":
        return "pending"
    return "\u2014"


def render_md(table: dict) -> str:
    rows = table["rows"]
    out = [
        "# Table 1 raw aggregation (cross-dataset attack detection)",
        "",
        "> Auto-generated by `code/repro/final_five_tables/aggregate_table1.py` from raw",
        "> result files only; no manual numbers. Unfilled cells keep status",
        "> pending/blocked/N-A with the reason recorded in `table1.json`.",
        "> \u2020 = diagnostic corpus (archived protocol marks MG/MINJA for GRAD as",
        "> diagnostic).",
        "",
        "Protocol: unit = trajectory/graph; for node-level sources the archived rule is",
        '"any node positive => trajectory positive". Binary baselines pool TP/FP/TN/FN',
        "across the units of a column before recomputing TPR/FPR. GRAD reuses",
        "`code/repro/unified_gnn_aggregate_20260915.py` (n_pos*n_neg fold weighting,",
        "seed-level SD across seeds 211-213).",
        "",
        "## Method identities and full-method status",
        "",
    ]
    for row_name in ROWS:
        row = rows[row_name]
        note = row.get("full_method_status")
        out.append(f"- **{row.get('display', row_name)}**"
                   + (f" — {note}" if note else ""))
    out += [
        "",
        "## A. AUROC",
        "",
        "| Method | " + " | ".join(COLUMNS) + " |",
        "|---" * (len(COLUMNS) + 1) + "|",
    ]
    for row_name in ROWS:
        row = rows[row_name]
        cells = row["cells"]
        line = [row.get("display", row_name)]
        for col in COLUMNS:
            c = cells[col]
            dagger = "\u2020" if c.get("diagnostic") else ""
            line.append(_fmt_auroc(c.get("auroc"), c["status"],
                                   bool(c.get("deterministic_across_seeds"))) + dagger)
        out.append("| " + " | ".join(line) + " |")

    out += [
        "",
        "## B. TPR / FPR (native or dev-calibrated operating point)",
        "",
        "| Method | " + " | ".join(COLUMNS) + " |",
        "|---" * (len(COLUMNS) + 1) + "|",
    ]
    for row_name in ROWS:
        row = rows[row_name]
        cells = row["cells"]
        line = [row.get("display", row_name)]
        for col in COLUMNS:
            c = cells[col]
            dagger = "\u2020" if c.get("diagnostic") else ""
            line.append(_fmt_tpr_fpr(c.get("tpr"), c.get("fpr"), c["status"]) + dagger)
        out.append("| " + " | ".join(line) + " |")

    out += [
        "",
        "## C. Confusion counts at the reported operating point",
        "",
        "| Method | Cell | TP | FP | TN | FN | n_pos | n_neg | threshold / note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row_name in ROWS:
        cells = rows[row_name]["cells"]
        for col in COLUMNS:
            c = cells[col]
            conf = c.get("confusion")
            if not conf:
                continue
            th = c.get("threshold") or {}
            note = th.get("label") or ""
            if c.get("kind") == "continuous_score":
                note += " (counts pooled across runs, each at its own dev threshold)" \
                    if "dev" in note else ""
            out.append(f"| {row_name} | {col} | {conf['tp']} | {conf['fp']} | "
                       f"{conf['tn']} | {conf['fn']} | {c.get('n_pos')} | "
                       f"{c.get('n_neg')} | {note} |")

    out += ["", "## D. Sequential Monitor variants (from raw JSON)", "",
            "| Variant | n_tasks | AUROC | TP/FP/TN/FN @0.5 | TPR/FPR @0.5 | source |",
            "|---|---|---|---|---|---|"]
    sm = rows["Sequential Monitor"]["cells"]["Decomp"]
    variants = {"decomp_fix2_cumulative (primary)": sm, **sm.get("variants", {})}
    for name, v in variants.items():
        if name.startswith("decomp_fix2_cumulative"):
            conf = v["confusion"]
            out.append(f"| {name} | {v['n_pos'] + v['n_neg']} | {v['auroc']['mean']:.4f} | "
                       f"{conf['tp']}/{conf['fp']}/{conf['tn']}/{conf['fn']} | "
                       f"{v['tpr']:.3f}/{v['fpr']:.3f} | `{v['sources'][0]}` |")
        else:
            conf = v["confusion_at_0.5"]
            out.append(f"| {name} | {v['n_tasks']} | {v['auroc']:.4f} | "
                       f"{conf['tp']}/{conf['fp']}/{conf['tn']}/{conf['fn']} | "
                       f"{v['tpr_at_0.5']:.3f}/{v['fpr_at_0.5']:.3f} | `{v['source']}` |")

    out += ["", "## E. Unfilled cells (machine-readable warnings in table1.json)", ""]
    if not table["warnings"]:
        out.append("- none")
    for w in table["warnings"]:
        metric = f" [{w['metric']}]" if w.get("metric") else ""
        out.append(f"- **{w['status']}** {w['row']} / {w['column']}{metric}: {w['reason']}")
    out.append("")
    return "\n".join(out)


# ===========================================================================
# REFERENCE CHECK
# ===========================================================================
def _check_cell(checks, method, path, expected, got, ndigits, kind="value"):
    ok = got is not None and round(got, ndigits) == expected
    checks.append({"method": method, "path": path, "expected": expected,
                   "got": got, "digits": ndigits, "kind": kind, "ok": ok})


def _check_int(checks, method, path, expected, got):
    ok = got == expected
    checks.append({"method": method, "path": path, "expected": expected,
                   "got": got, "kind": "int", "ok": ok})


def run_checks(table: dict) -> list[dict]:
    checks: list[dict] = []
    rows = table["rows"]

    # --- GRAD -------------------------------------------------------------
    grad = rows["GRAD"]["cells"]
    for col, exp in REFERENCE["GRAD"]["auroc"].items():
        a = grad[col]["auroc"]
        _check_cell(checks, "GRAD", f"{col}.auroc", exp, a["mean"], 4)
        _check_int(checks, "GRAD", f"{col}.n_seeds", REFERENCE["GRAD"]["n_seeds"],
                   a["n_seeds"])
    for col, exp in REFERENCE["GRAD"]["dev_tpr"].items():
        _check_cell(checks, "GRAD", f"{col}.dev_tpr", exp, grad[col]["tpr"], 3)
    for col, exp in REFERENCE["GRAD"]["dev_fpr"].items():
        _check_cell(checks, "GRAD", f"{col}.dev_fpr", exp, grad[col]["fpr"], 3)

    # --- AgentTrust+RiskChain --------------------------------------------
    trust = rows["AgentTrust+RiskChain"]["cells"]
    for col, exp in REFERENCE["AgentTrust+RiskChain"]["tpr"].items():
        _check_cell(checks, "AgentTrust+RiskChain", f"{col}.tpr", exp, trust[col]["tpr"], 3)
    for col, exp in REFERENCE["AgentTrust+RiskChain"]["fpr"].items():
        _check_cell(checks, "AgentTrust+RiskChain", f"{col}.fpr", exp, trust[col]["fpr"], 3)

    # --- AgentArmor -------------------------------------------------------
    armor = rows["AgentArmor"]["cells"]
    for col, exp in REFERENCE["AgentArmor"]["tpr"].items():
        _check_cell(checks, "AgentArmor", f"{col}.tpr", exp, armor[col]["tpr"], 3)
    for col, exp in REFERENCE["AgentArmor"]["fpr"].items():
        _check_cell(checks, "AgentArmor", f"{col}.fpr", exp, armor[col]["fpr"], 3)

    # --- AEGIS -------------------------------------------------------------
    aegis = rows["AEGIS"]["cells"]
    for col, (tp, fp, tn, fn) in REFERENCE["AEGIS"]["pooled"].items():
        got = aegis[col]["confusion"]
        _check_int(checks, "AEGIS", f"{col}.tp", tp, got["tp"])
        _check_int(checks, "AEGIS", f"{col}.fp", fp, got["fp"])
        _check_int(checks, "AEGIS", f"{col}.tn", tn, got["tn"])
        _check_int(checks, "AEGIS", f"{col}.fn", fn, got["fn"])
    for unit, (tp, fp, tn, fn) in REFERENCE["AEGIS"]["units"].items():
        found = None
        for col in COLUMNS:
            for u in aegis[col]["units"]:
                if u["unit"] == unit:
                    found = u
        if found is None:
            checks.append({"method": "AEGIS", "path": f"unit[{unit}]",
                           "expected": (tp, fp, tn, fn), "got": None,
                           "kind": "int4", "ok": False})
            continue
        _check_int(checks, "AEGIS", f"unit[{unit}].tp", tp, found["tp"])
        _check_int(checks, "AEGIS", f"unit[{unit}].fp", fp, found["fp"])
        _check_int(checks, "AEGIS", f"unit[{unit}].tn", tn, found["tn"])
        _check_int(checks, "AEGIS", f"unit[{unit}].fn", fn, found["fn"])

    # --- NeuroTaint -------------------------------------------------------
    nt = rows["NeuroTaint"]["cells"]
    for col, exp in REFERENCE["NeuroTaint"]["auroc"].items():
        _check_cell(checks, "NeuroTaint", f"{col}.auroc", exp, nt[col]["auroc"]["mean"], 4)
    for col, (tpr, fpr) in REFERENCE["NeuroTaint"]["pooled_tpr_fpr"].items():
        _check_cell(checks, "NeuroTaint", f"{col}.tpr", tpr, nt[col]["tpr"], 3)
        _check_cell(checks, "NeuroTaint", f"{col}.fpr", fpr, nt[col]["fpr"], 3)
    for col, (tp, fp, tn, fn) in REFERENCE["NeuroTaint"]["pooled_confusion"].items():
        got = nt[col]["confusion"]
        _check_int(checks, "NeuroTaint", f"{col}.tp", tp, got["tp"])
        _check_int(checks, "NeuroTaint", f"{col}.fp", fp, got["fp"])
        _check_int(checks, "NeuroTaint", f"{col}.tn", tn, got["tn"])
        _check_int(checks, "NeuroTaint", f"{col}.fn", fn, got["fn"])
    for col in REFERENCE["NeuroTaint"]["degenerate_columns"]:
        flag = nt[col].get("degenerate_dev_calibration")
        checks.append({"method": "NeuroTaint",
                       "path": f"{col}.degenerate_dev_calibration",
                       "expected": True, "got": flag, "kind": "flag",
                       "ok": flag is True})

    # --- Sequential Monitor ----------------------------------------------
    sm = rows["Sequential Monitor"]["cells"]["Decomp"]
    ref = REFERENCE["Sequential Monitor"]
    _check_cell(checks, "Sequential Monitor", "Decomp.auroc", ref["decomp_fix2_cumulative"]["auroc"],
                sm["auroc"]["mean"], 4)
    for k in ("tp", "fp", "tn", "fn"):
        _check_int(checks, "Sequential Monitor", f"Decomp.{k}",
                   ref["decomp_fix2_cumulative"][k], sm["confusion"][k])
    for name in ("official_cumulative", "official_separate"):
        v = sm["variants"][name]
        _check_cell(checks, "Sequential Monitor", f"{name}.auroc", ref[name]["auroc"],
                    v["auroc"], 4)
        for k in ("tp", "fp", "tn", "fn"):
            _check_int(checks, "Sequential Monitor", f"{name}.{k}", ref[name][k],
                       v["confusion_at_0.5"][k])

    # --- Agent-Sentry ------------------------------------------------------
    sentry = rows["Agent-Sentry"]["cells"]
    for col, exp in REFERENCE["Agent-Sentry"]["auroc"].items():
        _check_cell(checks, "Agent-Sentry", f"{col}.auroc", exp, sentry[col]["auroc"]["mean"], 4)
    for col, (tpr, fpr) in REFERENCE["Agent-Sentry"]["supplemental_native_tpr_fpr"].items():
        _check_cell(checks, "Agent-Sentry", f"{col}.tpr(archive supplemental)", tpr,
                    sentry[col]["tpr"], 4)
        _check_cell(checks, "Agent-Sentry", f"{col}.fpr(archive supplemental)", fpr,
                    sentry[col]["fpr"], 4)
    if sentry["MRT"]["tpr"] is not None:
        checks.append({"method": "Agent-Sentry", "path": "MRT.tpr_fpr",
                       "expected": "pending", "got": sentry["MRT"]["tpr"],
                       "kind": "pending", "ok": False})

    # --- structural: GRAD per-bench denominators --------------------------
    for bench, (n_pos, n_neg) in {"decomp": (174, 58), "mg": (600, 400),
                                  "minja": (75, 115), "mcp": (114, 118),
                                  "mrt": (50, 50)}.items():
        col = {"decomp": "Decomp", "mg": "MG", "minja": "MINJA",
               "mcp": "MCP", "mrt": "MRT"}[bench]
        c = grad[col]
        _check_int(checks, "GRAD", f"{col}.n_pos(3 seeds)", n_pos * 3, c["n_pos"])
        _check_int(checks, "GRAD", f"{col}.n_neg(3 seeds)", n_neg * 3, c["n_neg"])
    return checks


def print_check_report(checks: list[dict]) -> int:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for c in checks:
        by_method[c["method"]].append(c)
    print("\n=== --check: archived reference comparison ===")
    for method in ROWS:
        items = by_method.get(method)
        if not items:
            print(f"[SKIP] {method}: no reference values to check")
            continue
        failed = [c for c in items if not c["ok"]]
        for c in failed:
            print(f"[FAIL] {c['method']} {c['path']}: expected {c['expected']} "
                  f"got {c['got']} (kind={c['kind']})")
        status = "PASS" if not failed else f"FAIL ({len(failed)}/{len(items)} mismatches)"
        print(f"[{status}] {method} ({len(items)} checks)")
    failures = [c for c in checks if not c["ok"]]
    print(f"=== total checks={len(checks)} failures={len(failures)} ===")
    return 1 if failures else 0


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="assert the archived reference values; exit non-zero on mismatch")
    args = ap.parse_args()

    warn = Warnings()
    table = build_table(warn)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_JSON.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    TABLE_MD.write_text(render_md(table))
    print(f"wrote {rel(TABLE_JSON)}")
    print(f"wrote {rel(TABLE_MD)}")

    if not args.check:
        return 0
    checks = run_checks(table)
    return print_check_report(checks)


if __name__ == "__main__":
    raise SystemExit(main())
