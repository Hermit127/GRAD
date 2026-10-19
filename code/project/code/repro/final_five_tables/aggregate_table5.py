#!/usr/bin/env python3
"""Table 5 (core ablations) aggregator: raw runs -> JSON/MD/LaTeX/provenance.

Zero API. Reads only raw artifacts:
  Panel A: code/repro/out/table5_ablation/runs/<ds>_<cell>_<variant>_s<seed>.json
           (variants retrained on the frozen protocol),
           archived full runs in code/repro/out/unified_gnn/runs/,
           Information-matched Monitor cells in
           LOGS/final_five_tables/raw_results/lsm_infomatch.json.
  Panel B: LOGS/final_five_tables/raw_results/table5_panelb_verify.json
           (independent recomputation of the 40 p16 runs).

Commands:
  python aggregate_table5.py panel_a|panel_b|all

Outputs (default dir LOGS/final_five_tables/table5/):
  raw_results/table5_panel_a.json, raw_results/table5_panel_b.json,
  table5.json, table5.md, table5.tex, cell_provenance.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
ABL_RUNS = REPO / "code/repro/out/table5_ablation/runs"
RUNS_OVERRIDE: Path | None = None
ARCH_FULL = REPO / "code/repro/out/unified_gnn/runs"
LSM = REPO / "LOGS/final_five_tables/raw_results/lsm_infomatch.json"
PB_VERIFY = REPO / "LOGS/final_five_tables/raw_results/table5_panelb_verify.json"
OUT_DIR = REPO / "LOGS/final_five_tables/table5"

VARIANTS = [
    ("GRAD: text only", "text_only"),
    ("GRAD: structure only", "structure_only"),
    ("GRAD: chronological edges only", "chrono_only"),
    ("GRAD: without relation types", "no_relation_types"),
    ("GRAD: without keyword features", "no_keyword_features"),
    ("GRAD: full joint encoder", "full"),
]
APPENDIX_LABEL = "GRAD: w/o type semantics (appendix)"
APPENDIX_VARIANT = "no_type_semantics"
COLUMNS = ["Decomp", "MCP", "MRT"]
SEEDS = [211, 212, 213]
DATASET_CELLS = {
    "decomp": ["fold0"],
    "mcp": [f"fold{k}" for k in range(5)],
    "mrt": [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                                  "bank_transfer_2", "expenditure_accounting",
                                  "spam_filter_update")],
}
DATASET_COL = {"decomp": "Decomp", "mcp": "MCP", "mrt": "MRT"}
ARCH_FULL_NAME = {
    "decomp": lambda cell, seed: f"decomp_f0_s{seed}",
    "mcp": lambda cell, seed: f"mcp_{cell.replace('fold', 'f')}_s{seed}",
    "mrt": lambda cell, seed: f"mrt_{cell.replace('fold_', '')}_s{seed}",
}


def load_run(path: Path) -> dict:
    d = json.loads(path.read_text())
    test = d.get("scores", {}).get("test", [])
    if not test:
        return {"auroc": float("nan"), "pr_auc": float("nan"), "n_pos": 0,
                "n_neg": 0, "dev_threshold": None, "dev_tpr": None,
                "dev_fpr": None, "best_epoch": None,
                "ablation": d.get("ablation", {})}
    labels = [int(r["label"]) for r in test]
    vals = [float(r["score"]) for r in test]
    thr = d.get("threshold_reports", {}).get("fpr_10", {})
    return {
        "auroc": roc_auc_score(labels, vals) if len(set(labels)) > 1 else float("nan"),
        "pr_auc": average_precision_score(labels, vals) if len(set(labels)) > 1 else float("nan"),
        "n_pos": sum(labels), "n_neg": len(labels) - sum(labels),
        "dev_threshold": thr.get("threshold"),
        "dev_tpr": thr.get("test", {}).get("recall"),
        "dev_fpr": thr.get("test", {}).get("fpr"),
        "best_epoch": d.get("selected_checkpoint", {}).get("epoch"),
        "ablation": d.get("ablation", {}),
    }


def variant_runs(dataset: str, variant: str, runs_dir: Path | None = None) -> dict:
    base = Path(runs_dir or RUNS_OVERRIDE or ABL_RUNS)
    if not base.is_absolute():
        base = REPO / base
    out = {}
    for cell in DATASET_CELLS[dataset]:
        for seed in SEEDS:
            p = base / f"{dataset}_{cell}_{variant}_s{seed}.json"
            if p.exists():
                out[(cell, seed)] = {"path": p, "metrics": load_run(p)}
    return out


def archived_full_runs(dataset: str) -> dict:
    out = {}
    for cell in DATASET_CELLS[dataset]:
        for seed in SEEDS:
            p = ARCH_FULL / f"{ARCH_FULL_NAME[dataset](cell, seed)}.json"
            if p.exists():
                out[(cell, seed)] = {"path": p, "metrics": load_run(p)}
    return out


def aggregate_cells(runs: dict) -> dict | None:
    """AUROC/PR-AUC: fold-weighted by n_pos*n_neg within seed; mean/SD across seeds."""
    per_seed = {}
    for seed in SEEDS:
        items = [(c, v) for (c, s), v in runs.items() if s == seed]
        if not items:
            continue
        weights = [v["metrics"]["n_pos"] * v["metrics"]["n_neg"] for _, v in items]
        wsum = sum(weights)
        per_seed[seed] = {
            "auroc": sum(v["metrics"]["auroc"] * w for (_, v), w in zip(items, weights)) / wsum,
            "pr_auc": sum(v["metrics"]["pr_auc"] * w for (_, v), w in zip(items, weights)) / wsum,
            "dev_tpr": float(np.mean([v["metrics"]["dev_tpr"] for _, v in items
                                      if v["metrics"]["dev_tpr"] is not None])),
            "dev_fpr": float(np.mean([v["metrics"]["dev_fpr"] for _, v in items
                                      if v["metrics"]["dev_fpr"] is not None])),
            "n_cells": len(items),
            "n_pos": sum(v["metrics"]["n_pos"] for _, v in items),
            "n_neg": sum(v["metrics"]["n_neg"] for _, v in items),
        }
    if not per_seed:
        return None
    return {
        "auroc_mean": float(np.mean([v["auroc"] for v in per_seed.values()])),
        "auroc_std": float(np.std([v["auroc"] for v in per_seed.values()])),
        "pr_auc_mean": float(np.mean([v["pr_auc"] for v in per_seed.values()])),
        "pr_auc_std": float(np.std([v["pr_auc"] for v in per_seed.values()])),
        "dev_tpr_mean": float(np.mean([v["dev_tpr"] for v in per_seed.values()])),
        "dev_fpr_mean": float(np.mean([v["dev_fpr"] for v in per_seed.values()])),
        "n_seeds": len(per_seed),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "repetition_unit": "seed (fold-weighted within seed: 1 fold decomp, 5 mcp/mrt)",
    }


def _test_counts(run_path: Path) -> tuple[int, int]:
    """Return (n_pos, n_total) of the test score dump."""
    d = json.loads(run_path.read_text())
    test = d.get("scores", {}).get("test", [])
    npos = sum(int(r["label"]) for r in test)
    return npos, len(test)


def paired_vs_full(variant_map: dict, full_map: dict) -> dict:
    """Seed-level fold-weighted deltas (primary) + unweighted pair mean (secondary).

    The displayed cell means are fold-weighted within seed and then averaged over
    seeds, so the paired delta must follow the same path. The pair mean over
    (fold, seed) units is reported separately and was the source of the earlier
    0.029-vs-0.025 discrepancy.
    """
    common = sorted(set(variant_map) & set(full_map))
    diffs, seed_deltas = [], {}
    for c, s in common:
        va = variant_map[(c, s)]["metrics"]["auroc"]
        fa = full_map[(c, s)]["metrics"]["auroc"]
        diffs.append(va - fa)
        npos, n = _test_counts(variant_map[(c, s)]["path"])
        seed_deltas.setdefault(s, []).append((va - fa, npos * (n - npos)))
    if not diffs:
        return {}
    per_seed = {s: sum(v * w for v, w in items) / sum(w for _, w in items)
                for s, items in seed_deltas.items()}
    return {
        "n_pairs": len(diffs),
        "seed_level_delta_mean": float(np.mean(list(per_seed.values()))),
        "seed_level_delta_std": float(np.std(list(per_seed.values()))),
        "per_seed_delta": {str(k): round(v, 4) for k, v in per_seed.items()},
        "pair_mean_delta": float(np.mean(diffs)),
        "pair_std": float(np.std(diffs)),
        "all_seeds_same_sign": (all(v > 0 for v in per_seed.values())
                                or all(v < 0 for v in per_seed.values())),
        "note": "unit=seed (n=3); folds within a seed are correlated and are not "
                "treated as independent replicates; no strong significance claim",
    }


def monitor_aux() -> dict:
    """PR-AUC for the information-matched monitor from its stored per-task probs.

    Reconstruction must reproduce the stored AUROC exactly, otherwise the cell
    is skipped (no silently different aggregation).
    """
    import importlib
    import sys as _sys
    _sys.path.insert(0, str(REPO / "code/repro"))
    _sys.path.insert(0, str(REPO / "code/repro/static_big_table_v1"))
    try:
        mod = importlib.import_module("lsm_monitor_infomatch_20260917")
    except Exception:
        return {}
    out = {}
    src = REPO / "LOGS/lsm_infomatch_mcp_cumulative_20260917.json"
    if src.exists():
        d = json.loads(src.read_text())
        tasks = mod.load_tasks("mcp")
        probs = d["per_task_probs"]
        y, s = [], []
        for t in tasks:
            preds = probs[t["id"]]
            valid = [p for p in preds if p is not None]
            if t["label"] == 0 or not t["harm_index"]:
                y.append(t["label"]); s.append(max(valid) if valid else 0.0)
            else:
                rel = [p for p in preds[:t["harm_index"]] if p is not None]
                y.append(1); s.append(max(rel) if rel else 0.0)
        auroc = roc_auc_score(y, s)
        stored = json.loads(LSM.read_text())["cells"]["MCP"]["auroc"]
        if abs(auroc - stored) < 1e-9:
            out["MCP"] = {"pr_auc": float(average_precision_score(y, s)),
                          "n": len(y), "checked_auroc": auroc}
    return out


def panel_a() -> dict:
    lsm = json.loads(LSM.read_text())
    aux = monitor_aux()
    rows: dict = {}
    provenance: dict = {}

    rows["Information-matched Monitor"] = {
        "Decomp": {"auroc_mean": lsm["cells"]["Decomp"]["auroc"], "auroc_std": 0.0},
        "MCP": {"auroc_mean": lsm["cells"]["MCP"]["auroc"], "auroc_std": 0.0,
                "pr_auc_mean": aux.get("MCP", {}).get("pr_auc")},
        "MRT": None,
    }
    provenance["Information-matched Monitor"] = {
        "Decomp": {"status": "reused", "kind": "native interface", "n": 232,
                   "source": "LOGS/lsm_infomatch_report_20260917.md / "
                             "LOGS/final_five_tables/raw_results/lsm_infomatch.json"},
        "MCP": {"status": "new 2026-09-17", "kind": "information-matched adaptation",
                "n": 232, "deviation": "uses first node harm_step because the MCP graph "
                "harm_index is frozen at 1", "errors": 0},
        "MRT": {"status": "--", "reason": "blocked-cost: probe (1 graph x 20 prefixes, "
                "20 calls, CNY0.0309) projects CNY199.3 for the full 15,501-prefix run; "
                "0 formal spend; disposition awaiting PI"},
    }

    rows["GUARDIAN-adapted"] = {c: None for c in COLUMNS}
    provenance["GUARDIAN-adapted"] = {"Decomp": {"status": "--"}, "MCP": {"status": "--"},
                                      "MRT": {"status": "--"},
                                      "reason": "blocked (comparability): official repo "
                                      "ref/GUARDIAN@60d7b8b is an unsupervised multi-agent "
                                      "debate-graph autoencoder (BERT+GCN; nodes=agent replies); "
                                      "no drop-in mapping to tool-event graphs; adaptation needs "
                                      "new method decisions + GPU; PI approval pending "
                                      "(new_baselines_audit_20260916.md §2.5/§2.8)"}

    full_maps = {ds: (variant_runs(ds, "full") or archived_full_runs(ds))
                 for ds in ("decomp", "mcp", "mrt")}

    for label, variant in VARIANTS:
        row, prov = {}, {}
        for ds in ("decomp", "mcp", "mrt"):
            vr = variant_runs(ds, variant)
            used_archived = False
            if not vr and variant == "full":
                vr = archived_full_runs(ds)
                used_archived = True
            agg = aggregate_cells(vr) if vr else None
            if agg is None:
                reason = ("no formal runs yet" if ds != "mrt" else
                          "blocked: local CPU infeasible for MRT (35x MCP nodes, "
                          "~5-6h/run at 100 epochs); GPU host unreachable")
                row[DATASET_COL[ds]] = None
                prov[DATASET_COL[ds]] = {"status": "--", "reason": reason}
                continue
            if variant != "full" and vr and full_maps[ds]:
                prov[DATASET_COL[ds]] = {
                    "status": "reused-archived" if used_archived else "new",
                    "source_files": sorted(str(v["path"].relative_to(REPO))
                                           for v in vr.values()),
                    "paired_vs_full": paired_vs_full(vr, full_maps[ds]),
                }
            else:
                prov[DATASET_COL[ds]] = {
                    "status": "reused-archived" if used_archived else "new",
                    "source_files": sorted(str(v["path"].relative_to(REPO))
                                           for v in vr.values()),
                }
            row[DATASET_COL[ds]] = agg
        rows[label] = row
        provenance[label] = prov

    app = {}
    app_prov = {}
    for ds in ("decomp", "mcp"):
        runs = variant_runs(ds, APPENDIX_VARIANT)
        agg = aggregate_cells(runs) if runs else None
        app[DATASET_COL[ds]] = agg
        app_prov[DATASET_COL[ds]] = {"status": "new" if runs else "--"}
    app["MRT"] = None
    app_prov["MRT"] = {"status": "--", "reason": "deferred (same MRT blocker)"}
    rows[APPENDIX_LABEL] = app
    provenance[APPENDIX_LABEL] = app_prov

    return {"table": "table5_panel_a", "columns": COLUMNS, "rows": rows,
            "provenance": provenance,
            "runs_dir": str(RUNS_OVERRIDE or ABL_RUNS)}


def panel_b() -> dict:
    d = json.loads(PB_VERIFY.read_text())
    rows_map = [
        ("Shared detector: sequential fine-tuning", "naive"),
        ("Experts + random routing", "specialists_random"),
        ("Experts + all-expert aggregation", "specialists_all"),
        ("Experts + proposed routing", "specialists"),
        ("Experts + oracle routing", "specialists_oracle"),
    ]
    rows = {}
    for label, arm in rows_map:
        s = d["summary_recomputed"][arm]
        rows[label] = {
            "arm": arm,
            "final_acc": {
                "mean": s["macro_acc"]["mean_40runs"], "std": s["macro_acc"]["std_40runs"],
                "mean_seed_level": s["macro_acc"]["mean_20seeds"],
                "std_seed_level": s["macro_acc"]["std_20seeds"],
                "ci95_cluster_seed": s["macro_acc"]["ci95_cluster_seed"]},
            "forgetting": {
                "mean": s["forgetting"]["mean_40runs"], "std": s["forgetting"]["std_40runs"],
                "mean_seed_level": s["forgetting"]["mean_20seeds"],
                "std_seed_level": s["forgetting"]["std_20seeds"],
                "ci95_cluster_seed": s["forgetting"]["ci95_cluster_seed"]},
            "router_hit": s["router_hit"]["mean_40runs"],
            "paired_vs_shared": (d["paired_vs_naive_recomputed"].get(arm)
                                 if arm != "naive" else None),
        }
    return {
        "table": "table5_panel_b",
        "protocol": {
            "n_runs": 40, "units": "20 seeds x 2 orders (seeds 211-230)",
            "source": "code/repro/out/real_cl/p16_40",
            "verification": "40/40 runs recompute acc_matrix+F_T from per-graph "
                            "scores.jsonl with zero mismatch",
            "forgetting": "F_T=(1/T)*sum_j[max_{t=j..T} a_tj - a_Tj]",
            "oracle_warning": "diagnostic reference only; not deployable; not an "
                              "accuracy upper bound",
            "efficiency": "table5_efficiency.json (timing microbenchmark; p16 saved "
                          "no checkpoints, P1 pilot checkpoints used for cost only)",
        },
        "rows": rows,
        "pairwise": d["pairwise_recomputed"],
    }


def tex_panel_a(pa: dict) -> list[str]:
    order = ["Information-matched Monitor", "GUARDIAN-adapted", "GRAD: text only",
             "GRAD: structure only", "GRAD: chronological edges only",
             "GRAD: without relation types", "GRAD: without keyword features",
             "GRAD: full joint encoder", APPENDIX_LABEL]
    lines = [
        "\\begin{minipage}[t]{0.49\\textwidth}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "\\multicolumn{4}{@{}l}{\\textbf{(a) Semantic--Structural Modeling}} \\\\",
        "\\multicolumn{4}{@{}l}{\\footnotesize trajectory-level AUROC $\\uparrow$ ($[0,1]$), "
        "mean$\\pm$std over 3 seeds} \\\\",
        "\\midrule",
        "\\textit{Method} & Decomp & MCP & MRT \\\\",
        "\\midrule",
    ]
    for name in order:
        r = pa["rows"].get(name, {})
        if name == "GUARDIAN-adapted":
            lines.append("GUARDIAN-adapted$^{\\ddagger}$ & -- & -- & -- \\\\")
            continue
        label = name.replace("GRAD: w/o type semantics (appendix)", "GRAD: w/o type semantics$^{\\dagger}$")
        cells = []
        for c in COLUMNS:
            v = r.get(c)
            if v is None:
                cells.append("--")
            else:
                std = v.get("auroc_std", 0.0)
                cells.append(f"{v['auroc_mean']:.3f}$\\pm${std:.3f}" if std > 5e-4
                             else f"{v['auroc_mean']:.3f}")
        if name == APPENDIX_LABEL:
            lines.append("\\midrule")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{minipage}"]
    return lines


def tex_panel_b(pb: dict) -> list[str]:
    order = ["Shared detector: sequential fine-tuning", "Experts + random routing",
             "Experts + all-expert aggregation", "Experts + proposed routing",
             "Experts + oracle routing"]
    lines = [
        "\\begin{minipage}[t]{0.49\\textwidth}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "\\multicolumn{3}{@{}l}{\\textbf{(b) Antibody Experts and Routing}} \\\\",
        "\\multicolumn{3}{@{}l}{\\footnotesize MCP in-domain; "
        "40 runs $=$ 20 seeds $\\times$ 2 orders} \\\\",
        "\\midrule",
        "\\textit{Method} & Final ACC $\\uparrow$ & $F_T$ $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for name in order:
        r = pb["rows"][name]
        cells = (f"{r['final_acc']['mean']:.3f}$\\pm${r['final_acc']['std']:.3f} & "
                 f"{r['forgetting']['mean']:.3f}$\\pm${r['forgetting']['std']:.3f}")
        if name == "Shared detector: sequential fine-tuning":
            lines.append(f"\\rowcolor{{gray!15}} {name} & {cells} \\\\")
        elif name == "Experts + oracle routing":
            lines.append(f"\\textit{{{name}}} & {cells} \\\\")
        else:
            lines.append(f"{name} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{minipage}"]
    return lines


def build_latex(pa: dict, pb: dict) -> str:
    lines = [
        "% Table 5: core ablations. AUTO-GENERATED by aggregate_table5.py; do not edit.",
        "% Requires: booktabs, colortbl (xcolor), graphicx (for \\resizebox if used).",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Core ablations. (a) Semantic--structural modeling: trajectory-level AUROC "
        "in $[0,1]$ (mean$\\pm$std over seeds 211/212/213; Decomp uses the frozen single "
        "fold, MCP/MRT use the 5-fold cross-task splits, fold-weighted by $n_+n_-$ within "
        "each seed). (b) Antibody experts and routing on the MCP in-domain stream: "
        "$F_T=\\frac{1}{T}\\sum_j[\\max_{t\\ge j}a_{t,j}-a_{T,j}]$; std is over 40 runs "
        "(20 seeds $\\times$ 2 orders) while seed-clustered 95\\% CIs are in the provenance. "
        "Oracle routing is a diagnostic reference only.}",
        "\\label{tab:table5_ablations}",
        "\\setlength{\\tabcolsep}{4pt}",
    ]
    lines += tex_panel_a(pa)
    lines += ["\\hfill"]
    lines += tex_panel_b(pb)
    lines += [
        "\\par\\vspace{2pt}",
        "{\\footnotesize $^{\\dagger}$Appendix variant (type embedding unified and edge "
        "confidence zeroed). $^{\\ddagger}$Blocked on comparability; see the audit, no "
        "number entered. MRT `GRAD: full joint encoder` is reused from the archived "
        "57-run sweep (same frozen protocol); all other GRAD cells are new frozen-protocol "
        "runs (GPU). The monitor row is native on Decomp and an information-matched "
        "adaptation on MCP; MRT is blocked-cost. `--` means no number, never zero.}",
        "\\end{table*}", ""]
    return "\n".join(lines)


def fmt(v, digits=3):
    if v is None:
        return "--"
    if hasattr(v, "get") and v.get("auroc_mean") is not None:
        s = v["auroc_std"]
        return f"{v['auroc_mean']:.3f}" + (f"±{s:.3f}" if s > 5e-4 else "")
    return "--"


def write_outputs(pa: dict, pb: dict, out_dir: Path, eff_path: Path | None) -> None:
    raw = out_dir / "raw_results"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "table5_panel_a.json").write_text(json.dumps(pa, indent=1))
    (raw / "table5_panel_b.json").write_text(json.dumps(pb, indent=1))

    eff = json.loads(eff_path.read_text()) if eff_path and eff_path.exists() else None
    combined = {"table": "table5_ablations", "panel_a": pa, "panel_b": pb,
                "efficiency": eff}
    (out_dir / "table5.json").write_text(json.dumps(combined, indent=1))

    md = ["# Table 5 — Core Ablations (auto-generated; all numbers from raw runs)", "",
          "## Panel (a) Semantic–Structural Modeling (trajectory AUROC, [0,1])", "",
          "| Method | Decomp | MCP | MRT |", "|---|---|---|---|"]
    for name in (["Information-matched Monitor", "GUARDIAN-adapted", "GRAD: text only",
                  "GRAD: structure only", "GRAD: chronological edges only",
                  "GRAD: without relation types", "GRAD: without keyword features",
                  "GRAD: full joint encoder", APPENDIX_LABEL]):
        r = pa["rows"][name]
        md.append(f"| {name} | {fmt(r.get('Decomp'))} | {fmt(r.get('MCP'))} | {fmt(r.get('MRT'))} |")
    md += ["", "## Panel (b) Antibody Experts and Routing (MCP in-domain)", "",
           "| Method | Final ACC | F_T | Router hit |", "|---|---|---|---|"]
    for name, r in pb["rows"].items():
        rt = r["router_hit"]
        md.append(f"| {name} | {r['final_acc']['mean']:.3f}±{r['final_acc']['std']:.3f} "
                  f"| {r['forgetting']['mean']:.3f}±{r['forgetting']['std']:.3f} "
                  f"| {'--' if rt is None else f'{rt:.3f}'} |")
    md += ["", "## Auxiliary metrics (Panel a; PR-AUC and dev-selected operating point)", "",
           "| Method | Dataset | AUROC | PR-AUC | dev TPR@FPR10 | dev FPR@FPR10 |",
           "|---|---|---|---|---|---|"]
    for name, r in pa["rows"].items():
        for col in COLUMNS:
            v = r.get(col)
            if not isinstance(v, dict) or v.get("pr_auc_mean") is None:
                continue
            tpr = f"{v['dev_tpr_mean']:.3f}" if v.get("dev_tpr_mean") is not None else "--"
            fpr = f"{v['dev_fpr_mean']:.3f}" if v.get("dev_fpr_mean") is not None else "--"
            md.append(f"| {name} | {col} | {v['auroc_mean']:.3f} | {v['pr_auc_mean']:.3f} "
                      f"| {tpr} | {fpr} |")
    if eff:
        md += ["", "## Inference cost (identical hardware and batch size)", "",
               "| Mode | ms/graph |", "|---|---|"]
        for r in eff["results"]:
            md.append(f"| {r['mode']} | {r['ms_per_graph']:.2f} |")
    (out_dir / "table5.md").write_text("\n".join(md) + "\n")
    (out_dir / "table5.tex").write_text(build_latex(pa, pb))
    prov = {"generated_by": "code/repro/final_five_tables/aggregate_table5.py",
            "panel_a": pa["provenance"],
            "panel_b": {"protocol": pb["protocol"], "pairwise": pb["pairwise"]}}
    if eff:
        prov["efficiency"] = {"note": eff["note"], "params": eff["params"]}
    (out_dir / "cell_provenance.json").write_text(json.dumps(prov, indent=1))


def main() -> int:
    global RUNS_OVERRIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["panel_a", "panel_b", "all"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--runs-dir", default=None,
                    help="override the GRAD variant runs directory "
                         "(default code/repro/out/table5_ablation/runs; "
                         "use runs_cpu for the local CPU replication set)")
    ap.add_argument("--eff-json", default=str(OUT_DIR / "table5_efficiency.json"))
    args = ap.parse_args()
    if args.runs_dir:
        RUNS_OVERRIDE = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pa, pb = panel_a(), panel_b()
    if args.cmd == "panel_a":
        (out_dir / "raw_results").mkdir(exist_ok=True)
        (out_dir / "raw_results/table5_panel_a.json").write_text(json.dumps(pa, indent=1))
        print("panel_a written")
        return 0
    if args.cmd == "panel_b":
        (out_dir / "raw_results").mkdir(exist_ok=True)
        (out_dir / "raw_results/table5_panel_b.json").write_text(json.dumps(pb, indent=1))
        print("panel_b written")
        return 0
    write_outputs(pa, pb, out_dir, Path(args.eff_json))
    print(f"table5 written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
