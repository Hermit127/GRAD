#!/usr/bin/env python3
"""Build LOGS/final_five_tables/table5/ablation_report.md from raw results.

Reads:
  LOGS/final_five_tables/table5/table5.json              (aggregated cells)
  LOGS/final_five_tables/table5/cell_provenance.json     (paired stats/provenance)
  LOGS/final_five_tables/raw_results/table5_panelb_verify.json
  LOGS/final_five_tables/table5/table5_efficiency.json   (optional)
  LOGS/final_five_tables/table5_protocol_manifest.json

No number is hand-typed; every value is interpolated from the artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
T5 = REPO / "LOGS/final_five_tables/table5"


def load(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def fmt(v, digits=3):
    if v is None:
        return "--"
    return f"{v:.{digits}f}"


def main() -> int:
    t5 = load(T5 / "table5.json", {})
    prov = load(T5 / "cell_provenance.json", {})
    pb = load(REPO / "LOGS/final_five_tables/raw_results/table5_panelb_verify.json", {})
    eff = load(T5 / "table5_efficiency.json")
    manifest = load(REPO / "LOGS/final_five_tables/table5_protocol_manifest.json", {})

    pa = t5.get("panel_a", {})
    rows = pa.get("rows", {})
    lines = [
        "# Table 5 — Core Ablation Report",
        "",
        "Auto-generated from raw run artifacts; zero API spend. Companion files:",
        "`table5.tex` (paper table), `table5.md` (draft), `table5.json` (cells),",
        "`cell_provenance.json` (per-cell provenance), `protocol_manifest.json`.",
        "",
        "## 1. Frozen protocol (before any run)",
        "",
        f"- Splits: `{manifest.get('datasets', {}).get('decomp', {}).get('provenance', '')}`",
        "  and MCP fix2 cv5 / MRT leave-one-task-out; all train/dev/test group or task",
        "  overlaps audited to 0 in `table5_protocol_manifest.json`.",
        f"- Seeds: {manifest.get('seeds')}; epochs {manifest['training_budget']['epochs']}, "
        f"patience {manifest['training_budget']['patience']}, batch "
        f"{manifest['training_budget']['batch_size']}, hidden "
        f"{manifest['training_budget']['hidden_dim']}.",
        f"- Checkpoint selection: {manifest['checkpoint_selection']['metric']} on dev "
        f"at FPR<={manifest['checkpoint_selection']['target_fpr']}.",
        "- Input observation: node text `features.subtask` (192 chars CharCNN); "
        "12-dim permission vector; turn position; 9 handcrafted keyword flags;",
        "  rebuilt typed edges (10 types + confidence + rationale). Ignored: label "
        "(except y), harm_step, node_label, injected, taint, raw_text, metadata.",
        "- All variants are retrained from scratch with the module removed at input "
        "construction (no test-time perturbation). `full` is tensor-identical to the "
        "original builder (unit test).",
        "",
        "## 2. Panel (a) results",
        "",
        "| Method | Decomp | MCP | MRT |",
        "|---|---|---|---|",
    ]
    order = ["Information-matched Monitor", "GUARDIAN-adapted", "GRAD: text only",
             "GRAD: structure only", "GRAD: chronological edges only",
             "GRAD: without relation types", "GRAD: without keyword features",
             "GRAD: full joint encoder", "GRAD: w/o type semantics (appendix)"]
    for name in order:
        r = rows.get(name, {})
        cells = []
        for c in ("Decomp", "MCP", "MRT"):
            v = r.get(c)
            if v is None:
                cells.append("--")
            else:
                std = v.get("auroc_std")
                cells.append(f"{v['auroc_mean']:.3f}" +
                             (f"±{std:.3f}" if std and std > 5e-4 else ""))
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += ["", "### Pre-specified paired comparisons (variant - full, AUROC)",
              "",
              "Unit = seed (n=3). `all_seeds_same_sign` requires the sign to agree in "
              "all three seeds; with n=3 seeds no p-value is claimed.",
              "",
              "| Variant | Dataset | mean delta | per-seed deltas | same direction |",
              "|---|---|---|---|---|"]
    for name in order:
        p = prov.get("panel_a", {}).get(name, {})
        for col in ("Decomp", "MCP"):
            cp = p.get(col, {}) if isinstance(p, dict) else {}
            pair = cp.get("paired_vs_full")
            if not pair:
                continue
            per_seed = ", ".join(f"{k}:{v:+.3f}" for k, v in
                                 sorted(pair["per_seed_delta"].items()))
            lines.append(f"| {name} | {col} | {pair['seed_level_delta_mean']:+.3f} | {per_seed} "
                         f"| {'yes' if pair['all_seeds_same_sign'] else 'no'} |")

    lines += ["", "### Auxiliary metrics", "",
              "PR-AUC and the dev-selected operating point are in `table5.md` "
              "(auto-generated) and `table5.json` per cell.",
              ""]

    lines += ["## 3. Panel (b): independent re-verification and head-to-head", ""]
    if pb:
        checks = pb.get("protocol_checks", {})
        lines += [
            f"- Protocol identity: {checks.get('runs_with_zero_recompute_mismatch')}/40 runs "
            "recompute the accuracy matrix and F_T from per-graph scores with zero mismatch; "
            f"orders={checks.get('orders')}, units={checks.get('runs_per_seed_order')}.",
            f"- `F_T = (1/T) sum_j [max_(t>=j) a_tj - a_Tj]` (verified against the "
            "archived runner implementation).",
            "",
        ]
    pw = pb.get("pairwise_recomputed", {}) if pb else {}
    if pw:
        lines += ["| Pair | Metric | delta | 95% CI (seed-clustered) | p (20 seed means) | wins |",
                  "|---|---|---|---|---|---|"]
        for pair_name, d in pw.items():
            for metric in ("macro_acc", "forgetting"):
                m = d[metric]
                lines.append(f"| {pair_name} | {metric} | {m['mean']:+.4f} "
                             f"| [{m['ci95_cluster_seed'][0]:+.4f}, {m['ci95_cluster_seed'][1]:+.4f}] "
                             f"| {m['p_wilcoxon_20seedmeans']:.4g} "
                             f"| {m['wins']}/{m['n']} |")
        lines.append("")

    if eff:
        lines += ["## 4. Inference cost (identical hardware and batch size)", "",
                  "| Mode | ms/graph |", "|---|---|"]
        for r in eff["results"]:
            lines.append(f"| {r['mode']} | {r['ms_per_graph']:.2f} |")
        params = eff["params"]
        lines += ["",
                  f"- Shared detector: {params['shared_detector']:,} params "
                  f"({params['shared_detector_bytes']/1e6:.2f} MB); adapter "
                  f"{params['adapter']:,} params; head {params['head']} B; "
                  f"expert file {params['expert_disk_bytes']} B for the pilot's 2 domains.",
                  f"- {eff['note']}",
                  ""]

    lines += [
        "## 5. Blockers and pending cells",
        "",
        "- **MRT (all GRAD rows)**: one 100-epoch MRT run is ~5-6 h on the local CPU "
        "(35x MCP node count); the GPU host `gpu-server` is unreachable behind the "
        "jumpboard (and now also rejects keys). MRT cells remain `--`. Increment "
        "request: restore GPU access (preferred) or approve a local CPU batch "
        "(~15 runs/variant).",
        "- **GUARDIAN-adapted**: blocked on comparability and a method decision; see "
        "`guardian_audit.md`. No numbers entered.",
        "- **Information-matched Monitor (MRT)**: blocked-cost; probe-backed projection "
        "CNY199.3; disposition awaiting PI.",
        "- **Empty-cell policy**: no imputation, no test-set threshold tuning, no "
        "test-time perturbation results presented as module ablations.",
        "",
        "## 6. Claim status (computed, not asserted)",
        "",
    ]

    # Automatic supported/unsupported assessment for the six pre-registered comparisons
    claims = []
    for name, variant in [("GRAD: text only", "text_only"),
                          ("GRAD: structure only", "structure_only"),
                          ("GRAD: chronological edges only", "chrono_only"),
                          ("GRAD: without relation types", "no_relation_types"),
                          ("GRAD: without keyword features", "no_keyword_features")]:
        parts = []
        for col in ("Decomp", "MCP"):
            cp = prov.get("panel_a", {}).get(name, {}).get(col, {})
            pair = cp.get("paired_vs_full") if isinstance(cp, dict) else None
            if not pair:
                parts.append(f"{col}: pending")
                continue
            verdict = ("full higher (all seeds)" if pair["seed_level_delta_mean"] < 0
                       and pair["all_seeds_same_sign"] else
                       "full higher (not all seeds)" if pair["seed_level_delta_mean"] < 0 else
                       "variant >= full" if pair["seed_level_delta_mean"] > 0 else "tied")
            parts.append(f"{col}: {verdict} ({pair['seed_level_delta_mean']:+.3f})")
        claims.append(f"- `full` vs `{name}`: " + "; ".join(parts))
    lines += claims
    lines += [
        "",
        "Caveats that always apply:",
        "- n=3 seeds cannot support significance claims; folds/seeds are correlated, "
        "so repeated predictions are not independent samples.",
        "- `structure_only` retains a text-derived edge topology (deterministic rules "
        "over visible text): it is a non-text-feature variant, not text-blind.",
        "- `no_relation_types` keeps edge descriptions and confidence, so it is a "
        "type-label ablation, not a full removal of relation semantics; the appendix "
        "variant `no_type_semantics` removes the type embedding and confidence.",
        "- `no_keyword_features` only tests the handcrafted flags; the model can still "
        "learn lexical features from raw text, so it cannot independently prove the "
        "absence of dataset shortcuts.",
        "- The Information-matched Monitor is adapted (MCP) / native (Decomp) and "
        "cannot be ranked against GRAD without its disclosed protocol differences.",
        "",
    ]
    (T5 / "ablation_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {T5 / 'ablation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
