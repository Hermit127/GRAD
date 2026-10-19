#!/usr/bin/env python3
"""Auto-generate the Phase B instrumentation report from raw instrumentation JSONs."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KW = REPO / "code/repro/out/table5_keyword"
OUT = REPO / "LOGS/final_five_tables/table5"


def load_all():
    out = {}
    for p in sorted((KW / "runs").glob("instr_*.json")):
        out[p.stem.replace("instr_", "")] = json.loads(p.read_text())
    local = KW / "instr_mcp_fold0_s211" / "instrumentation.json"
    if local.exists():
        out.setdefault("mcp_fold0_s211_cpu", json.loads(local.read_text()))
    return out


def main() -> int:
    data = load_all()
    md = ["# Keyword channel instrumentation (Phase B, dev only)", "",
          "Short diagnostic runs (15 epochs, best-dev-AUROC state; NOT the frozen "
          "selection rule). No test file is read; masking/permutation are sensitivity "
          "probes, not ablations.", ""]
    md += ["| run | normal | keyword zeroed | keyword permuted | norm s/k/char | "
           "grad s/k/char (E1) |", "|---|---|---|---|---|---|"]
    for name, d in data.items():
        h = d["history"]
        e1 = h[0]
        s = d["sensitivity_dev_auroc"]
        md.append(f"| {name} | {s['normal']:.4f} | {s['keyword_zeroed']:.4f} "
                  f"({s['keyword_zeroed']-s['normal']:+.4f}) | {s['keyword_permuted']:.4f} "
                  f"({s['keyword_permuted']-s['normal']:+.4f}) | "
                  f"{e1['norm_struct']:.2f}/{e1['norm_keyword']:.2f}/{e1['norm_char']:.2f} | "
                  f"{e1['grad_struct']:.2e}/{e1['grad_keyword']:.2e}/{e1['grad_char']:.2e} |")
    md += ["", "## Interpretation (evidence-tagged)", "",
           "- **Scale imbalance (supported)**: the char-text block enters `node_proj` with "
           "a norm 10-20x larger than the keyword/struct blocks (Decomp ~8.4-9.3 vs 0.61; "
           "MCP ~7-11 vs 0.46), and gradient mass concentrates in the text channel "
           "(~4-6x keyword at epoch 1 on both corpora).",
           "- **Marginal-prior use / lexical shortcut (partially supported)**: on MCP, "
           "permuting keyword flags across graphs does not hurt dev AUROC (-0.017 for "
           "zeroing vs +0.008 for permutation), i.e. the model uses keyword marginals "
           "more than per-graph alignment; on Decomp permutation hurts by up to -0.016, "
           "so a small per-graph signal exists there.",
           "- **Redundancy (supported)**: several keyword dims are dead on both corpora "
           "(Decomp group_3 all-zero; MCP groups 2 and 7 all-zero; MCP group_7 rate "
           "0.009), so part of the channel carries no information at all.",
           "- **Optimization randomness (cannot be excluded)**: the 3-seed Decomp deltas "
           "are small (+0.002..+0.010) and per-seed test-graph bootstrap CIs include 0; "
           "the 0.005 gap must not be described as a settled mechanism defect.",
           "", "These four explanations motivate K2/K3 (train-fitted normalization + a "
           "separate small projection + channel dropout) and keep K1 (removal) as the "
           "simplest accepted option if it wins.", ""]
    (OUT / "keyword_instrumentation_report.md").write_text("\n".join(md) + "\n")
    print("wrote keyword_instrumentation_report.md from", list(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
