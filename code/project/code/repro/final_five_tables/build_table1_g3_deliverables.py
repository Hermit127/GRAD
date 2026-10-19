#!/usr/bin/env python3
"""Build Table 1 (G3 alignment) deliverables from table1_g3.json (no training)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "LOGS/final_five_tables/table1_g3"
T1 = REPO / "LOGS/final_five_tables/raw_results/table1.json"
G3JSON = OUT / "table1_g3.json"
SPLIT = REPO / "code/repro/out/unified_gnn/fix2_splits"


def sha256(p: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with h.open if False else open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:n]


def fmt(x, d=4):
    return "--" if x is None else f"{x:.{d}f}"


def main() -> int:
    data = json.loads(G3JSON.read_text())
    old_t1 = json.loads(T1.read_text())
    import subprocess
    git_head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    code_hashes = {f: sha256(REPO / f) for f in (
        "code/train_gnn_v2.py", "code/repro/final_five_tables/train_gnn_ablate.py",
        "code/repro/final_five_tables/train_gnn_ablate2.py")}
    data.setdefault("model_versions", {})["git_head"] = git_head
    data["model_versions"]["code_sha256_16"] = code_hashes
    data["model_versions"]["batch_trainer"] = {
        "old_archived": "code/train_gnn_v2.py (pre-out-dir plumbing; same behavior)",
        "paired_K0": "code/repro/final_five_tables/train_gnn_ablate.py --variant full",
        "g3": "code/repro/final_five_tables/train_gnn_ablate2.py --candidate A_moddrop --moddrop 0.25",
    }
    G3JSON.write_text(json.dumps(data, indent=1))
    cells = data["cells"]
    old, paired, g3 = cells["old_archived"], cells["paired_K0"], cells["g3"]

    # split hashes for the alignment record
    splits = {}
    for name, path in {
        "decomp_fold0_train": SPLIT / "decomp/fold0/train.jsonl",
        "decomp_fold0_dev": SPLIT / "decomp/fold0/dev.jsonl",
        "decomp_fold0_test": SPLIT / "decomp/fold0/test.jsonl",
        "mcp_fold0_train": SPLIT / "mcp/fold0/train.jsonl",
        "mcp_fold0_test": SPLIT / "mcp/fold0/test.jsonl",
    }.items():
        splits[name] = {"sha256_16": sha256(path),
                        "mtime": path.stat().st_mtime}

    # ---------------- alignment md ----------------
    md = ["# Table 1 with G3_p25 — version and protocol alignment", "",
          "Read-only re-aggregation of existing per-sample results; no training, no API.",
          "Old main table (`raw_results/table1.{json,md}`) is preserved unchanged.", "",
          "## 1. Run batches", "",
          "| batch | directory | role | n runs used |", "|---|---|---|---|"]
    for name, key, note in (
        ("old_archived", "old_archived", "original 57-run sweep; source of the old Table 1 GRAD row"),
        ("paired_K0", "paired_K0", "same-protocol re-run; pairs with G3 (K0 full)"),
        ("g3", "g3", "frozen G3_p25 confirmation runs"),
    ):
        n = len([u for col in cells[key] if cells[key][col]
                 for u in cells[key][col]["units"]])
        md.append(f"| {name} | `{data['batches'][key]['path']}` | {note} | {n} |")
    md += ["", "## 2. Protocol identity evidence", "",
           "### 2.1 Splits (same files for all three batches)", "",
           "| split file | sha256 (16) | mtime |", "|---|---|---|"]
    for k, v in splits.items():
        md.append(f"| {k} | `{v['sha256_16']}` | {v['mtime']:.0f} (2026-09-15) |")
    md += ["", "Split files were written 2026-09-15 15:27-15:28, before both training "
               "sweeps; all run configs point to these exact paths.",
           "",
           "### 2.2 Configs", "",
           "Diffing a representative old run (`unified_gnn/runs/mcp_f0_s211.json`) with a "
           "paired run (`table5_ablation/runs/mcp_fold0_full_s211.json`) leaves only "
           "`out_dir`, `save_checkpoint`, `variant` (fields added by the paired harness); "
           "all shared fields are identical: epochs 100, patience 25, batch 48, "
           "hidden 192, lr 1e-3, seed set, checkpoint metric `fixed_fpr_f1` @FPR<=0.10.",
           "",
           "### 2.3 Code", "",
           "`git diff 87240497 2da1b5af -- code/train_gnn_v2.py` shows a single change: "
           "the `--out-dir` output plumbing (6 insertions, 1 deletion). No model, feature, "
           "training-loop, evaluation or threshold code changed between the two batches. "
           "The G3 harness `train_gnn_ablate2.py` shares the input builder with the frozen "
           "trainer (tensor-identity unit test) and the same protocol constants.",
           "",
           "### 2.4 Preprocessing and seeds", "",
           "- Node text 192 chars / edge rationale 160 chars / char vocab 128 in all trainers.",
           "- Seeds 211/212/213; seed 42 pilot excluded (archived convention).",
           "- Aggregation: fold-weighted by n_pos*n_neg within seed, mean and sample SD "
           "(n-1) across seeds; operating point = dev `threshold_reports.fpr_10` threshold "
           "(or ablate2 `val.dev_selected_threshold`), applied to test, counts pooled.",
           "",
           "## 3. Why G3's K0 differs from the old Table 1", "",
           "The old Table 1 GRAD row (Decomp 0.9516, MCP 0.6210, MRT 0.7400) comes from "
           "the original sweep; the G3 report's K0 (0.9575 / 0.6410 / 0.7400) comes from "
           "the paired re-run batch. The protocol is identical (S2.1-S2.4); the gap is "
           "run-to-run variance between two batches of the same experiment.", "",
           "Per-seed AUROC (seed : old | paired | G3):", ""]
    for col in ("Decomp", "MCP", "MRT"):
        o = (old.get(col) or {}).get("per_seed", {})
        p = (paired.get(col) or {}).get("per_seed", {})
        g = (g3.get(col) or {}).get("per_seed", {})
        if not o and not g:
            continue
        md.append(f"- **{col}**: " + "; ".join(
            f"s{s}: {fmt(o.get(s))} | {fmt(p.get(s))} | {fmt(g.get(s))}"
            for s in ("211", "212", "213")))
    md += ["", "Aggregate means (old | paired | G3): Decomp "
               f"{fmt((old.get('Decomp') or {}).get('auroc_mean'))} | "
               f"{fmt((paired.get('Decomp') or {}).get('auroc_mean'))} | "
               f"{fmt((g3.get('Decomp') or {}).get('auroc_mean'))}; MCP "
               f"{fmt((old.get('MCP') or {}).get('auroc_mean'))} | "
               f"{fmt((paired.get('MCP') or {}).get('auroc_mean'))} | "
               f"{fmt((g3.get('MCP') or {}).get('auroc_mean'))}; MRT "
               f"{fmt((old.get('MRT') or {}).get('auroc_mean'))} | "
               f"{fmt((paired.get('MRT') or {}).get('auroc_mean'))} | "
               f"{fmt((g3.get('MRT') or {}).get('auroc_mean'))}.",
           "",
           "**MRT mapping**: the paired batch did not retrain the full model on MRT; the "
           "archived runs are reused by construction, so old and paired MRT are the same "
           "15 run files (0.7400). G3's MRT row is new (0.8730).",
           "",
           "**AUROC source caveat**: all cells here are recomputed from the per-sample "
           "scores with a rank AUROC (ties averaged). The trainers' stored "
           "`test_graph.auroc` differs on 7/15 old MCP runs by up to 0.0027 (two forward "
           "passes, microscopic numeric differences; threshold-0.5 confusion metrics are "
           "identical). Recomputation reproduces the archived aggregates to 0.0007 on MCP "
           "and exactly on Decomp/MRT; the archived values are kept in the JSON for "
           "reference (`auroc_mean_trainer_reported`).",
           "",
           "## 4. G3 coverage", "",
           "G3_p25 has runs for **Decomp (fix2 fold0, test 232)**, **MCP (fix2 cv5, "
           "5 folds)** and **MRT (cross-task, 5 folds)**, seeds 211-213. It has no runs "
           "for MG / MINJA / InjecAgent / ASB / AgentHarm; those cells are gaps and are "
           "not filled from the old GRAD batches (see `missing_runs.md`).",
           "",
           "## 5. Baselines", "",
           "Non-GRAD baseline cells are copied unchanged from `raw_results/table1.json` "
           "with their original provenance/status (including pending/blocked/N-A); their "
           "protocol-match status is recorded per cell in `table1_g3.json`.",
           ""]
    (OUT / "table1_g3_alignment.md").write_text("\n".join(md))

    # ---------------- tex ----------------
    def disp_of(row, name):
        return row.get("display", name)

    def auroc_cell(c, with_n=True):
        if not c:
            return "--"
        s = f"{c['auroc_mean']:.4f}$\\pm${c['auroc_sd_sample']:.4f}"
        return s

    def op_cell(c):
        if not c or not c.get("confusion_pooled"):
            return "--"
        p = c["confusion_pooled"]
        return (f"{p['tpr']:.3f}/{p['fpr']:.3f} "
                f"({p['tp']}/{p['fp']}/{p['tn']}/{p['fn']}, "
                f"n$_{{pos}}$={p['tp']+p['fn']}, n$_{{neg}}$={p['fp']+p['tn']})")

    baselines = old_t1["rows"]
    order = ["Sequential Monitor", "Agent-Sentry", "AgentTrust+RiskChain",
             "NeuroTaint", "AgentArmor", "AEGIS", "AuthGraph"]
    lines = [
        "% Updated detection table with G3_p25. AUTO-GENERATED; do not edit.",
        "% Old table (raw_results/table1.json) is preserved unchanged.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Cross-dataset attack detection. Baseline rows are reused unchanged "
        "from the archived Table 1 aggregation (own protocols as disclosed there). "
        "GRAD rows: original archived batch, the same-protocol paired batch, and the "
        "frozen G3_p25 candidate; all GRAD cells are recomputed from per-sample scores "
        "(fold-weighted $n_+n_-$ within seed, mean/sample SD over seeds 211--213). "
        "G3 has no runs on the extension corpora (gaps, not zeros).}",
        "\\label{tab:table1_g3}",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        " & \\multicolumn{3}{c}{AUROC $\\uparrow$ (trajectory level)} \\\\",
        "\\cmidrule(lr){2-4}",
        "\\textit{Method} & Decomp & MCP & MRT \\\\",
        "\\midrule",
    ]
    for name in order:
        row = baselines[name]
        disp = row.get("display", name)
        cs = []
        for col in ("Decomp", "MCP", "MRT"):
            cell = (row.get("cells") or {}).get(col) or {}
            a = cell.get("auroc") or {}
            if isinstance(a, dict) and a.get("mean") is not None:
                sd = a.get("seed_sd")
                cs.append(f"{a['mean']:.4f}" + (f"$\\pm${sd:.4f}" if sd else ""))
            else:
                st = cell.get("status", "--")
                cs.append("\\textit{" + st + "}" if st not in ("filled",) else "--")
        lines.append(f"{disp} & {' & '.join(cs)} \\\\")
    lines += ["\\midrule"]
    for label, key in (("GRAD (original, archived batch)", "old_archived"),
                       ("GRAD (original, paired batch)", "paired_K0"),
                       ("\\textbf{GRAD (G3\\_p25, frozen)}", "g3")):
        cs = []
        for c in ("Decomp", "MCP", "MRT"):
            v = auroc_cell(cells[key].get(c) or None)
            if key == "paired_K0" and c == "MRT":
                arch_ = auroc_cell(cells["old_archived"].get("MRT"))
                v = f"{arch_}$^{{\\ddagger}}$"
            cs.append(v)
        lines.append(f"{label} & {' & '.join(cs)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "",
              "\\vspace{2pt}",
              "\\begin{tabular}{@{}lccc@{}}",
              "\\toprule",
              " & \\multicolumn{3}{c}{TPR / FPR at the reported operating point} \\\\",
              "\\cmidrule(lr){2-4}",
              "\\textit{Method} & Decomp & MCP & MRT \\\\",
              "\\midrule"]
    for name in order:
        row = baselines[name]
        cs = []
        for col in ("Decomp", "MCP", "MRT"):
            cell = (row.get("cells") or {}).get(col) or {}
            if cell.get("tpr") is not None and cell.get("fpr") is not None:
                cs.append(f"{cell['tpr']:.3f}/{cell['fpr']:.3f}")
            else:
                st = cell.get("status", "--")
                cs.append("\\textit{" + str(st) + "}" if st != "filled" else "n/a")
        lines.append(f"{disp_of(row, name)} & {' & '.join(cs)} \\\\")
    lines += ["\\midrule"]
    for label, key in (("GRAD original (archived)", "old_archived"),
                       ("GRAD original (paired)", "paired_K0"),
                       ("\\textbf{GRAD G3\\_p25}", "g3")):
        cs = []
        for c in ("Decomp", "MCP", "MRT"):
            c_ = cells[key].get(c)
            if key == "paired_K0" and c == "MRT":
                cs.append("0.793/0.653$^{\\ddagger}$")
            elif c_ and c_.get("confusion_pooled"):
                p_ = c_["confusion_pooled"]
                cs.append(f"{p_['tpr']:.3f}/{p_['fpr']:.3f}")
            else:
                cs.append("--")
        lines.append(f"{label} & {' & '.join(cs)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "",
              "\\vspace{2pt}",
              "\\begin{tabular}{@{}lll@{}}",
              "\\toprule",
              "\\multicolumn{3}{@{}l}{\\textbf{GRAD operating points at the dev-selected "
              "threshold (TPR/FPR, pooled TP/FP/TN/FN)}} \\\\",
              "\\midrule",
              "\\textit{Run batch} & Decomp & MCP / MRT \\\\",
              "\\midrule",
              f"GRAD original (archived) & {op_cell(old.get('Decomp'))} & "
              f"{op_cell(old.get('MCP'))} / \\textit{{n/a (reused)}} \\\\",
              f"GRAD original (paired) & {op_cell(paired.get('Decomp'))} & "
              f"{op_cell(paired.get('MCP'))} / \\textit{{reused}} \\\\",
              f"\\textbf{{GRAD G3\\_p25}} & {op_cell(g3.get('Decomp'))} & "
              f"{op_cell(g3.get('MCP'))} / {op_cell(g3.get('MRT'))} \\\\",
              "\\bottomrule", "\\end{tabular}", "",
              "\\vspace{2pt}",
              "{\\footnotesize $^{\\ddagger}$Paired batch reuses the archived MRT runs "
              "(full model not retrained); operating point likewise reused. All GRAD "
              "cells recomputed from per-sample scores; dev-selected thresholds only, "
              "counts pooled over runs (3 seeds x folds). MRT FPR transfer is poor for "
              "all batches (calibration 32 graphs vs admission 20/fold) and is disclosed "
              "in Table 1 notes.}",
              "\\end{table*}", ""]
    (OUT / "table1_g3.tex").write_text("\n".join(lines))

    # ---------------- missing runs ----------------
    mm = ["# Missing runs for a G3 version of Table 1 (read-only audit)", "",
          "Only genuinely missing work is listed; nothing is auto-run.", "",
          "## A. Re-aggregation only (raw results already exist)", "",
          "- Decomp / MCP / MRT G3_p25 cells: already produced here "
          "(`table1_g3.json`); no missing work.",
          "- Old-batch MG (5 folds x 3 seeds) and MINJA (3 folds x 3 seeds) GRAD cells: "
          "already aggregated in the archived Table 1 (`table1.json`); they are NOT G3 "
          "and cannot be relabelled. If a G3-era MG/MINJA row is wanted, training is "
          "required (section C).",
          "",
          "## B. Inference from existing checkpoints", "",
          "- None available. The G3 confirmation runs did not save checkpoints "
          "(`run_g3_gpu.sh` does not pass `--save-checkpoint`; no `.pt` exists under "
          "`code/repro/out/table5_g3/`). The archived sweep also did not persist "
          "checkpoints. Any new-corpus inference requires retraining.",
          "",
          "## C. Needs training (no paid API required; GPU only)", "",
          "| corpus | G3 runs needed | seeds | est. GPU time |", "|---|---|---|---|"]
    mm += ["| MG (fix2 cv5) | 5 folds x 3 seeds = 15 | 211-213 | ~45-60 min on idle GPUs |",
           "| MINJA (fix2 cv3) | 3 x 3 = 9 | 211-213 | ~30-40 min |",
           "| InjecAgent (x4) | 1 fold x 3 = 3 | 211-213 | ~2.5 h (2728 train graphs) |",
           "| ASB (x4) | 3 | 211-213 | ~10 min |",
           "| AgentHarm (x4) | 3 | 211-213 | ~30 min |",
           "",
           "## D. Baseline cells that remain blocked/pending (unchanged from Table 1)",
           "",
           "- AuthGraph: implementation verified offline; evaluation needs a paid API "
           "budget (not requested in this task).",
           "- Agent-Sentry full L1+L2+L3: blocked on native L1 training positives "
           "(unpublished benchmark data).",
           "- AgentArmor full method: blocked on the unpublished registry/dependency "
           "definitions (mechanism comparison only).",
           "- Sequential Monitor on non-Decomp corpora: no native subtask sequence "
           "interface (information-matched adaptation lives in Table 5, not here).",
           ""]
    (OUT / "missing_runs.md").write_text("\n".join(mm))
    print("wrote table1_g3_alignment.md, table1_g3.tex, missing_runs.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
