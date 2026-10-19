#!/usr/bin/env python3
"""P0: GRAD-512 operating points + checkpoint inventory (zero training/API).

PI 2026-09-21 (GPT plan, P0): read the 33 `G3_512_*` confirm runs, verify per file
that a valid dev-selected threshold exists, apply it to test scores (rule from the
harness: pred = 1 if score >= threshold), compute per-run TP/FP/TN/FN, pool by
dataset, and report TPR/FPR with run paths/denominators/config hashes. No G3-192
thresholds, no test-time threshold fitting, no AUROC-derived operating points.

Also inventories server checkpoints (*.pt/*.pth) with hash/config/seed/selection basis.

Outputs:
  LOGS/final_five_tables/table5/grad512_operating_points.{json,md,tex}
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "code/repro/out/grad_repair/confirm"
OUT = REPO / "LOGS/final_five_tables/table5"
PROTOCOL_KEYS = ["candidate", "lambda_aux", "moddrop", "max_node_chars", "max_edge_chars",
                 "epochs", "patience", "batch_size", "lr", "target_fpr", "hidden_dim",
                 "edge_mode", "edge_dropout", "edge_text", "ablate", "dev_only"]
CKPT_DIRS = [REPO / "code/repro/out" / d for d in
             ("grad_repair", "table5_g3", "table5_confirm", "table5_ablation")]


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            if limit and f.tell() > limit:
                break
    return h.hexdigest()


def cfg_hash(cfg: dict, keys: list[str] | None = None) -> str:
    c = {k: cfg.get(k) for k in keys} if keys else cfg
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()


def dataset_of(fname: str) -> str:
    if "_mcp_" in fname:
        return "mcp"
    if "_decomp_" in fname:
        return "decomp"
    if "_mrt_" in fname:
        return "mrt"
    return "unknown"


def pool(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_pos": tp + fn, "n_neg": fp + tn, "n": tp + fp + tn + fn,
            "tpr": tp / max(1, tp + fn), "fpr": fp / max(1, fp + tn),
            "precision": tp / max(1, tp + fp),
            "acc": (tp + tn) / max(1, tp + fp + tn + fn)}


def audit_run(path: Path) -> dict:
    d = json.loads(path.read_text())
    cfg = d.get("config") or {}
    val_rows = d.get("scores", {}).get("val", [])
    test_rows = d.get("scores", {}).get("test", [])
    thr = d.get("val", {}).get("dev_selected_threshold")
    stored_test_thr = (d.get("test") or {}).get("dev_selected_threshold")
    val_scores = {float(r["score"]) for r in val_rows}
    test_scores = {float(r["score"]) for r in test_rows}
    thr = float(thr) if thr is not None else None

    def confuse(rows, t):
        if t is None or not rows:
            return 0, 0, 0, 0
        pred = [1 if float(r["score"]) >= t else 0 for r in rows]
        tp = sum(1 for r, p in zip(rows, pred) if int(r["label"]) == 1 and p == 1)
        fn = sum(1 for r, p in zip(rows, pred) if int(r["label"]) == 1 and p == 0)
        fp = sum(1 for r, p in zip(rows, pred) if int(r["label"]) == 0 and p == 1)
        tn = sum(1 for r, p in zip(rows, pred) if int(r["label"]) == 0 and p == 0)
        return tp, fp, tn, fn

    val_tp, val_fp, val_tn, val_fn = confuse(val_rows, thr)
    tp, fp, tn, fn = confuse(test_rows, thr)
    stored_val = (d.get("val") or {}).get("metrics_at_threshold") or {}
    reproduced = (abs(stored_val.get("recall", -1) - val_tp / max(1, val_tp + val_fn)) < 1e-9
                  and abs(stored_val.get("fpr", -1) - val_fp / max(1, val_fp + val_tn)) < 1e-9)
    fname = path.name
    return {
        "run": str(path.relative_to(REPO)), "dataset": dataset_of(fname),
        "seed": int(fname.split("_s")[-1].split(".")[0]),
        "config_hash": cfg_hash(cfg), "protocol_hash": cfg_hash(cfg, PROTOCOL_KEYS),
        "run_file_sha256": sha256_file(path),
        "val_n": len(val_rows), "test_n": len(test_rows),
        "dev_threshold": thr,
        "dev_threshold_is_observed_val_score": bool(thr in val_scores),
        "stored_test_threshold": stored_test_thr,
        "stored_test_threshold_equals_dev": stored_test_thr == thr,
        "harness_rule": "pred = 1 if score >= threshold",
        "val_metrics_reproduced": reproduced,
        "best_fixed_epoch": (d.get("best_fixed") or {}).get("epoch"),
        "selection_basis": "dev fixed-FPR F1 @FPR<=%.2f (harness best_fixed)" % cfg.get("target_fpr", 0.10),
        "val": {"tp": val_tp, "fp": val_fp, "tn": val_tn, "fn": val_fn},
        "test": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def checkpoint_inventory() -> list[dict]:
    inv = []
    for base in CKPT_DIRS:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".pt", ".pth"):
                inv.append({"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size,
                            "sha256": sha256_file(p)})
    return sorted(inv, key=lambda r: r["path"])


def write_tex(report: dict) -> None:
    L = ["% GRAD-512 operating points (dev-selected threshold; test confusion)",
         "\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "Dataset & Runs & TP & FP & TN & FN & TPR & FPR \\\\", "\\midrule"]
    for ds in ("decomp", "mcp", "mrt"):
        p = report["pooled_test"].get(ds)
        if not p:
            continue
        L.append(f"{ds} & {report['n_runs'][ds]} & {p['tp']} & {p['fp']} & {p['tn']} & {p['fn']} "
                 f"& {p['tpr']:.4f} & {p['fpr']:.4f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "grad512_operating_points.tex").write_text("\n".join(L) + "\n")


def main() -> None:
    files = sorted(RUNS.glob("G3_512_*.json"))
    audits = [audit_run(p) for p in files]
    per_ds = defaultdict(list)
    for a in audits:
        per_ds[a["dataset"]].append(a)
    report = {
        "note": "strict provenance audit: deviation from declared schema",
        "n_runs": {ds: len(v) for ds, v in per_ds.items()},
        "audits": audits,
        "pooled_test": {ds: pool([dict(a["test"], ) for a in v]) for ds, v in per_ds.items()},
        "per_seed_test": {},
        "pooled_val": {ds: pool([{**a["val"], } for a in v]) for ds, v in per_ds.items()},
    }
    for ds, v in per_ds.items():
        by_seed = defaultdict(list)
        for a in v:
            by_seed[a["seed"]].append(a)
        report["per_seed_test"][ds] = {s: pool([dict(a["test"]) for a in r])
                                       for s, r in sorted(by_seed.items())}
    report["checkpoints"] = checkpoint_inventory()
    report["checkpoint_note"] = ("server-side scan of code/repro/out {grad_repair, table5_g3, "
                                 "table5_confirm, table5_ablation}; local raw dirs are not synced")
    # audit flags that need attention
    issues = [a for a in audits if (not a["dev_threshold_is_observed_val_score"]
                                    or not a["harness_rule"] or a["test_n"] == 0)]
    report["audit_issues"] = [a["run"] for a in issues]
    (OUT / "grad512_operating_points.json").write_text(json.dumps(report, indent=2))
    write_md(report)
    write_tex(report)
    print("runs:", len(audits), "issues:", report["audit_issues"])


def write_md(report: dict) -> None:
    L = ["# GRAD-512 operating points (P0)", "",
         "Dev-selected threshold from each run's `val.dev_selected_threshold`, applied to the",
         "stored test scores with the harness rule `pred = 1 if score >= threshold`.",
         "No test-time threshold selection; no G3-192 thresholds. Confirmation batch label:",
         "frozen-scheme replication (legacy test, exposed).", ""]
    L.append("## Pooled test confusion (all cells x 3 seeds per dataset)")
    L.append("")
    L.append("| dataset | runs | TP | FP | TN | FN | TPR | FPR | precision | acc |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in ("decomp", "mcp", "mrt"):
        p = report["pooled_test"].get(ds)
        if not p:
            continue
        L.append(f"| {ds} | {report['n_runs'][ds]} | {p['tp']} | {p['fp']} | {p['tn']} | {p['fn']} "
                 f"| {p['tpr']:.4f} | {p['fpr']:.4f} | {p['precision']:.4f} | {p['acc']:.4f} |")
    L.append("")
    L.append("## Per-seed pooled test")
    L.append("")
    L.append("| dataset | seed | TP | FP | TN | FN | TPR | FPR |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, seeds in report["per_seed_test"].items():
        for s, p in seeds.items():
            L.append(f"| {ds} | {s} | {p['tp']} | {p['fp']} | {p['tn']} | {p['fn']} "
                     f"| {p['tpr']:.4f} | {p['fpr']:.4f} |")
    L.append("")
    L.append("## Pooled validation (dev) at the same thresholds")
    L.append("")
    L.append("| dataset | TP | FP | TN | FN | TPR | FPR |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, p in report["pooled_val"].items():
        L.append(f"| {ds} | {p['tp']} | {p['fp']} | {p['tn']} | {p['fn']} | {p['tpr']:.4f} | {p['fpr']:.4f} |")
    L.append("")
    L.append(f"## Audit issues: {report['audit_issues'] or 'none'}")
    L.append("")
    L.append("Per-run provenance (path / config hash / run sha256 / threshold flags) in "
             "`grad512_operating_points.json`; checkpoint inventory below.")
    L.append("")
    L.append("## Server checkpoint inventory")
    L.append("")
    if report["checkpoints"]:
        L.append("| path | MB | sha256 |")
        L.append("|---|---|---|")
        for c in report["checkpoints"]:
            L.append(f"| {c['path']} | {c['bytes'] / 1e6:.1f} | {c['sha256'][:16]}… |")
    else:
        L.append("No `*.pt`/`*.pth` found under the scanned directories.")
    L.append("")
    (OUT / "grad512_operating_points.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
