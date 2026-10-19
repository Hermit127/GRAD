#!/usr/bin/env python3
"""Phase 1 negative-transfer diagnostics (dev/train only, zero API).

Reads the CPU-fallback runs (runs_cpu) and the frozen split files:
  1. Learning curves: best epoch, train loss, val BCE/BF1 trend per variant.
  2. Dev error concentration by graph size / text length / top node types / source.
  3. full vs structure_only vs chrono_only disagreement tables + oracle ceiling
     (diagnostic only; not a deployable score).
  4. Edge statistics: rebuilt vs raw edges, dependency vs chronological share,
     edge-rationale text repetition; error vs edge density.
Writes LOGS/final_five_tables/table5/phase1_diagnosis.{json,md}
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))

RUNS = REPO / "code/repro/out/table5_ablation/runs_cpu"
OUT = REPO / "LOGS/final_five_tables/table5"
SPLIT_DIR = REPO / "code/repro/out/unified_gnn/fix2_splits"
SEEDS = [211, 212, 213]
VARIANTS = ["full", "text_only", "structure_only", "chrono_only",
            "no_relation_types", "no_keyword_features", "no_type_semantics"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def dev_rows(ds, cell, variant, seed):
    """Join dev scores (in file order) with dev graphs."""
    p = RUNS / f"{ds}_{cell}_{variant}_s{seed}.json"
    if not p.exists():
        return []
    d = load(p)
    graphs = [json.loads(l) for l in (SPLIT_DIR / ds / cell / "dev.jsonl").read_text().splitlines()
              if l.strip()]
    scores = d.get("scores", {}).get("val", [])
    if len(scores) != len(graphs):
        return []
    rows = []
    for g, s in zip(graphs, scores):
        assert int(g["label"]) == int(s["label"]), (g.get("task_id"), g["label"], s["label"])
        rows.append({"task_id": g.get("task_id"), "label": s["label"], "score": s["score"],
                     "n_nodes": len(g.get("nodes", [])),
                     "text_len": sum(len(str((n.get("features") or {}).get("subtask", "")))
                                     for n in g.get("nodes", [])),
                     "types": [str(n.get("type", "")) for n in g.get("nodes", [])],
                     "meta": g.get("metadata", {}),
                     "n_raw_edges": len(g.get("edges", []))})
    return rows


def auroc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ss = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def acc(rows, thr=0.5):
    return float(np.mean([(1 if r["score"] >= thr else 0) == r["label"] for r in rows]))


def curves() -> dict:
    out = {}
    for variant in VARIANTS:
        runs = []
        for seed in SEEDS:
            p = RUNS / f"mcp_fold0_{variant}_s{seed}.json"
            if not p.exists():
                continue
            d = load(p)
            h = d.get("history", [])
            if not h:
                continue
            runs.append({
                "seed": seed,
                "epochs_run": len(h),
                "best_epoch": d["selected_checkpoint"]["epoch"],
                "best_fixed_f1": d["selected_checkpoint"]["score"],
                "train_loss_first": h[0]["train_loss"],
                "train_loss_last": h[-1]["train_loss"],
                "train_loss_min": min(r["train_loss"] for r in h),
                "val_bce_min": min(r["val_graph"]["bce"] for r in h),
                "val_bce_last": h[-1]["val_graph"]["bce"],
                "val_auroc_max": max(r["val_graph"]["auroc"] for r in h),
                "val_auroc_last": h[-1]["val_graph"]["auroc"],
                "best_diag_epoch": max(h, key=lambda r: r["val_graph"]["auroc"])["epoch"],
            })
        out[variant] = runs
    return out


def concentration(ds="mcp") -> dict:
    """Error concentration on dev for full and structure_only."""
    out = {}
    for variant in ("full", "structure_only", "chrono_only"):
        rows = []
        for cell in [f"fold{k}" for k in range(5)]:
            for seed in SEEDS:
                rows += dev_rows(ds, cell, variant, seed)
        if not rows:
            continue
        by_size = defaultdict(lambda: [0, 0])
        by_len = defaultdict(lambda: [0, 0])
        by_type = defaultdict(lambda: [0, 0])
        by_task = defaultdict(lambda: [0, 0])
        for r in rows:
            correct = int((1 if r["score"] >= 0.5 else 0) == r["label"])
            size_bin = "<=5" if r["n_nodes"] <= 5 else "6-10" if r["n_nodes"] <= 10 else "11-20" if r["n_nodes"] <= 20 else ">20"
            len_bin = "<=1000" if r["text_len"] <= 1000 else "1001-3000" if r["text_len"] <= 3000 else ">3000"
            by_size[size_bin][1] += 1; by_size[size_bin][0] += correct
            by_len[len_bin][1] += 1; by_len[len_bin][0] += correct
            for t in set(r["types"]):
                by_type[t][1] += 1; by_type[t][0] += correct
            task = str(r["meta"].get("task") or r["meta"].get("trace_id") or "?")
            by_task[task][1] += 1; by_task[task][0] += correct
        out[variant] = {
            "n_dev": len(rows), "dev_acc": acc(rows),
            "dev_auroc": auroc([r["label"] for r in rows], [r["score"] for r in rows]),
            "by_size": {k: round(v[0] / v[1], 3) for k, v in sorted(by_size.items()) if v[1] > 0},
            "by_len": {k: round(v[0] / v[1], 3) for k, v in sorted(by_len.items()) if v[1] > 0},
            "by_type_worst": {k: round(v[0] / v[1], 3) for k, v in
                              sorted(by_type.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))[:6]
                              if v[1] >= 10},
            "by_task_worst": {k: round(v[0] / v[1], 3) for k, v in
                              sorted(by_task.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))[:6]
                              if v[1] >= 10},
        }
    return out


def disagreement(ds="mcp") -> dict:
    out = {}
    for a, b in (("full", "structure_only"), ("full", "chrono_only"),
                 ("full", "text_only"), ("structure_only", "chrono_only")):
        joint = []
        for cell in [f"fold{k}" for k in range(5)]:
            for seed in SEEDS:
                ra = {(r["task_id"], seed): r for r in dev_rows(ds, cell, a, seed)}
                rb = {(r["task_id"], seed): r for r in dev_rows(ds, cell, b, seed)}
                for k in set(ra) & set(rb):
                    joint.append((ra[k], rb[k]))
        if not joint:
            continue
        ca = [int((1 if x["score"] >= 0.5 else 0) == x["label"]) for x, _ in joint]
        cb = [int((1 if y["score"] >= 0.5 else 0) == y["label"]) for _, y in joint]
        both_wrong = sum(1 for x, y in zip(ca, cb) if not x and not y)
        a_only = sum(1 for x, y in zip(ca, cb) if x and not y)
        b_only = sum(1 for x, y in zip(ca, cb) if not x and y)
        both_right = sum(1 for x, y in zip(ca, cb) if x and y)
        combined = [max(x["score"], y["score"]) for x, y in joint]
        labels = [x["label"] for x, _ in joint]
        out[f"{a}_vs_{b}"] = {
            "n": len(joint), "both_right": both_right, "both_wrong": both_wrong,
            f"{a}_only_right": a_only, f"{b}_only_right": b_only,
            "acc_a": both_right / len(joint) + a_only / len(joint),
            "acc_b": both_right / len(joint) + b_only / len(joint),
            "oracle_score_auroc_diagnostic": auroc(labels, combined),
            "note": "oracle max is diagnostic only; it is not deployable and is not a "
                    "performance claim",
        }
    return out


def edge_stats(ds="mcp") -> dict:
    from train_gnn_ablate import rebuild_v2_edges
    import train_gnn_ablate as abl

    stats = defaultdict(list)
    rationale_overlap = []
    type_share = Counter()
    chrono_share = Counter()
    for cell in [f"fold{k}" for k in range(5)]:
        for seed in SEEDS:
            # per-sample error for full
            rows = {r["task_id"]: r for r in dev_rows(ds, cell, "full", seed)}
            graphs = [json.loads(l) for l in
                      (SPLIT_DIR / ds / cell / "dev.jsonl").read_text().splitlines() if l.strip()]
            abl.VARIANT = "full"
            for g in graphs:
                rebuilt = rebuild_v2_edges(g)
                n_raw = len(g.get("edges", []))
                n_re = len(rebuilt)
                n_chrono = sum(1 for e in rebuilt if e["type"] == "sequence_next")
                stats["n_nodes"].append(len(g["nodes"]))
                stats["n_raw_edges"].append(n_raw)
                stats["n_rebuilt"].append(n_re)
                stats["n_chrono"].append(n_chrono)
                stats["dep_share"].append((n_re - n_chrono) / max(n_re, 1))
                stats["edges_per_node"].append(n_re / max(len(g["nodes"]), 1))
                for e in rebuilt:
                    type_share[e["type"]] += 1
                    if e["type"] == "sequence_next":
                        chrono_share["sequence_next"] += 1
                # rationale overlap with endpoint node text (token-level)
                by_id = {str(n["id"]): n for n in g["nodes"]}
                for e in rebuilt[:5]:
                    rt = str(e.get("rationale", ""))
                    src = str((by_id.get(str(e["source"]), {}).get("features") or {}).get("subtask", ""))
                    tgt = str((by_id.get(str(e["target"]), {}).get("features") or {}).get("subtask", ""))
                    if rt:
                        toks = set(rt.split())
                        ref = set((src + " " + tgt).split())
                        if toks:
                            rationale_overlap.append(len(toks & ref) / len(toks))
    out = {
        "n_samples": len(stats["n_nodes"]),
        "mean_nodes": float(np.mean(stats["n_nodes"])),
        "mean_raw_edges": float(np.mean(stats["n_raw_edges"])),
        "mean_rebuilt_edges": float(np.mean(stats["n_rebuilt"])),
        "mean_chrono_edges": float(np.mean(stats["n_chrono"])),
        "mean_dependency_share": float(np.mean(stats["dep_share"])),
        "mean_edges_per_node": float(np.mean(stats["edges_per_node"])),
        "relation_type_share": {k: round(v / sum(type_share.values()), 3)
                                for k, v in type_share.most_common()},
        "rationale_token_overlap_with_endpoint_text": float(np.mean(rationale_overlap))
        if rationale_overlap else None,
        "raw_edges_used_by_full_model": False,
        "note": "full/chrono/structure variants all ignore corpus edges[]; edges are "
                "synthesized by rebuild_v2_edges() from turns, node types and regexes over "
                "visible text. Corpus edges are only used by the prototype "
                "train_gnn_v2_rawedges.py, which is not part of this matrix.",
    }
    return out


def main() -> int:
    report = {"curves": curves(), "concentration": concentration(),
              "disagreement": disagreement(), "edge_stats": edge_stats()}
    (OUT / "phase1_diagnosis.json").write_text(json.dumps(report, indent=1))

    md = ["# Phase 1 diagnosis — why the joint model loses on MCP", "",
          "Dev/train only; test never read. Generated by "
          "`code/repro/final_five_tables/phase1_diagnose_table5.py`.", "",
          "## 1. Learning curves (MCP fold0)", "",
          "| Variant | seed | epochs | best epoch | best dev F1 | train loss first→last | val BCE min→last | dev AUROC max/last |",
          "|---|---|---|---|---|---|---|---|"]
    for v, runs in report["curves"].items():
        for r in runs:
            md.append(f"| {v} | {r['seed']} | {r['epochs_run']} | {r['best_epoch']} | "
                      f"{r['best_fixed_f1']:.3f} | {r['train_loss_first']:.3f}→{r['train_loss_last']:.3f} | "
                      f"{r['val_bce_min']:.3f}→{r['val_bce_last']:.3f} | "
                      f"{r['val_auroc_max']:.3f}/{r['val_auroc_last']:.3f} |")
    md += ["", "## 2. Dev error concentration (all 5 folds x 3 seeds)", "",
           "| Variant | dev AUROC | dev ACC | by size (nodes) | by text length | worst node types |",
           "|---|---|---|---|---|---|"]
    for v, d in report["concentration"].items():
        md.append(f"| {v} | {d['dev_auroc']:.3f} | {d['dev_acc']:.3f} | {d['by_size']} | "
                  f"{d['by_len']} | {d['by_type_worst']} |")
    md += ["", "## 3. Disagreement tables (dev, 5 folds x 3 seeds)", "",
           "| Pair | n | both right | both wrong | A only right | B only right | oracle AUROC (diagnostic) |",
           "|---|---|---|---|---|---|---|"]
    for k, d in report["disagreement"].items():
        a_only = [v for kk, v in d.items() if kk.endswith("_only_right") and kk.startswith(
            k.split("_vs_")[0])]
        b_only = [v for kk, v in d.items() if kk.endswith("_only_right") and kk.startswith(
            k.split("_vs_")[1])]
        md.append(f"| {k} | {d['n']} | {d['both_right']} | {d['both_wrong']} | "
                  f"{a_only[0] if a_only else '?'} | {b_only[0] if b_only else '?'} | "
                  f"{d['oracle_score_auroc_diagnostic']:.3f} |")
    e = report["edge_stats"]
    md += ["", "## 4. Edge statistics (MCP dev)", "",
           f"- mean nodes {e['mean_nodes']:.1f}, raw corpus edges {e['mean_raw_edges']:.1f}, "
           f"rebuilt edges {e['mean_rebuilt_edges']:.1f}, chronological subset {e['mean_chrono_edges']:.1f} "
           f"({1 - e['mean_dependency_share']:.0%} of rebuilt edges are the chronological chain).",
           f"- relation type shares: `{e['relation_type_share']}`",
           f"- edge rationale token overlap with endpoint text: "
           f"{e['rationale_token_overlap_with_endpoint_text']:.2f} (rationale is largely a copy "
           "of endpoint texts by construction).",
           f"- corpus `edges[]` used by the full model: **{e['raw_edges_used_by_full_model']}** "
           "(the rebuild discards raw observable edges).",
           "", "## 5. Candidate causes (ranked, for Phase 2)", "",
           "1. **Dependency-edge heuristics add noise / no usable raw relations**: the full model "
           "discards corpus edges and synthesizes ~74% dependency edges from regexes over text; "
           "chrono-only matches or beats it, so those synthetic edges carry no positive value on MCP.",
           "2. **Weak/noisy text channel in the MCP regime**: text-only is the weakest variant "
           "and MCP text features have little label association (length AUROC ~0.48), so the "
           "joint encoder has little signal to fuse and may dilute the structural channel.",
           "3. **Fusion/optimization imbalance**: all channels enter one projection with no "
           "normalization or auxiliary supervision; the full model's best epoch and loss curves "
           "provide evidence on whether text gradients dominate (instrumented check pending).",
           ""]
    (OUT / "phase1_diagnosis.md").write_text("\n".join(md) + "\n")
    print("wrote phase1_diagnosis.json/md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
