#!/usr/bin/env python3
"""Phase 0 audit for Table 5 Panel A negative transfer (zero API).

A. Implementation identity: configs across variants (only variant/seed/paths differ),
   param-count accounting, input-contract spot checks on real graphs.
B. Aggregation: recompute per-seed fold-weighted cells and paired deltas from raw
   unrounded scores; explain the delta discrepancy vs pair-mean.
C. Data shortcuts: label association of size/length/type/keyword features (fit on
   train, evaluate on dev), source/template overlap across splits, train-only fits.

Writes LOGS/final_five_tables/table5/phase0_audit.{json,md}.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code/repro"))
sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))

RUNS_CPU = REPO / "code/repro/out/table5_ablation/runs_cpu"
OUT = REPO / "LOGS/final_five_tables/table5"
CELLS = {
    "decomp": ["fold0"],
    "mcp": [f"fold{k}" for k in range(5)],
}
SEEDS = [211, 212, 213]
VARIANTS = ["full", "text_only", "structure_only", "chrono_only",
            "no_relation_types", "no_keyword_features", "no_type_semantics"]
SPLIT_DIR = REPO / "code/repro/out/unified_gnn/fix2_splits"


def auroc(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(scores)) + 1
    # average ranks for ties
    order = np.argsort(scores)
    sorted_scores = scores[order]
    i = 0
    ranks_avg = np.empty(len(scores))
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = (i + j) / 2 + 1
        ranks_avg[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks_avg[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def load_run(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------- A. implementation ----------------

def audit_implementation() -> dict:
    out: dict = {"config_fields_differing": [], "per_variant": {}, "errors": []}
    ref_cfg = None
    for variant in VARIANTS:
        rows = []
        for ds, cells in CELLS.items():
            for cell in cells:
                for seed in SEEDS:
                    p = RUNS_CPU / f"{ds}_{cell}_{variant}_s{seed}.json"
                    if not p.exists():
                        continue
                    d = load_run(p)
                    rows.append({"file": p.name, "ds": ds, "cell": cell, "seed": seed,
                                 "config": d.get("config", {}),
                                 "ablation": d.get("ablation", {})})
        if not rows:
            out["errors"].append(f"{variant}: no runs")
            continue
        params = {r["ablation"].get("model_params") for r in rows}
        classes = {r["ablation"].get("model_class") for r in rows}
        audits = defaultdict(list)
        for r in rows:
            splits = r["ablation"].get("splits", {})
            audits["train_edges_used"].append(splits.get("train", {}).get("n_edges_used"))
            audits["train_kw_used"].append(splits.get("train", {}).get("keyword_features_used"))
            audits["train_text_enc"].append(splits.get("train", {}).get("node_text_encoded"))
            audits["train_edge_text"].append(splits.get("train", {}).get("edge_text_encoded"))
        out["per_variant"][variant] = {
            "n_runs": len(rows),
            "model_params": sorted(params),
            "model_class": sorted(classes),
            "train_edges_used_range": [min(audits["train_edges_used"]),
                                       max(audits["train_edges_used"])],
            "keyword_features_used": sorted(set(audits["train_kw_used"])),
            "node_text_encoded": sorted(set(audits["train_text_enc"])),
            "edge_text_encoded": sorted(set(audits["train_edge_text"])),
        }
        cfg_keys = set(rows[0]["config"])
        for r in rows:
            for k in cfg_keys:
                a, b = rows[0]["config"].get(k), r["config"].get(k)
                if a != b and k not in ("variant", "seed", "train", "val", "test",
                                        "out_dir", "save_checkpoint"):
                    out["config_fields_differing"].append(
                        {"variant": variant, "file": r["file"], "field": k, "a": a, "b": b})
    out["config_fields_differing"] = out["config_fields_differing"][:20]
    # expected param deltas vs full
    full = json.loads((RUNS_CPU / "mcp_fold0_full_s211.json").read_text())
    fb = full["ablation"]["node_base_dim"]
    fe = full["ablation"]["edge_base_dim"]
    out["param_check"] = {"full_node_base": fb, "full_edge_base": fe,
                          "expected": {}, "observed": {}}
    for variant, exp_delta in (("text_only", 192 * (fb - 1)),
                               ("no_keyword_features", 192 * (fb - 13)),
                               ("structure_only", None)):
        p = RUNS_CPU / f"mcp_fold0_{variant}_s211.json"
        if p.exists():
            obs = json.loads(p.read_text())["ablation"]["model_params"]
            out["param_check"]["observed"][variant] = obs
    out["param_check"]["full_params"] = full["ablation"]["model_params"]
    return out


# ---------------- B. aggregation ----------------

def audit_aggregation() -> dict:
    per_cell = defaultdict(dict)
    for variant in VARIANTS:
        for ds, cells in CELLS.items():
            for cell in cells:
                for seed in SEEDS:
                    p = RUNS_CPU / f"{ds}_{cell}_{variant}_s{seed}.json"
                    if p.exists():
                        d = json.loads(p.read_text())
                        per_cell[(ds, variant)][(cell, seed)] = {
                            "auroc": d["test_graph"]["auroc"],
                            "n_pos": d["test_graph"].get("n_pos"),
                            "n_neg": d["test_graph"].get("n_neg"),
                            "n": len(d.get("scores", {}).get("test", [])),
                        }
    # use per-cell pos/neg counts from the test labels where available
    def cell_counts(ds, cell, seed):
        p = RUNS_CPU / f"{ds}_{cell}_full_s{seed}.json"
        d = json.loads(p.read_text())
        test = d["scores"]["test"]
        npos = sum(r["label"] for r in test)
        return npos, len(test) - npos

    out = {"cells": {}, "paired": {}, "note":
           "seed-level delta = fold-weighted AUROC difference per seed, then mean over "
           "the 3 seeds (matches the displayed cell means); pair-mean is the unweighted "
           "mean over all (fold, seed) pairs and is reported separately."}
    for ds, cells in CELLS.items():
        out["cells"].setdefault(ds, {})
        for variant in VARIANTS:
            data = per_cell.get((ds, variant), {})
            if not data:
                continue
            seed_vals = []
            for seed in SEEDS:
                items = [(c, data[(c, seed)]) for c in cells if (c, seed) in data]
                if not items:
                    continue
                ws = []
                for c, m in items:
                    npos, nneg = cell_counts(ds, c, seed)
                    ws.append(npos * nneg)
                wsum = sum(ws)
                seed_vals.append(sum(m["auroc"] * w for (c, m), w in zip(items, ws)) / wsum)
            out["cells"][ds][variant] = {
                "seed_level_mean": float(np.mean(seed_vals)),
                "seed_level_std": float(np.std(seed_vals)),
                "per_seed": {str(s): round(v, 6) for s, v in zip(SEEDS, seed_vals)},
            }
        full_map = per_cell.get((ds, "full"), {})
        if not full_map:
            continue
        out["paired"][ds] = {}
        for variant in VARIANTS:
            data = per_cell.get((ds, variant), {})
            if not data or variant == "full":
                continue
            pair_deltas, seed_deltas = [], {}
            for seed in SEEDS:
                items = [(c, data[(c, seed)], full_map[(c, seed)]) for c in cells
                         if (c, seed) in data and (c, seed) in full_map]
                if not items:
                    continue
                ws = []
                for c, _, _ in items:
                    npos, nneg = cell_counts(ds, c, seed)
                    ws.append(npos * nneg)
                wsum = sum(ws)
                sd = sum((m["auroc"] - f["auroc"]) * w for (c, m, f), w in zip(items, ws)) / wsum
                seed_deltas[seed] = sd
                pair_deltas += [m["auroc"] - f["auroc"] for (c, m, f) in items]
            out["paired"][ds][variant] = {
                "seed_level_delta_mean": float(np.mean(list(seed_deltas.values()))),
                "seed_level_delta_std": float(np.std(list(seed_deltas.values()))),
                "per_seed_delta": {str(k): round(v, 6) for k, v in seed_deltas.items()},
                "pair_mean_delta": float(np.mean(pair_deltas)),
                "pair_std": float(np.std(pair_deltas)),
                "n_pairs": len(pair_deltas),
                "all_seeds_same_sign": (all(v > 0 for v in seed_deltas.values())
                                        or all(v < 0 for v in seed_deltas.values())),
            }
    return out


# ---------------- C. data ----------------

def graph_features(g: dict) -> dict:
    nodes = g.get("nodes", [])
    text = " ".join(str((n.get("features") or {}).get("subtask", "")) for n in nodes)
    types = Counter(str(n.get("type", "")) for n in nodes)
    return {
        "n_nodes": len(nodes),
        "text_len": len(text),
        "n_edges_raw": len(g.get("edges", [])),
        "types": types,
        "label": int(g.get("label", 0)),
        "task_id": g.get("task_id"),
        "metadata": g.get("metadata", {}),
    }


def audit_data() -> dict:
    from train_gnn import content_features
    out: dict = {"per_dataset": {}, "source_overlap": {}}
    for ds, cells in CELLS.items():
        feats = defaultdict(lambda: {"train": [], "dev": []})
        type_feats = defaultdict(lambda: {"train": [], "dev": []})
        for cell in cells:
            for split in ("train", "dev"):
                p = SPLIT_DIR / ds / cell / f"{split}.jsonl"
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    g = json.loads(line)
                    f = graph_features(g)
                    kw = sum(content_features(" ".join(
                        str((n.get("features") or {}).get("subtask", "")) for n in g["nodes"])))
                    feats["n_nodes"][split].append((f["label"], f["n_nodes"]))
                    feats["text_len"][split].append((f["label"], f["text_len"]))
                    feats["n_edges_raw"][split].append((f["label"], f["n_edges_raw"]))
                    feats["keyword_flags"][split].append((f["label"], kw))
                    for t, c in f["types"].items():
                        type_feats[t][split].append((f["label"], c))
        out["per_dataset"][ds] = {}
        for name, data in feats.items():
            tr, dv = data["train"], data["dev"]
            out["per_dataset"][ds][name] = {
                "dev_auroc": auroc([y for y, _ in dv], [v for _, v in dv]),
                "train_auroc": auroc([y for y, _ in tr], [v for _, v in tr]),
                "dev_label1_mean": float(np.mean([v for y, v in dv if y == 1])),
                "dev_label0_mean": float(np.mean([v for y, v in dv if y == 0])),
            }
        per_type = {}
        for t, data in type_feats.items():
            tr, dv = data["train"], data["dev"]
            if not dv:
                continue
            per_type[t] = {
                "dev_auroc": auroc([y for y, _ in dv], [v for _, v in dv]),
                "train_auroc": auroc([y for y, _ in tr], [v for _, v in tr]),
                "dev_pos_rate": float(np.mean([v for y, v in dv if y == 1])),
                "dev_neg_rate": float(np.mean([v for y, v in dv if y == 0])),
            }
        out["per_dataset"][ds]["node_type_features"] = dict(sorted(
            per_type.items(),
            key=lambda kv: abs((kv[1]["dev_auroc"] or 0.5) - 0.5), reverse=True)[:8])
    
    # source/template overlap across splits (metadata fields)
    for ds, cells in CELLS.items():
        overlaps = {}
        for cell in cells:
            meta_keys = set()
            per_split = {}
            for split in ("train", "dev", "test"):
                p = SPLIT_DIR / ds / cell / f"{split}.jsonl"
                metas = []
                for line in p.read_text().splitlines():
                    if line.strip():
                        metas.append(json.loads(line).get("metadata", {}))
                per_split[split] = metas
                for m in metas:
                    meta_keys |= set(m)
            per_key = {}
            for key in sorted(meta_keys):
                sets = {s: {str(m.get(key)) for m in per_split[s] if m.get(key) is not None}
                        for s in per_split}
                per_key[key] = {
                    "n_unique": {s: len(v) for s, v in sets.items()},
                    "train_test_overlap": len(sets["train"] & sets["test"]),
                    "train_dev_overlap": len(sets["train"] & sets["dev"]),
                }
            overlaps[cell] = per_key
        out["source_overlap"][ds] = overlaps
    return out


def main() -> int:
    report = {
        "implementation": audit_implementation(),
        "aggregation": audit_aggregation(),
        "data": audit_data(),
        "notes": [
            "structure_only vocab is built from the TRAIN split only (build_node_type_vocab).",
            "no normalization statistics are fitted anywhere; CharCNN is a fixed ord() tokenizer.",
            "MCP source strings are isolated per-graph row; only metadata overlap is reported.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase0_audit.json").write_text(json.dumps(report, indent=1))

    md = ["# Phase 0 audit — Table 5 Panel A negative transfer", "",
          "Auto-generated by `code/repro/final_five_tables/phase0_audit_table5.py`", ""]
    md += ["## A. Implementation identity", "",
           "| Variant | runs | params | class | train edges | keywords used | node text | edge text |",
           "|---|---|---|---|---|---|---|---|"]
    for v, d in report["implementation"]["per_variant"].items():
        md.append(f"| {v} | {d['n_runs']} | {d['model_params']} | {d['model_class']} | "
                  f"{d['train_edges_used_range']} | {d['keyword_features_used']} | "
                  f"{d['node_text_encoded']} | {d['edge_text_encoded']} |")
    md += ["", f"Config fields differing across runs: "
               f"{report['implementation']['config_fields_differing'] or 'none (besides variant/seed/paths)'}",
           "", "## B. Aggregation (raw, unrounded)", ""]
    for ds, variants in report["aggregation"]["paired"].items():
        md += [f"### {ds}", "",
               "| Variant | seed-level delta (mean±std) | per-seed | pair-mean | n_pairs | same sign |",
               "|---|---|---|---|---|---|"]
        for v, d in variants.items():
            md.append(f"| {v} | {d['seed_level_delta_mean']:+.4f}±{d['seed_level_delta_std']:.4f} | "
                      f"{d['per_seed_delta']} | {d['pair_mean_delta']:+.4f} | {d['n_pairs']} | "
                      f"{d['all_seeds_same_sign']} |")
        md.append("")
    md += ["## C. Data / shortcut associations (dev AUROC of trivial features)", "",
           "| Dataset | feature | train AUROC | dev AUROC | label1 mean | label0 mean |",
           "|---|---|---|---|---|---|"]
    for ds, feats in report["data"]["per_dataset"].items():
        for name, d in feats.items():
            if name == "node_type_features":
                continue
            md.append(f"| {ds} | {name} | {d['train_auroc']:.3f} | {d['dev_auroc']:.3f} | "
                      f"{d['dev_label1_mean']:.2f} | {d['dev_label0_mean']:.2f} |")
    md += ["", "Top node-type count features (dev AUROC, |AUC-0.5| ranked):", "",
           "| Dataset | node type | train AUROC | dev AUROC | pos rate | neg rate |",
           "|---|---|---|---|---|---|"]
    for ds, feats in report["data"]["per_dataset"].items():
        for t, d in (feats.get("node_type_features") or {}).items():
            md.append(f"| {ds} | {t} | {d['train_auroc']:.3f} | {d['dev_auroc']:.3f} | "
                      f"{d['dev_pos_rate']:.3f} | {d['dev_neg_rate']:.3f} |")
    md += ["", "## D. Metadata overlap across splits", ""]
    for ds, cells in report["data"]["source_overlap"].items():
        for cell, keys in cells.items():
            for key, d in keys.items():
                if d["train_test_overlap"] or d["train_dev_overlap"]:
                    md.append(f"- {ds}/{cell} `{key}`: train∩test={d['train_test_overlap']} "
                              f"train∩dev={d['train_dev_overlap']} unique={d['n_unique']}")
    (OUT / "phase0_audit.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT / 'phase0_audit.json'} and phase0_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
