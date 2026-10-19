#!/usr/bin/env python3
"""Freeze the Table 5 Panel A protocol before any formal run (zero API).

Writes LOGS/final_five_tables/table5_protocol_manifest.json with:
  - exact train/dev/test paths, sha256, graph/node/edge/pos counts
  - group/family/task layout and split provenance
  - overlap audits (group-disjoint for decomp/mcp, task-disjoint for mrt)
  - seeds, training budget, checkpoint selection rule, input observation scope
  - code hashes (train_gnn_v2.py, train_gnn_ablate.py, tests, aggregators)
  - aggregation rule and leakage caveats
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "LOGS/final_five_tables/table5_protocol_manifest.json"

DECOMP = REPO / "code/repro/out/unified_gnn/fix2_splits/decomp/fold0"
MCP = REPO / "code/repro/out/unified_gnn/fix2_splits/mcp"
MRT = REPO / "results/mrt_cross_task_v1"

CODE_FILES = [
    "code/train_gnn_v2.py",
    "code/repro/final_five_tables/train_gnn_ablate.py",
    "code/repro/final_five_tables/test_train_gnn_ablate.py",
    "code/repro/final_five_tables/verify_table5_panelb.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_stats(path: Path, split: str, group_key: str) -> dict:
    n = pos = nodes = edges = 0
    groups: set[str] = set()
    task_ids: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        n += 1
        pos += int(g.get("label", 0))
        nodes += len(g.get("nodes", []))
        edges += len(g.get("edges", []))
        task_ids.add(str(g.get("task_id")))
        if group_key:
            groups.add(str(g.get(group_key) or g.get("metadata", {}).get(group_key, "")))
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": sha256(path),
        "split": split,
        "n_graphs": n,
        "n_pos": pos,
        "n_neg": n - pos,
        "n_nodes": nodes,
        "n_edges": edges,
        "n_groups": len(groups) if group_key else None,
        "n_task_ids": len(task_ids),
    }


def overlap(sets: dict[str, set[str]]) -> dict:
    names = sorted(sets)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}_x_{b}"] = len(sets[a] & sets[b])
    return out


def main() -> int:
    manifest: dict = {
        "table": "table5_panel_a_protocol_frozen",
        "frozen_at_utc": "2026-09-17T17:20:00Z",
        "datasets": {},
        "seeds": [211, 212, 213],
        "training_budget": {
            "epochs": 100, "patience": 25, "batch_size": 48,
            "hidden_dim": 192, "lr": 1e-3, "weight_decay": 1e-4,
            "optimizer": "AdamW", "grad_clip": 1.0,
            "hard_benign_weight": 3.0, "hard_benign_top_k": 128,
            "hard_benign_start_epoch": 2,
        },
        "checkpoint_selection": {
            "metric": "fixed_fpr_f1",
            "target_fpr": 0.10,
            "rule": "dev epoch maximizing F1 at the largest threshold with dev FPR<=0.10",
            "early_stop": "epoch - best_epoch_of_selected_metric >= patience",
        },
        "input_observation": {
            "node_text": "features.subtask truncated to 192 chars, CharCNN (char id = min(ord,127))",
            "node_struct": "12-dim permission vector from static TOOL_TO_PERM + turn/max_nodes + 9 CONTENT_KEYWORDS flags",
            "edges": "rebuilt deterministically from node order/type and regex rules over visible node text; 10 types + confidence; rationale text <=300 chars (160 encoded)",
            "ignored_fields": ["label (except y)", "harm_step", "node_label", "injected",
                               "taint", "raw_text/raw_payload (not read by the model)",
                               "metadata/attack_category/source"],
            "char_window_note": "192-char node window is the frozen unified-GRAD observation; "
                                "231/485 AgentDojo nodes exceed it in the Table 2 hook, not this corpus",
        },
        "aggregation": {
            "auroc": "fold-weighted by n_pos*n_neg within seed; mean/SD across seeds",
            "dev_point": "threshold selected on dev at FPR<=10%; TPR/FPR evaluated on test",
            "aux_metrics": ["PR-AUC (added by table5 aggregator)", "TPR/FPR at dev-selected threshold"],
        },
        "hardware": {
            "archived_full_runs": "server GPU7 (gpu-server), results in code/repro/out/unified_gnn/runs",
            "new_variant_runs": "Apple M2 CPU, 8 cores (GPU host unreachable as of 2026-09-17/18)",
            "mixing_caveat": "all Panel A rows include a harness-rerun `full` on the same hardware; "
                             "archived GPU full runs are used only as a cross-check",
        },
        "code_hashes": {f: sha256(REPO / f) for f in CODE_FILES},
        "panel_b_protocol": {
            "source": "code/repro/out/real_cl/p16_40 (40 runs = 20 seeds x 2 orders)",
            "seeds": list(range(211, 231)),
            "orders": ["forward", "reversed"],
            "updates_per_stage": 200, "batch": 16, "buffer": 100,
            "ewc_lambda": 100.0, "joint_mult": 3, "data_expanded": True,
            "forgetting": "F_T=(1/T)*sum_j[max_{t=j..T} a_tj - a_Tj]",
            "router": "nearest-centroid over 25-dim observable graph signature (train signatures only)",
            "verified": "code/repro/final_five_tables/verify_table5_panelb.py: "
                        "40/40 runs recompute acc_matrix+F_T from scores.jsonl with zero mismatch",
        },
    }

    decomp = {}
    for split in ("train", "dev", "test"):
        decomp[split] = graph_stats(DECOMP / f"{split}.jsonl", split, "group")
    manifest["datasets"]["decomp"] = {
        "splits": decomp,
        "provenance": "code/repro/unified_gnn_materialize_20260915.py from "
                      "static_big_table_v1 fix2_20260910 split_ids/decomp.json "
                      "(group=(harmful|benign)_<id>; role ratios 60/15/10/5/10; "
                      "gate_fit + threshold_cal excluded from train/dev/test)",
        "overlap_groups": overlap({s: {json.loads(l)["group"] for l in (DECOMP / f"{s}.jsonl").read_text().splitlines() if l.strip()}
                                   for s in ("train", "dev", "test")}),
    }

    mcp_folds = {}
    for fold in sorted(MCP.glob("fold*")):
        splits = {}
        for split in ("train", "dev", "test"):
            splits[split] = graph_stats(fold / f"{split}.jsonl", split, "group")
        mcp_folds[fold.name] = {
            "splits": splits,
            "overlap_groups": overlap({s: {str(json.loads(l).get("group")) for l in (fold / f"{s}.jsonl").read_text().splitlines() if l.strip()}
                                       for s in ("train", "dev", "test")}),
        }
    manifest["datasets"]["mcp"] = {
        "folds": mcp_folds,
        "provenance": "mcp_trace_to_graph.py defense-matched 232-graph corpus; "
                      "fix2_20260910 cv5 split_ids/mcp.json group=metadata.task stem",
    }

    mrt_folds = {}
    for fold in sorted(p for p in MRT.glob("fold_*") if p.is_dir()):
        splits = {}
        for name, split in (("train", "train"), ("dev", "calibration"), ("test", "admission")):
            splits[name] = graph_stats(fold / f"{split}.jsonl", name, "")
        mrt_folds[fold.name] = {
            "splits": splits,
            "overlap_task_ids": overlap({s: {str(json.loads(l)["task_id"])
                                             for l in (fold / f"{f}.jsonl").read_text().splitlines() if l.strip()}
                                         for s, f in (("train", "train"), ("dev", "calibration"), ("test", "admission"))}),
        }
    manifest["datasets"]["mrt"] = {
        "folds": mrt_folds,
        "provenance": "build_mrt_cross_task_protocol.py over SHADE-Arena scale440 observable pool; "
                      "leave-one-task-out (5 tasks); train 280 / cal 32 / admission 20",
        "budget_blocker": "one 100-epoch MRT run is ~5-6h on local CPU (35x MCP nodes); "
                          "15 runs/variant => local CPU infeasible; GPU host unreachable",
    }

    OUT.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {OUT}")
    for ds in ("decomp", "mcp", "mrt"):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
