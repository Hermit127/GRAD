#!/usr/bin/env python3
"""Protocol A window alignment on Decomp: GRAD prefix scores vs stored LSM
per-prefix probabilities, under identical decision points and aggregation.

GRAD is the canonical SSUM backbone (code/train_gnn_v2.py) trained from
scratch on the frozen fix2 decomp fold0 split. Prefix inputs are produced by
the shared pre_call contract (candidate visible, its result and current-event
edges absent). The LSM side is re-aggregated from stored per-prefix
probabilities; no API calls.

Windows reported for both sides:
  local_prefix1    decision points k=0.. (prefix 1 included)
  official_prefix2 decision points k=1.. (official LSM window)
Aggregation for both: harmful = max over k <= deadline; benign = max over all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

_REPO_CODE = Path(__file__).resolve().parents[2]
if str(_REPO_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_gnn_v2 import (V2GraphDataset, CausalCharGNN,  # noqa: E402
                          graph_to_v2_data, threshold_for_target_fpr)
from graphbuild import prefix_graph  # noqa: E402
from protocol_a import graph_index, ranking_metrics, trajectory_scores  # noqa: E402

DEVICE = torch.device("cpu")


def load_graphs(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = CausalCharGNN(cfg["node_base_dim"], cfg["edge_base_dim"],
                          hidden_dim=cfg.get("hidden_dim", 192),
                          dropout=cfg.get("dropout", 0.35) if "dropout" in cfg else 0.35)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def score_prefixes(model, graphs, min_index=0):
    out = {}
    for g in graphs:
        nodes = sorted(g["nodes"], key=lambda n: (n["turn"], n["id"]))
        per_prefix = [None] * len(nodes)
        datas, ks = [], []
        for k in range(min_index, len(nodes)):
            pg = prefix_graph(g, k, candidate_phase="pre_call")
            datas.append(graph_to_v2_data(pg))
            ks.append(k)
        for start in range(0, len(datas), 64):
            chunk = datas[start:start + 64]
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(chunk).to(DEVICE)
            logits = model(batch.x_base, batch.node_chars, batch.edge_index,
                           batch.edge_attr_base, batch.edge_chars, batch.batch)
            probs = torch.sigmoid(logits).tolist()
            for k, p in zip(ks[start:start + 64], probs):
                per_prefix[k] = float(p)
        out[g["task_id"]] = per_prefix
    return out


def aggregate(probs_by_task, labels_by_task, deadlines_by_task, min_index):
    labels, scores = [], []
    for tid, per_prefix in probs_by_task.items():
        s = trajectory_scores(per_prefix, deadlines_by_task.get(tid),
                              labels_by_task[tid], min_index=min_index)
        if s is None:
            continue
        labels.append(labels_by_task[tid])
        scores.append(s)
    return labels, scores


def op_point(labels, scores, threshold):
    pred = [1 if s >= threshold else 0 for s in scores]
    pos = sum(labels)
    neg = len(labels) - pos
    tp = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
    return {"threshold": threshold, "tpr": tp / max(pos, 1), "fpr": fp / max(neg, 1),
            "tp": tp, "fp": fp, "pos": pos, "neg": neg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True,
                    help="dir containing decomp fold0 train/dev/test.jsonl")
    ap.add_argument("--lsm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-fpr", type=float, default=0.10)
    args = ap.parse_args()

    splits = Path(args.splits)
    model, cfg, ckpt = load_model(args.checkpoint)
    dev_graphs = load_graphs(splits / "dev.jsonl")
    test_graphs = load_graphs(splits / "test.jsonl")
    dev_labels, dev_deadlines = graph_index(dev_graphs)
    test_labels, test_deadlines = graph_index(test_graphs)

    dev_probs = score_prefixes(model, dev_graphs, min_index=0)
    test_probs = score_prefixes(model, test_graphs, min_index=0)

    lsm = json.load(open(args.lsm))
    lsm_probs = lsm["per_task_probs"]

    report = {"checkpoint": args.checkpoint, "config_subset": {
        k: cfg.get(k) for k in ("seed", "train", "val", "test", "epochs",
                                "checkpoint_metric", "hidden_dim")},
        "model": {}, "lsm": {}, "alignment": {}}

    for tag, min_index in (("local_prefix1", 0), ("official_prefix2", 1)):
        dl, ds = aggregate(dev_probs, dev_labels, dev_deadlines, min_index)
        tl, ts = aggregate(test_probs, test_labels, test_deadlines, min_index)
        chosen = threshold_for_target_fpr(dl, ds, args.target_fpr)[0]
        report["model"][tag] = {
            "dev": ranking_metrics(dl, ds), "test": ranking_metrics(tl, ts),
            "dev_threshold": chosen, "test_at_dev_threshold": op_point(tl, ts, chosen),
        }
        ll, ls = aggregate(lsm_probs, test_labels, test_deadlines, min_index)
        report["lsm"][tag] = {
            "test": ranking_metrics(ll, ls),
            "at_0.5": op_point(ll, ls, 0.5),
        }
        report["alignment"][tag] = {
            "n_model": report["model"][tag]["test"]["n"],
            "n_lsm": report["lsm"][tag]["test"]["n"],
            "auroc_model": report["model"][tag]["test"]["auroc"],
            "auroc_lsm": report["lsm"][tag]["test"]["auroc"],
            "delta": report["model"][tag]["test"]["auroc"] - report["lsm"][tag]["test"]["auroc"],
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True, default=float)
    print(json.dumps(report, indent=1, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
