#!/usr/bin/env python3
"""Task 1 dev-pilot trainer/evaluator.

Trains every arm from scratch on the same generated split with the same seed
set, optimizer budget and dev-selection rule, then reports:
  - AUROC / PR-AUC / TPR@dev-FPR<=10% / test FPR
  - pair-order accuracy and mean margin (harm vs safe within a pair)
  - relation-change response by counterfactual kind
  - frozen-model structural perturbation deltas (P1/P2/P3)
Outputs per-run raw scores (task_id, pair_id, label, score) plus an automatic
aggregation table and a provenance manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader as PyGDataLoader

from graphbuild import (build_graph, perturb_degree_swap, perturb_endpoint_rewire,
                        perturb_irrelevant)
from models_text import (MAX_TOKENS, TokenTextClassifier, build_vocab,  # noqa: E402
                         encode_text)
from serialize import chrono_text, relation_text
from graphbuild import EDGE_TYPES
from variants import (TEXT_CONFIGS, VARIANT_CONFIGS, build_node_type_vocab,
                      graph_to_data, make_model, model_forward,
                      node_base_dim as variants_base_dim)

DEVICE = torch.device("cpu")


def set_device(name):
    global DEVICE
    DEVICE = torch.device(name)
    return DEVICE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_scenarios(pairs_dir: Path, split: str):
    path = pairs_dir / f"pairs_{split}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_graphs(scenarios, mode, drop_relation_types=False, drop_edges=(),
                wrong_source=False):
    return [build_graph(s, mode=mode, drop_relation_types=drop_relation_types,
                        drop_edges=drop_edges, wrong_source=wrong_source)
            for s in scenarios]


def threshold_for_target_fpr(labels, scores, target):
    candidates = sorted(set(scores), reverse=True)
    candidates = [float("inf")] + candidates + [float("-inf")]
    best = None
    for t in candidates:
        pred = [1 if s >= t else 0 for s in scores]
        neg = sum(1 for y in labels if y == 0)
        fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
        tp = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 1)
        pos = sum(1 for y in labels if y == 1)
        fpr = fp / max(neg, 1)
        tpr = tp / max(pos, 1)
        if fpr <= target + 1e-12 and (best is None or tpr > best["tpr"]):
            best = {"threshold": t, "tpr": tpr, "fpr": fpr,
                    "tp": tp, "fp": fp, "pos": pos, "neg": neg}
    return best


def metrics_at(labels, scores, threshold):
    pred = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
    pos = sum(1 for y in labels if y == 1)
    neg = sum(1 for y in labels if y == 0)
    return {"threshold": threshold, "tpr": tp / max(pos, 1), "fpr": fp / max(neg, 1),
            "tp": tp, "fp": fp, "pos": pos, "neg": neg}


def score_metrics(labels, scores):
    return {
        "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else float(np.mean(labels)),
        "n": len(labels), "n_pos": int(sum(labels)), "n_neg": int(len(labels) - sum(labels)),
    }


def pair_rows(rows, kind_by_pair=None):
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[r["pair_id"]][r["label"]] = r["score"]
    out = []
    for pid, d in by_pair.items():
        if 1 not in d or 0 not in d:
            continue
        out.append({"pair_id": pid, "margin": d[1] - d[0],
                    "ordered": 1.0 if d[1] > d[0] else (0.5 if d[1] == d[0] else 0.0),
                    "kind": (kind_by_pair or {}).get(pid)})
    return out


def pair_stats(pairs):
    if not pairs:
        return {"n_pairs": 0, "order_acc": None, "mean_margin": None}
    return {
        "n_pairs": len(pairs),
        "order_acc": float(np.mean([p["ordered"] for p in pairs])),
        "mean_margin": float(np.mean([p["margin"] for p in pairs])),
    }


def bootstrap_pairs(pairs, n_draw=2000, seed=0):
    rng = np.random.default_rng(seed)
    accs, margins = [], []
    for _ in range(n_draw):
        idx = rng.integers(0, len(pairs), len(pairs))
        sample = [pairs[i] for i in idx]
        accs.append(np.mean([p["ordered"] for p in sample]))
        margins.append(np.mean([p["margin"] for p in sample]))
    return {
        "order_acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "mean_margin_ci95": [float(np.percentile(margins, 2.5)), float(np.percentile(margins, 97.5))],
    }


def train_gnn(cfg, graphs_train, graphs_dev, seed, epochs, batch_size, lr,
              patience, node_type_vocab, hidden_dim, char_dim, dropout):
    set_seed(seed)
    train_data = [graph_to_data(g, cfg, node_type_vocab) for g in graphs_train]
    dev_data = [graph_to_data(g, cfg, node_type_vocab) for g in graphs_dev]
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = PyGDataLoader(dev_data, batch_size=batch_size)
    model = make_model(cfg, node_type_vocab, hidden_dim=hidden_dim,
                       char_dim=char_dim, dropout=dropout).to(DEVICE)
    pos = sum(1 for g in graphs_train if int(g["label"]) == 1)
    neg = len(graphs_train) - pos
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = {"auroc": -1, "state": None, "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model_forward(model, batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch.y.view(-1), pos_weight=pos_weight.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        dev_scores, dev_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(DEVICE)
                dev_scores.extend(torch.sigmoid(model_forward(model, batch)).tolist())
                dev_labels.extend(batch.y.view(-1).tolist())
        auroc = roc_auc_score(dev_labels, dev_scores) if len(set(dev_labels)) > 1 else 0.5
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_auroc": float(auroc)})
        if auroc > best["auroc"]:
            best = {"auroc": float(auroc), "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": epoch}
        if epoch - best["epoch"] >= patience:
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    def score(graphs):
        data = [graph_to_data(g, cfg, node_type_vocab) for g in graphs]
        loader = PyGDataLoader(data, batch_size=batch_size)
        rows = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                probs = torch.sigmoid(model_forward(model, batch)).tolist()
                for prob, tid, pid, y in zip(probs, batch.task_id, batch.pair_id,
                                             batch.y.view(-1).tolist()):
                    rows.append({"task_id": tid, "pair_id": pid, "label": int(y),
                                 "score": float(prob)})
        return rows

    return model, best, history, score, {"node_type_vocab": node_type_vocab,
                                         "hidden_dim": hidden_dim,
                                         "char_dim": char_dim,
                                         "dropout": dropout}


def train_text(view_cfg, graphs_train, graphs_dev, seed, epochs, batch_size, lr,
               patience):
    set_seed(seed)
    render = (chrono_text if view_cfg["view"] == "chrono" else
              (lambda g: relation_text(g, include_types=view_cfg["include_types"])))
    vocab = build_vocab([render(g) for g in graphs_train])

    def to_tensor(graphs):
        ids = torch.tensor([encode_text(render(g), vocab) for g in graphs],
                           dtype=torch.long)
        labels = torch.tensor([float(g["label"]) for g in graphs])
        return ids, labels

    tr_x, tr_y = to_tensor(graphs_train)
    dv_x, dv_y = to_tensor(graphs_dev)
    pos = float((tr_y == 1).sum())
    neg = float((tr_y == 0).sum())
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32)
    model = TokenTextClassifier(len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = {"auroc": -1, "state": None, "epoch": 0}
    history = []
    n = len(tr_x)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(tr_x[idx].to(DEVICE))
            loss = F.binary_cross_entropy_with_logits(
                logits, tr_y[idx].to(DEVICE), pos_weight=pos_weight.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            dev_scores = torch.sigmoid(model(dv_x.to(DEVICE))).tolist()
        auroc = roc_auc_score(dv_y.tolist(), dev_scores) if len(set(dv_y.tolist())) > 1 else 0.5
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_auroc": float(auroc)})
        if auroc > best["auroc"]:
            best = {"auroc": float(auroc), "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": epoch}
        if epoch - best["epoch"] >= patience:
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    def score(graphs):
        x, _ = to_tensor(graphs)
        with torch.no_grad():
            probs = torch.sigmoid(model(x.to(DEVICE))).tolist()
        return [{"task_id": g["task_id"], "pair_id": g["pair_id"],
                 "label": int(g["label"]), "score": float(p)} for g, p in zip(graphs, probs)]

    return model, best, history, score, {"vocab": vocab}


def evaluate_run(name, cfg, graphs, seed, args, node_type_vocab):
    tr = graphs["train"]
    dv = graphs["dev"]
    te = graphs[args.eval_split]
    t0 = time.time()
    if cfg.get("arm") == "text":
        model, best, history, scorer, meta = train_text(
            cfg, tr, dv, seed, args.epochs, args.batch_size, args.lr, args.patience)
    else:
        model, best, history, scorer, meta = train_gnn(
            cfg, tr, dv, seed, args.epochs, args.batch_size, args.lr, args.patience,
            node_type_vocab, args.hidden_dim, args.char_dim, args.dropout)
    dev_rows = scorer(dv)
    test_rows = scorer(te)
    dev_labels = [r["label"] for r in dev_rows]
    dev_scores = [r["score"] for r in dev_rows]
    thr = threshold_for_target_fpr(dev_labels, dev_scores, args.target_fpr)
    test_labels = [r["label"] for r in test_rows]
    test_scores = [r["score"] for r in test_rows]
    te_metrics = metrics_at(test_labels, test_scores, thr["threshold"])
    metrics = {
        "dev": {"auroc": best["auroc"], "selected_epoch": best["epoch"],
                "threshold_at_target_fpr": thr},
        "test": {**score_metrics(test_labels, test_scores), "at_dev_threshold": te_metrics},
    }
    kind_by_pair = {g["pair_id"]: g["kind"] for g in te}
    pairs = pair_rows(test_rows, kind_by_pair)
    metrics["test"]["pair"] = {**pair_stats(pairs), **bootstrap_pairs(pairs, seed=seed)}
    for kind in sorted({g["kind"] for g in te}):
        kp = [p for p in pairs if p["kind"] == kind]
        metrics["test"][f"pair_{kind}"] = pair_stats(kp)
    out = {
        "run": name, "seed": seed, "config": {k: v for k, v in cfg.items() if k != "arm"},
        "history": history, "metrics": metrics,
        "scores": {"dev": dev_rows, "test": test_rows},
        "wall_s": time.time() - t0,
    }
    return out, model, scorer, meta


def run_perturbations(model, scorer, graphs_test, out):
    results = {}
    for kind, fn in (("endpoint_rewire", perturb_endpoint_rewire),
                     ("degree_swap", perturb_degree_swap),
                     ("irrelevant_chrono", perturb_irrelevant)):
        pert = [fn(g, seed=0) for g in graphs_test]
        rows = scorer(pert)
        base_by_task = {r["task_id"]: r["score"] for r in out["scores"]["test"]}
        deltas = []
        for r in rows:
            if r["task_id"] in base_by_task:
                deltas.append(r["score"] - base_by_task[r["task_id"]])
        pairs = pair_rows(rows)
        results[kind] = {
            "mean_abs_delta": float(np.mean([abs(d) for d in deltas])) if deltas else None,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "score_flip_rate": float(np.mean([abs(d) > 0.5 for d in deltas])) if deltas else None,
            "changed_edges_mean": float(np.mean([g.get("perturbation", {}).get("edges_changed", 0)
                                                 for g in pert])),
            "pair": pair_stats(pairs),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="211,212,213")
    ap.add_argument("--arms", default="gnn,text")
    ap.add_argument("--variants", default=",".join(VARIANT_CONFIGS.keys()))
    ap.add_argument("--text-views", default=",".join(TEXT_CONFIGS.keys()))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-fpr", type=float, default=0.10)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--char-dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="persist the selected model state under OUT/checkpoints")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-split", default="test",
                    help="split used for reported test metrics (e.g. confirm)")
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()

    set_device(args.device)
    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    split_names = ["train", "dev", args.eval_split]
    scenarios = {s: load_scenarios(pairs_dir, s) for s in split_names}
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    variant_names = [v for v in args.variants.split(",") if v]

    provenance = {
        "pairs_dir": str(pairs_dir.resolve()),
        "pairs_manifest": json.load(open(pairs_dir / "pairs_manifest.json")),
        "splits": {s: len(v) for s, v in scenarios.items()},
        "seeds": seeds, "configs": variant_names,
        "text_views": args.text_views, "args": vars(args),
    }
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=1, sort_keys=True)
    hasher = hashlib.sha256()
    for s in split_names:
        hasher.update((pairs_dir / f"pairs_{s}.jsonl").read_bytes())
    provenance["pairs_sha256"] = hasher.hexdigest()
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=1, sort_keys=True)

    summary = {"runs": [], "perturbations": {}}
    n_runs = 0
    for arm in arms:
        names = variant_names if arm == "gnn" else [t for t in args.text_views.split(",") if t]
        for name in names:
            cfg = dict(VARIANT_CONFIGS.get(name) or TEXT_CONFIGS[name])
            cfg["arm"] = arm
            mode = cfg.get("edge_mode", "provenance")
            graphs = {s: make_graphs(scenarios[s], mode,
                                     cfg.get("drop_relation_types", False),
                                     tuple(cfg.get("drop_edges", ())),
                                     bool(cfg.get("wrong_source", False)))
                      for s in scenarios}
            node_type_vocab = (build_node_type_vocab(graphs["train"])
                               if cfg.get("use_type_onehot") else None)
            for seed in seeds:
                if args.max_runs and n_runs >= args.max_runs:
                    break
                run_name = f"{name}_s{seed}"
                out, model, scorer, meta = evaluate_run(run_name, cfg, graphs, seed, args,
                                                        node_type_vocab)
                if args.save_checkpoint:
                    ckpt_dir = out_dir / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "run": run_name, "arm": arm, "config": cfg, "meta": meta,
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "threshold": out["metrics"]["dev"]["threshold_at_target_fpr"]["threshold"],
                        "node_base_dim": (variants_base_dim(cfg, len(node_type_vocab or []))
                                          if arm == "gnn" else None),
                        "edge_base_dim": (len(EDGE_TYPES) + 1 if arm == "gnn" else None),
                    }, ckpt_dir / f"{run_name}.pt")
                if args.perturb and name == "full_provenance" and seed == seeds[0]:
                    out["perturbations"] = run_perturbations(model, scorer, graphs[args.eval_split], out)
                    summary["perturbations"] = out["perturbations"]
                with open(runs_dir / f"{run_name}.json", "w") as f:
                    json.dump(out, f, indent=1, sort_keys=True)
                summary["runs"].append({
                    "run": run_name, "arm": arm,
                    "auroc": out["metrics"]["test"]["auroc"],
                    "pr_auc": out["metrics"]["test"]["pr_auc"],
                    "tpr_at_dev_fpr": out["metrics"]["test"]["at_dev_threshold"]["tpr"],
                    "test_fpr": out["metrics"]["test"]["at_dev_threshold"]["fpr"],
                    "pair_order_acc": out["metrics"]["test"]["pair"]["order_acc"],
                    "mean_margin": out["metrics"]["test"]["pair"]["mean_margin"],
                    "dev_auroc": out["metrics"]["dev"]["auroc"],
                })
                print(json.dumps(summary["runs"][-1]), flush=True)
                n_runs += 1
            if args.max_runs and n_runs >= args.max_runs:
                break
        if args.max_runs and n_runs >= args.max_runs:
            break

    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    by_arm = defaultdict(list)
    for r in summary["runs"]:
        by_arm[r["run"].rsplit("_s", 1)[0]].append(r)
    lines = ["# Task 1 dev pilot (auto-aggregated)", "",
             "| arm | AUROC mean±sd | PR-AUC | TPR@devFPR10 | test FPR | pair-order | margin |",
             "|---|---|---|---|---|---|---|"]
    for arm, rs in by_arm.items():
        def ms(key):
            vals = [r[key] for r in rs if r[key] is not None]
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "n/a"
        lines.append(f"| {arm} (n={len(rs)}) | {ms('auroc')} | {ms('pr_auc')} | "
                     f"{ms('tpr_at_dev_fpr')} | {ms('test_fpr')} | "
                     f"{ms('pair_order_acc')} | {ms('mean_margin')} |")
    (out_dir / "aggregate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
