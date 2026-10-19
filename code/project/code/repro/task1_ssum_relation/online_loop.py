#!/usr/bin/env python3
"""Layer-4 closed-loop evaluation in a controlled execution environment.

Protocol (frozen before running, see freeze_online.py):
  * Decision input per method = the scenario graph, which contains only
    proposal-time information: executed events, the proposed candidate, and
    proposal-intrinsic provenance (which source supplied the argument, which
    authorization covers it). No execution result, no label, no future event.
  * Arms: none / rule verifier / GRAD full provenance / GRAD chronological /
    same-information text monitor. Every learned arm uses a frozen checkpoint
    and a dev-selected threshold.
  * Environment continues after intervention: a blocked attack action is
    replanned to the trusted alternative (task completes without harm); a
    blocked benign action cannot be replaced (utility loss).
  * Outcomes are computed by the environment verifier (task authorization +
    actual parameter provenance), never by the detector's own rule.

Reported: ASR, utility under attack (UA), benign utility (BU), false-block
rate, and per-decision defense latency (p50/p95).

This is a script-driven deterministic environment: it is a *controlled
execution environment closed loop*, not an LLM-agent deployment loop. The text
monitor is a learned same-information monitor, not the ICLR'26 prompt-based
Sequential Monitor (which requires paid API calls).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_FINAL_FIVE = Path(__file__).resolve().parents[2] / "repro" / "final_five_tables"
if str(_REPO_FINAL_FIVE) not in sys.path:
    sys.path.insert(0, str(_REPO_FINAL_FIVE))

from cfgen2 import verify_policy  # noqa: E402
from graphbuild import build_graph  # noqa: E402
from models_text import MAX_TOKENS, TokenTextClassifier, encode_text  # noqa: E402
from rule_baseline import rule_decision  # noqa: E402
from serialize import relation_text  # noqa: E402
from train_gnn_ablate import CausalCharGNN, CausalCharGNNNoText  # noqa: E402
from variants import VARIANT_CONFIGS, graph_to_data, make_model, model_forward  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_freeze(freeze_path):
    freeze = json.load(open(freeze_path))
    checks = []
    pairs = Path(freeze["pairs_file"])
    checks.append(("pairs", sha256(pairs) == freeze["pairs_sha256"]))
    if freeze.get("pairs_manifest_sha256"):
        pm = Path(freeze["pairs_manifest"])
        checks.append(("pairs_manifest", sha256(pm) == freeze["pairs_manifest_sha256"]))
    for arm, d in freeze["defenses"].items():
        checks.append((f"{arm}.ckpt", sha256(d["checkpoint"]) == d["checkpoint_sha256"]))
        checks.append((f"{arm}.run", sha256(d["run_json"]) == d["run_json_sha256"]))
    here = Path(__file__).resolve().parent
    for name, digest in freeze.get("loop_code", {}).items():
        p = here / name
        checks.append((f"code.{name}", p.exists() and sha256(p) == digest))
    bad = [name for name, ok in checks if not ok]
    if bad:
        raise SystemExit(f"freeze verification failed: {bad}")
    return freeze


def load_gnn(entry):
    ckpt = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    meta = ckpt["meta"]
    node_type_vocab = meta["node_type_vocab"]
    model = make_model(cfg, node_type_vocab, hidden_dim=meta["hidden_dim"],
                       char_dim=meta["char_dim"], dropout=meta["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, node_type_vocab, float(ckpt["threshold"])


def load_text(entry):
    ckpt = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    vocab = ckpt["meta"]["vocab"]
    model = TokenTextClassifier(len(vocab))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, float(ckpt["threshold"])


def score_gnn_batch(model, graphs, cfg, node_type_vocab):
    from torch_geometric.data import Batch
    dev = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for g in graphs:
            data = graph_to_data(g, cfg, node_type_vocab)
            batch = Batch.from_data_list([data]).to(dev)
            scores.append(float(torch.sigmoid(model_forward(model, batch)).item()))
    return scores


def score_text_batch(model, graphs, vocab):
    dev = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for g in graphs:
            ids = torch.tensor([encode_text(relation_text(g), vocab)], dtype=torch.long)
            scores.append(float(torch.sigmoid(model(ids.to(dev))).item()))
    return scores


def outcome(decision, variant):
    """Closed-loop environment outcome for one scenario.

    variant: harm (unauthorized/untrusted action proposed) or safe.
    allow  -> harm executes (attack) / benign completes
    block  -> attack replans to the trusted alternative (no harm, task
              completes); benign has no alternative (task incomplete).
    """
    if variant == "harm":
        harm_executed = decision == "allow"
        task_completed = True
        utility_under_attack = decision == "block"
        return {"harm_executed": harm_executed, "task_completed": task_completed,
                "utility_under_attack": utility_under_attack, "false_block": False}
    task_completed = decision == "allow"
    return {"harm_executed": False, "task_completed": task_completed,
            "utility_under_attack": None, "false_block": decision == "block"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    freeze = verify_freeze(args.freeze)
    device = torch.device(args.device)
    rows = [json.loads(l) for l in open(freeze["pairs_file"]) if l.strip()]

    arms = {"none": None, "rule": None}
    gnn_arms, text_arms = {}, {}
    for arm, entry in freeze["defenses"].items():
        if entry["config"].get("arm") == "text" or arm == "text_relation":
            model, vocab, thr = load_text(entry)
            model.to(device)
            text_arms[arm] = (model, vocab, thr, entry["config"])
        else:
            model, cfg, ntv, thr = load_gnn(entry)
            model.to(device)
            gnn_arms[arm] = (model, cfg, ntv, thr, entry["config"])

    # Pre-build per-arm decision graphs (identical scenario list order).
    grad_arm_names = list(gnn_arms.keys()) + list(text_arms.keys())
    graph_cache = {}
    for arm in grad_arm_names:
        if arm in gnn_arms:
            cfg = gnn_arms[arm][1]
            graph_cache[arm] = [build_graph(r, mode=cfg.get("edge_mode", "provenance"),
                                            drop_relation_types=cfg.get("drop_relation_types", False),
                                            drop_edges=tuple(cfg.get("drop_edges", ())),
                                            wrong_source=bool(cfg.get("wrong_source", False)))
                                for r in rows]
        else:
            graph_cache[arm] = [build_graph(r, mode="provenance") for r in rows]

    rule_graphs = [build_graph(r, mode="provenance") for r in rows]
    results = {"freeze": args.freeze, "device": args.device, "per_scenario": [],
               "arms": {}}
    latencies = defaultdict(list)
    for i, r in enumerate(rows):
        variant = r["variant"]
        record = {"pair_id": r["pair_id"], "kind": r["kind"], "variant": variant,
                  "label": int(r["label"])}
        decisions = {}
        decisions["none"] = "allow"
        decisions["rule"] = "block" if rule_decision(rule_graphs[i]) else "allow"
        for arm, (model, cfg, ntv, thr, _c) in gnn_arms.items():
            t0 = time.perf_counter()
            score = score_gnn_batch(model, [graph_cache[arm][i]], cfg, ntv)[0]
            latencies[arm].append((time.perf_counter() - t0) * 1000)
            decisions[arm] = "block" if score >= thr else "allow"
            record[f"score_{arm}"] = score
        for arm, (model, vocab, thr, _c) in text_arms.items():
            t0 = time.perf_counter()
            score = score_text_batch(model, [graph_cache[arm][i]], vocab)[0]
            latencies[arm].append((time.perf_counter() - t0) * 1000)
            decisions[arm] = "block" if score >= thr else "allow"
            record[f"score_{arm}"] = score
        for arm, decision in decisions.items():
            rec = dict(record)
            rec["arm"] = arm
            rec["decision"] = decision
            rec.update(outcome(decision, variant))
            results["per_scenario"].append(rec)

    for arm in ["none", "rule"] + grad_arm_names:
        rs = [x for x in results["per_scenario"] if x["arm"] == arm]
        atk = [x for x in rs if x["variant"] == "harm"]
        ben = [x for x in rs if x["variant"] == "safe"]
        lat = np.array(latencies.get(arm, [0.0]))
        results["arms"][arm] = {
            "n_attack": len(atk), "asr": float(np.mean([x["harm_executed"] for x in atk])),
            "ua": float(np.mean([x["utility_under_attack"] for x in atk])),
            "n_benign": len(ben), "bu": float(np.mean([x["task_completed"] for x in ben])),
            "false_block_rate": float(np.mean([x["false_block"] for x in ben])),
            "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    report = Path(args.out).with_suffix(".md")
    lines = ["# Layer-4 closed-loop results (controlled execution environment)", "",
             "| arm | ASR↓ | UA↑ | BU↑ | false-block↓ | lat p50/p95 (ms) |",
             "|---|---|---|---|---|---|"]
    for arm, m in results["arms"].items():
        lines.append(f"| {arm} | {m['asr']:.3f} | {m['ua']:.3f} | {m['bu']:.3f} | "
                     f"{m['false_block_rate']:.3f} | {m['latency_ms_p50']:.2f}/{m['latency_ms_p95']:.2f} |")
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
