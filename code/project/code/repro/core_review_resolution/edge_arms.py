#!/usr/bin/env python3
"""B-line attribution (reviewer question 2): E / Z / R edge arms.

Installed as a runtime patch over the FROZEN W1 trainer
(`code/repro/final_five_tables/train_grad_repair.py`) and the frozen scoring
script (`code/repro/w1_final_confirmation/score_frozen.py`); no frozen file is
modified, and the same transform is used for train / dev / test.

Arms (frozen in LOGS/core_review_resolution/b_prereg.md BEFORE any training):

  E  original candidate relations (rebuilt_v2 edges) == frozen W1 GRAD input.
  Z  remove cross-node edges entirely; the only edges left are the self-loops
     added internally by GINEConv (`add_self_loops=True`), i.e. the
     implementation-required self-loops.
  R  random rewiring that keeps, per edge type, the exact source list and the
     exact target multiset. This preserves edge count, per-type edge counts,
     out-degree exactly and the per-type in-degree multiset exactly. Targets
     are permuted with a pre-registered deterministic seed derived from the
     graph content (no search over perturbations; one fixed rule).

Environment variable `CRR_EDGE_ARM` selects the arm (default E). The frozen
training/scoring code paths and all other inputs are untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Iterable

# Pre-registered rewiring salt (frozen in b_prereg.md; never changed).
REWIRE_SALT = "crr-b-attribution-20260926"
ARM_ENV = "CRR_EDGE_ARM"


def stable_graph_key(graph: dict) -> str:
    payload = {
        "task_id": graph.get("task_id"),
        "edges": sorted([str(e.get("source")), str(e.get("target")), str(e.get("type"))]
                        for e in graph.get("edges", [])),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def stable_rewire_seed(graph: dict) -> int:
    h = hashlib.sha256((REWIRE_SALT + "|" + stable_graph_key(graph)).encode()).hexdigest()
    return int(h[:16], 16)


def rewire_edges(edges: list[dict], seed: int) -> list[dict]:
    """Permute targets within same-type groups (sources stay in place)."""
    rng = random.Random(seed)
    by_type: dict[str, list[int]] = {}
    for i, e in enumerate(edges):
        by_type.setdefault(str(e.get("type")), []).append(i)
    out = [dict(e) for e in edges]
    for etype in sorted(by_type):
        idxs = by_type[etype]
        targets = [edges[i].get("target") for i in idxs]
        rng.shuffle(targets)
        for i, tgt in zip(idxs, targets):
            out[i]["target"] = tgt
    return out


def transform_edges(edges: list[dict], graph: dict, arm: str) -> list[dict]:
    if arm == "E":
        return edges
    if arm == "Z":
        return []
    if arm == "R":
        return rewire_edges(edges, stable_rewire_seed(graph))
    raise ValueError(f"unknown edge arm: {arm!r}")


def install_patch(module) -> str:
    """Patch `module.merged_edges` to apply the env-selected arm."""
    arm = os.environ.get(ARM_ENV, "E").upper()
    if arm not in ("E", "Z", "R"):
        raise SystemExit(f"invalid {ARM_ENV}={arm!r}")
    original = module.merged_edges

    def patched(graph, cfg):
        return transform_edges(original(graph, cfg), graph, arm)

    module.merged_edges = patched
    return arm
