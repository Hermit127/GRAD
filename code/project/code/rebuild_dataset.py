#!/usr/bin/env python3
"""Rebuild DecomposedHarm graph dataset with:
1. Taint propagation from harm_index (bidirectional decay)
2. Benign node injection into attack graphs (mixed chains)
Output: mixed JSONL files for train/val/test.
"""
from __future__ import annotations

import json, random, sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "ref" / "dataset"
EDGE_PROPAGATE = {"data_flow", "causal_derivation", "writes_to", "reads_from"}
SEED = 42
random.seed(SEED)


def load_jsonl(path: str) -> list[dict]:
    graphs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                graphs.append(json.loads(line))
    return graphs


def save_jsonl(graphs: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for g in graphs:
            f.write(json.dumps(g) + "\n")


def propagate_taint_on_graph(graph: dict) -> dict:
    """Bidirectional propagation from harm_index node. Mutates in-place."""
    if graph["label"] != 1:
        # Benign: all nodes taint=0, node_label=0
        for n in graph["nodes"]:
            n["taint"] = 0.0
            n.setdefault("features", {})["node_label"] = 0
        return graph

    harm_idx = graph.get("harm_index", -1) - 1  # 1-based → 0-based
    n_nodes = len(graph["nodes"])
    if harm_idx < 0 or harm_idx >= n_nodes:
        harm_idx = max(0, n_nodes - 1)

    # Reset all taint to 0
    for n in graph["nodes"]:
        n["taint"] = 0.0

    # Build adjacency from edges
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    for e in graph["edges"]:
        if e["type"] in EDGE_PROPAGATE:
            out_edges[e["source"]].append(e["target"])
            in_edges[e["target"]].append(e["source"])

    # BFS from harm node, bidirectional, multiplicative decay
    node_ids = [n["id"] for n in graph["nodes"]]
    source_id = node_ids[harm_idx]
    graph["nodes"][harm_idx]["taint"] = 1.0

    max_hops = 5
    decay = 0.7
    visited = {source_id: 1.0}  # node_id → best_taint
    queue = [(source_id, 1.0, 0)]  # (node_id, taint, hops)

    while queue:
        cur_id, cur_taint, cur_hops = queue.pop(0)
        if cur_hops >= max_hops:
            continue
        next_taint = cur_taint * decay
        next_hops = cur_hops + 1

        # Forward
        for tgt_id in out_edges.get(cur_id, []):
            if next_taint > visited.get(tgt_id, 0):
                visited[tgt_id] = next_taint
                idx = node_ids.index(tgt_id)
                graph["nodes"][idx]["taint"] = next_taint
                queue.append((tgt_id, next_taint, next_hops))

        # Backward
        for src_id in in_edges.get(cur_id, []):
            if next_taint > visited.get(src_id, 0):
                visited[src_id] = next_taint
                idx = node_ids.index(src_id)
                graph["nodes"][idx]["taint"] = next_taint
                queue.append((src_id, next_taint, next_hops))

    # Set node_label = 1 for tainted nodes only (harm subgraph), 0 for others
    for n in graph["nodes"]:
        n.setdefault("features", {})["node_label"] = 1 if n["taint"] > 0 else 0

    return graph


def build_benign_pool(graphs: list[dict]) -> list[dict]:
    """Extract all nodes from benign graphs."""
    pool = []
    for g in graphs:
        if g["label"] == 0:
            for n in g["nodes"]:
                pool.append({"type": n["type"], "features": n["features"]})
    return pool


def inject_benign_into_attack(graph: dict, benign_pool: list[dict],
                               inject_ratio: float = 0.5) -> dict:
    """Inject benign nodes into attack graph at random positions.
    Injected nodes get node_label=0, taint=0.
    """
    if graph["label"] != 1 or len(benign_pool) == 0:
        return graph

    n_original = len(graph["nodes"])
    num_inject = max(1, int(n_original * inject_ratio))
    num_inject = min(num_inject, len(benign_pool))
    injected = random.sample(benign_pool, num_inject)

    # Random insertion points with replacement
    insert_counts = defaultdict(int)
    for p in random.choices(range(n_original + 1), k=num_inject):
        insert_counts[p] += 1

    new_nodes = []
    old_to_new = {}
    injected_idx = 0
    new_idx = 0

    for i in range(n_original + 1):
        for _ in range(insert_counts.get(i, 0)):
            bn = injected[injected_idx]
            new_nodes.append({
                "id": f"mixed_b_{injected_idx}",
                "type": bn["type"], "taint": 0.0, "turn": new_idx,
                "features": {**bn.get("features", {}), "node_label": 0, "injected": True},
            })
            injected_idx += 1
            new_idx += 1
        if i < n_original:
            old_to_new[i] = new_idx
            n = dict(graph["nodes"][i])
            n["turn"] = new_idx
            new_nodes.append(n)
            new_idx += 1

    # Remap edges
    new_edges = []
    for e in graph.get("edges", []):
        src_orig = next((j for j, n in enumerate(graph["nodes"]) if n["id"] == e["source"]), None)
        tgt_orig = next((j for j, n in enumerate(graph["nodes"]) if n["id"] == e["target"]), None)
        if src_orig is not None and tgt_orig is not None:
            if src_orig in old_to_new and tgt_orig in old_to_new:
                new_edges.append({
                    "source": new_nodes[old_to_new[src_orig]]["id"],
                    "target": new_nodes[old_to_new[tgt_orig]]["id"],
                    "type": e["type"],
                })
    # Chain all
    for j in range(len(new_nodes) - 1):
        new_edges.append({
            "source": new_nodes[j]["id"], "target": new_nodes[j+1]["id"],
            "type": "data_flow",
        })

    # Update harm_index
    new_harm = graph.get("harm_index", -1)
    if new_harm > 0 and (new_harm - 1) in old_to_new:
        new_harm = old_to_new[new_harm - 1] + 1

    return {
        "task_id": graph["task_id"] + f"_mix{inject_ratio}",
        "label": 1, "harm_index": new_harm,
        "num_nodes": len(new_nodes), "num_edges": len(new_edges),
        "nodes": new_nodes, "edges": new_edges,
    }


def main():
    print("=" * 60)
    print("Rebuilding dataset with taint propagation + mixed chains")
    print("=" * 60)

    # Step 0: Collect benign pool from train only (no data leakage)
    train = load_jsonl(str(DATA_DIR / "decomp_graphs_train_full.jsonl"))
    val = load_jsonl(str(DATA_DIR / "decomp_graphs_validation_full.jsonl"))
    test = load_jsonl(str(DATA_DIR / "decomp_graphs_test_full.jsonl"))
    benign_pool = build_benign_pool(train)
    print(f"Benign pool: {len(benign_pool)} nodes from train")

    # Step 1: Propagate taint on all attack graphs
    for name, graphs in [("train", train), ("val", val), ("test", test)]:
        for g in graphs:
            propagate_taint_on_graph(g)
        n_tainted = sum(1 for g in graphs if g["label"] == 1
                        for n in g["nodes"] if n["taint"] > 0)
        print(f"  {name}: {n_tainted} tainted nodes across "
              f"{sum(1 for g in graphs if g['label']==1)} attack graphs "
              f"({n_tainted/max(1,sum(1 for g in graphs if g['label']==1)):.1f}/graph)")

    # Step 2: Generate mixed attack graphs (attack + injected benign)
    # Mix ratios: 50%, 100% extra benign nodes (drop 150% to reduce noise)
    mixed = []
    for g in train:
        if g["label"] == 1:
            for ratio in [0.5, 1.0]:
                mg = inject_benign_into_attack(dict(g), benign_pool, ratio)
                mg["task_id"] = mg["task_id"] + f"_r{ratio}"
                mixed.append(mg)

    # Combine: original + mixed (shuffled)
    all_train = train + mixed
    random.shuffle(all_train)
    print(f"\n  Train: {len(train)} original + {len(mixed)} mixed = {len(all_train)} total")

    # Step 3: Generate mixed val/test (using train benign pool, no val/test benign leak)
    for name, orig in [("validation", val), ("test", test)]:
        mixed = []
        for g in orig:
            if g["label"] == 1:
                for ratio in [0.5, 1.0]:
                    mg = inject_benign_into_attack(dict(g), benign_pool, ratio)
                    mg["task_id"] = mg["task_id"] + f"_r{ratio}"
                    mixed.append(mg)
        result = orig + mixed
        random.shuffle(result)
        print(f"  {name}: {len(orig)} original + {len(mixed)} mixed = {len(result)} total")
        save_jsonl(result, str(DATA_DIR / f"decomp_graphs_{name}_mixed.jsonl"))

    # Save mixed train
    save_jsonl(all_train, str(DATA_DIR / "decomp_graphs_train_mixed.jsonl"))

    # Also save base files with propagated taint (for reference)
    for name, graphs in [("train", train), ("validation", val), ("test", test)]:
        save_jsonl(graphs, str(DATA_DIR / f"decomp_graphs_{name}_tainted.jsonl"))

    print(f"\nDone. Mixed datasets saved to {DATA_DIR}/")
    print(f"  Train: decomp_graphs_train_mixed.jsonl")
    print(f"  Val:   decomp_graphs_validation_mixed.jsonl")
    print(f"  Test:  decomp_graphs_test_mixed.jsonl")


if __name__ == "__main__":
    main()
