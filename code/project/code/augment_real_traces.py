#!/usr/bin/env python3
"""Augment real MemoryGraft payloads into 200 attack + 200 benign behavior graphs.
Each real payload becomes 20 variants by randomizing gap nodes and context."""
import json, random
from pathlib import Path

SEEDS_PATH = Path(__file__).resolve().parent.parent / "ref" / "MemoryGraft" / \
             "metagpt_attack_poc" / "payloads" / "experience_seeds.json"
OUT = Path(__file__).resolve().parent.parent / "ref" / "dataset" / "memorygraft_augmented.jsonl"

random.seed(42)

BENIGN_GAP_NODES = [
    ("file_read", "Read project documentation"),
    ("web_search", "Search: documentation reference"),
    ("information_search", "Look up API reference"),
    ("data_aggregate", "Aggregate intermediate results"),
    ("file_write", "Save intermediate output"),
    ("database_query", "Query validation database"),
    ("web_download", "Download reference material"),
]

ATTACK_GAP_NODES = [
    ("file_read", "Read configuration file"),
    ("web_search", "Search: quick EDA techniques"),
    ("information_search", "Look up validation bypass"),
    ("data_aggregate", "Prepare dataset for export"),
    ("file_write", "Save preprocessed data"),
]


def classify_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ['curl', 'wget', 'pipe', '| bash', '| sh',
                              'os.system', 'subprocess', 'exec(']):
        return "shell_exec"
    if any(k in t for k in ['python', 'script', 'pandas', 'df.', 'import ',
                              'numpy', 'plt.', 'sklearn', 'csv', 'parquet',
                              'monkey', 'set ', 'builtins']):
        return "code_exec"
    if any(k in t for k in ['upload', 'remote', 'external', 'endpoint']):
        return "web_request"
    if any(k in t for k in ['store', 'cache', 'save', 'persist', 'memory']):
        return "memory_write"
    return "tool_execute"


def make_graph(task_id: str, label: int, experience: dict, n_gaps: int = 0) -> dict:
    nodes, edges = [], []
    tid = 0

    def add(ntype, subtask):
        nonlocal tid
        nid = f"{task_id}_s{tid}"
        nodes.append({"id": nid, "type": ntype, "taint": 1.0 if label==1 else 0.0,
                      "turn": tid, "features": {"subtask": subtask, "harm_step": label==1,
                                                "node_label": label}})
        if tid > 0:
            edges.append({"source": f"{task_id}_s{tid-1}", "target": nid, "type": "data_flow"})
        tid += 1

    # Pre-gap benign nodes
    for _ in range(n_gaps):
        ntype, subtask = random.choice(BENIGN_GAP_NODES)
        add(ntype, subtask)

    # Core attack/benign: memory_write(poisoned, attack only) → memory_read → execute
    req = experience["req"]
    resp = experience["resp"]
    if label == 1:
        # MemoryGraft attack mechanism: poisoned experience is *written* to
        # persistent memory first (build_poisoned_store), then retrieved and
        # acted on. The write step was previously dropped, so attack graphs
        # contained no memory_write node and any memory-centric detector
        # (e.g. NeuroTaint) had no taint source. Restore it to match
        # memorygraft_bench.jsonl structure (memory_write -> memory_read -> exec).
        add("memory_write", f"Store poisoned experience: {req}")
    add("memory_read", f"Retrieve: {req}")
    exec_type = classify_type(resp)
    add(exec_type, resp[:200])

    # Post-gap nodes
    for _ in range(n_gaps):
        gap_pool = ATTACK_GAP_NODES if label == 1 else BENIGN_GAP_NODES
        ntype, subtask = random.choice(gap_pool)
        add(ntype, subtask)

    # Context edges: poison write -> read-back (the memory-poison semantic edge).
    # Node order: pre-gap (n_gaps) -> memory_write -> memory_read -> exec -> post-gap (n_gaps).
    # Fix: explicit indices; previous code used nodes[-3]/[-2]/[-1] which pointed at
    # the wrong nodes whenever n_gaps >= 1 (audit EXP-2026W35-010 F2).
    if label == 1:
        mw_id = nodes[n_gaps]["id"]
        mr_id = nodes[n_gaps + 1]["id"]
        edges.append({"source": mw_id, "target": mr_id, "type": "writes_to"})

    return {"task_id": task_id, "label": label, "harm_index": 1 if label else -1,
            "num_nodes": len(nodes), "num_edges": len(edges),
            "nodes": nodes, "edges": edges, "source_type": "memorygraft_real"}


def main():
    with open(SEEDS_PATH) as f:
        data = json.load(f)

    benign = data["benign_experiences"]
    poisoned = data["poisoned_experiences"]

    graphs = []

    # Attack: 10 real payloads × 60 variants = 600
    for base_idx, exp in enumerate(poisoned):
        for var in range(60):
            n_gaps = random.randint(1, 4)
            g = make_graph(f"mg_att_{base_idx:02d}_v{var:02d}", 1, exp, n_gaps)
            graphs.append(g)

    # Benign: 100 real payloads × 4 variants = 400
    for base_idx, exp in enumerate(benign):
        for var in range(4):
            n_gaps = random.randint(0, 3)
            g = make_graph(f"mg_ben_{base_idx:03d}_v{var:02d}", 0, exp, n_gaps)
            graphs.append(g)

    random.shuffle(graphs)
    with open(OUT, "w") as f:
        for g in graphs:
            f.write(json.dumps(g) + "\n")

    n_att = sum(1 for g in graphs if g["label"] == 1)
    avg_n = sum(g["num_nodes"] for g in graphs) / len(graphs)
    print(f"Saved {len(graphs)} graphs ({n_att} att / {len(graphs)-n_att} ben) → {OUT}")
    print(f"  Avg nodes: {avg_n:.1f}")
    print(f"  Attack seeds: {len(poisoned)}, variants per seed: 60")
    print(f"  Benign seeds: {len(benign)}, variants per seed: 4")


if __name__ == "__main__":
    main()
