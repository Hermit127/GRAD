#!/usr/bin/env python3
"""DecomposedHarm agent tasks → Behavior Graph → JSONL for GNN training.

Each task becomes a graph: subtask steps → nodes, harm_index → tainted node.
Output: one serialized dict per line ({nodes: [...], edges: [...], label: 0/1}).
"""
from __future__ import annotations

import json, os, re, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
from behavior_graph import BehaviorGraph, GraphNode, GraphEdge

# ===== Subtask → Node type (keyword-based, specific patterns first) =====
TYPE_MAP = [
    # Email [7 §2.3 Fraud/Harassment]
    (["email", "send.*mail", "compose.*mail", "draft.*mail", "inbox", "forward.*mail"], "email_send"),
    # Social media [7 §2.3 Disinformation/Hate]
    (["social media", "facebook", "twitter", "instagram", "post on", "tweet",
      "share on", "linkedin", "forum", "reddit"], "social_media"),
    # Document [7 §2.3 Fraud/Forgery]
    (["fake.*(document|invoice|certificate|id|letter)", "forge", "counterfeit",
      "draft.*(document|report|letter)", "create.*(invoice|template|certificate)",
      "compile.*report", "write.*report"], "document_create"),
    # Account access [7 §2.3 Cybercrime/Harassment]
    (["login", "log in", "sign in", "access.*account", "impersonate",
      "hack", "unauthorized", "bypass.*password"], "account_access"),
    # Messaging [7 §2.3 Harassment/Fraud]
    (["message", "text ", "sms", "dm ", "chat", "reply", "respond",
      "contact via", "reach out"], "message_send"),
    # Web download
    (["download.*(file|software)", "wget", "curl.*-o", "fetch.*file", "save.*file"], "web_download"),
    # Data aggregation
    (["aggregate", "organize.*data", "collect.*list", "extract.*data",
      "gather.*list", "spreadsheet", "compile.*list"], "data_aggregate"),
    # Tool/script execution
    (["execute", "run script", "run program", "launch", "deploy", "start.*tool",
      "open.*application", "use.*tool"], "tool_execute"),
    # Information gathering (default fallback — catch-all for SEARCH patterns)
    (["search engine", "google search", "look up", "find.*information",
      "research", "gather.*information", "browse", "locate.*(website|page)",
      "search for", "query", "look for", "find on"], "information_search"),
    # File operations [1][2]
    (["delete", "remove", "rm ", "erase", "wipe"], "file_delete"),
    (["read file", "open file", "view.*file", "cat ", "head ", "tail "], "file_read"),
    (["write file", "save to", "edit", "modify", "create file", "append", ">>", "> "], "file_write"),
    # Web [1][4]
    (["curl", "wget", "api", "http", "fetch url", "request", "post data"], "web_request"),
    # Shell/code [2][3]
    (["command", "bash", "shell", "terminal", "run ", "python", "node", "npm", "pip", "git"], "shell_exec"),
    # System [1][4]
    (["install", "setup", "configure", "set up", "build", "compile"], "system_config"),
    # Memory [1][6]
    (["remember", "store", "memorize", "save.*memory", "persist"], "memory_write"),
    (["recall", "retrieve", "remember.*past", "look up.*memory", "history"], "memory_read"),
    # Agent [3][4]
    (["ask", "request.*human", "delegate", "assign.*task", "spawn"], "agent_spawn"),
    (["notify", "alert", "inform", "tell", "announce"], "agent_message"),
]


def classify_subtask(text: str) -> str:
    """Map natural language subtask to a graph node type."""
    lower = text.lower().strip()
    for keywords, ntype in TYPE_MAP:
        for kw in keywords:
            if re.search(kw, lower):
                return ntype
    return "information_search"


def task_to_dict(task: dict, task_id: int = 0) -> dict | None:
    """Convert one task to a serializable graph dict. Returns None if empty.
    
    Node labels: all nodes in attack chains → 1, all nodes in benign chains → 0.
    taint is computed via propagation from harm_index (kept for L1 API compat).
    """
    decomposition = task.get("decomposition", "")
    label = 1 if task.get("label") == "harmful" else 0
    sid = task.get("id", f"task_{task_id}")

    lines = [l.strip() for l in decomposition.split("\n") if l.strip()]
    if not lines:
        return None

    harm_index = task.get("harm_index", -1)
    if harm_index is None:
        harm_index = -1
    harm_index = int(float(harm_index))

    g = BehaviorGraph()
    prev_id = None

    for i, line in enumerate(lines):
        ntype = classify_subtask(line)
        nid = f"{sid}_s{i}"
        is_harm = (i + 1) == harm_index
        # taint propagation: harm node → 1.0, decay @0.7 per hop along data_flow
        taint = 0.0

        g.add_node(GraphNode(
            id=nid, type=ntype, session_id=sid[:20], turn=i,
            features={"subtask": line[:200], "harm_step": is_harm,
                      "node_label": label},
            taint=taint,
        ))

        if prev_id:
            g.add_edge(GraphEdge(source=prev_id, target=nid, type="data_flow"))
            # memory edges
            prev_node = g.nodes.get(prev_id)
            if prev_node:
                if prev_node.type == "memory_write" and ntype == "memory_read":
                    g.add_edge(GraphEdge(source=prev_id, target=nid, type="reads_from"))
                if prev_node.type in ("file_write", "shell_exec") and ntype == "memory_write":
                    g.add_edge(GraphEdge(source=prev_id, target=nid, type="writes_to"))
        prev_id = nid

    # Propagate taint backward from harm node along data_flow edges
    if label == 1 and 1 <= harm_index <= len(lines):
        g.set_taint_from_source(harm_index - 1, max_hops=5, decay=0.7)

    return {
        "task_id": sid,
        "label": label,
        "harm_index": harm_index,
        "num_nodes": len(g.nodes),
        "num_edges": len(g.edges),
        "nodes": [{"id": n.id, "type": n.type, "taint": n.taint, "turn": n.turn,
                    "features": n.features} for n in g.nodes.values()],
        "edges": [{"source": e.source, "target": e.target, "type": e.type}
                  for e in g.edges],
    }


def convert_dataset(split: str = "validation", max_tasks: int = None,
                    out_path: str = None) -> list[dict]:
    """Convert DecomposedHarm split to list of serializable graph dicts."""
    from datasets import load_dataset

    ds = load_dataset('YuehHanChen/DecomposedHarm', split=split,
                      token=os.environ.get("HF_TOKEN"))
    data = [dict(d) for d in ds]

    agents = data  # use ALL task types, not just agent
    if max_tasks:
        agents = agents[:max_tasks]

    graphs = []
    for i, task in enumerate(agents):
        d = task_to_dict(task, i)
        if d:
            graphs.append(d)

    if out_path and graphs:
        out_dir = Path(out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for g in graphs:
                f.write(json.dumps(g) + "\n")
        print(f"  Saved {len(graphs)} graphs to {out_path}")

    return graphs


def main():
    print("=" * 60)
    print("DecomposedHarm → Behavior Graph Converter")
    print("=" * 60)

    # Convert all splits
    for split, max_n in [("validation", 20), ("train", 100), ("test", 20)]:
        out = f"LOGS/decomp_graphs_{split}.jsonl"
        graphs = convert_dataset(split, max_tasks=max_n, out_path=out)
        if not graphs:
            continue

        total_n = sum(g["num_nodes"] for g in graphs)
        total_e = sum(g["num_edges"] for g in graphs)
        types_used = set()
        for g in graphs:
            for n in g["nodes"]:
                types_used.add(n["type"])
        harmful = sum(1 for g in graphs if g["label"] == 1)

        print(f"\n  {split}: {len(graphs)} graphs")
        print(f"    Harmful: {harmful}, Benign: {len(graphs)-harmful}")
        print(f"    Avg nodes: {total_n/len(graphs):.1f}, edges: {total_e/len(graphs):.1f}")
        print(f"    Types: {sorted(types_used)}")


if __name__ == "__main__":
    main()
