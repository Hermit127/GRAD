#!/usr/bin/env python3
"""Compare GNN vs AgentTrust-style LLM vs NeuroTaint vs L1 on MemoryGraft data."""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "ref" / "dataset"
MODEL_DIR = Path(__file__).resolve().parent.parent / "code" / "models"
KEY_FILE = Path(__file__).resolve().parent.parent / "ref/MINJA/rap/OpenAI_api_key.txt"


# ==== Baseline 1: GNN (ours) ====
def baseline_gnn(graphs: list[dict]) -> list[int]:
    import torch
    from train_gnn import (WindowGINE, _graph_to_full_data, extract_window,
                           WINDOW_SIZE)
    ckpt = torch.load(MODEL_DIR / "gnn_window_classifier.pt", map_location="cpu")
    cfg = ckpt["config"]
    model = WindowGINE(in_dim=cfg["in_dim"], edge_dim=cfg["edge_dim"],
                       hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"], heads=4)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    preds = []
    for g in graphs:
        fd = _graph_to_full_data(g)
        ws = [extract_window(fd, s, WINDOW_SIZE) for s in range(max(0, fd.x.size(0) - 1))]
        ws = [w for w in ws if w is not None]
        if not ws:
            preds.append(0); continue
        probs = [torch.sigmoid(model(w.x, w.edge_index, w.edge_attr,
                       torch.zeros(w.x.size(0), dtype=torch.long))).item() for w in ws]
        preds.append(1 if max(probs) > 0.5 else 0)
    return preds


# ==== Baseline 2: AgentTrust (rule-only, actual source) ====
NODE_TO_AGENTTRUST = {
    "shell_exec": "shell_command", "code_exec": "code_execution",
    "tool_execute": "code_execution", "file_read": "file_read",
    "file_write": "file_write", "file_delete": "file_delete",
    "web_request": "network_request", "web_search": "network_request",
    "web_download": "network_request", "account_access": "credential_access",
    "database_query": "database_query", "information_search": "file_read",
    "data_aggregate": "file_read", "memory_write": "file_write",
    "memory_read": "file_read", "memory_reflect": "file_read",
    "version_control": "file_write", "agent_spawn": "api_call",
    "agent_message": "api_call", "email_send": "api_call",
    "message_send": "api_call", "social_media": "api_call",
    "document_create": "file_write", "prompt_inject": "unknown",
    "system_config": "system_config",
}

def baseline_agenttrust(graphs: list[dict]) -> list[int]:
    """Use actual AgentTrust TrustInterceptor.verify() on each action.

    Per-session chain semantics: AgentTrust's SessionTracker keeps a single
    global action history and matches RiskChain patterns greedily in order,
    so one interceptor instance per graph (session isolation) is required.
    Nodes are fed in graph order to preserve the per-session action sequence.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref" / "AgentTrust"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref" / "AgentTrust" / "src"))
    from agent_trust.core.interceptor import TrustInterceptor
    from agent_trust.core.types import Action, ActionType
    from datetime import datetime, timezone

    preds = []
    for g in graphs:
        interceptor = TrustInterceptor()
        blocked = False
        for n in g["nodes"]:
            at_type = ActionType(NODE_TO_AGENTTRUST.get(n["type"], "unknown"))
            subtask = n.get("features", {}).get("subtask", "")[:200]
            action = Action(
                action_type=at_type, tool_name=n["type"],
                description=subtask, raw_content=subtask,
                session_id=g["task_id"],
                timestamp=datetime.now(timezone.utc),
            )
            try:
                report = interceptor.verify(action)
                if report.verdict.value in ("block", "warn"):  # lowercase (TrustVerdict enum)
                    blocked = True
                    break
            except Exception:
                pass
        preds.append(1 if blocked else 0)
    return preds


# ==== Baseline 3: NeuroTaint-style (memory taint propagation) ====
# Taint source is inferred from node text (poisoned/malicious markers), not a
# nonexistent "taint" field. Propagates memory_write -> memory_read -> exec.
TAINT_MARKERS = (
    "poison", "malicious", "backdoor", "steal", "exfil", "leak",
    "bypass", "remote", "sanitizer", "curl", "wget", "payload",
)


def baseline_neurotaint(graphs: list[dict]) -> list[int]:
    """NeuroTaint-style: taint seeds from memory_write content markers, then
    propagate taint along data_flow / writes_to / reads_from edges until it
    reaches a dangerous exec sink (shell_exec / code_exec / tool_execute).

    Prior bug: propagation stopped at memory_read and the final check only
    tested whether the tainted node itself was an exec type, so the verdict
    was always False (taint never reached the sink).
    """
    preds = []
    for g in graphs:
        # Build adjacency (out-edges per node)
        adj = {}
        edge_types = {}
        for e in g.get("edges", []):
            adj.setdefault(e["source"], []).append(e["target"])
            edge_types[(e["source"], e["target"])] = e.get("type", "")

        # Taint seeds: memory_write whose text matches a marker
        tainted = set()
        for n in g["nodes"]:
            if n.get("type") == "memory_write":
                text = str(n.get("features", {}).get("subtask", "")).lower()
                if any(marker in text for marker in TAINT_MARKERS):
                    tainted.add(n["id"])

        # Propagate taint along taint-capable edges until fixpoint
        if tainted:
            frontier = list(tainted)
            while frontier:
                u = frontier.pop()
                for v in adj.get(u, []):
                    if (u, v) in edge_types and edge_types[(u, v)] in ("data_flow", "writes_to", "reads_from"):
                        if v not in tainted:
                            tainted.add(v)
                            frontier.append(v)

        # Sink check: any exec node reached by taint
        dangerous = any(
            n["id"] in tainted and n.get("type") in ("shell_exec", "code_exec", "tool_execute")
            for n in g["nodes"]
        )
        preds.append(1 if dangerous else 0)
    return preds


# ==== Baseline 4: Regex L1 ====
def baseline_l1_regex(graphs: list[dict]) -> list[int]:
    import re
    L1 = re.compile(r"curl\b|wget\b|\|\s*bash|rm\s+-rf|chmod\s+777|sudo\b|"
                    r"/etc/passwd|eval\(|exec\(|os\.system|subprocess|"
                    r"pip install.*--no-verify", re.I)
    preds = []
    for g in graphs:
        hit = False
        for n in g["nodes"]:
            text = n.get("features", {}).get("subtask", "")
            if L1.search(text):
                hit = True; break
        preds.append(1 if hit else 0)
    return preds


# ==== Main ====
def main():
    graphs = [json.loads(l) for l in open(DATA_DIR / "memorygraft_augmented.jsonl") if l.strip()]

    # Seed-level split (same as training)
    from collections import defaultdict
    def seed_id(tid): return tid.split('_v')[0]
    att_by_seed = defaultdict(list); ben_by_seed = defaultdict(list)
    for g in graphs:
        if g['label']==1: att_by_seed[seed_id(g['task_id'])].append(g)
        else: ben_by_seed[seed_id(g['task_id'])].append(g)

    att_seeds = sorted(att_by_seed.keys()); ben_seeds = sorted(ben_by_seed.keys())
    import random; random.seed(42)
    random.shuffle(att_seeds); random.shuffle(ben_seeds)

    test_g = []
    for s in att_seeds[8:]: test_g.extend(att_by_seed[s])
    for s in ben_seeds[80:]: test_g.extend(ben_by_seed[s])

    labels = [g["label"] for g in test_g]

    print(f"Test: {len(test_g)} graphs ({sum(labels)} att / {len(test_g)-sum(labels)} ben)", flush=True)

    baselines = [
        ("GNN (ours)", baseline_gnn),
        ("AgentTrust (source)", baseline_agenttrust),
        ("NeuroTaint", baseline_neurotaint),
        ("L1 Regex", baseline_l1_regex),
    ]

    results = {}
    for name, fn in baselines:
        print(f"\n=== {name} ===", flush=True)
        preds = fn(test_g)
        results[name] = {
            "acc": accuracy_score(labels, preds),
            "pre": precision_score(labels, preds, zero_division=0),
            "rec": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }
        m = results[name]
        print(f"  Acc={m['acc']:.4f}  Pre={m['pre']:.4f}  Rec={m['rec']:.4f}  F1={m['f1']:.4f}", flush=True)

    print(f"\n{'='*60}")
    print(f"{'Method':20s} {'Acc':>8s} {'Pre':>8s} {'Rec':>8s} {'F1':>8s}")
    print(f"{'='*60}")
    for name in [x[0] for x in baselines]:
        m = results[name]
        print(f"{name:20s} {m['acc']:8.4f} {m['pre']:8.4f} {m['rec']:8.4f} {m['f1']:8.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
