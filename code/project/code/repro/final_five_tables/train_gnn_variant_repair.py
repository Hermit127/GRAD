#!/usr/bin/env python3
"""Table 5 Panel A ablation trainer.

Grad-repair copy 2026-09-21: only addition is --max-node-chars (rebinds the
module global before datasets are built); frozen queue code is not touched.

Surgically derived from code/train_gnn_v2.py (unified GRAD backbone). The
`--variant full` path is a code-identical copy of the original data builder and
model; `test_train_gnn_ablate.py` asserts tensor-identity for full vs original.
Every variant removes information at *training input construction* time (not by
test-time perturbation) and is retrained from scratch on the frozen splits.

Variants (exact removal scope is recorded in the result JSON `ablation` block):
  full                 : unchanged unified model.
  text_only            : node char text + event position only; all edges and
                         edge descriptions removed; permission vector and
                         handcrafted keyword features removed.
  structure_only       : permission vector + position + node-type one-hot +
                         rebuilt edge topology with type one-hot and confidence;
                         node/edge CharCNN and keyword features removed.
                         NOTE: edge topology/types are produced by deterministic
                         rules over visible text by the shared builder, so this
                         is a *non-text-feature* variant, not text-blind.
  chrono_only          : node inputs unchanged; only the chronological
                         sequence_next chain is kept (directed i->i+1 in stable
                         turn order, no reverse edges, no self-loops); all
                         dependency edges/descriptions/types/confidence removed.
  no_relation_types    : node inputs and edge topology/descriptions/confidence
                         unchanged; relation-type embedding unified to one
                         category. This removes the explicit type label only.
  no_keyword_features  : the 9 handcrafted CONTENT_KEYWORDS flags are removed
                         from node features; raw text, structure and edge
                         builder unchanged.
  no_type_semantics    : appendix variant; type embedding unified AND edge
                         confidence zeroed (full explicit type semantics gone).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

_REPO_CODE = Path(__file__).resolve().parents[2]
if str(_REPO_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_CODE))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, TransformerConv, global_add_pool

from train_gnn import (
    CONTENT_KEYWORDS,
    DEFAULT_PERM,
    PERMISSION,
    TOOL_TO_PERM,
    content_features,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "ref" / "dataset"
OUT_DIR = Path(__file__).resolve().parent.parent / "LOGS"
MODEL_DIR = Path(__file__).resolve().parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V2_EDGE_TYPES = [
    "sequence_next",
    "data_flow",
    "control_trigger",
    "memory_store",
    "memory_retrieve",
    "cross_turn",
    "sink_trigger",
    "spawns",
    "delegates_message",
    "writes_to",
]
EDGE2IDX = {t: i for i, t in enumerate(V2_EDGE_TYPES)}

SOURCE_TYPES = {
    "web_request", "web_search", "web_download", "information_search",
    "file_read", "database_query", "account_access", "prompt_inject",
}
INTERMEDIATE_TYPES = {
    "memory_write", "memory_read", "file_write", "data_aggregate",
    "document_create", "agent_message",
}
SINK_TYPES = {
    "shell_exec", "code_exec", "tool_execute", "email_send", "message_send",
    "social_media", "document_create", "account_access", "system_config",
    "web_request", "file_write", "file_delete", "agent_spawn",
}

DATA_FLOW_RE = re.compile(
    r"\b(use|using|based on|from|with|retriev|extract|gather|collect|compile|download|attach|"
    r"include|reference|process|submit|send|post|update|modify)\b",
    re.I,
)
CONTROL_RE = re.compile(r"\b(trigger|schedule|execute|run|launch|deploy|start|restart|enable|disable|automate)\b", re.I)
SINK_RE = re.compile(
    r"\b(send|post|execute|run|delete|modify|redirect|transfer|upload|install|disable|schedule|"
    r"impersonat|fake|fraud|phish|backdoor|credential|payment|invoice)\b",
    re.I,
)

MAX_NODE_CHARS = 192
MAX_EDGE_CHARS = 160
CHAR_VOCAB = 128

ABLATION_VARIANTS = [
    "full",
    "text_only",
    "structure_only",
    "chrono_only",
    "no_relation_types",
    "no_keyword_features",
    "no_type_semantics",
]
VARIANT = "full"

VARIANT_CONTRACTS: dict[str, dict] = {
    "full": {
        "node_text_char_cnn": True, "edge_text_char_cnn": True,
        "permission_vector": True, "position": True, "keyword_features": True,
        "node_type_onehot": False, "edges": "rebuilt_full",
        "edge_type_embedding": True, "edge_confidence": True,
    },
    "text_only": {
        "node_text_char_cnn": True, "edge_text_char_cnn": False,
        "permission_vector": False, "position": True, "keyword_features": False,
        "node_type_onehot": False, "edges": "none",
        "edge_type_embedding": False, "edge_confidence": False,
    },
    "structure_only": {
        "node_text_char_cnn": False, "edge_text_char_cnn": False,
        "permission_vector": True, "position": True, "keyword_features": False,
        "node_type_onehot": True, "edges": "rebuilt_full",
        "edge_type_embedding": True, "edge_confidence": True,
    },
    "chrono_only": {
        "node_text_char_cnn": True, "edge_text_char_cnn": True,
        "permission_vector": True, "position": True, "keyword_features": True,
        "node_type_onehot": False, "edges": "sequence_next_only",
        "edge_type_embedding": True, "edge_confidence": True,
        "note": "edge rationale for sequence_next is the constant 'adjacent steps'",
    },
    "no_relation_types": {
        "node_text_char_cnn": True, "edge_text_char_cnn": True,
        "permission_vector": True, "position": True, "keyword_features": True,
        "node_type_onehot": False, "edges": "rebuilt_full",
        "edge_type_embedding": False, "edge_confidence": True,
        "note": "edge descriptions/confidence retained -> type-label ablation only",
    },
    "no_keyword_features": {
        "node_text_char_cnn": True, "edge_text_char_cnn": True,
        "permission_vector": True, "position": True, "keyword_features": False,
        "node_type_onehot": False, "edges": "rebuilt_full",
        "edge_type_embedding": True, "edge_confidence": True,
        "note": "edge-builder regexes stay (they define topology, shared with full)",
    },
    "no_type_semantics": {
        "node_text_char_cnn": True, "edge_text_char_cnn": True,
        "permission_vector": True, "position": True, "keyword_features": True,
        "node_type_onehot": False, "edges": "rebuilt_full",
        "edge_type_embedding": False, "edge_confidence": False,
        "note": "appendix variant: type embedding unified and confidence zeroed",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def node_text(node: dict) -> str:
    return node.get("features", {}).get("subtask", "") or node.get("type", "")


def char_ids(text: str, max_len: int) -> list[int]:
    text = (text or "")[:max_len]
    ids = [min(ord(ch), CHAR_VOCAB - 1) for ch in text]
    return ids + [0] * (max_len - len(ids))


def add_edge(edges: list[dict], seen: set[tuple[str, str, str]], source: str, target: str,
             etype: str, confidence: float, rationale: str) -> None:
    key = (source, target, etype)
    if key in seen:
        return
    seen.add(key)
    edges.append({
        "source": source,
        "target": target,
        "type": etype,
        "confidence": float(confidence),
        "rationale": rationale[:300],
    })


def rebuild_v2_edges(g: dict) -> list[dict]:
    nodes = sorted(g.get("nodes", []), key=lambda n: n.get("turn", 0))
    edges: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for prev, cur in zip(nodes[:-1], nodes[1:]):
        ptxt, ctxt = node_text(prev), node_text(cur)
        add_edge(edges, seen, prev["id"], cur["id"], "sequence_next", 1.0, "adjacent steps")

        if prev["type"] == "memory_write":
            add_edge(edges, seen, prev["id"], cur["id"], "memory_store", 0.85, ptxt)
        if cur["type"] == "memory_read":
            add_edge(edges, seen, prev["id"], cur["id"], "memory_retrieve", 0.85, ctxt)
        if prev["type"] in ("file_write", "document_create") and cur["type"] in SINK_TYPES:
            add_edge(edges, seen, prev["id"], cur["id"], "writes_to", 0.7, ptxt + " -> " + ctxt)
        if prev["type"] == "agent_spawn":
            add_edge(edges, seen, prev["id"], cur["id"], "spawns", 0.75, ptxt)
        if prev["type"] == "agent_message" or cur["type"] == "agent_message":
            add_edge(edges, seen, prev["id"], cur["id"], "delegates_message", 0.65, ptxt + " -> " + ctxt)

        if DATA_FLOW_RE.search(ctxt) or prev["type"] in SOURCE_TYPES | INTERMEDIATE_TYPES:
            add_edge(edges, seen, prev["id"], cur["id"], "data_flow", 0.55, ptxt + " -> " + ctxt)
        if CONTROL_RE.search(ctxt) or cur["type"] in ("shell_exec", "code_exec", "tool_execute", "system_config"):
            add_edge(edges, seen, prev["id"], cur["id"], "control_trigger", 0.55, ptxt + " -> " + ctxt)
        if cur["type"] in SINK_TYPES and SINK_RE.search(ctxt):
            add_edge(edges, seen, prev["id"], cur["id"], "sink_trigger", 0.6, ctxt)

    # Longer-range candidate causal edges. These are path-proposer edges,
    # not ground-truth causal labels.
    for i, src in enumerate(nodes):
        stxt = node_text(src)
        for dst in nodes[i + 2:min(i + 6, len(nodes))]:
            dtxt = node_text(dst)
            if src["type"] == "memory_write" and dst["type"] == "memory_read":
                add_edge(edges, seen, src["id"], dst["id"], "memory_retrieve", 0.9, stxt + " -> " + dtxt)
                add_edge(edges, seen, src["id"], dst["id"], "cross_turn", 0.55, "delayed memory use")
            if src["type"] in SOURCE_TYPES and dst["type"] in INTERMEDIATE_TYPES and DATA_FLOW_RE.search(dtxt):
                add_edge(edges, seen, src["id"], dst["id"], "data_flow", 0.45, stxt + " -> " + dtxt)
            if src["type"] in INTERMEDIATE_TYPES and dst["type"] in SINK_TYPES and SINK_RE.search(dtxt):
                add_edge(edges, seen, src["id"], dst["id"], "sink_trigger", 0.5, stxt + " -> " + dtxt)

    return edges


def has_candidate_path(nodes: list[dict], edges: list[dict], node_ids: set[str]) -> bool:
    node_by_id = {n["id"]: n for n in nodes if n["id"] in node_ids}
    if not node_by_id:
        return False
    adj: dict[str, list[tuple[str, str]]] = {}
    for e in edges:
        if e["source"] in node_ids and e["target"] in node_ids:
            adj.setdefault(e["source"], []).append((e["target"], e["type"]))

    starts = [nid for nid, n in node_by_id.items() if n["type"] in SOURCE_TYPES or n["type"] in INTERMEDIATE_TYPES]
    for start in starts:
        queue = [(start, 0, False)]
        seen = {(start, False)}
        while queue:
            cur, depth, has_causal = queue.pop(0)
            n = node_by_id[cur]
            if depth >= 1 and n["type"] in SINK_TYPES and has_causal:
                return True
            if depth >= 4:
                continue
            for nxt, etype in adj.get(cur, []):
                causal = has_causal or etype in {
                    "data_flow", "control_trigger", "memory_retrieve", "sink_trigger",
                    "writes_to", "spawns", "delegates_message",
                }
                state = (nxt, causal)
                if state not in seen:
                    seen.add(state)
                    queue.append((nxt, depth + 1, causal))
    return False


def build_node_type_vocab(path: str | Path) -> list[str]:
    """Node-type vocabulary from the TRAIN split only (frozen, no val/test leakage)."""
    types: set[str] = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                for n in json.loads(line).get("nodes", []):
                    types.add(str(n.get("type", "")))
    return sorted(types)


def variant_edges(g: dict) -> list[dict]:
    if VARIANT == "text_only":
        return []
    edges = rebuild_v2_edges(g)
    if VARIANT == "chrono_only":
        return [e for e in edges if e["type"] == "sequence_next"]
    return edges


def graph_to_v2_data(g: dict, node_type_vocab: list[str] | None = None) -> Data:
    nodes = sorted(g.get("nodes", []), key=lambda n: n.get("turn", 0))
    edges = variant_edges(g)
    n_nodes = len(nodes)
    if VARIANT == "text_only":
        base_dim = 1
    elif VARIANT == "no_keyword_features":
        base_dim = len(PERMISSION) + 1
    elif VARIANT == "structure_only":
        base_dim = len(PERMISSION) + 1 + (len(node_type_vocab) if node_type_vocab else 0)
    else:
        base_dim = len(PERMISSION) + 1 + len(CONTENT_KEYWORDS)
    x_base = torch.zeros(n_nodes, base_dim)
    node_chars = torch.zeros(n_nodes, MAX_NODE_CHARS, dtype=torch.long)
    id_to_idx: dict[str, int] = {}
    type_index = {t: i for i, t in enumerate(node_type_vocab or [])}
    use_text = VARIANT != "structure_only"
    use_keywords = VARIANT not in ("text_only", "structure_only", "no_keyword_features")
    use_perm = VARIANT not in ("text_only",)

    for i, n in enumerate(nodes):
        if use_perm:
            perm = TOOL_TO_PERM.get(n["type"], DEFAULT_PERM)
            x_base[i, :len(PERMISSION)] = torch.tensor(perm, dtype=torch.float32)
            x_base[i, len(PERMISSION)] = n.get("turn", i) / max(n_nodes, 1)
        else:
            x_base[i, 0] = n.get("turn", i) / max(n_nodes, 1)
        if use_keywords:
            x_base[i, len(PERMISSION) + 1:] = torch.tensor(content_features(node_text(n)), dtype=torch.float32)
        if VARIANT == "structure_only":
            ti = type_index.get(str(n.get("type", "")))
            if ti is not None:
                x_base[i, len(PERMISSION) + 1 + ti] = 1.0
        if use_text:
            node_chars[i] = torch.tensor(char_ids(node_text(n), MAX_NODE_CHARS), dtype=torch.long)
        id_to_idx[n["id"]] = i

    if VARIANT == "text_only":
        edges = []
    edge_index, edge_attr, edge_chars = [], [], []
    for e in edges:
        src, tgt = id_to_idx.get(e["source"]), id_to_idx.get(e["target"])
        if src is None or tgt is None:
            continue
        et = torch.zeros(len(V2_EDGE_TYPES) + 1, dtype=torch.float32)
        if VARIANT in ("no_relation_types", "no_type_semantics"):
            et[0] = 1.0
        else:
            et[EDGE2IDX.get(e["type"], 0)] = 1.0
        et[-1] = 0.0 if VARIANT == "no_type_semantics" else float(e.get("confidence", 1.0))
        edge_index.append([src, tgt])
        edge_attr.append(et)
        if VARIANT == "structure_only":
            edge_chars.append(torch.zeros(MAX_EDGE_CHARS, dtype=torch.long))
        else:
            edge_chars.append(torch.tensor(char_ids(e.get("rationale", e["type"]), MAX_EDGE_CHARS), dtype=torch.long))

    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.stack(edge_attr) if edge_attr else torch.zeros(0, len(V2_EDGE_TYPES) + 1)
    ec = torch.stack(edge_chars) if edge_chars else torch.zeros(0, MAX_EDGE_CHARS, dtype=torch.long)
    return Data(
        x=torch.empty(n_nodes, 0),
        x_base=x_base,
        node_chars=node_chars,
        edge_index=ei,
        edge_attr_base=ea,
        edge_chars=ec,
        y=torch.tensor([int(g.get("label", 0))], dtype=torch.float32),
        y_graph=torch.tensor([int(g.get("label", 0))], dtype=torch.float32),
        num_nodes_v2=n_nodes,
        num_nodes=n_nodes,
    )


def extract_window(data: Data, start: int, k: int, graph_label: int) -> Data | None:
    n = data.x_base.size(0)
    end = min(start + k, n)
    if end - start < 2:
        return None
    old_nodes = list(range(start, end))
    old2new = {old: new for new, old in enumerate(old_nodes)}
    node_ids = {str(i) for i in old_nodes}
    real_node_ids = set()

    sub_edges, sub_ea, sub_ec = [], [], []
    for j in range(data.edge_index.size(1)):
        s, t = data.edge_index[0, j].item(), data.edge_index[1, j].item()
        if s in old2new and t in old2new:
            sub_edges.append([old2new[s], old2new[t]])
            sub_ea.append(data.edge_attr_base[j])
            sub_ec.append(data.edge_chars[j])

    sub_ei = torch.tensor(sub_edges, dtype=torch.long).t().contiguous() if sub_edges else torch.zeros(2, 0, dtype=torch.long)
    sub_ea = torch.stack(sub_ea) if sub_ea else torch.zeros(0, len(V2_EDGE_TYPES) + 1)
    sub_ec = torch.stack(sub_ec) if sub_ec else torch.zeros(0, MAX_EDGE_CHARS, dtype=torch.long)

    # Weak path label: positive graphs only get positive windows when the
    # path proposer finds a causal/sink structure inside the window.
    nodes_marker = getattr(data, "nodes_marker", None)
    edges_marker = getattr(data, "edges_marker", None)
    if nodes_marker is not None and edges_marker is not None:
        real_node_ids = {nodes_marker[i]["id"] for i in old_nodes}
        win_has_path = has_candidate_path(nodes_marker, edges_marker, real_node_ids)
    else:
        win_has_path = sub_ei.size(1) > 0
    label = 0.95 if graph_label == 1 and win_has_path else 0.05

    return Data(
        x=torch.empty(len(old_nodes), 0),
        x_base=data.x_base[old_nodes],
        node_chars=data.node_chars[old_nodes],
        edge_index=sub_ei,
        edge_attr_base=sub_ea,
        edge_chars=sub_ec,
        y=torch.tensor([label], dtype=torch.float32),
        num_nodes=len(old_nodes),
    )


class V2GraphDataset(Dataset):
    def __init__(self, path: str | Path, node_type_vocab: list[str] | None = None):
        super().__init__()
        self.graphs: list[dict] = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.graphs.append(json.loads(line))
        self.data = [graph_to_v2_data(g, node_type_vocab) for g in self.graphs]
        self.sample_weights = [1.0 for _ in self.graphs]

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int) -> Data:
        d = self.data[idx]
        d.sample_weight = torch.tensor([self.sample_weights[idx]], dtype=torch.float32)
        d.sample_idx = torch.tensor([idx], dtype=torch.long)
        return d

    def set_hard_benign(self, indices: set[int], weight: float) -> None:
        self.sample_weights = [
            weight if i in indices and int(g.get("label", 0)) == 0 else 1.0
            for i, g in enumerate(self.graphs)
        ]


class V2WindowDataset(Dataset):
    def __init__(self, path: str | Path, k: int, train: bool, stride: int):
        super().__init__()
        self.k = k
        self.train = train
        self.stride = stride if train else 1
        self.graphs: list[dict] = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.graphs.append(json.loads(line))
        self.full_data = []
        self.graph_labels = []
        self.v2_edges = []
        for g in self.graphs:
            d = graph_to_v2_data(g)
            d.nodes_marker = sorted(g.get("nodes", []), key=lambda n: n.get("turn", 0))
            d.edges_marker = rebuild_v2_edges(g)
            self.full_data.append(d)
            self.graph_labels.append(int(g.get("label", 0)))
            self.v2_edges.append(d.edges_marker)

        self.window_index: list[tuple[int, int, float]] = []
        self._graph_nw = []
        self._cum_nw = [0]
        for gi, d in enumerate(self.full_data):
            n = d.x_base.size(0)
            count = 0
            step = self.stride if train else 1
            for start in range(0, n - 1, step):
                w = extract_window(d, start, self.k, self.graph_labels[gi])
                if w is None:
                    continue
                self.window_index.append((gi, start, float(w.y.item())))
                count += 1
            self._graph_nw.append(count if not train else max(0, n - 1))
            self._cum_nw.append(self._cum_nw[-1] + max(0, n - 1))
        if train:
            random.shuffle(self.window_index)

    def len(self) -> int:
        return len(self.window_index) if self.train else self._cum_nw[-1]

    def get(self, idx: int) -> Data:
        if self.train:
            gi, start, _ = self.window_index[idx]
            return extract_window(self.full_data[gi], start, self.k, self.graph_labels[gi])
        gi = 0
        for lo, hi in zip(self._cum_nw[:-1], self._cum_nw[1:]):
            if lo <= idx < hi:
                start = idx - lo
                return extract_window(self.full_data[gi], start, self.k, self.graph_labels[gi])
            gi += 1
        return extract_window(self.full_data[0], 0, self.k, self.graph_labels[0])


class CharCNN(torch.nn.Module):
    def __init__(self, emb_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.emb = torch.nn.Embedding(CHAR_VOCAB, emb_dim, padding_idx=0)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(emb_dim, out_dim, kernel_size=k, padding=0)
            for k in (3, 5, 7)
        ])
        self.proj = torch.nn.Linear(out_dim * len(self.convs), out_dim)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        if chars.numel() == 0:
            return torch.zeros(chars.size(0), self.proj.out_features, device=chars.device)
        x = self.emb(chars).transpose(1, 2)
        outs = []
        for conv in self.convs:
            y = F.gelu(conv(x))
            outs.append(F.max_pool1d(y, kernel_size=y.size(-1)).squeeze(-1))
        return self.proj(torch.cat(outs, dim=-1))


def encode_char_rows(encoder: CharCNN, chars: torch.Tensor) -> torch.Tensor:
    """Bucket rows by text length without changing max-pooled convolutions."""
    if chars.size(0) == 0:
        return encoder(chars)
    max_kernel = max(conv.kernel_size[0] for conv in encoder.convs)
    lengths = (chars != 0).sum(dim=1).detach().cpu().tolist()
    buckets: dict[int, list[int]] = {}
    for row, length in enumerate(lengths):
        target = min(chars.size(1), int(length) + max_kernel)
        keep = max_kernel
        while keep < target:
            keep *= 2
        keep = min(chars.size(1), keep)
        buckets.setdefault(keep, []).append(row)
    encoded = torch.zeros(
        chars.size(0), encoder.proj.out_features, device=chars.device, dtype=torch.float32
    )
    for keep, rows in sorted(buckets.items()):
        indices = torch.tensor(rows, dtype=torch.long, device=chars.device)
        values = encoder(chars.index_select(0, indices)[:, :keep])
        encoded = encoded.index_copy(0, indices, values)
    return encoded


class CausalCharGNN(torch.nn.Module):
    def __init__(self, node_base_dim: int, edge_base_dim: int,
                 hidden_dim: int = 192, char_dim: int = 64,
                 dropout: float = 0.35, heads: int = 4):
        super().__init__()
        self.node_char = CharCNN(out_dim=char_dim)
        self.edge_char = CharCNN(out_dim=char_dim)
        self.node_proj = torch.nn.Linear(node_base_dim + char_dim, hidden_dim)
        self.edge_proj = torch.nn.Linear(edge_base_dim + char_dim, hidden_dim)
        self.gine = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ), edge_dim=hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.tf1 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                   edge_dim=hidden_dim, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.tf2 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                   edge_dim=hidden_dim, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.attn = torch.nn.Linear(hidden_dim, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, x_base, node_chars, edge_index, edge_attr_base, edge_chars, batch):
        nc = encode_char_rows(self.node_char, node_chars)
        x = self.node_proj(torch.cat([x_base, nc], dim=-1))
        if edge_attr_base.size(0) > 0:
            ec = encode_char_rows(self.edge_char, edge_chars)
            edge_attr = self.edge_proj(torch.cat([edge_attr_base, ec], dim=-1))
        else:
            edge_attr = torch.zeros(0, x.size(-1), device=x.device)

        residual = x
        x = self.gine(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual

        residual = x
        x = self.tf1(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual

        residual = x
        x = self.tf2(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual

        a = self.attn(x).sigmoid()
        pooled = global_add_pool(x * a, batch) / global_add_pool(a, batch).clamp(min=1e-8)
        return self.classifier(pooled).squeeze(-1)


class CausalCharGNNNoText(torch.nn.Module):
    """Same stack as CausalCharGNN with the char encoders removed (structure_only).

    Node inputs are the non-text feature vector (permission + position + node
    type one-hot); edge inputs are the type one-hot + confidence. Char tensors
    are accepted for interface compatibility but never read.
    """

    def __init__(self, node_base_dim: int, edge_base_dim: int,
                 hidden_dim: int = 192, dropout: float = 0.35, heads: int = 4):
        super().__init__()
        self.node_proj = torch.nn.Linear(node_base_dim, hidden_dim)
        self.edge_proj = torch.nn.Linear(edge_base_dim, hidden_dim)
        self.gine = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ), edge_dim=hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.tf1 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                   edge_dim=hidden_dim, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.tf2 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                   edge_dim=hidden_dim, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.attn = torch.nn.Linear(hidden_dim, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, x_base, node_chars, edge_index, edge_attr_base, edge_chars, batch):
        x = self.node_proj(x_base)
        if edge_attr_base.size(0) > 0:
            edge_attr = self.edge_proj(edge_attr_base)
        else:
            edge_attr = torch.zeros(0, x.size(-1), device=x.device)

        residual = x
        x = self.gine(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual

        residual = x
        x = self.tf1(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual

        residual = x
        x = self.tf2(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual

        a = self.attn(x).sigmoid()
        pooled = global_add_pool(x * a, batch) / global_add_pool(a, batch).clamp(min=1e-8)
        return self.classifier(pooled).squeeze(-1)


def move_batch(batch: Data) -> Data:
    return batch.to(DEVICE)


def batch_logits(model: CausalCharGNN, batch: Data) -> torch.Tensor:
    return model(
        batch.x_base,
        batch.node_chars,
        batch.edge_index,
        batch.edge_attr_base,
        batch.edge_chars,
        batch.batch,
    )


def compute_pos_weight(ds: V2WindowDataset) -> float:
    pos = sum(1 for _, _, y in ds.window_index if y > 0.5)
    neg = len(ds.window_index) - pos
    return neg / max(pos, 1)


def train_epoch(model, loader, optimizer, pos_weight: float) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch)
        optimizer.zero_grad()
        logits = batch_logits(model, batch)
        loss_vec = F.binary_cross_entropy_with_logits(
            logits, batch.y, pos_weight=torch.tensor(pos_weight, device=DEVICE),
            reduction="none")
        if hasattr(batch, "sample_weight"):
            loss_vec = loss_vec * batch.sample_weight.view_as(loss_vec)
        loss = loss_vec.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_window(model, loader) -> dict:
    model.eval()
    probs, labels = [], []
    total_bce, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch)
        logits = batch_logits(model, batch)
        total_bce += F.binary_cross_entropy_with_logits(logits, batch.y, reduction="sum").item()
        p = torch.sigmoid(logits)
        probs.extend(p.cpu().tolist())
        labels.extend((batch.y > 0.5).long().cpu().tolist())
        n += batch.num_graphs
    pred = [1 if p > 0.5 else 0 for p in probs]
    return {
        "bce": total_bce / max(n, 1),
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "auroc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5,
    }


@torch.no_grad()
def evaluate_graph(model, ds: V2WindowDataset) -> dict:
    model.eval()
    loader = DataLoader(ds, batch_size=64)
    probs = []
    for batch in loader:
        batch = move_batch(batch)
        probs.extend(torch.sigmoid(batch_logits(model, batch)).cpu().tolist())

    labels, scores = [], []
    for gi, g in enumerate(ds.graphs):
        s, e = ds._cum_nw[gi], ds._cum_nw[gi + 1]
        gs = max(probs[s:e]) if e > s else 0.0
        labels.append(int(g.get("label", 0)))
        scores.append(gs)
    pred = [1 if s > 0.5 else 0 for s in scores]
    benign_n = sum(1 for y in labels if y == 0)
    fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "auroc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5,
        "fpr": fp / max(benign_n, 1),
    }


def metrics_at_threshold(labels: list[int], scores: list[float], threshold: float) -> dict:
    pred = [1 if s >= threshold else 0 for s in scores]
    benign_n = sum(1 for y in labels if y == 0)
    fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "fpr": fp / max(benign_n, 1),
    }


def threshold_for_target_fpr(labels: list[int], scores: list[float], target_fpr: float) -> tuple[float, dict]:
    candidates = sorted(set(scores), reverse=True)
    candidates = [float("inf")] + candidates + [float("-inf")]
    best_threshold = candidates[0]
    best_metrics = metrics_at_threshold(labels, scores, best_threshold)
    for threshold in candidates:
        m = metrics_at_threshold(labels, scores, threshold)
        if m["fpr"] <= target_fpr + 1e-12:
            if (m["recall"], m["f1"], m["precision"]) > (
                best_metrics["recall"], best_metrics["f1"], best_metrics["precision"]
            ):
                best_threshold, best_metrics = threshold, m
    return best_threshold, best_metrics


def fixed_fpr_report(labels: list[int], scores: list[float], targets: list[float]) -> dict:
    return {
        f"fpr_{int(t * 100):02d}": threshold_for_target_fpr(labels, scores, t)[1]
        for t in targets
    }


@torch.no_grad()
def predict_loader(model, loader) -> tuple[list[float], list[int], list[int]]:
    model.eval()
    probs, labels, indices = [], [], []
    for batch in loader:
        batch = move_batch(batch)
        logits = batch_logits(model, batch)
        p = torch.sigmoid(logits)
        probs.extend(p.cpu().tolist())
        labels.extend((batch.y > 0.5).long().cpu().tolist())
        if hasattr(batch, "sample_idx"):
            indices.extend(batch.sample_idx.cpu().tolist())
        else:
            indices.extend(range(len(indices), len(indices) + batch.num_graphs))
    return probs, labels, indices


@torch.no_grad()
def evaluate_graph_loader(model, loader, fixed_fpr_targets: list[float] | None = None) -> dict:
    probs, labels, _ = predict_loader(model, loader)
    logits = torch.logit(torch.tensor(probs).clamp(1e-6, 1 - 1e-6))
    y = torch.tensor(labels, dtype=torch.float32)
    total_bce = F.binary_cross_entropy_with_logits(logits, y, reduction="mean").item()
    default_metrics = metrics_at_threshold(labels, probs, 0.5)
    result = {
        "bce": total_bce,
        **{k: v for k, v in default_metrics.items() if k != "threshold"},
        "threshold": 0.5,
        "auroc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5,
    }
    if fixed_fpr_targets:
        result["fixed_fpr"] = fixed_fpr_report(labels, probs, fixed_fpr_targets)
    return result


def mine_hard_benign(model, loader, threshold: float, top_k: int) -> set[int]:
    scores, labels, indices = predict_loader(model, loader)
    candidates = [
        (score, idx)
        for score, label, idx in zip(scores, labels, indices)
        if label == 0 and score >= threshold
    ]
    candidates.sort(reverse=True)
    if top_k > 0:
        candidates = candidates[:top_k]
    return {idx for _, idx in candidates}


def apply_threshold(labels: list[int], scores: list[float], threshold: float) -> dict:
    m = metrics_at_threshold(labels, scores, threshold)
    return {
        **m,
        "auroc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5,
    }


def audit_variant_inputs(train_graphs: list[dict], val_graphs: list[dict],
                         test_graphs: list[dict], node_type_vocab: list[str] | None) -> dict:
    """Input-level evidence that the variant removes/adds exactly what it claims."""
    vocab = set(node_type_vocab or [])
    audit: dict[str, dict] = {}
    for split, graphs in (("train", train_graphs), ("dev", val_graphs), ("test", test_graphs)):
        n_nodes = n_edges_raw = n_edges_used = n_char_nonzero = n_kw_nonzero = 0
        types_seen: set[str] = set()
        edge_types_used: set[str] = set()
        for g in graphs:
            nodes = g.get("nodes", [])
            n_nodes += len(nodes)
            n_edges_raw += len(g.get("edges", []))
            used = variant_edges(g)
            n_edges_used += len(used)
            for e in used:
                edge_types_used.add(e["type"])
            for n in nodes:
                types_seen.add(str(n.get("type", "")))
                n_char_nonzero += 1 if node_text(n).strip() else 0
                n_kw_nonzero += sum(content_features(node_text(n)))
        audit[split] = {
            "n_graphs": len(graphs),
            "n_pos": sum(1 for g in graphs if int(g.get("label", 0)) == 1),
            "n_nodes": n_nodes,
            "n_edges_raw_corpus": n_edges_raw,
            "n_edges_used": n_edges_used,
            "edge_types_used": sorted(edge_types_used),
            "node_text_nonempty": n_char_nonzero,
            "keyword_flags_present_in_text": n_kw_nonzero,
            "node_types_seen": len(types_seen),
            "node_types_unseen_in_train_vocab": (len(types_seen - vocab)
                                                 if node_type_vocab is not None else None),
        }
    contract = VARIANT_CONTRACTS[VARIANT]
    for split in audit:
        audit[split]["node_text_encoded"] = contract["node_text_char_cnn"]
        audit[split]["keyword_features_used"] = contract["keyword_features"]
        audit[split]["edge_text_encoded"] = contract["edge_text_char_cnn"]
    return {
        "variant": VARIANT,
        "contract": contract,
        "node_type_vocab_size": len(node_type_vocab) if node_type_vocab is not None else None,
        "splits": audit,
    }


def main() -> int:
    global VARIANT, MAX_NODE_CHARS
    parser = argparse.ArgumentParser(description="Train causal-edge CharCNN+GNN v2")
    parser.add_argument("--train", default=str(DATA_DIR / "decomp_graphs_train_mixed.jsonl"))
    parser.add_argument("--val", default=str(DATA_DIR / "decomp_graphs_validation_mixed.jsonl"))
    parser.add_argument("--test", default=str(DATA_DIR / "decomp_graphs_test_mixed.jsonl"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mode", choices=["graph", "window"], default="graph")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.10,
                        help="Primary dev FPR target for checkpoint selection.")
    parser.add_argument("--fixed-fpr-targets", default="0.05,0.10,0.20",
                        help="Comma-separated FPR targets for threshold reports.")
    parser.add_argument("--checkpoint-metric",
                        choices=["fixed_fpr_f1", "val_graph_f1", "val_bce"],
                        default="fixed_fpr_f1",
                        help="Validation metric used for checkpoint selection.")
    parser.add_argument("--hard-benign-weight", type=float, default=3.0)
    parser.add_argument("--hard-benign-top-k", type=int, default=128)
    parser.add_argument("--hard-benign-start-epoch", type=int, default=2)
    parser.add_argument("--out-dir", default=None,
                        help="Directory for the result JSON (default: LOGS). "
                             "Use a per-run directory to avoid concurrent file races.")
    parser.add_argument("--max-node-chars", type=int, default=MAX_NODE_CHARS,
                        help="char window override (grad-repair confirmation)")
    parser.add_argument("--variant", choices=ABLATION_VARIANTS, default="full",
                        help="Table 5 Panel A ablation variant (training-input removal).")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="Persist the selected checkpoint under out_dir "
                             "(default: no checkpoint, old code/models untouched).")
    args = parser.parse_args()

    VARIANT = args.variant
    MAX_NODE_CHARS = args.max_node_chars

    set_seed(args.seed)
    print("=== Causal-edge CharCNN+GNN v2 ===", flush=True)
    print(f"Device={DEVICE} seed={args.seed} epochs={args.epochs} window={args.window_size}", flush=True)
    fixed_targets = [float(x) for x in args.fixed_fpr_targets.split(",") if x.strip()]

    t0 = time.time()
    node_type_vocab: list[str] | None = None
    if args.variant == "structure_only":
        node_type_vocab = build_node_type_vocab(args.train)
        print(f"node_type_vocab(train-only) n={len(node_type_vocab)}", flush=True)
    if args.mode == "graph":
        train_ds = V2GraphDataset(args.train, node_type_vocab)
        val_ds = V2GraphDataset(args.val, node_type_vocab)
        test_ds = V2GraphDataset(args.test, node_type_vocab)
        print(f"Loaded ds in {time.time()-t0:.1f}s: train_graphs={train_ds.len()} "
              f"val_graphs={val_ds.len()} test_graphs={test_ds.len()}", flush=True)
        pos = sum(1 for g in train_ds.graphs if int(g.get("label", 0)) == 1)
        neg = train_ds.len() - pos
        pos_weight = neg / max(pos, 1)
    else:
        train_ds = V2WindowDataset(args.train, args.window_size, train=True, stride=args.stride)
        val_ds = V2WindowDataset(args.val, args.window_size, train=False, stride=1)
        test_ds = V2WindowDataset(args.test, args.window_size, train=False, stride=1)
        print(f"Loaded ds in {time.time()-t0:.1f}s: train_windows={train_ds.len()} "
              f"val_windows={val_ds.len()} test_windows={test_ds.len()}", flush=True)
        pos_weight = compute_pos_weight(train_ds)
    print(f"pos_weight={pos_weight:.3f}", flush=True)

    input_audit = audit_variant_inputs(train_ds.graphs, val_ds.graphs, test_ds.graphs,
                                       node_type_vocab) if args.mode == "graph" else {}

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    if args.variant == "text_only":
        node_base_dim = 1
    elif args.variant == "no_keyword_features":
        node_base_dim = len(PERMISSION) + 1
    elif args.variant == "structure_only":
        node_base_dim = len(PERMISSION) + 1 + len(node_type_vocab or [])
    else:
        node_base_dim = len(PERMISSION) + 1 + len(CONTENT_KEYWORDS)
    edge_base_dim = len(V2_EDGE_TYPES) + 1
    model_cls = CausalCharGNNNoText if args.variant == "structure_only" else CausalCharGNN
    model = model_cls(node_base_dim, edge_base_dim, hidden_dim=args.hidden_dim).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params={params:,} node_base={node_base_dim} edge_base={edge_base_dim}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    best_bce, best_bce_state, best_bce_epoch = float("inf"), None, 0
    best_fixed_score, best_fixed_state, best_fixed_epoch = -1.0, None, 0
    best_fixed_threshold, best_fixed_metrics = 0.5, {}
    best_graph_f1, best_graph_f1_state, best_graph_f1_epoch = -1.0, None, 0
    history = []
    hard_benign_indices: set[int] = set()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight)
        if args.mode == "graph":
            val_graph = evaluate_graph_loader(model, val_loader, fixed_targets)
            val_win = val_graph
        else:
            val_win = evaluate_window(model, val_loader)
            val_graph = evaluate_graph(model, val_ds)
        scheduler.step(val_win["bce"])
        fixed_key = f"fpr_{int(args.target_fpr * 100):02d}"
        fixed_metrics = val_graph.get("fixed_fpr", {}).get(fixed_key)
        fixed_score = -1.0
        fixed_threshold = 0.5
        if fixed_metrics:
            fixed_score = fixed_metrics["f1"]
            fixed_threshold = fixed_metrics["threshold"]

        if val_win["bce"] < best_bce:
            best_bce = val_win["bce"]
            best_bce_epoch = epoch
            best_bce_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if fixed_score > best_fixed_score:
            best_fixed_score = fixed_score
            best_fixed_epoch = epoch
            best_fixed_threshold = fixed_threshold
            best_fixed_metrics = fixed_metrics or {}
            best_fixed_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_graph["f1"] > best_graph_f1:
            best_graph_f1 = val_graph["f1"]
            best_graph_f1_epoch = epoch
            best_graph_f1_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        mined_count = 0
        if args.mode == "graph" and epoch >= args.hard_benign_start_epoch:
            hard_benign_indices = mine_hard_benign(
                model, train_eval_loader, threshold=fixed_threshold,
                top_k=args.hard_benign_top_k)
            train_ds.set_hard_benign(hard_benign_indices, args.hard_benign_weight)
            mined_count = len(hard_benign_indices)

        row = {
            "epoch": epoch,
            "train_loss": loss,
            "val_window": val_win,
            "val_graph": val_graph,
            "fixed_selection": {
                "target_fpr": args.target_fpr,
                "score": fixed_score,
                "threshold": fixed_threshold,
                "metrics": fixed_metrics,
            },
            "hard_benign": {
                "count": mined_count,
                "weight": args.hard_benign_weight if mined_count else 1.0,
            },
        }
        history.append(row)
        print(f"E{epoch:03d} loss={loss:.4f} val_bce={val_win['bce']:.4f} "
              f"val_win_f1={val_win['f1']:.3f} val_graph_f1={val_graph['f1']:.3f} "
              f"val_fpr={val_graph['fpr']:.3f} "
              f"fixed_{fixed_key}_f1={fixed_score:.3f} hard_benign={mined_count}", flush=True)
        if args.checkpoint_metric == "val_graph_f1":
            best_epoch_for_stop = best_graph_f1_epoch
        elif args.checkpoint_metric == "val_bce":
            best_epoch_for_stop = best_bce_epoch
        else:
            best_epoch_for_stop = best_fixed_epoch
        if epoch - best_epoch_for_stop >= args.patience:
            print(f"Early stop at epoch {epoch}; best_fixed={best_fixed_epoch} "
                  f"best_graph_f1={best_graph_f1_epoch} best_bce={best_bce_epoch}", flush=True)
            break

    selected_epoch = best_fixed_epoch
    selected_score = best_fixed_score
    if args.checkpoint_metric == "val_graph_f1":
        selected_state = best_graph_f1_state or best_fixed_state or best_bce_state
        selected_epoch = best_graph_f1_epoch
        selected_score = best_graph_f1
    elif args.checkpoint_metric == "val_bce":
        selected_state = best_bce_state or best_fixed_state
        selected_epoch = best_bce_epoch
        selected_score = best_bce
    else:
        selected_state = best_fixed_state or best_bce_state
    if selected_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in selected_state.items()})
    if args.mode == "graph":
        val_scores, val_labels, _ = predict_loader(model, val_loader)
        test_scores, test_labels, _ = predict_loader(model, test_loader)
        score_dump = {
            "val": [{"label": int(y), "score": float(s)} for y, s in zip(val_labels, val_scores)],
            "test": [{"label": int(y), "score": float(s)} for y, s in zip(test_labels, test_scores)],
        }
        threshold_reports = {}
        for target in fixed_targets:
            threshold, val_m = threshold_for_target_fpr(val_labels, val_scores, target)
            test_m = apply_threshold(test_labels, test_scores, threshold)
            threshold_reports[f"fpr_{int(target * 100):02d}"] = {
                "threshold": threshold,
                "val": val_m,
                "test": test_m,
            }
        test_graph = evaluate_graph_loader(model, test_loader, fixed_targets)
        test_win = test_graph
    else:
        test_win = evaluate_window(model, test_loader)
        test_graph = evaluate_graph(model, test_ds)
        threshold_reports = {}
        score_dump = {}
    print("\nWINDOW TEST", flush=True)
    for k, v in test_win.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}", flush=True)
    print("GRAPH TEST", flush=True)
    for k, v in test_graph.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}", flush=True)
    if threshold_reports:
        print("FIXED-FPR TEST", flush=True)
        for key, report in threshold_reports.items():
            t = report["test"]
            print(f"  {key}: threshold={report['threshold']:.4f} "
                  f"test_fpr={t['fpr']:.3f} recall={t['recall']:.3f} "
                  f"precision={t['precision']:.3f} f1={t['f1']:.3f}", flush=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(args),
        "ablation": {
            **input_audit,
            "node_base_dim": node_base_dim,
            "edge_base_dim": edge_base_dim,
            "model_params": params,
            "model_class": model_cls.__name__,
            "source_trainer": "code/train_gnn_v2.py",
        },
        "edge_types": V2_EDGE_TYPES,
        "best_epoch": selected_epoch,
        "selected_checkpoint": {
            "metric": args.checkpoint_metric,
            "epoch": selected_epoch,
            "score": selected_score,
        },
        "best_graph_f1": {
            "epoch": best_graph_f1_epoch,
            "score": best_graph_f1,
        },
        "best_bce_epoch": best_bce_epoch,
        "best_val_bce": best_bce,
        "best_fixed": {
            "epoch": best_fixed_epoch,
            "score": best_fixed_score,
            "threshold": best_fixed_threshold,
            "metrics": best_fixed_metrics,
        },
        "test_window": test_win,
        "test_graph": test_graph,
        "threshold_reports": threshold_reports,
        "scores": score_dump,
        "history": history,
    }
    result_path = out_dir / f"gnn_v2_results_{run_id}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    if args.save_checkpoint:
        torch.save({
            "model_state": selected_state,
            "config": {
                **vars(args),
                "node_base_dim": node_base_dim,
                "edge_base_dim": edge_base_dim,
                "edge_types": V2_EDGE_TYPES,
                "node_type_vocab": node_type_vocab,
            },
            "best_fixed": result["best_fixed"],
            "test_window": test_win,
            "test_graph": test_graph,
            "threshold_reports": threshold_reports,
            "history": history,
        }, out_dir / f"gnn_ablate_{run_id}.pt")
    print(f"\nSaved {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
