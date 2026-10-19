#!/usr/bin/env python3
"""Model-input variants and tensorization for the Task 1 pilot.

All removals happen at input-construction time and all arms are trained from
scratch under the same split, seed set, optimizer budget and dev-selection
rule. Names mirror the required ablation list.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

_REPO_CODE = Path(__file__).resolve().parents[2]
_REPO_FINAL_FIVE = _REPO_CODE / "repro" / "final_five_tables"
for _p in (str(_REPO_CODE), str(_REPO_FINAL_FIVE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_gnn_ablate import (  # noqa: E402
    CHAR_VOCAB,
    CONTENT_KEYWORDS,
    DEFAULT_PERM,
    MAX_EDGE_CHARS,
    MAX_NODE_CHARS,
    PERMISSION,
    TOOL_TO_PERM,
    CausalCharGNN,
    CausalCharGNNNoText,
    encode_char_rows,
)
from graphbuild import EDGE_TYPES, EDGE2IDX  # noqa: E402

VARIANT_CONFIGS = {
    "full_heuristic": dict(edge_mode="heuristic", use_node_text=True,
                           use_keywords=True, use_perm=True,
                           use_type_onehot=False, use_edge_text=True,
                           drop_relation_types=False),
    "full_provenance": dict(edge_mode="provenance", use_node_text=True,
                            use_keywords=True, use_perm=True,
                            use_type_onehot=False, use_edge_text=True,
                            drop_relation_types=False),
    "text_only": dict(edge_mode="none", use_node_text=True,
                      use_keywords=False, use_perm=False,
                      use_type_onehot=False, use_edge_text=False,
                      drop_relation_types=False),
    "structure_only": dict(edge_mode="provenance", use_node_text=False,
                           use_keywords=False, use_perm=True,
                           use_type_onehot=True, use_edge_text=False,
                           drop_relation_types=False),
    "chrono_only": dict(edge_mode="chrono", use_node_text=True,
                        use_keywords=True, use_perm=True,
                        use_type_onehot=False, use_edge_text=True,
                        drop_relation_types=False),
    "no_relation_types": dict(edge_mode="provenance", use_node_text=True,
                              use_keywords=True, use_perm=True,
                              use_type_onehot=False, use_edge_text=True,
                              drop_relation_types=True),
    "no_keyword_features": dict(edge_mode="provenance", use_node_text=True,
                                use_keywords=False, use_perm=True,
                                use_type_onehot=False, use_edge_text=True,
                                drop_relation_types=False),
    "no_read_link": dict(edge_mode="provenance", use_node_text=True,
                         use_keywords=True, use_perm=True,
                         use_type_onehot=False, use_edge_text=True,
                         drop_relation_types=False,
                         drop_edges=("reads_from",)),
    "no_param_link": dict(edge_mode="provenance", use_node_text=True,
                          use_keywords=True, use_perm=True,
                          use_type_onehot=False, use_edge_text=True,
                          drop_relation_types=False,
                          drop_edges=("param_source",)),
    "wrong_source": dict(edge_mode="provenance", use_node_text=True,
                         use_keywords=True, use_perm=True,
                         use_type_onehot=False, use_edge_text=True,
                         drop_relation_types=False, wrong_source=True),
    # DISC-2026W39-002 §3 A3 (additive; existing arms unchanged):
    # node-attribute arm — full attribute input, zero relations, pooling only.
    "node_attr": dict(edge_mode="none", use_node_text=True,
                      use_keywords=True, use_perm=True,
                      use_type_onehot=True, use_edge_text=False,
                      drop_relation_types=False),
}

TEXT_CONFIGS = {
    "text_chrono": dict(view="chrono", include_types=False),
    "text_relation": dict(view="relation", include_types=True),
}


def char_ids(text, max_len):
    text = (text or "")[:max_len]
    ids = [min(ord(ch), CHAR_VOCAB - 1) for ch in text]
    return ids + [0] * (max_len - len(ids))


def node_text(node):
    return (node.get("features", {}) or {}).get("subtask", "") or node.get("type", "")


def node_base_dim(cfg, vocab_size=0):
    dim = 0
    if cfg["use_perm"]:
        dim += len(PERMISSION)
    dim += 1
    if cfg["use_keywords"]:
        dim += len(CONTENT_KEYWORDS)
    if cfg["use_type_onehot"]:
        dim += vocab_size
    return dim


def graph_to_data(graph, cfg, node_type_vocab=None, tension=None):
    nodes = sorted(graph["nodes"], key=lambda n: (n["turn"], n["id"]))
    vocab = list(node_type_vocab or [])
    type_index = {t: i for i, t in enumerate(vocab)}
    base_dim = node_base_dim(cfg, len(vocab))
    n = len(nodes)
    x_base = torch.zeros(n, base_dim)
    node_chars = torch.zeros(n, MAX_NODE_CHARS, dtype=torch.long)
    id_to_idx = {}
    use_keywords = cfg["use_keywords"]
    for i, nd in enumerate(nodes):
        col = 0
        if cfg["use_perm"]:
            perm = TOOL_TO_PERM.get(nd["type"], DEFAULT_PERM)
            x_base[i, col:col + len(PERMISSION)] = torch.tensor(perm, dtype=torch.float32)
            col += len(PERMISSION)
        x_base[i, col] = nd.get("turn", i) / max(n, 1)
        col += 1
        if use_keywords:
            feats = [1 if any(k in node_text(nd).lower() for k in kws) else 0
                     for kws in CONTENT_KEYWORDS]
            x_base[i, col:col + len(CONTENT_KEYWORDS)] = torch.tensor(feats, dtype=torch.float32)
            col += len(CONTENT_KEYWORDS)
        if cfg["use_type_onehot"]:
            ti = type_index.get(str(nd.get("type", "")))
            if ti is not None:
                x_base[i, col + ti] = 1.0
        if cfg["use_node_text"]:
            node_chars[i] = torch.tensor(char_ids(node_text(nd), MAX_NODE_CHARS))
        id_to_idx[nd["id"]] = i

    edge_index, edge_attr, edge_chars = [], [], []
    n_edge_types = len(EDGE_TYPES)
    for e in graph["edges"]:
        src, tgt = id_to_idx.get(e["source"]), id_to_idx.get(e["target"])
        if src is None or tgt is None:
            continue
        et = torch.zeros(n_edge_types + 1, dtype=torch.float32)
        if cfg["drop_relation_types"]:
            et[0] = 1.0
        else:
            et[EDGE2IDX.get(e["type"], 0)] = 1.0
        et[-1] = float(e.get("confidence", 1.0))
        edge_index.append([src, tgt])
        edge_attr.append(et)
        if cfg["use_edge_text"]:
            rationale = e.get("rationale", e["type"])
            if cfg["drop_relation_types"]:
                rationale = "connection"
            edge_chars.append(torch.tensor(char_ids(rationale, MAX_EDGE_CHARS)))
        else:
            edge_chars.append(torch.zeros(MAX_EDGE_CHARS, dtype=torch.long))
    ei = (torch.tensor(edge_index, dtype=torch.long).t().contiguous()
          if edge_index else torch.zeros(2, 0, dtype=torch.long))
    ea = (torch.stack(edge_attr) if edge_attr
          else torch.zeros(0, n_edge_types + 1))
    ec = (torch.stack(edge_chars) if edge_chars
          else torch.zeros(0, MAX_EDGE_CHARS, dtype=torch.long))
    data = Data(
        x=torch.empty(n, 0),
        x_base=x_base,
        node_chars=node_chars,
        edge_index=ei,
        edge_attr_base=ea,
        edge_chars=ec,
        y=torch.tensor([int(graph["label"])], dtype=torch.float32),
        pair_id=graph["pair_id"],
        task_id=graph["task_id"],
        num_nodes=n,
    )
    data.edge_base_dim = n_edge_types + 1
    if tension is not None:
        data.tension = torch.tensor([float(tension)], dtype=torch.float32)
    return data


def build_node_type_vocab(graphs):
    types = set()
    for g in graphs:
        for nd in g["nodes"]:
            types.add(str(nd.get("type", "")))
    return sorted(types)


def make_model(cfg, node_type_vocab, hidden_dim=128, char_dim=48, dropout=0.3):
    base_dim = node_base_dim(cfg, len(node_type_vocab or []))
    edge_dim = len(EDGE_TYPES) + 1
    if cfg["use_node_text"]:
        return CausalCharGNN(base_dim, edge_dim, hidden_dim=hidden_dim,
                             char_dim=char_dim, dropout=dropout)
    return CausalCharGNNNoText(base_dim, edge_dim, hidden_dim=hidden_dim,
                               dropout=dropout)


def model_forward(model, batch):
    return model(batch.x_base, batch.node_chars, batch.edge_index,
                 batch.edge_attr_base, batch.edge_chars, batch.batch)
