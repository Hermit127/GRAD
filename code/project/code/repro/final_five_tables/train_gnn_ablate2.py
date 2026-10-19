#!/usr/bin/env python3
"""Phase 2 candidate trainer (experiment `table5-phase2-20260918`).

Separate from code/repro/final_five_tables/train_gnn_ablate.py so the frozen
replication queue is never touched. Implemented candidates (preregistered in
LOGS/final_five_tables/table5/phase2_prereg_20260918.md):

  A_norm_aux   : per-modality LayerNorm before fusion + auxiliary text/struct
                 heads, L = L_joint + lambda*(L_text + L_struct), lambda in {0.1,0.3}
  A_moddrop    : A + per-graph modality dropout p=0.1 during training
  B_gate       : s = s_struct + alpha*semantic, alpha learned from deployable
                 inputs only; aux loss 0.5*BCE(s_struct); dumps s_struct/semantic/alpha
  C_raw_rel    : chronological chain + mapped corpus edges (no regex edges),
                 constant edge rationale; optional edge dropout on non-chrono edges
  full_ref     : the frozen full model reproduced with this trainer (dev reference)

Protocol identical to the frozen replication (seeds/epochs/patience/batch/checkpoint).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, TransformerConv, global_add_pool

import sys
_REPO_CODE = Path(__file__).resolve().parents[2]
if str(_REPO_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_CODE))

from train_gnn import (  # noqa: E402
    CONTENT_KEYWORDS,
    DEFAULT_PERM,
    PERMISSION,
    TOOL_TO_PERM,
    content_features,
)
from train_gnn_ablate import (  # noqa: E402
    V2_EDGE_TYPES,
    EDGE2IDX,
    CHAR_VOCAB,
    MAX_NODE_CHARS,
    MAX_EDGE_CHARS,
    CharCNN,
    encode_char_rows,
    rebuild_v2_edges,
    char_ids,
    node_text,
    set_seed,
    metrics_at_threshold,
    threshold_for_target_fpr,
    build_node_type_vocab,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_CANDIDATES = {"K2_kwnorm", "K3_kwdrop25", "K3_kwdrop50"}

RAW_EDGE_MAP = {
    "temporal_adjacency": "sequence_next",
    "causal_derivation": "data_flow",
    "triggers": "control_trigger",
    "reads_from": "memory_retrieve",
    "memory_read": "memory_retrieve",
    "memory_write": "memory_store",
    "writes_to": "writes_to",
    "data_flow": "data_flow",
    "derived_from": "data_flow",
    "control_trigger": "control_trigger",
    "sink_trigger": "sink_trigger",
    "spawns": "spawns",
    "delegates_message": "delegates_message",
    "cross_turn": "cross_turn",
    "sequence_next": "sequence_next",
}


def raw_edges(g: dict) -> list[dict]:
    out, seen = [], set()
    for e in g.get("edges", []):
        t = RAW_EDGE_MAP.get(str(e.get("type", "")), "data_flow")
        key = (str(e.get("source")), str(e.get("target")), t)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": key[0], "target": key[1], "type": t,
                    "confidence": 1.0, "rationale": "raw relation"})
    return out


def merged_edges(g: dict, cfg) -> list[dict]:
    if cfg["edge_mode"] == "rebuilt":
        return rebuild_v2_edges(g)
    chrono = [e for e in rebuild_v2_edges(g) if e["type"] == "sequence_next"]
    out, seen = [], set()
    for e in chrono + raw_edges(g):
        key = (e["source"], e["target"], e["type"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def graph_to_data(g: dict, cfg: dict, vocab: list[str] | None) -> Data:
    nodes = sorted(g.get("nodes", []), key=lambda n: n.get("turn", 0))
    edges = merged_edges(g, cfg)
    n = len(nodes)
    type_index = {t: i for i, t in enumerate(vocab or [])}
    if cfg["candidate"] == "S3_reg":
        x_struct = torch.zeros(n, len(PERMISSION) + 1 + (len(vocab) if vocab else 0))
        x_text = None
        node_chars = torch.zeros(n, MAX_NODE_CHARS, dtype=torch.long)
        for i, node in enumerate(nodes):
            perm = TOOL_TO_PERM.get(node["type"], DEFAULT_PERM)
            x_struct[i, :len(PERMISSION)] = torch.tensor(perm, dtype=torch.float32)
            x_struct[i, len(PERMISSION)] = node.get("turn", i) / max(n, 1)
            if vocab:
                ti = type_index.get(str(node.get("type", "")))
                if ti is not None:
                    x_struct[i, len(PERMISSION) + 1 + ti] = 1.0
    elif cfg["candidate"] in K_CANDIDATES:
        # K2/K3: structure block = permissions + position (13); the 9 handcrafted
        # keyword flags go through a separate normalized channel (`x_kw`), fitted
        # on the TRAIN split only (normalization applied in CData.__init__).
        x_struct = torch.zeros(n, len(PERMISSION) + 1)
        x_kw = torch.zeros(n, len(CONTENT_KEYWORDS))
        x_text = None
        node_chars = torch.zeros(n, MAX_NODE_CHARS, dtype=torch.long)
        for i, node in enumerate(nodes):
            perm = TOOL_TO_PERM.get(node["type"], DEFAULT_PERM)
            x_struct[i, :len(PERMISSION)] = torch.tensor(perm, dtype=torch.float32)
            x_struct[i, len(PERMISSION)] = node.get("turn", i) / max(n, 1)
            x_kw[i] = torch.tensor(content_features(node_text(node)), dtype=torch.float32)
            node_chars[i] = torch.tensor(char_ids(node_text(node), MAX_NODE_CHARS),
                                         dtype=torch.long)
    elif cfg["candidate"] == "B_gate":
        x_struct = torch.zeros(n, len(PERMISSION) + 1 + (len(vocab) if vocab else 0))
        x_text = torch.zeros(n, 1)
        node_chars = torch.zeros(n, MAX_NODE_CHARS, dtype=torch.long)
        for i, node in enumerate(nodes):
            perm = TOOL_TO_PERM.get(node["type"], DEFAULT_PERM)
            x_struct[i, :len(PERMISSION)] = torch.tensor(perm, dtype=torch.float32)
            x_struct[i, len(PERMISSION)] = node.get("turn", i) / max(n, 1)
            if vocab:
                ti = type_index.get(str(node.get("type", "")))
                if ti is not None:
                    x_struct[i, len(PERMISSION) + 1 + ti] = 1.0
            x_text[i, 0] = node.get("turn", i) / max(n, 1)
            node_chars[i] = torch.tensor(char_ids(node_text(node), MAX_NODE_CHARS),
                                         dtype=torch.long)
    else:
        x_struct = torch.zeros(n, len(PERMISSION) + 1 + len(CONTENT_KEYWORDS))
        x_text = None
        node_chars = torch.zeros(n, MAX_NODE_CHARS, dtype=torch.long)
        for i, node in enumerate(nodes):
            perm = TOOL_TO_PERM.get(node["type"], DEFAULT_PERM)
            x_struct[i, :len(PERMISSION)] = torch.tensor(perm, dtype=torch.float32)
            x_struct[i, len(PERMISSION)] = node.get("turn", i) / max(n, 1)
            x_struct[i, len(PERMISSION) + 1:] = torch.tensor(
                content_features(node_text(node)), dtype=torch.float32)
            node_chars[i] = torch.tensor(char_ids(node_text(node), MAX_NODE_CHARS),
                                         dtype=torch.long)

    id_to_idx = {nd["id"]: i for i, nd in enumerate(nodes)}
    edge_index, edge_attr, edge_chars, is_chrono = [], [], [], []
    use_edge_text = cfg.get("edge_text", "copy") == "copy"
    for e in edges:
        src, tgt = id_to_idx.get(e["source"]), id_to_idx.get(e["target"])
        if src is None or tgt is None:
            continue
        et = torch.zeros(len(V2_EDGE_TYPES) + 1, dtype=torch.float32)
        et[EDGE2IDX.get(e["type"], 0)] = 1.0
        et[-1] = float(e.get("confidence", 1.0))
        edge_index.append([src, tgt])
        edge_attr.append(et)
        if use_edge_text:
            txt = e.get("rationale", e["type"])
        else:
            txt = "raw relation" if cfg.get("edge_text") == "constant" else ""
        edge_chars.append(torch.tensor(char_ids(txt, MAX_EDGE_CHARS), dtype=torch.long))
        is_chrono.append(1 if e["type"] == "sequence_next" else 0)
    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.stack(edge_attr) if edge_attr else torch.zeros(0, len(V2_EDGE_TYPES) + 1)
    ec = torch.stack(edge_chars) if edge_chars else torch.zeros(0, MAX_EDGE_CHARS, dtype=torch.long)
    d = Data(
        x=torch.empty(n, 0), x_base=x_struct, node_chars=node_chars,
        edge_index=ei, edge_attr_base=ea, edge_chars=ec,
        is_chrono=torch.tensor(is_chrono, dtype=torch.bool),
        y=torch.tensor([int(g.get("label", 0))], dtype=torch.float32),
        num_nodes=n,
    )
    if cfg["candidate"] == "B_gate":
        d.x_text = x_text
    if cfg["candidate"] in K_CANDIDATES:
        d.x_kw = x_kw
    return d


def compute_kw_stats(train_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std of the 9 keyword dims over the TRAIN split only."""
    vals = []
    for line in Path(train_path).read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        txt = " ".join(str((n.get("features") or {}).get("subtask", ""))
                       for n in g.get("nodes", []))
        vals.append(content_features(txt))
    arr = np.asarray(vals, dtype=np.float64)
    mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(arr.std(axis=0), dtype=torch.float32)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)   # dead dims -> identity
    return mean, std


class CData(Dataset):
    def __init__(self, path, cfg, vocab, train=False, kw_stats=None):
        super().__init__()
        self.graphs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        self.cfg = cfg
        self.vocab = vocab
        self.train = train
        self.data = [graph_to_data(g, cfg, vocab) for g in self.graphs]
        if cfg["candidate"] in K_CANDIDATES:
            assert kw_stats is not None, "K candidates need train-split kw stats"
            mean, std = kw_stats
            std = torch.where(torch.as_tensor(std) < 1e-8,
                              torch.ones_like(torch.as_tensor(std)),
                              torch.as_tensor(std))
            for d in self.data:
                d.x_kw = (d.x_kw - mean) / std
        self.sample_weights = [1.0] * len(self.graphs)

    def len(self):
        return len(self.data)

    def get(self, idx):
        d = self.data[idx]
        if self.train and self.cfg.get("edge_dropout", 0) > 0 and d.edge_index.size(1) > 0:
            keep = torch.ones(d.edge_index.size(1), dtype=torch.bool)
            mask = ~d.is_chrono
            drop = torch.rand(mask.sum()) < self.cfg["edge_dropout"]
            keep[mask] = ~drop
            d = Data(x=d.x, x_base=d.x_base, node_chars=d.node_chars,
                     edge_index=d.edge_index[:, keep], edge_attr_base=d.edge_attr_base[keep],
                     edge_chars=d.edge_chars[keep], is_chrono=d.is_chrono[keep],
                     y=d.y, num_nodes=d.num_nodes)
            if hasattr(d, "x_text") or self.cfg["candidate"] == "B_gate":
                d.x_text = self.data[idx].x_text
        d.sample_weight = torch.tensor([self.sample_weights[idx]], dtype=torch.float32)
        d.sample_idx = torch.tensor([idx], dtype=torch.long)
        return d

    def set_hard_benign(self, indices, weight):
        self.sample_weights = [weight if i in indices and int(g.get("label", 0)) == 0 else 1.0
                               for i, g in enumerate(self.graphs)]


def attn_pool(h: torch.Tensor, batch: torch.Tensor, attn: torch.nn.Module) -> torch.Tensor:
    a = attn(h).sigmoid()
    return global_add_pool(h * a, batch) / global_add_pool(a, batch).clamp(min=1e-8)


class ModalityNormGNN(torch.nn.Module):
    """Candidate A/G3: per-modality LayerNorm + auxiliary heads (aux train-only)."""

    def __init__(self, node_base_dim, edge_base_dim, hidden=192, dropout=0.35, heads=4,
                 use_ln=True, use_aux=True):
        super().__init__()
        self.use_ln = use_ln
        self.use_aux = use_aux
        self.node_char = CharCNN(out_dim=64)
        self.edge_char = CharCNN(out_dim=64)
        self.ln_base = torch.nn.LayerNorm(node_base_dim) if use_ln else torch.nn.Identity()
        self.ln_char = torch.nn.LayerNorm(64) if use_ln else torch.nn.Identity()
        self.ln_edge = torch.nn.LayerNorm(edge_base_dim) if use_ln else torch.nn.Identity()
        self.ln_edge_char = torch.nn.LayerNorm(64) if use_ln else torch.nn.Identity()
        self.node_proj = torch.nn.Linear(node_base_dim + 64, hidden)
        self.edge_proj = torch.nn.Linear(edge_base_dim + 64, hidden)
        self.gine = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden)), edge_dim=hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.tf1 = TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.tf2 = TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden)
        self.attn = torch.nn.Linear(hidden, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden // 2, 1))
        # auxiliary heads (training-only; not executed or deployed at inference)
        if use_aux:
            self.aux_text_proj = torch.nn.Sequential(torch.nn.Linear(64, hidden), torch.nn.GELU())
            self.aux_text_attn = torch.nn.Linear(hidden, 1)
            self.aux_text_head = torch.nn.Linear(hidden, 1)
            self.aux_struct_proj = torch.nn.Sequential(
                torch.nn.Linear(len(PERMISSION) + 1, hidden), torch.nn.GELU())
            self.aux_struct_attn = torch.nn.Linear(hidden, 1)
            self.aux_struct_head = torch.nn.Linear(hidden, 1)
        self.n_struct_dims = len(PERMISSION) + 1
        self.node_base_dim = node_base_dim
        self.dropout = dropout

    def forward(self, batch, moddrop_p=0.0, training=False):
        xb = batch.x_base
        nc = encode_char_rows(self.node_char, batch.node_chars)
        xb_m = xb
        nc_m = nc
        if training and moddrop_p > 0:
            # structural modality = permission+position dims; text modality = keyword
            # flags + char-text (per-graph masks, independent)
            graph_mask_s = (torch.rand(batch.num_graphs, device=xb.device) < moddrop_p).float()
            graph_mask_t = (torch.rand(batch.num_graphs, device=xb.device) < moddrop_p).float()
            s_mask = graph_mask_s[batch.batch].unsqueeze(-1)
            t_mask = graph_mask_t[batch.batch].unsqueeze(-1)
            xb_m = xb.clone()
            xb_m[:, :self.n_struct_dims] = xb[:, :self.n_struct_dims] * (1 - s_mask)
            xb_m[:, self.n_struct_dims:] = xb[:, self.n_struct_dims:] * (1 - t_mask)
            nc_m = nc * (1 - t_mask)
        x = self.node_proj(torch.cat([self.ln_base(xb_m), self.ln_char(nc_m)], dim=-1))
        if batch.edge_attr_base.size(0) > 0:
            ec = encode_char_rows(self.edge_char, batch.edge_chars)
            edge_attr = self.edge_proj(torch.cat([self.ln_edge(batch.edge_attr_base),
                                                  self.ln_edge_char(ec)], dim=-1))
        else:
            edge_attr = torch.zeros(0, x.size(-1), device=x.device)
        residual = x
        x = F.gelu(self.bn1(self.gine(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        residual = x
        x = F.gelu(self.bn2(self.tf1(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        residual = x
        x = F.gelu(self.bn3(self.tf2(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        pooled = attn_pool(x, batch.batch, self.attn)
        logit = self.classifier(pooled).squeeze(-1)
        out = {"logit": logit}
        if self.use_aux and training:
            h_t = self.aux_text_proj(self.ln_char(nc_m))
            s_t = self.aux_text_head(attn_pool(h_t, batch.batch, self.aux_text_attn)).squeeze(-1)
            h_s = self.aux_struct_proj(xb_m[:, :self.n_struct_dims])
            s_s = self.aux_struct_head(attn_pool(h_s, batch.batch, self.aux_struct_attn)).squeeze(-1)
            out.update({"s_t": s_t, "s_s": s_s})
        return out


class StackBranch(torch.nn.Module):
    """Residual MLP stack + attention pool returning logit and pooled embedding."""

    def __init__(self, in_dim, hidden=192, dropout=0.35, depth=3):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, hidden)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                torch.nn.Dropout(dropout)) for _ in range(depth)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden) for _ in range(depth)])
        self.attn = torch.nn.Linear(hidden, 1)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
                                        torch.nn.Dropout(dropout), torch.nn.Linear(hidden // 2, 1))

    def forward(self, x, batch_vec):
        h = self.proj(x)
        for block, bn in zip(self.blocks, self.bns):
            h = F.gelu(bn(block(h))) + h
        pooled = attn_pool(h, batch_vec, self.attn)
        return self.head(pooled).squeeze(-1), pooled


class StructBranch(torch.nn.Module):
    def __init__(self, node_base_dim, edge_base_dim, hidden=192, dropout=0.35, heads=4):
        super().__init__()
        self.node_proj = torch.nn.Linear(node_base_dim, hidden)
        self.edge_proj = torch.nn.Linear(edge_base_dim, hidden)
        self.gine = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden)), edge_dim=hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.tf1 = TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.tf2 = TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden)
        self.attn = torch.nn.Linear(hidden, 1)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
                                        torch.nn.Dropout(dropout), torch.nn.Linear(hidden // 2, 1))
        self.dropout = dropout

    def forward(self, x_base, edge_index, edge_attr, batch_vec):
        x = self.node_proj(x_base)
        residual = x
        x = F.gelu(self.bn1(self.gine(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        residual = x
        x = F.gelu(self.bn2(self.tf1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        residual = x
        x = F.gelu(self.bn3(self.tf2(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        pooled = attn_pool(x, batch_vec, self.attn)
        return self.head(pooled).squeeze(-1), pooled


class ResidualGateModel(torch.nn.Module):
    """Candidate B: structure-main + learned semantic residual gate."""

    def __init__(self, struct_dim, edge_base_dim, hidden=192, dropout=0.35):
        super().__init__()
        self.char = CharCNN(out_dim=64)
        self.struct_branch = StructBranch(struct_dim, edge_base_dim, hidden, dropout)
        self.text_branch = StackBranch(1 + 64, hidden, dropout)
        self.gate = torch.nn.Sequential(torch.nn.Linear(2 * hidden, hidden // 2),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(hidden // 2, 1))
        torch.nn.init.normal_(self.gate[-1].weight, std=0.01)
        torch.nn.init.constant_(self.gate[-1].bias, -2.0)
        self.beta = 0.5

    def forward(self, batch, moddrop_p=0.0, training=False):
        nc = encode_char_rows(self.char, batch.node_chars)
        x_text = torch.cat([batch.x_text, nc], dim=-1)
        if batch.edge_attr_base.size(0) > 0:
            edge_attr = self.struct_branch.edge_proj(batch.edge_attr_base)
        else:
            edge_attr = torch.zeros(0, self.struct_branch.node_proj.out_features,
                                    device=batch.x_base.device)
        s_s, h_s = self.struct_branch(batch.x_base, batch.edge_index, edge_attr, batch.batch)
        s_t, h_t = self.text_branch(x_text, batch.batch)
        alpha = torch.sigmoid(self.gate(torch.cat([h_s, h_t], dim=-1))).squeeze(-1)
        logit = s_s + alpha * s_t
        return {"logit": logit, "s_s": s_s, "s_t": s_t, "alpha": alpha}


class KeywordNormGNN(torch.nn.Module):
    """K2/K3: normalized keyword channel with an independent small projection."""

    def __init__(self, struct_dim, kw_dim, edge_base_dim, hidden=192,
                 dropout=0.35, heads=4, kw_hidden=16):
        super().__init__()
        self.node_char = CharCNN(out_dim=64)
        self.edge_char = CharCNN(out_dim=64)
        self.kw_proj = torch.nn.Sequential(
            torch.nn.Linear(kw_dim, kw_hidden), torch.nn.GELU())
        self.node_proj = torch.nn.Linear(struct_dim + kw_hidden + 64, hidden)
        self.edge_proj = torch.nn.Linear(edge_base_dim + 64, hidden)
        self.gine = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden)), edge_dim=hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.tf1 = TransformerConv(hidden, hidden // heads, heads=heads,
                                   edge_dim=hidden, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.tf2 = TransformerConv(hidden, hidden // heads, heads=heads,
                                   edge_dim=hidden, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden)
        self.attn = torch.nn.Linear(hidden, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden // 2, 1))
        self.dropout = dropout

    def forward(self, batch, kw_drop_p=0.0, training=False):
        nc = encode_char_rows(self.node_char, batch.node_chars)
        kw = self.kw_proj(batch.x_kw)
        if training and kw_drop_p > 0:
            keep = (torch.rand(batch.num_graphs, device=kw.device) >= kw_drop_p).float()
            kw = kw * keep[batch.batch].unsqueeze(-1)
        x = self.node_proj(torch.cat([batch.x_base, kw, nc], dim=-1))
        if batch.edge_attr_base.size(0) > 0:
            ec = encode_char_rows(self.edge_char, batch.edge_chars)
            edge_attr = self.edge_proj(torch.cat([batch.edge_attr_base, ec], dim=-1))
        else:
            edge_attr = torch.zeros(0, x.size(-1), device=x.device)
        residual = x
        x = F.gelu(self.bn1(self.gine(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        residual = x
        x = F.gelu(self.bn2(self.tf1(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        residual = x
        x = F.gelu(self.bn3(self.tf2(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        pooled = attn_pool(x, batch.batch, self.attn)
        return {"logit": self.classifier(pooled).squeeze(-1)}


class StructRegGNN(torch.nn.Module):
    """Fairness baseline: structure-only input with the same regularization family
    as G3 (LayerNorm, auxiliary structure head, per-graph structure dropout)."""

    def __init__(self, node_base_dim, edge_base_dim, hidden=192, dropout=0.35,
                 heads=4, use_ln=True, use_aux=True):
        super().__init__()
        self.use_ln = use_ln
        self.use_aux = use_aux
        self.ln_base = torch.nn.LayerNorm(node_base_dim) if use_ln else torch.nn.Identity()
        self.node_proj = torch.nn.Linear(node_base_dim, hidden)
        self.edge_proj = torch.nn.Linear(edge_base_dim, hidden)
        self.gine = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden)), edge_dim=hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.tf1 = TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.tf2 = TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden)
        self.attn = torch.nn.Linear(hidden, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden // 2, 1))
        if use_aux:
            self.aux_struct_proj = torch.nn.Sequential(
                torch.nn.Linear(node_base_dim, hidden), torch.nn.GELU())
            self.aux_struct_attn = torch.nn.Linear(hidden, 1)
            self.aux_struct_head = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, batch, moddrop_p=0.0, training=False):
        xb = batch.x_base
        if training and moddrop_p > 0:
            keep = (torch.rand(batch.num_graphs, device=xb.device) >= moddrop_p).float()
            xb = xb * keep[batch.batch].unsqueeze(-1)
        x = self.node_proj(self.ln_base(xb))
        if batch.edge_attr_base.size(0) > 0:
            edge_attr = self.edge_proj(batch.edge_attr_base)
        else:
            edge_attr = torch.zeros(0, x.size(-1), device=x.device)
        residual = x
        x = F.gelu(self.bn1(self.gine(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        residual = x
        x = F.gelu(self.bn2(self.tf1(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        residual = x
        x = F.gelu(self.bn3(self.tf2(x, batch.edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training) + residual
        pooled = attn_pool(x, batch.batch, self.attn)
        out = {"logit": self.classifier(pooled).squeeze(-1)}
        if self.use_aux and training:
            h_s = self.aux_struct_proj(self.ln_base(xb))
            out["s_s"] = self.aux_struct_head(
                attn_pool(h_s, batch.batch, self.aux_struct_attn)).squeeze(-1)
        return out


def batch_logits(model, batch, cfg, training=False):
    if cfg["candidate"] == "A_norm_aux" or cfg["candidate"] == "A_moddrop":
        md = cfg.get("moddrop", 0.0) if cfg["candidate"] == "A_moddrop" else 0.0
        return model(batch, moddrop_p=md, training=training)
    if cfg["candidate"] == "B_gate":
        return model(batch, training=training)
    if cfg["candidate"] == "S3_reg":
        return model(batch, moddrop_p=cfg.get("moddrop", 0.25), training=training)
    elif cfg["candidate"] in K_CANDIDATES:
        kd = {"K3_kwdrop25": 0.25, "K3_kwdrop50": 0.5}.get(cfg["candidate"], 0.0)
        return model(batch, kw_drop_p=kd, training=training)
    return {"logit": model(batch.x_base, batch.node_chars, batch.edge_index,
                           batch.edge_attr_base, batch.edge_chars, batch.batch)}


def compute_loss(out, y, cfg, sample_weight=None, pos_weight=1.0):
    lw = torch.tensor(pos_weight, device=y.device)
    loss = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=lw, reduction="none")
    if sample_weight is not None:
        loss = loss * sample_weight.view_as(loss)
    loss = loss.mean()
    parts = {"joint": float(loss.item())}
    if cfg["candidate"] in ("A_norm_aux", "A_moddrop") and "s_t" in out:
        lt = F.binary_cross_entropy_with_logits(out["s_t"], y, pos_weight=lw)
        ls = F.binary_cross_entropy_with_logits(out["s_s"], y, pos_weight=lw)
        lam = cfg.get("lambda_aux", 0.1)
        loss = loss + lam * (lt + ls)
        parts.update({"text": float(lt.item()), "struct": float(ls.item()), "lambda": lam})
    if cfg["candidate"] == "B_gate":
        ls = F.binary_cross_entropy_with_logits(out["s_s"], y, pos_weight=lw)
        loss = loss + 0.5 * ls
        parts.update({"struct": float(ls.item()), "beta": 0.5})
    if cfg["candidate"] == "S3_reg" and "s_s" in out:
        ls = F.binary_cross_entropy_with_logits(out["s_s"], y, pos_weight=lw)
        lam = cfg.get("lambda_aux", 0.1)
        loss = loss + lam * ls
        parts.update({"struct": float(ls.item()), "lambda": lam})
    return loss, parts


def train_epoch(model, loader, optimizer, cfg, pos_weight):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = batch_logits(model, batch, cfg, training=True)
        sw = batch.sample_weight if hasattr(batch, "sample_weight") else None
        loss, _ = compute_loss(out, batch.y, cfg, sw, pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()); n += 1
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, cfg):
    model.eval()
    probs, labels, indices, extra = [], [], [], {"s_s": [], "s_t": [], "alpha": []}
    for batch in loader:
        batch = batch.to(DEVICE)
        out = batch_logits(model, batch, cfg)
        p = torch.sigmoid(out["logit"])
        probs.extend(p.cpu().tolist())
        labels.extend((batch.y > 0.5).long().cpu().tolist())
        if hasattr(batch, "sample_idx"):
            indices.extend(batch.sample_idx.cpu().tolist())
        else:
            indices.extend(range(len(indices), len(indices) + batch.num_graphs))
        for k in extra:
            if k in out:
                extra[k].extend(torch.sigmoid(out[k]).cpu().tolist() if k != "alpha"
                                else out[k].cpu().tolist())
    return probs, labels, indices, extra


def fixed_fpr_f1(labels, scores, target=0.10):
    t, m = threshold_for_target_fpr(labels, scores, target)
    return t, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--candidate", required=True,
                    choices=["full_ref", "A_norm_aux", "A_moddrop", "B_gate",
                             "C_raw_rel", "C_raw_rel_drop",
                             "K2_kwnorm", "K3_kwdrop25", "K3_kwdrop50", "S3_reg"])
    ap.add_argument("--lambda-aux", type=float, default=0.1)
    ap.add_argument("--moddrop", type=float, default=0.1)
    ap.add_argument("--edge-mode", default="rebuilt", choices=["rebuilt", "raw_chrono"])
    ap.add_argument("--edge-dropout", type=float, default=0.0)
    ap.add_argument("--edge-text", default="copy", choices=["copy", "constant", "none"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target-fpr", type=float, default=0.10)
    ap.add_argument("--hard-benign-weight", type=float, default=3.0)
    ap.add_argument("--hard-benign-top-k", type=int, default=128)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ablate", default="none",
                    choices=["none", "no_ln", "no_aux", "no_moddrop"],
                    help="component ablation for the A_moddrop/G3 family")
    ap.add_argument("--dev-only", action="store_true",
                    help="evaluate dev only and store no test scores (Phase 2 dev runs)")
    args = ap.parse_args()

    cfg = vars(args)
    if args.candidate in ("A_moddrop", "S3_reg"):
        if args.ablate == "no_ln":
            cfg["use_ln"] = False
        elif args.ablate == "no_aux":
            cfg["use_aux"] = False
            cfg["lambda_aux"] = 0.0
        elif args.ablate == "no_moddrop":
            cfg["moddrop"] = 0.0
    if args.candidate in ("C_raw_rel", "C_raw_rel_drop"):
        cfg["edge_mode"] = "raw_chrono"
        cfg["edge_text"] = "constant"
    if args.candidate == "C_raw_rel_drop":
        cfg["edge_dropout"] = 0.2

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== phase2 candidate {args.candidate} seed={args.seed} device={DEVICE} ===", flush=True)
    vocab = (build_node_type_vocab(args.train)
             if args.candidate in ("B_gate", "S3_reg") else None)
    kw_stats = compute_kw_stats(args.train) if args.candidate in K_CANDIDATES else None
    train_ds = CData(args.train, cfg, vocab, train=True, kw_stats=kw_stats)
    val_ds = CData(args.val, cfg, vocab, kw_stats=kw_stats)
    test_ds = CData(args.test, cfg, vocab, kw_stats=kw_stats)
    kw_stats_report = ({"mean": [round(float(x), 6) for x in kw_stats[0]],
                        "std": [round(float(x), 6) for x in kw_stats[1]],
                        "fitted_on": "train split only"} if kw_stats else None)
    pos = sum(1 for g in train_ds.graphs if int(g.get("label", 0)) == 1)
    neg = train_ds.len() - pos
    pos_weight = neg / max(pos, 1)
    print(f"loaded train={train_ds.len()} val={val_ds.len()} test={test_ds.len()} "
          f"pos_weight={pos_weight:.3f} vocab={len(vocab or [])}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    if args.candidate in ("A_norm_aux", "A_moddrop"):
        model = ModalityNormGNN(
            len(PERMISSION) + 1 + len(CONTENT_KEYWORDS),
            len(V2_EDGE_TYPES) + 1, args.hidden_dim,
            use_ln=cfg.get("use_ln", True), use_aux=cfg.get("use_aux", True)).to(DEVICE)
    elif args.candidate == "B_gate":
        model = ResidualGateModel(len(PERMISSION) + 1 + len(vocab or []),
                                  len(V2_EDGE_TYPES) + 1, args.hidden_dim).to(DEVICE)
    elif args.candidate in K_CANDIDATES:
        model = KeywordNormGNN(len(PERMISSION) + 1, len(CONTENT_KEYWORDS),
                               len(V2_EDGE_TYPES) + 1, args.hidden_dim).to(DEVICE)
    elif args.candidate == "S3_reg":
        model = StructRegGNN(len(PERMISSION) + 1 + len(vocab or []),
                             len(V2_EDGE_TYPES) + 1, args.hidden_dim,
                             use_ln=cfg.get("use_ln", True),
                             use_aux=cfg.get("use_aux", True)).to(DEVICE)
    else:
        from train_gnn_ablate import CausalCharGNN
        model = CausalCharGNN(len(PERMISSION) + 1 + len(CONTENT_KEYWORDS),
                              len(V2_EDGE_TYPES) + 1, hidden_dim=args.hidden_dim).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"model params={params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    best_bce, best_bce_state = float("inf"), None
    best_fixed_score, best_fixed_state, best_fixed_epoch = -1.0, None, 0
    best_fixed_threshold, best_fixed_metrics = 0.5, {}
    history = []
    hard_benign = set()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, cfg, pos_weight)
        probs, labels, _, _ = predict(model, val_loader, cfg)
        val_bce = float(F.binary_cross_entropy(
            torch.tensor(probs).clamp(1e-6, 1 - 1e-6), torch.tensor(labels, dtype=torch.float32)).item())
        thr, m = fixed_fpr_f1(labels, probs, args.target_fpr)
        scheduler.step(val_bce)
        if val_bce < best_bce:
            best_bce = val_bce
            best_bce_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if m["f1"] > best_fixed_score:
            best_fixed_score, best_fixed_threshold, best_fixed_metrics = m["f1"], thr, m
            best_fixed_epoch = epoch
            best_fixed_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mined = 0
        if epoch >= 2:
            scores, labs, idxs, _ = predict(model, train_eval_loader, cfg)
            cand = sorted([(s, i) for s, l, i in zip(scores, labs, idxs)
                           if l == 0 and s >= thr], reverse=True)[:args.hard_benign_top_k]
            hard_benign = {i for _, i in cand}
            train_ds.set_hard_benign(hard_benign, args.hard_benign_weight)
            mined = len(hard_benign)
        history.append({"epoch": epoch, "train_loss": loss, "val_bce": val_bce,
                        "fixed_f1": m["f1"], "hard_benign": mined})
        print(f"E{epoch:03d} loss={loss:.4f} val_bce={val_bce:.4f} fixed_f1={m['f1']:.3f}", flush=True)
        if epoch - best_fixed_epoch >= args.patience:
            print(f"early stop at {epoch}", flush=True)
            break

    if best_fixed_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_fixed_state.items()})

    def full_eval(loader, split):
        probs, labels, idxs, extra = predict(model, loader, cfg)
        auroc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        thr, m = fixed_fpr_f1(labels, probs, args.target_fpr)
        scores = []
        for i, (p, l) in enumerate(zip(probs, labels)):
            row = {"label": int(l), "score": float(p)}
            for k, v in extra.items():
                if i < len(v):
                    row[k] = float(v[i])
            scores.append(row)
        return {"auroc": auroc, "dev_selected_threshold": thr, "metrics_at_threshold": m}, scores

    val_metrics, val_scores = full_eval(val_loader, "val")
    if args.dev_only:
        test_metrics, test_scores = None, []
    else:
        test_metrics, test_scores = full_eval(test_loader, "test")
    # dev-fitted fixed-weight fusion baseline for B (uses branch scores only)
    fusion_report = None
    if args.candidate == "B_gate":
        ys = np.array([r["label"] for r in val_scores])
        s_s = np.array([r["s_s"] for r in val_scores])
        s_t = np.array([r["s_t"] for r in val_scores])
        best_w, best_auc = 0.0, -1.0
        for w in np.arange(0, 1.0001, 0.05):
            auc = roc_auc_score(ys, (1 - w) * s_s + w * s_t) if len(set(ys)) > 1 else float("nan")
            if auc > best_auc:
                best_auc, best_w = auc, float(w)
        fusion_report = {
            "dev_best_weight_semantic": best_w,
            "dev_best_fusion_auroc": float(best_auc),
            "note": "w selected on dev only; the fixed-weight comparator for the gate. "
                    "Test fusion is computed only at confirmation time.",
        }
        if not args.dev_only:
            yt = np.array([r["label"] for r in test_scores])
            s_st = np.array([r["s_s"] for r in test_scores])
            s_tt = np.array([r["s_t"] for r in test_scores])
            fusion_report["test_fusion_auroc"] = (
                float(roc_auc_score(yt, (1 - best_w) * s_st + best_w * s_tt))
                if len(set(yt)) > 1 else float("nan"))
    alphas = [r.get("alpha") for r in val_scores if "alpha" in r]
    result = {
        "config": cfg,
        "candidate_meta": {
            "experiment_id": "table5-phase2-20260918",
            "mechanism": {
                "A_norm_aux": "per-modality LayerNorm + aux heads",
                "A_moddrop": "A + modality dropout 0.1",
                "B_gate": "s = s_struct + sigmoid(gate)*s_semantic",
                "C_raw_rel": "chrono + raw corpus edges, constant edge text",
                "C_raw_rel_drop": "C_raw_rel + non-chrono edge dropout 0.2",
                "full_ref": "frozen full reference reproduced in this trainer",
                "K2_kwnorm": "train-fitted keyword normalization + independent 9->16 projection",
                "K3_kwdrop25": "K2 + per-graph keyword-channel dropout 0.25 (train only)",
                "K3_kwdrop50": "K2 + per-graph keyword-channel dropout 0.5 (train only)",
            }.get(args.candidate, f"custom candidate {args.candidate}"),
            "params": params,
            "alpha_mean": float(np.mean(alphas)) if alphas else None,
            "alpha_std": float(np.std(alphas)) if alphas else None,
            "ablation": cfg.get("ablate", "none"),
            "use_ln": cfg.get("use_ln", True),
            "use_aux": cfg.get("use_aux", True),
            "kw_stats": kw_stats_report,
            "kw_dropout": {"K3_kwdrop25": 0.25, "K3_kwdrop50": 0.5}.get(
                args.candidate, 0.0),
        },
        "best_fixed": {"epoch": best_fixed_epoch, "score": best_fixed_score,
                       "threshold": best_fixed_threshold, "metrics": best_fixed_metrics},
        "val": val_metrics,
        "test": test_metrics,
        "fusion_comparator": fusion_report,
        "scores": {"val": val_scores, "test": test_scores},
        "history": history,
        "input_audit": {
            "n_train": train_ds.len(), "n_val": val_ds.len(), "n_test": test_ds.len(),
            "edge_mode": cfg["edge_mode"], "edge_dropout": cfg["edge_dropout"],
            "edge_text": cfg["edge_text"],
        },
    }
    rid = time.strftime("%Y%m%d_%H%M%S")
    p = out_dir / f"phase2_{args.candidate}_s{args.seed}_{rid}.json"
    p.write_text(json.dumps(result, indent=1))
    print(f"saved {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
