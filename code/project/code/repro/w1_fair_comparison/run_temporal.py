#!/usr/bin/env python3
"""W1 temporal arm: order-aware sequence model over the same char encoder.

Contract: identical node text + normalized position as the frozen text_only
variant (no edges, no permission vector, no keyword flags); CharCNN per step
(reused class from the frozen trainer), then a Transformer encoder over the
turn-ordered step sequence, masked attention pooling, same classifier head.

Runs the frozen variant trainer main() with the model class patched. Every
other detail (splits, budget, hard-benign mining, dev fixed-FPR F1 selection,
determinism wrapper) is inherited unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if os.environ.get("DET_MODE", "full") == "full":
    torch.use_deterministic_algorithms(True)

_FT = Path(__file__).resolve().parents[1] / "final_five_tables"
sys.path.insert(0, str(_FT))
import train_gnn_variant_repair as base  # noqa: E402


class TemporalCharNet(torch.nn.Module):
    def __init__(self, node_base_dim: int, edge_base_dim: int,
                 hidden_dim: int = 192, char_dim: int = 64,
                 dropout: float = 0.35, heads: int = 4, layers: int = 2):
        super().__init__()
        self.node_char = base.CharCNN(out_dim=char_dim)
        self.pos_proj = torch.nn.Linear(node_base_dim, hidden_dim)
        self.in_proj = torch.nn.Linear(char_dim, hidden_dim)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.attn = torch.nn.Linear(hidden_dim, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, x_base, node_chars, edge_index, edge_attr_base,
                edge_chars, batch_vector):
        h = self.in_proj(base.encode_char_rows(self.node_char, node_chars))
        h = h + self.pos_proj(x_base)
        device = h.device
        b = batch_vector.to(device)
        n_graphs = int(b.max().item()) + 1
        counts = torch.bincount(b, minlength=n_graphs)
        max_len = int(counts.max().item())
        offsets = torch.cumsum(counts, 0) - counts
        pos = torch.arange(h.size(0), device=device) - offsets[b]
        flat = b * max_len + pos
        x = torch.zeros(n_graphs * max_len, h.size(-1), device=device)
        x = x.index_copy(0, flat, h).view(n_graphs, max_len, h.size(-1))
        pad = torch.arange(max_len, device=device)[None, :] >= counts[:, None].to(device)
        x = self.encoder(x, src_key_padding_mask=pad)
        a = self.attn(x).sigmoid().masked_fill(pad.unsqueeze(-1), 0.0)
        pooled = (x * a).sum(1) / a.sum(1).clamp(min=1e-8)
        return self.classifier(pooled).squeeze(-1)


if os.environ.get("W1_PATCH", "1") == "1":
    base.CausalCharGNN = TemporalCharNet
if os.environ.get("W1_DEVICE"):
    base.DEVICE = torch.device(os.environ["W1_DEVICE"])

if __name__ == "__main__":
    raise SystemExit(base.main())
