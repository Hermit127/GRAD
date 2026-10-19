#!/usr/bin/env python3
"""Text baselines for the Task 1 pilot.

The strong text control receives the full serialized relation set; to make the
control informative rather than trivially weak it uses a small token-level
Transformer encoder (2 layers, 4 heads) over the serialized text. Both text
arms (chronological and relation-serialized) share architecture, budget, split
and dev-selection rule with each other and with the graph arms' protocol.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

MAX_TOKENS = 512
TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+|[^\sA-Za-z\d_]")
PAD, UNK = 0, 1


def tokenize(text):
    return TOKEN_RE.findall(text or "")


def build_vocab(texts, max_size=8000, min_count=1):
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))
    items = [(tok, c) for tok, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for tok, _ in items[:max_size - 2]:
        vocab[tok] = len(vocab)
    return vocab


def encode_text(text, vocab):
    ids = [vocab.get(tok, UNK) for tok in tokenize(text)][:MAX_TOKENS]
    return ids + [PAD] * (MAX_TOKENS - len(ids))


def encode_text_audit(text, vocab):
    toks = tokenize(text)
    ids = [vocab.get(tok, UNK) for tok in toks][:MAX_TOKENS]
    return ids + [PAD] * (MAX_TOKENS - len(ids)), len(toks), len(toks) > MAX_TOKENS


class TokenTextClassifier(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, emb_dim=64, layers=2,
                 heads=4, dropout=0.3):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.pos = torch.nn.Embedding(MAX_TOKENS, emb_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=layers)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, ids):
        pad_mask = ids.eq(PAD)
        x = self.emb(ids) + self.pos(torch.arange(ids.size(1), device=ids.device))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(pooled).view(-1)
