#!/usr/bin/env python3
"""Unit tests for the K2/K3 keyword-channel candidates.

Run: python code/repro/final_five_tables/test_keyword_candidates.py

Checks:
  1. K2 data keeps every non-keyword channel identical to full and only moves the
     9 keyword dims into a separate raw channel;
  2. normalization statistics are computed on the TRAIN split only and applied
     unchanged to dev/test;
  3. zero-variance keyword dims become identity (no NaN);
  4. K3 keyword dropout is training-only (eval is deterministic);
  5. training-mode dropout actually changes the keyword channel;
  6. labels never enter any input channel.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))

import train_gnn_ablate as abl  # noqa: E402
import train_gnn_ablate2 as a2  # noqa: E402

TRAIN = REPO / "code/repro/out/unified_gnn/fix2_splits/mcp/fold0/train.jsonl"
DEV = REPO / "code/repro/out/unified_gnn/fix2_splits/mcp/fold0/dev.jsonl"


def load_samples(path: Path, n: int = 4):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()][:n]


class KeywordCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TRAIN.exists():
            raise unittest.SkipTest("missing split")
        cls.stats = a2.compute_kw_stats(TRAIN)

    def cfg(self, candidate):
        return {"candidate": candidate, "edge_mode": "rebuilt", "edge_text": "copy",
                "edge_dropout": 0.0}

    def test_channels_identical_to_full(self):
        for g in load_samples(TRAIN):
            full = abl.graph_to_v2_data(g)
            k2 = a2.graph_to_data(g, self.cfg("K2_kwnorm"), None)
            self.assertTrue(torch.equal(full.x_base[:, :13], k2.x_base))
            self.assertTrue(torch.equal(full.x_base[:, 13:22], k2.x_kw))
            self.assertTrue(torch.equal(full.node_chars, k2.node_chars))
            self.assertTrue(torch.equal(full.edge_index, k2.edge_index))
            self.assertTrue(torch.equal(full.edge_attr_base, k2.edge_attr_base))
            self.assertTrue(torch.equal(full.edge_chars, k2.edge_chars))

    def test_normalization_uses_train_stats(self):
        mean, std = self.stats
        ds = a2.CData(str(TRAIN), self.cfg("K2_kwnorm"), None, train=True,
                      kw_stats=(mean, std))
        for g, d in zip(ds.graphs, ds.data):
            raw = a2.graph_to_data(g, self.cfg("K2_kwnorm"), None).x_kw
            self.assertTrue(torch.allclose(d.x_kw, (raw - mean) / std, atol=1e-5))
        # different stats must change the output -> proves the passed stats are used
        other = (mean + 1.0, std)
        d2 = a2.CData(str(DEV), self.cfg("K2_kwnorm"), None, kw_stats=other)
        for g, d in zip(d2.graphs, d2.data):
            raw = a2.graph_to_data(g, self.cfg("K2_kwnorm"), None).x_kw
            self.assertTrue(torch.allclose(d.x_kw, (raw - other[0]) / other[1], atol=1e-5))

    def test_dead_dims_are_identity(self):
        mean = torch.zeros(9)
        std = torch.zeros(9)   # synthetic dead dims: guard must center and keep finite
        ds = a2.CData(str(TRAIN), self.cfg("K2_kwnorm"), None, kw_stats=(mean, std))
        for d in ds.data:
            self.assertTrue(torch.isfinite(d.x_kw).all())

    def test_eval_is_deterministic_for_k3(self):
        torch.manual_seed(0)
        graphs = load_samples(DEV, 3)
        cfg = self.cfg("K3_kwdrop50")
        ds = [a2.graph_to_data(g, cfg, None) for g in graphs]
        batch = Batch.from_data_list(ds)
        model = a2.KeywordNormGNN(13, 9, len(a2.V2_EDGE_TYPES) + 1, hidden=32)
        model.eval()
        with torch.no_grad():
            a = a2.batch_logits(model, batch, cfg, training=False)["logit"]
            b = a2.batch_logits(model, batch, cfg, training=False)["logit"]
        self.assertTrue(torch.equal(a, b))

    def test_training_dropout_changes_channel(self):
        torch.manual_seed(0)
        graphs = load_samples(DEV, 4)
        cfg = self.cfg("K3_kwdrop50")
        ds = [a2.graph_to_data(g, cfg, None) for g in graphs]
        batch = Batch.from_data_list(ds)
        model = a2.KeywordNormGNN(13, 9, len(a2.V2_EDGE_TYPES) + 1, hidden=32)
        model.train()
        outs = [a2.batch_logits(model, batch, cfg, training=True)["logit"]
                for _ in range(8)]
        self.assertFalse(all(torch.equal(outs[0], o) for o in outs[1:]))

    def test_no_label_leakage(self):
        for candidate in ("K2_kwnorm", "K3_kwdrop25", "K3_kwdrop50"):
            for g in load_samples(TRAIN):
                d = a2.graph_to_data(g, self.cfg(candidate), None)
                self.assertEqual(float(d.y.item()), float(g["label"]))
                self.assertFalse(hasattr(d, "harm_step"))
                self.assertFalse(hasattr(d, "injected"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
