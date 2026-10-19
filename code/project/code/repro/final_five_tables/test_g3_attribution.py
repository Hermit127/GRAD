#!/usr/bin/env python3
"""Unit tests for G3 attribution: aux heads off at inference, ablation flags,
and the regularized structure-only baseline (S3_reg).

Run: python code/repro/final_five_tables/test_g3_attribution.py
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


def samples(path, n=3):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()][:n]


class G3AttributionTests(unittest.TestCase):
    def cfg(self, **kw):
        base = {"candidate": "A_moddrop", "edge_mode": "rebuilt", "edge_text": "copy",
                "edge_dropout": 0.0, "moddrop": 0.25, "lambda_aux": 0.1,
                "use_ln": True, "use_aux": True, "ablate": "none"}
        base.update(kw)
        return base

    def test_aux_heads_not_executed_at_inference(self):
        graphs = samples(DEV)
        ds = [a2.graph_to_data(g, self.cfg(), None) for g in graphs]
        batch = Batch.from_data_list(ds)
        model = a2.ModalityNormGNN(22, len(a2.V2_EDGE_TYPES) + 1, hidden=32)
        calls = {"aux_text": 0, "aux_struct": 0}

        model.aux_text_proj.register_forward_hook(lambda *_: calls.__setitem__(
            "aux_text", calls["aux_text"] + 1))
        model.aux_struct_proj.register_forward_hook(lambda *_: calls.__setitem__(
            "aux_struct", calls["aux_struct"] + 1))
        model.eval()
        with torch.no_grad():
            out = a2.batch_logits(model, batch, self.cfg(), training=False)
        self.assertNotIn("s_t", out)
        self.assertNotIn("s_s", out)
        self.assertEqual(calls["aux_text"], 0)
        self.assertEqual(calls["aux_struct"], 0)
        # training mode must execute them (loss needs the aux logits)
        model.train()
        out2 = a2.batch_logits(model, batch, self.cfg(), training=True)
        self.assertIn("s_t", out2)
        self.assertIn("s_s", out2)
        self.assertGreater(calls["aux_text"], 0)
        self.assertGreater(calls["aux_struct"], 0)

    def test_ablation_flags(self):
        m_ln = a2.ModalityNormGNN(22, 11, hidden=32, use_ln=True, use_aux=True)
        self.assertIsInstance(m_ln.ln_base, torch.nn.LayerNorm)
        m_noln = a2.ModalityNormGNN(22, 11, hidden=32, use_ln=False, use_aux=True)
        self.assertIsInstance(m_noln.ln_base, torch.nn.Identity)
        m_noaux = a2.ModalityNormGNN(22, 11, hidden=32, use_ln=True, use_aux=False)
        self.assertFalse(hasattr(m_noaux, "aux_text_proj"))
        # no_aux forward must not emit aux keys even in training mode
        graphs = samples(TRAIN)
        ds = [a2.graph_to_data(g, self.cfg(ablate="no_aux"), None) for g in graphs]
        batch = Batch.from_data_list(ds)
        m_noaux.train()
        out = a2.batch_logits(m_noaux, batch, self.cfg(ablate="no_aux"), training=True)
        self.assertNotIn("s_t", out)

    def test_s3_reg_uses_structure_only_inputs(self):
        vocab = a2.build_node_type_vocab(str(TRAIN))
        for g in samples(TRAIN):
            full = abl.graph_to_v2_data(g)
            s3 = a2.graph_to_data(g, self.cfg(candidate="S3_reg"), vocab)
            self.assertEqual(s3.x_base.size(1), 13 + len(vocab))
            self.assertFalse(s3.node_chars.any())          # no text
            self.assertTrue(torch.equal(s3.edge_index, full.edge_index))
            # permissions + position block identical to full
            self.assertTrue(torch.equal(full.x_base[:, :13], s3.x_base[:, :13]))
            self.assertFalse(hasattr(s3, "x_kw"))

    def test_s3_reg_no_keyword_and_no_text_leakage(self):
        g = samples(TRAIN)[0]
        vocab = a2.build_node_type_vocab(str(TRAIN))
        s3 = a2.graph_to_data(g, self.cfg(candidate="S3_reg"), vocab)
        # tail block is a node-type one-hot (sum in {0,1}), not keyword flags
        tail = s3.x_base[:, 13:]
        sums = tail.sum(dim=1)
        self.assertTrue(bool(((sums == 1) | (sums == 0)).all()))
        self.assertFalse(s3.node_chars.any())
        self.assertEqual(float(s3.y.item()), float(g["label"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
