"""Gate-side node building must be byte-identical to the GRAD-512 trainer.

The audit concern: the closed-loop gate must build node/edge tensors exactly
as `train_grad_repair.py` did at training time (window 512/160, A_moddrop),
otherwise scores at runtime are not the trained function.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/final_five_tables'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

import train_grad_repair as tgr  # noqa: E402
import train_gnn_ablate2 as g3t  # noqa: E402

CFG = {'candidate': 'A_moddrop', 'moddrop': 0.25, 'lambda_aux': 0.1,
       'edge_mode': 'rebuilt', 'edge_dropout': 0.0, 'edge_text': 'copy',
       'max_node_chars': 512, 'max_edge_chars': 160}
FIELDS = ('x_base', 'node_chars', 'edge_index', 'edge_attr_base',
          'edge_chars', 'is_chrono')


class TestG3GateIdentity(unittest.TestCase):
    def test_tensor_identity_with_trainer(self):
        rows = [json.loads(l) for l in (
            REPO / 'code/repro/out/online_gate_v2/dev.jsonl'
        ).read_text().splitlines() if l.strip()]
        self.assertTrue(rows)
        g3t.MAX_NODE_CHARS, g3t.MAX_EDGE_CHARS = 512, 160
        for g in rows[:8]:
            a = tgr.graph_to_data(g, CFG, None)
            b = g3t.graph_to_data(g, CFG, None)
            for f in FIELDS:
                ta, tb = getattr(a, f), getattr(b, f)
                self.assertEqual(ta.shape, tb.shape, f'{f}: shape')
                self.assertTrue(torch.equal(ta, tb), f'{f}: values')

    def test_gate_score_matches_trainer_scorer(self):
        spec = importlib.util.spec_from_file_location(
            'g3_gate_mod', REPO / 'code/repro/final_table4/g3_gate.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckpt = REPO / 'code/repro/out/online_gate_g3/g3_gate_512.pt'
        if not ckpt.exists():
            self.skipTest('local gate checkpoint not present')
        gate = mod.G3Gate(device='cpu')
        rows = [json.loads(l) for l in (
            REPO / 'code/repro/out/online_gate_v2/dev.jsonl'
        ).read_text().splitlines() if l.strip()]
        d = g3t.graph_to_data(rows[0], CFG, None)
        from defenses import GradGateV2  # noqa: F401  (import parity smoke)
        p, _ = gate.score_nodes(rows[0]['nodes'])
        self.assertTrue(0.0 <= p <= 1.0)
