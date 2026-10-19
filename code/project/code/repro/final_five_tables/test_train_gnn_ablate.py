#!/usr/bin/env python3
"""Input-level contract tests for Table 5 Panel A ablation variants.

Run: python code/repro/final_five_tables/test_train_gnn_ablate.py

Checks (per variant, on real corpus samples):
  1. variant="full" is tensor-identical to the original code/train_gnn_v2.py;
  2. every removal actually removes the claimed information;
  3. everything not claimed as removed stays identical to full;
  4. labels / harm_step / attack-family annotations never enter model inputs;
  5. edge indices are in-bounds and edge tensors are row-aligned;
  6. a short forward+backward works for every variant (empty-edge text_only).
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
sys.path.insert(0, str(REPO / "code" / "repro" / "final_five_tables"))

import train_gnn_v2 as orig  # noqa: E402
import train_gnn_ablate as abl  # noqa: E402

CORPORA = {
    "decomp": REPO / "code/repro/out/unified_gnn/fix2_splits/decomp/fold0/train.jsonl",
    "mcp": REPO / "code/repro/out/unified_gnn/fix2_splits/mcp/fold0/train.jsonl",
    "mrt": REPO / "results/mrt_cross_task_v1/fold_spam_filter_update/train.jsonl",
}


def load_samples(path: Path, n: int = 6) -> list[dict]:
    graphs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                graphs.append(json.loads(line))
            if len(graphs) >= n:
                break
    return graphs


def tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a, b)


class AblationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = {}
        for name, path in CORPORA.items():
            if path.exists():
                cls.samples[name] = load_samples(path)
            else:
                raise unittest.SkipTest(f"missing corpus {path}")

    def build(self, variant: str, g: dict, vocab=None):
        abl.VARIANT = variant
        return abl.graph_to_v2_data(g, vocab)

    def test_full_identical_to_original(self):
        for name, graphs in self.samples.items():
            for g in graphs:
                with self.subTest(corpus=name, task=g.get("task_id")):
                    d_orig = orig.graph_to_v2_data(g)
                    d_new = self.build("full", g)
                    for key in ("x_base", "node_chars", "edge_index",
                                "edge_attr_base", "edge_chars", "y"):
                        self.assertTrue(
                            tensors_equal(getattr(d_orig, key), getattr(d_new, key)),
                            f"{name}: full variant differs from original on {key}")

    def test_text_only_removals(self):
        for name, graphs in self.samples.items():
            for g in graphs:
                with self.subTest(corpus=name):
                    full = self.build("full", g)
                    d = self.build("text_only", g)
                    self.assertEqual(d.x_base.size(1), 1, "text_only keeps only position")
                    self.assertEqual(d.edge_index.numel(), 0, "text_only must have no edges")
                    self.assertEqual(d.edge_attr_base.size(0), 0)
                    self.assertEqual(d.edge_chars.size(0), 0)
                    self.assertTrue(tensors_equal(full.node_chars, d.node_chars),
                                    "text_only keeps the same visible text channel")

    def test_structure_only_removals_and_retention(self):
        for name, graphs in self.samples.items():
            vocab = abl.build_node_type_vocab(CORPORA[name])
            for g in graphs:
                with self.subTest(corpus=name):
                    full = self.build("full", g)
                    d = self.build("structure_only", g, vocab)
                    self.assertEqual(d.x_base.size(1), len(abl.PERMISSION) + 1 + len(vocab))
                    self.assertFalse(d.node_chars.any(), "node text must be removed")
                    self.assertFalse(d.edge_chars.any(), "edge text must be removed")
                    self.assertTrue(tensors_equal(full.edge_index, d.edge_index),
                                    "structure_only keeps the full topology")
                    self.assertTrue(tensors_equal(full.edge_attr_base, d.edge_attr_base),
                                    "structure_only keeps edge type+confidence")
                    n = d.x_base.size(0)
                    self.assertTrue(torch.equal(d.x_base[:, :len(abl.PERMISSION)],
                                                full.x_base[:, :len(abl.PERMISSION)]))
                    self.assertTrue(torch.equal(d.x_base[:, len(abl.PERMISSION)],
                                                full.x_base[:, len(abl.PERMISSION)]))
                    type_block = d.x_base[:, len(abl.PERMISSION) + 1:]
                    self.assertTrue(bool(((type_block.sum(dim=1) == 1)
                                          | (type_block.sum(dim=1) == 0)).all()))

    def test_chrono_only_removals_and_retention(self):
        for name, graphs in self.samples.items():
            for g in graphs:
                with self.subTest(corpus=name):
                    full = self.build("full", g)
                    d = self.build("chrono_only", g)
                    self.assertTrue(tensors_equal(full.x_base, d.x_base))
                    self.assertTrue(tensors_equal(full.node_chars, d.node_chars))
                    seq = [e for e in abl.rebuild_v2_edges(g) if e["type"] == "sequence_next"]
                    self.assertEqual(d.edge_index.size(1), len(seq))
                    full_edges = {tuple(full.edge_index[:, i].tolist())
                                  for i in range(full.edge_index.size(1))}
                    for i in range(d.edge_index.size(1)):
                        key = tuple(d.edge_index[:, i].tolist())
                        self.assertIn(key, full_edges, "chrono edges must be full-edge subset")
                    self.assertFalse((d.edge_attr_base[:, 1:-1] != 0).any(),
                                     "chrono_only must not carry dependency type labels")
                    self.assertTrue(
                        bool((d.edge_attr_base[:, 0] == 1.0).all()),
                        "chrono_only uses the single sequence_next channel")
                    self.assertTrue(
                        bool((d.edge_attr_base[:, -1] == 1.0).all()),
                        "chrono_only confidence is the constant 1.0")

    def test_no_relation_types_removals_and_retention(self):
        for name, graphs in self.samples.items():
            for g in graphs:
                with self.subTest(corpus=name):
                    full = self.build("full", g)
                    d = self.build("no_relation_types", g)
                    self.assertTrue(tensors_equal(full.edge_index, d.edge_index),
                                    "topology must be unchanged")
                    self.assertTrue(tensors_equal(full.node_chars, d.node_chars))
                    self.assertTrue(tensors_equal(full.x_base, d.x_base))
                    self.assertTrue(tensors_equal(full.edge_chars, d.edge_chars))
                    self.assertTrue(bool((d.edge_attr_base[:, 0] == 1.0).all()))
                    self.assertFalse(bool((d.edge_attr_base[:, 1:-1] != 0).any()),
                                     "type one-hot must be unified")
                    self.assertTrue(tensors_equal(full.edge_attr_base[:, -1:],
                                                  d.edge_attr_base[:, -1:]),
                                    "confidence retained -> type-label ablation only")

    def test_no_keyword_removals_and_retention(self):
        for name, graphs in self.samples.items():
            for g in graphs:
                with self.subTest(corpus=name):
                    full = self.build("full", g)
                    d = self.build("no_keyword_features", g)
                    keep = len(abl.PERMISSION) + 1
                    self.assertEqual(d.x_base.size(1), keep)
                    self.assertTrue(tensors_equal(full.x_base[:, :keep], d.x_base))
                    self.assertTrue(tensors_equal(full.node_chars, d.node_chars))
                    self.assertTrue(tensors_equal(full.edge_index, d.edge_index))
                    self.assertTrue(tensors_equal(full.edge_attr_base, d.edge_attr_base))

    def test_labels_never_enter_inputs(self):
        for name, graphs in self.samples.items():
            for g in graphs:
                for variant in abl.ABLATION_VARIANTS:
                    with self.subTest(corpus=name, variant=variant):
                        d = self.build(variant, g)
                        self.assertEqual(float(d.y.item()), float(g.get("label", 0)))
                        self.assertFalse(hasattr(d, "harm_step"))
                        self.assertFalse(hasattr(d, "injected"))
                        self.assertFalse(hasattr(d, "node_label"))

    def test_edge_bounds_and_alignment(self):
        for name, graphs in self.samples.items():
            vocab = abl.build_node_type_vocab(CORPORA[name])
            for g in graphs:
                for variant in abl.ABLATION_VARIANTS:
                    with self.subTest(corpus=name, variant=variant):
                        d = self.build(variant, g, vocab)
                        n = d.x_base.size(0)
                        if d.edge_index.numel():
                            self.assertGreaterEqual(int(d.edge_index.min()), 0)
                            self.assertLess(int(d.edge_index.max()), n)
                        self.assertEqual(d.edge_index.size(1), d.edge_attr_base.size(0))
                        self.assertEqual(d.edge_index.size(1), d.edge_chars.size(0))

    def test_structure_only_unseen_type_is_zero(self):
        g = {
            "task_id": "synthetic", "label": 1,
            "nodes": [
                {"id": "a", "type": "never_seen_type", "turn": 0, "features": {"subtask": "x"}},
                {"id": "b", "type": "memory_write", "turn": 1, "features": {"subtask": "y"}},
            ],
            "edges": [{"source": "a", "target": "b", "type": "data_flow"}],
        }
        vocab = ["memory_write"]
        d = self.build("structure_only", g, vocab)
        type_block = d.x_base[:, len(abl.PERMISSION) + 1:]
        self.assertTrue(torch.equal(type_block[0], torch.zeros_like(type_block[0])))
        self.assertEqual(float(type_block[1].sum()), 1.0)

    def test_forward_backward_all_variants(self):
        torch.manual_seed(0)
        name = "mcp"
        graphs = self.samples[name][:2]
        vocab = abl.build_node_type_vocab(CORPORA[name])
        for variant in abl.ABLATION_VARIANTS:
            with self.subTest(variant=variant):
                abl.VARIANT = variant
                ds = [abl.graph_to_v2_data(g, vocab) for g in graphs]
                batch = Batch.from_data_list(ds)
                if variant == "text_only":
                    node_base_dim = 1
                elif variant == "no_keyword_features":
                    node_base_dim = len(abl.PERMISSION) + 1
                elif variant == "structure_only":
                    node_base_dim = len(abl.PERMISSION) + 1 + len(vocab)
                else:
                    node_base_dim = len(abl.PERMISSION) + 1 + len(abl.CONTENT_KEYWORDS)
                model_cls = (abl.CausalCharGNNNoText if variant == "structure_only"
                             else abl.CausalCharGNN)
                model = model_cls(node_base_dim, len(abl.V2_EDGE_TYPES) + 1, hidden_dim=32)
                logits = abl.batch_logits(model, batch)
                self.assertEqual(logits.shape[0], len(ds))
                loss = logits.sum()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
