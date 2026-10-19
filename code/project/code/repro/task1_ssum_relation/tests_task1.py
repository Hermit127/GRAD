#!/usr/bin/env python3
"""Contract tests for the Task 1 relation pilot (no training)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfgen import FAMILIES, check_pair, make_pair, verify_harm  # noqa: E402
from graphbuild import (build_graph, heuristic_edges, perturb_degree_swap,  # noqa: E402
                        perturb_endpoint_rewire, perturb_irrelevant, prefix_graph)
from serialize import chrono_text, relation_text  # noqa: E402
from variants import VARIANT_CONFIGS, build_node_type_vocab, graph_to_data  # noqa: E402
from protocol_a import trajectory_scores  # noqa: E402
from online_loop import outcome, verify_freeze  # noqa: E402


def sample_pair(kind="read_target", family=0, pair=3, seed=211):
    a, b = make_pair(family, pair, kind, seed)
    assert check_pair(a, b)
    return a, b


class TestCfgen(unittest.TestCase):
    def test_pair_invariants_kinds(self):
        for kind in ("read_target", "param_source", "auth_source"):
            a, b = sample_pair(kind)
            ga, gb = build_graph(a), build_graph(b)
            ea = sorted((e["source"], e["target"], e["type"]) for e in ga["edges"])
            eb = sorted((e["source"], e["target"], e["type"]) for e in gb["edges"])
            self.assertEqual([t for *_, t in ea], [t for *_, t in eb])
            self.assertEqual(len([e for e in ea if e not in eb]), 1)

    def test_label_from_semantics_not_edge_change(self):
        a, b = sample_pair("read_target")
        self.assertEqual(verify_harm(a), 1)
        self.assertEqual(verify_harm(b), 0)
        edge = [e for e in a["edges"] if e["type"] == "reads_from"][0]
        other = [e for e in b["edges"] if e["type"] == "reads_from"][0]
        edge["target"] = other["target"]
        self.assertEqual(verify_harm(a), 0)

    def test_chrono_text_identical(self):
        a, b = sample_pair("param_source")
        ga, gb = build_graph(a), build_graph(b)
        self.assertEqual(chrono_text(ga), chrono_text(gb))
        ra = relation_text(ga).splitlines()
        rb = relation_text(gb).splitlines()
        diff = [x for x in ra if x not in rb]
        self.assertEqual(len(diff), 1)


class TestCfgenV2(unittest.TestCase):
    def test_pair_invariants_all_kinds(self):
        from cfgen2 import KINDS, check_pair_v2, make_pair_v2, verify_policy
        for ki, kind in enumerate(KINDS):
            a, b = make_pair_v2(0, ki, kind, 211)
            self.assertTrue(check_pair_v2(a, b))
            self.assertEqual(verify_policy(a), 1)
            self.assertEqual(verify_policy(b), 0)

    def test_authorized_legitimate_untrusted_both_sides(self):
        from cfgen2 import closure, make_pair_v2, UNTRUSTED_TYPES
        a, b = make_pair_v2(0, 3, "authorized", 211)
        for s in (a, b):
            cl, by = closure(s, s["proposed"],
                             ("param_source", "reads_from", "derived_from"))
            self.assertTrue(any(by.get(x, {}).get("type") in UNTRUSTED_TYPES
                                for x in cl))
        self.assertEqual(a["label"], 1)
        self.assertEqual(b["label"], 0)

    def test_label_from_policy_not_reachability(self):
        from cfgen2 import make_pair_v2, verify_policy
        a, b = make_pair_v2(1, 3, "authorized", 212)
        auth = [e for e in b["edges"] if e["type"] == "authorization"][0]
        other = [e for e in a["edges"] if e["type"] == "authorization"][0]
        auth["target"] = other["target"]
        self.assertEqual(verify_policy(b), 1)


class TestRepresentationGap(unittest.TestCase):
    def test_heuristic_edges_blind_to_relation(self):
        a, b = sample_pair("read_target")
        ha, hb = heuristic_edges(a), heuristic_edges(b)
        sig_a = sorted((e["source"], e["target"], e["type"]) for e in ha)
        sig_b = sorted((e["source"], e["target"], e["type"]) for e in hb)
        self.assertEqual(sig_a, sig_b)

    def test_provenance_edges_visible(self):
        a, b = sample_pair("read_target")
        pa, pb = build_graph(a)["edges"], build_graph(b)["edges"]
        sig_a = sorted((e["source"], e["target"], e["type"]) for e in pa)
        sig_b = sorted((e["source"], e["target"], e["type"]) for e in pb)
        self.assertNotEqual(sig_a, sig_b)
        self.assertEqual(len([e for e in sig_a if e not in sig_b]), 1)

    def test_wrong_source_matched_statistics(self):
        a, b = sample_pair("read_target")
        for scenario in (a, b):
            base = build_graph(scenario)
            wrong = build_graph(scenario, wrong_source=True)
            self.assertEqual(len(base["edges"]), len(wrong["edges"]))
            self.assertEqual(sorted(e["type"] for e in base["edges"]),
                             sorted(e["type"] for e in wrong["edges"]))
        wa = build_graph(a, wrong_source=True)
        wb = build_graph(b, wrong_source=True)
        sa = sorted((e["source"], e["target"], e["type"]) for e in wa["edges"])
        sb = sorted((e["source"], e["target"], e["type"]) for e in wb["edges"])
        self.assertEqual(sa, sb)

    def test_drop_edges(self):
        a, _ = sample_pair("param_source")
        g = build_graph(a)
        nr = build_graph(a, drop_edges=("reads_from",))
        np_ = build_graph(a, drop_edges=("param_source",))
        self.assertEqual([e for e in nr["edges"] if e["type"] == "reads_from"], [])
        self.assertEqual([e for e in np_["edges"] if e["type"] == "param_source"], [])
        self.assertEqual(len(g["edges"]) - len(nr["edges"]), 1)
        self.assertEqual(len(g["edges"]) - len(np_["edges"]), 1)


class TestTensorization(unittest.TestCase):
    def test_shapes_and_no_leak(self):
        a, _ = sample_pair()
        g = build_graph(a)
        cfg = VARIANT_CONFIGS["full_provenance"]
        vocab = build_node_type_vocab([g])
        d = graph_to_data(g, cfg, vocab)
        self.assertEqual(d.x_base.size(0), len(g["nodes"]))
        self.assertEqual(d.edge_index.size(1), len(g["edges"]))
        blob = d.x_base.abs().sum().item() + d.node_chars.sum().item() + \
            d.edge_attr_base.sum().item() + d.edge_chars.sum().item()
        self.assertNotIn("harm", json.dumps(g["nodes"]))
        self.assertNotIn("safe", json.dumps(g["nodes"]))
        self.assertTrue(blob >= 0)

    def test_text_only_has_no_edges(self):
        a, _ = sample_pair()
        g = build_graph(a, mode="none")
        cfg = VARIANT_CONFIGS["text_only"]
        d = graph_to_data(g, cfg)
        self.assertEqual(d.edge_index.numel(), 0)


class TestPrefixContract(unittest.TestCase):
    def test_pre_call_hides_candidate_edges_and_return(self):
        a, _ = sample_pair()
        g = build_graph(a)
        g["nodes"][-1]["features"]["return"] = "LEAK_TOKEN"
        k = len(g["nodes"]) - 1
        pg = prefix_graph(g, k, candidate_phase="pre_call")
        self.assertNotIn("return", pg["nodes"][-1]["features"])
        candidate_id = pg["nodes"][-1]["id"]
        for e in pg["edges"]:
            self.assertNotEqual(e["source"], candidate_id)
            self.assertNotEqual(e["target"], candidate_id)
        pg2 = prefix_graph(g, k, candidate_phase="post_return")
        self.assertIn("return", pg2["nodes"][-1]["features"])

    def test_prefix_edges_visible_only(self):
        a, _ = sample_pair()
        g = build_graph(a)
        pg = prefix_graph(g, 1, candidate_phase="pre_call")
        ids = {n["id"] for n in pg["nodes"]}
        for e in pg["edges"]:
            self.assertIn(e["source"], ids)
            self.assertIn(e["target"], ids)


class TestPerturbations(unittest.TestCase):
    def test_keep_counts(self):
        a, _ = sample_pair()
        g = build_graph(a)
        for fn in (perturb_endpoint_rewire, perturb_degree_swap, perturb_irrelevant):
            p = fn(g, seed=1)
            self.assertEqual(len(p["nodes"]), len(g["nodes"]))
            self.assertEqual(sorted(e["type"] for e in p["edges"]),
                             sorted(e["type"] for e in g["edges"]))


class TestWindowAggregation(unittest.TestCase):
    def test_deadline_and_benign_rule(self):
        probs = [0.1, 0.2, 0.9, 0.95]
        self.assertAlmostEqual(trajectory_scores(probs, deadline=2, label=1), 0.9)
        self.assertAlmostEqual(trajectory_scores(probs, deadline=2, label=1,
                                                 min_index=1), 0.9)
        self.assertAlmostEqual(trajectory_scores(probs, deadline=None, label=0), 0.95)
        probs2 = [0.9, 0.1, 0.2, 0.3]
        self.assertAlmostEqual(trajectory_scores(probs2, deadline=2, label=1), 0.9)
        self.assertIsNone(trajectory_scores([None], deadline=0, label=1))


class TestOnlineLoop(unittest.TestCase):
    def test_outcome_semantics(self):
        from online_loop import outcome
        a = outcome("allow", "harm")
        self.assertTrue(a["harm_executed"] and a["task_completed"]
                        and not a["utility_under_attack"])
        b = outcome("block", "harm")
        self.assertFalse(b["harm_executed"])
        self.assertTrue(b["task_completed"] and b["utility_under_attack"])
        c = outcome("block", "safe")
        self.assertFalse(c["task_completed"])
        self.assertTrue(c["false_block"])
        d = outcome("allow", "safe")
        self.assertTrue(d["task_completed"] and not d["false_block"])

    def test_freeze_verification_detects_tamper(self):
        import hashlib
        import tempfile
        from online_loop import verify_freeze
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pairs = td / "pairs.jsonl"
            pairs.write_text('{"a":1}\n')
            ckpt = td / "m.pt"
            ckpt.write_bytes(b"weights")
            run = td / "runs" / "full_provenance_s211.json"
            run.parent.mkdir()
            run.write_text(json.dumps({"metrics": {"dev": {"threshold_at_target_fpr":
                                                           {"threshold": 0.5}}},
                                       "config": {}}))
            h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            freeze = {"pairs_file": str(pairs), "pairs_sha256": h(pairs),
                      "pairs_manifest": None, "pairs_manifest_sha256": None,
                      "defenses": {"full_provenance": {
                          "checkpoint": str(ckpt), "checkpoint_sha256": h(ckpt),
                          "run_json": str(run), "run_json_sha256": h(run)}},
                      "loop_code": {}}
            fp = td / "freeze.json"
            fp.write_text(json.dumps(freeze))
            self.assertTrue(verify_freeze(fp))
            ckpt.write_bytes(b"tampered")
            with self.assertRaises(SystemExit):
                verify_freeze(fp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
