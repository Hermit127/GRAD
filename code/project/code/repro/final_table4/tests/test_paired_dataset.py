"""Validation tests for the paired-authorization dataset (zero API)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))

PAIRS = REPO / 'code/repro/out/paired_repair_v1'


@unittest.skipUnless((PAIRS / 'manifest.json').exists(),
                     'paired dataset not built in this checkout')
class TestPairedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((PAIRS / 'manifest.json').read_text())
        cls.pairs = {role: [json.loads(l) for l in
                            (PAIRS / f'{role}.jsonl').read_text().splitlines()
                            if l.strip()]
                     for role in ('train', 'dev', 'test_probe')}

    def test_leakage(self):
        leak = self.manifest['leakage']
        self.assertEqual(leak['train_test_overlap'], [])
        self.assertEqual(leak['dev_test_overlap'], [])
        self.assertEqual(leak['train_dev_overlap'], [])

    def test_candidate_swap_changes_only_last_node(self):
        pair = next(p for p in self.pairs['dev']
                    if p['type'] == 'candidate_swap')
        a, b = pair['nodes_a'], pair['nodes_b']
        self.assertEqual(len(a), len(b))
        for na, nb in zip(a[:-1], b[:-1]):
            self.assertEqual(na, nb, 'prefix nodes must be identical')
        self.assertNotEqual(a[-1]['features']['subtask'],
                            b[-1]['features']['subtask'])
        self.assertIn('status=proposed', a[-1]['features']['subtask'])
        self.assertIn('status=proposed', b[-1]['features']['subtask'])
        self.assertEqual(pair['label_a_harm'], 1)
        self.assertEqual(pair['label_b_harm'], 0)
        self.assertEqual(pair['expected_order'], 'A>B')

    def test_evidence_swap_changes_no_candidate(self):
        pairs = [p for p in self.pairs['dev']
                 if p['type'] == 'evidence_swap']
        self.assertTrue(pairs)
        pair = pairs[0]
        self.assertEqual(pair['nodes_a'][-1], pair['nodes_b'][-1])
        a_text = ' '.join(n['features']['subtask'] for n in pair['nodes_a'])
        b_text = ' '.join(n['features']['subtask'] for n in pair['nodes_b'])
        self.assertNotEqual(a_text, b_text)

    def test_pair_counts_match_manifest(self):
        for role in ('train', 'dev', 'test_probe'):
            self.assertEqual(len(self.pairs[role]),
                             self.manifest['splits'][role]['n_pairs'])

    def test_label_sanity(self):
        for role in ('train', 'dev'):
            for p in self.pairs[role]:
                if p['type'] == 'candidate_swap':
                    self.assertEqual(p['label_a_harm'], 1)
                    self.assertEqual(p['label_b_harm'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
