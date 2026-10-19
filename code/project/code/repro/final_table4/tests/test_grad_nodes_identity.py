"""Zero-API identity test: live GRAD v2 node builder == training builder.

Compares ``grad_nodes.v10_trace_nodes`` (deployment) against
``build_gate_dataset_full_20260917.full_nodes`` (training) on every historical
v10 no-gate run, and against the stored ``online_gate_v2`` split rows.

Run:
  python -m unittest \
      code.repro.final_table4.tests.test_grad_nodes_identity
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

import build_gate_dataset_full_20260917 as trainer  # noqa: E402
from envlib import ENVS, valid_authorization  # noqa: E402
from grad_nodes import v10_trace_nodes  # noqa: E402

SOURCE_FILES = trainer.DEFAULT_FILES
SPLITS = REPO / 'code/repro/out/online_gate_v2'


def _node_signature(nodes):
    return [(n['type'], n.get('status'), n['features']['subtask'],
             n['features'].get('source_kind'))
            for n in nodes]


class TestGradNodeIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        for role in ('train', 'dev', 'test'):
            p = SPLITS / f'{role}.jsonl'
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        cls.rows.setdefault(rec['task_id'], []).append(rec)
        cls.n_runs = 0
        cls.n_compared = 0
        cls.n_rows_matched = 0

    def _iter_runs(self, name):
        data = json.loads((REPO / 'LOGS' / name).read_text())
        for rec in data.get('runs', []):
            if rec.get('arm') != 'none' or rec.get('error'):
                continue
            self.n_runs += 1
            yield rec

    def test_builder_identity(self):
        for name in SOURCE_FILES:
            path = REPO / 'LOGS' / name
            if not path.exists():
                self.skipTest(f'missing source {name}')
                continue
            for rec in self._iter_runs(name):
                env = ENVS[rec['env']]
                trace = rec['trace']
                events = trace.get('events', [])
                idx = None
                for i, ev in enumerate(events):
                    if ev.get('tool') == env['action']:
                        idx = i
                if idx is None:
                    continue
                tool, arg = events[idx]['tool'], events[idx]['arg']
                if valid_authorization(trace, env, tool, arg):
                    continue
                expect = trainer.full_nodes(trace, events[:idx],
                                            candidate=(tool, arg))
                prefix = dict(trace)
                prefix['events'] = events[:idx]
                got = v10_trace_nodes(prefix, candidate=(tool, arg))
                self.assertEqual(_node_signature(expect), _node_signature(got),
                                 msg=f'{rec["run_id"]} nodes differ')
                self.n_compared += 1
                task_id = f"{rec['run_id']}|{idx}"
                rows = self.rows.get(task_id)
                if rows:
                    sig = _node_signature(got)
                    if any(_node_signature(r['nodes']) == sig for r in rows):
                        self.n_rows_matched += 1
        print(f'[identity] runs={self.n_runs} compared={self.n_compared} '
              f'stored_rows_matched={self.n_rows_matched}')
        self.assertGreater(self.n_compared, 0)
        self.assertGreater(self.n_rows_matched, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
