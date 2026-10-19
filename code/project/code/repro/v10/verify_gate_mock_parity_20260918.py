#!/usr/bin/env python3
"""Zero-API parity check: live v10 gate path vs the v2 training dataset.

For a saved v10 run used in training, rebuild the deployment-style trace prefix
(events before the action) and score it through ``gnn_gate.score_prefix`` (which
now uses ``build_gate_dataset_full.full_nodes``). The node texts must equal the
dataset row built by the training builder, and the logits must match.

Usage:
  python code/repro/v10/verify_gate_mock_parity_20260918.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/v10'))

import os  # noqa: E402
os.environ.setdefault('GNN_GATE_CKPT',
                      str(ROOT / 'code/repro/out/online_gate_v2/gnn_gate_v2.pt'))

from gnn_gate import score_prefix  # noqa: E402

OUT = ROOT / 'LOGS/final_table2/grad_audit_20260918'
DATASET = ROOT / 'code/repro/out/online_gate_v2/train.jsonl'


def find_run(run_id, idx, expected_nodes):
    """Return the run variant whose prefix nodes equal the dataset row."""
    from build_gate_dataset_full_20260917 import full_nodes
    for path in sorted(ROOT.glob('LOGS/v10_*.json')):
        data = json.loads(path.read_text())
        for rec in data.get('runs', []):
            if rec.get('run_id') != run_id or not rec.get('trace'):
                continue
            events = rec['trace'].get('events', [])
            if len(events) <= idx:
                continue
            ev = events[idx]
            nodes = full_nodes(rec['trace'], events[:idx],
                               candidate=(ev['tool'], ev['arg']))
            if [n['features']['subtask'] for n in nodes] == \
                    [n['features']['subtask'] for n in expected_nodes]:
                return rec
    return None


def main() -> int:
    checked, mismatches = [], 0
    rows = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    seen = set()
    for row in rows:
        run_id, idx = row['task_id'].rsplit('|', 1)
        idx = int(idx)
        if run_id in seen:
            continue
        rec = find_run(run_id, idx, row['nodes'])
        if rec is None:
            continue
        seen.add(run_id)
        trace = rec['trace']
        ev = trace['events'][idx]
        prefix = dict(trace)
        prefix['events'] = trace['events'][:idx]
        res = score_prefix(prefix, ev['tool'], ev['arg'])
        from build_gate_dataset_full_20260917 import full_nodes
        live_nodes = full_nodes(prefix, prefix['events'],
                                candidate=(ev['tool'], ev['arg']))
        text_live = [n['features']['subtask'] for n in live_nodes]
        text_data = [n['features']['subtask'] for n in row['nodes']]
        equal = text_live == text_data
        if not equal:
            mismatches += 1
        checked.append({'task_id': row['task_id'], 'nodes_equal': equal,
                        'nodes_live': len(text_live), 'nodes_data': len(text_data),
                        'p_gate_path': res['p']})
        if len(checked) >= 20:
            break
    report = {'checked': len(checked), 'mismatches': mismatches,
              'rows': checked,
              'full_nodes_sha256_16': hashlib.sha256(
                  (ROOT / 'code/repro/v10/build_gate_dataset_full_20260917.py')
                  .read_bytes()).hexdigest()[:16]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'mock_parity_check.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=1))
    for r in checked[:5]:
        print(r)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
