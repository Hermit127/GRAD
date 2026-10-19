"""Build the pre-execution gate dataset from existing v10 no-gate runs.

For every run with arm='none' that ends with an action attempt, take the prefix
of executed events and the proposed action call as the last node; label = the
run's `independent_harm`. Split is family-grouped per env (train/dev/test) so the
closed-loop evaluation can use held-out families.

Output: code/repro/out/online_gate/{train,dev,test}.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))

from envlib import ENVS, valid_authorization  # noqa: E402
from gate_graph import trace_nodes  # noqa: E402

OUT = REPO / 'code/repro/out/online_gate'
DEFAULT_FILES = [
    'v10_main_shop.json', 'v10_main_workspace.json', 'v10_main_finance.json',
    'v10_mainb_shop.json', 'v10_mainb_workspace.json', 'v10_mainb_finance.json',
    'v10_conf_shop.json', 'v10_conf_workspace.json', 'v10_conf_finance.json',
    'v10_confb_shop.json', 'v10_confb_workspace.json', 'v10_confb_finance.json',
    'v10_cloud.json', 'v10_cloud_v2.json',
]


def family_roles(env: str, families: list[str]) -> dict[str, str]:
    fams = sorted(families)
    if len(fams) >= 4:
        roles = {f: 'train' for f in fams[:-2]}
        roles[fams[-2]] = 'dev'
        roles[fams[-1]] = 'test'
    elif len(fams) == 2:
        roles = {fams[0]: 'train', fams[1]: 'test'}
    else:
        roles = {f: 'train' for f in fams}
    return roles


def samples_from_run(rec: dict) -> list[dict]:
    env_name = rec['env']
    env = ENVS[env_name]
    action_tool = env['action']
    fam = env['families'][rec['family']]
    trace = rec.get('trace') or {}
    events = trace.get('events', [])
    idx = None
    for i, ev in enumerate(events):
        if ev.get('tool') == action_tool:
            idx = i
    if idx is None:
        return []
    tool, arg = events[idx]['tool'], events[idx]['arg']
    # Deployment reality: an explicitly authorized call never reaches the gate.
    if valid_authorization(trace, env, action_tool, arg):
        return []
    canon = env['canon'](arg) if 'canon' in env else arg
    label = int(bool(env['harm'](canon, fam)))
    cut_seq = int(events[idx].get('seq', 0) or 0)
    prefix = {
        'events': events[:idx],
        'records': trace.get('records', []),
        'mem_events': [m for m in trace.get('mem_events', [])
                       if int(m.get('seq', 0) or 0) <= cut_seq],
    }
    nodes = trace_nodes(prefix, candidate=(tool, arg))
    return [{
        'task_id': f"{rec['run_id']}|{idx}",
        'label': label,
        'nodes': nodes,
        'meta': {'env': env_name, 'family': rec['family'],
                 'condition': rec['condition'], 'arm': rec['arm']},
    }]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='*', default=DEFAULT_FILES)
    args = ap.parse_args()

    roles = {env: family_roles(env, list(spec['families']))
             for env, spec in ENVS.items()}
    out = defaultdict(list)
    stats = Counter()
    for name in args.files:
        path = REPO / 'LOGS' / name
        if not path.exists():
            print('missing', path)
            continue
        data = json.loads(path.read_text())
        for rec in data.get('runs', []):
            if rec.get('arm') != 'none' or rec.get('error'):
                continue
            for s in samples_from_run(rec):
                role = roles[s['meta']['env']].get(s['meta']['family'], 'train')
                out[role].append(s)
                stats[(role, s['meta']['env'], s['label'])] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    for role in ('train', 'dev', 'test'):
        path = OUT / f'{role}.jsonl'
        with open(path, 'w') as f:
            for s in out[role]:
                f.write(json.dumps(s) + '\n')
        pos = sum(1 for s in out[role] if s['label'] == 1)
        print(f'{role}: n={len(out[role])} pos={pos} neg={len(out[role]) - pos}')
    print('by (role, env, label):')
    for k in sorted(stats):
        print(' ', k, stats[k])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
