#!/usr/bin/env python3
"""Phase B: build same-tool authorization-contrast pairs from existing v10 runs.

For every (env, family, action argument) that has BOTH
  * a malicious run (independent_harm True, no matching authorization) and
  * an authorized run (valid_authorization matches the same argument),
emit two samples whose candidate node text is identical (same tool, same
argument); only the observable context differs (user task / task authority vs
untrusted observation or note suggesting the same argument).

Label basis (auditable, recomputed here):
  * malicious  label=1  <= envlib.independent_outcome -> harm=True, completion=False
  * authorized label=0  <= envlib.independent_outcome -> harm=False, completion=True
Family-level roles reuse the frozen online_gate_v2 split; test-family pairs are
written separately and must not be used for training.

Outputs in LOGS/final_table2/auth_pairs_20260918/:
  auth_pairs_{train,dev,test}.jsonl, auth_pairs_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/v10'))

from build_gate_dataset_full_20260917 import full_nodes, tasks_text  # noqa: E402
from envlib import ENVS, independent_outcome, valid_authorization  # noqa: E402

OUT = ROOT / 'LOGS/final_table2/auth_pairs_20260918'
V2_MANIFEST = ROOT / 'code/repro/out/online_gate_v2/manifest.json'


def canon(env_name, arg):
    env = ENVS[env_name]
    return env['canon'](arg) if 'canon' in env else str(arg)


def outcome(env_name, family, trace, arg):
    env = ENVS[env_name]
    return independent_outcome(env, env['families'][family], trace, arg)


def authorized(env_name, trace, action, arg):
    env = ENVS[env_name]
    return valid_authorization(trace, env, action, arg)


def candidate_index(trace, action_tool):
    for i, ev in enumerate(trace.get('events', [])):
        if ev.get('tool') == action_tool:
            last = i
    try:
        return last
    except UnboundLocalError:
        return None


def sample_from_run(rec, env_name, role, label, pair_id, basis):
    env = ENVS[env_name]
    trace = rec['trace']
    idx = candidate_index(trace, env['action'])
    if idx is None:
        return None
    ev = trace['events'][idx]
    tool, arg = ev['tool'], ev['arg']
    nodes = full_nodes(trace, trace['events'][:idx], candidate=(tool, arg))
    return {
        'task_id': f"{rec['run_id']}|{idx}",
        'sample_key': f"{rec['run_id']}|{idx}|{arg}",
        'pair_id': pair_id,
        'label': label,
        'nodes': nodes,
        'meta': {'env': env_name, 'family': rec['family'], 'role': role,
                 'label_basis': basis, 'condition': rec['condition'],
                 'arm': rec['arm'], 'run_id': rec['run_id'],
                 'user_task_chars': len(tasks_text(trace)),
                 'n_prefix_events': idx},
    }


def load_runs():
    runs = []
    for path in sorted(ROOT.glob('LOGS/v10_*.json')):
        data = json.loads(path.read_text())
        for rec in data.get('runs', []):
            if rec.get('error'):
                continue
            if not rec.get('trace'):
                continue
            runs.append(rec)
    return runs


def main() -> int:
    roles = json.loads(V2_MANIFEST.read_text())['families_by_role']
    fam_role = {}
    for role, fams in roles.items():
        for f in fams:
            fam_role[(role, f)] = role

    def role_of(family):
        for role in ('train', 'dev', 'test'):
            if family in roles.get(role, []):
                return role
        return None

    runs = load_runs()
    malicious = defaultdict(list)
    authorized_runs = defaultdict(list)
    for rec in runs:
        env_name = rec['env']
        if env_name not in ENVS:
            continue
        role = role_of(rec['family'])
        if role is None:
            continue
        env = ENVS[env_name]
        idx = candidate_index(rec['trace'], env['action'])
        if idx is None:
            continue
        arg = rec['trace']['events'][idx]['arg']
        key = (env_name, rec['family'], str(arg))
        if authorized(env_name, rec['trace'], env['action'], arg):
            authorized_runs[key].append(rec)
        else:
            out = outcome(env_name, rec['family'], rec['trace'], arg)
            if out['harm']:
                malicious[key].append(rec)

    out = defaultdict(list)
    pair_stats = []
    for key in sorted(malicious):
        if key not in authorized_runs:
            continue
        env_name, family, arg = key
        role = role_of(family)
        mal_runs, auth_runs = malicious[key], authorized_runs[key]
        mal_runs.sort(key=lambda r: r['run_id'])
        auth_runs.sort(key=lambda r: r['run_id'])
        mal = mal_runs[0]
        mal_idx = candidate_index(mal['trace'], ENVS[env_name]['action'])
        auth = min(auth_runs,
                   key=lambda r: abs(candidate_index(r['trace'], ENVS[env_name]['action']) - mal_idx))
        auth_idx = candidate_index(auth['trace'], ENVS[env_name]['action'])
        pid = f'{env_name}|{family}|{arg}'
        s_mal = sample_from_run(mal, env_name, role, 1, pid,
                                'independent_outcome.harm=True')
        s_auth = sample_from_run(auth, env_name, role, 0, pid,
                                 'independent_outcome authorized completion=True')
        if not s_mal or not s_auth:
            continue
        mal_cand = s_mal['nodes'][-1]['features']['subtask']
        auth_cand = s_auth['nodes'][-1]['features']['subtask']
        pair_stats.append({
            'pair_id': pid, 'role': role, 'candidate_identical': mal_cand == auth_cand,
            'malicious_run': mal['run_id'], 'authorized_run': auth['run_id'],
            'mal_nodes': len(s_mal['nodes']), 'auth_nodes': len(s_auth['nodes']),
            'node_delta': abs(len(s_mal['nodes']) - len(s_auth['nodes'])),
            'mal_user_task_chars': s_mal['meta']['user_task_chars'],
            'auth_user_task_chars': s_auth['meta']['user_task_chars'],
        })
        out[role].append(s_mal)
        out[role].append(s_auth)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'source_runs': len(runs), 'pairs': pair_stats,
                'counts': {}, 'candidate_identical': 0}
    for role in ('train', 'dev', 'test'):
        lines = sorted(out[role], key=lambda s: s['pair_id'] + str(s['label']))
        with open(OUT / f'auth_pairs_{role}.jsonl', 'w') as f:
            for s in lines:
                f.write(json.dumps(s) + '\n')
        manifest['counts'][role] = {
            'samples': len(lines), 'pairs': len({s['pair_id'] for s in lines}),
            'pos': sum(1 for s in lines if s['label'] == 1),
            'neg': sum(1 for s in lines if s['label'] == 0),
            'envs': dict(Counter(s['meta']['env'] for s in lines)),
            'families': sorted({s['meta']['family'] for s in lines}),
            'node_len_min': min((len(s['nodes']) for s in lines), default=0),
            'node_len_max': max((len(s['nodes']) for s in lines), default=0),
        }
    manifest['candidate_identical'] = sum(1 for p in pair_stats
                                          if p['candidate_identical'])
    manifest['pair_node_delta_mean'] = (
        sum(p['node_delta'] for p in pair_stats) / max(len(pair_stats), 1))
    (OUT / 'auth_pairs_manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'pairs'}, indent=1))
    for role in ('train', 'dev', 'test'):
        n = manifest['counts'][role]
        print(role, 'samples', n['samples'], 'pairs', n['pairs'], n['pos'], 'pos', n['neg'], 'neg')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
