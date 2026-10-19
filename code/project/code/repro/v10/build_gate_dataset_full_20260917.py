"""Build the FULL-CONTEXT pre-execution gate dataset from existing v10 no-gate runs.

Same sample selection / labels / family splits as
`build_gate_dataset_20260915.py`, but nodes carry full context:

  * first node: the run's user task (``trace['tasks']`` joined with ' | '),
    type ``user_task``, text ``grad_map.task_node_text``;
  * executed events (excluding ``end``): text ``grad_map.format_event_text``
    with the complete arguments and the full observation (no truncation at
    this layer; the encoder window is a model property), status=executed;
  * memory-write nodes exactly as the original builder (record content
    materialized as ``memory_write`` with ``source_kind``);
  * candidate proposed action node: same formatter, status=proposed.

Training and deployment therefore share one text format: the live hook
(``grad_map.trace_nodes_full``) produces the same strings for the same
semantic inputs (v10 mock single-arg events are wrapped as ``{"arg": ...}``).

Output: code/repro/out/online_gate_v2/{train,dev,test}.jsonl + manifest.json
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

import grad_map  # noqa: E402
from envlib import ENVS, valid_authorization  # noqa: E402
from gate_graph import TOOL_TYPE, _node  # noqa: E402

OUT = REPO / 'code/repro/out/online_gate_v2'
ORIG = REPO / 'code/repro/out/online_gate'
DEFAULT_FILES = [
    'v10_main_shop.json', 'v10_main_workspace.json', 'v10_main_finance.json',
    'v10_mainb_shop.json', 'v10_mainb_workspace.json', 'v10_mainb_finance.json',
    'v10_conf_shop.json', 'v10_conf_workspace.json', 'v10_conf_finance.json',
    'v10_confb_shop.json', 'v10_confb_workspace.json', 'v10_confb_finance.json',
    'v10_cloud.json', 'v10_cloud_v2.json',
]

NODE_TEXT_SCHEMA = {
    'user_task': "features.subtask = grad_map.task_node_text(tasks joined ' | '); "
                 "type='user_task'; tool_name='user_task'",
    'tool_event': "features.subtask = grad_map.format_event_text(tool, {'arg': arg}, "
                  "observation, 'executed') (shared with the live hook); "
                  "type=gate_graph.TOOL_TYPE[tool]",
    'memory_write': "features.subtask = 'remember <record content[:200]>'; "
                    "type='memory_write'; source_kind=record.source_kind "
                    "(v10-only node type; AgentDojo has no memory writes)",
    'candidate': "features.subtask = grad_map.format_event_text(tool, {'arg': arg}, "
                 "None, 'proposed'); type=gate_graph.TOOL_TYPE[tool]",
}


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


def tasks_text(trace: dict) -> str:
    """Join the run's user-task entries with ' | ' (list of str or of dicts)."""
    tasks = trace.get('tasks') or []
    parts = []
    for t in tasks:
        if isinstance(t, str):
            parts.append(t)
        elif isinstance(t, dict):
            parts.append(str(t.get('content') or t.get('text') or ''))
    return ' | '.join(p for p in parts if p)


def full_nodes(trace: dict, prefix_events: list[dict],
               candidate: tuple[str, str]) -> list[dict]:
    """Full-context nodes: user task + full prefix events/observations + candidate."""
    records = {(r['record_id'], r.get('version', 1)): r
               for r in trace.get('records', [])}
    writes_by_event: dict[str, list[dict]] = {}
    for m in trace.get('mem_events', []):
        if m.get('op') == 'write':
            writes_by_event.setdefault(str(m.get('source_event_id', '')), []).append(m)

    nodes: list[dict] = []
    task_text = tasks_text(trace)
    if task_text:
        nodes.append({
            'id': f'n{len(nodes)}', 'type': 'user_task', 'taint': 0,
            'turn': len(nodes), 'status': 'context',
            'features': {'subtask': grad_map.task_node_text(task_text),
                         'tool_name': 'user_task',
                         'command': ''},
        })
    for ev in prefix_events:
        tool = ev.get('tool')
        if tool in (None, 'end'):
            continue
        arg = str(ev.get('arg', ''))
        obs = str(ev.get('observation', ''))
        node = _node(len(nodes), str(tool), arg)
        node['status'] = 'executed'
        node['features']['subtask'] = grad_map.format_event_text(
            str(tool), {'arg': arg}, obs, 'executed')
        nodes.append(node)
        for m in writes_by_event.get(str(ev.get('event_id', '')), []):
            rec = records.get((m['record_id'], m.get('version', 1)))
            content = str(rec.get('content', '')) if rec else ''
            mem_node = _node(len(nodes), 'remember', content[:200])
            mem_node['type'] = 'memory_write'
            mem_node['features']['source_kind'] = (
                rec.get('source_kind', '') if rec else '')
            nodes.append(mem_node)
    cand_node = _node(len(nodes), str(candidate[0]), str(candidate[1]))
    cand_node['status'] = 'proposed'
    cand_node['features']['subtask'] = grad_map.format_event_text(
        str(candidate[0]), {'arg': str(candidate[1])}, None, 'proposed')
    nodes.append(cand_node)
    return nodes


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
    nodes = full_nodes(trace, events[:idx], candidate=(tool, arg))
    return [{
        'task_id': f"{rec['run_id']}|{idx}",
        'sample_key': f"{rec['run_id']}|{idx}|{arg}",
        'label': label,
        'nodes': nodes,
        'meta': {'env': env_name, 'family': rec['family'],
                 'condition': rec['condition'], 'arm': rec['arm']},
    }]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='*', default=DEFAULT_FILES)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)

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

    # Parity check against the original 110-char dataset (same selection).
    parity = {}
    for role in ('train', 'dev', 'test'):
        orig_path = ORIG / f'{role}.jsonl'
        orig_ids = set()
        if orig_path.exists():
            orig_ids = {json.loads(l)['task_id']
                        for l in orig_path.read_text().splitlines() if l.strip()}
        new_ids = {s['task_id'] for s in out[role]}
        parity[role] = {'orig_n': len(orig_ids), 'new_n': len(new_ids),
                        'same_ids': orig_ids == new_ids}

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'source_files': args.files, 'splits': {}, 'schema': NODE_TEXT_SCHEMA,
                'selection_parity_vs_20260915': parity,
                'families_by_role': {}, 'out_dir': str(out_dir)}
    for role in ('train', 'dev', 'test'):
        path = out_dir / f'{role}.jsonl'
        with open(path, 'w') as f:
            for s in out[role]:
                f.write(json.dumps(s) + '\n')
        pos = sum(1 for s in out[role] if s['label'] == 1)
        fams = sorted({s['meta']['family'] for s in out[role]})
        envs = sorted({s['meta']['env'] for s in out[role]})
        n_nodes = [len(s['nodes']) for s in out[role]]
        manifest['splits'][role] = {
            'n': len(out[role]), 'pos': pos, 'neg': len(out[role]) - pos,
            'envs': envs, 'families': fams,
            'nodes_total': sum(n_nodes),
            'nodes_per_sample': {'min': min(n_nodes, default=0),
                                 'max': max(n_nodes, default=0),
                                 'mean': round(sum(n_nodes) / max(len(n_nodes), 1), 2)},
        }
        manifest['families_by_role'][role] = fams
        keys = [s.get('sample_key', s['task_id']) for s in out[role]]
        manifest['splits'][role]['unique_sample_keys'] = len(set(keys))
        manifest['splits'][role]['duplicate_task_ids'] = len(out[role]) - len(
            {s['task_id'] for s in out[role]})
        print(f'{role}: n={len(out[role])} pos={pos} neg={len(out[role]) - pos} '
              f'unique_keys={len(set(keys))} '
              f'dup_task_ids={manifest["splits"][role]["duplicate_task_ids"]} '
              f'families={fams}')
    print('parity vs 20260915:', json.dumps(parity))
    print('by (role, env, label):')
    for k in sorted(stats):
        print(' ', k, stats[k])

    # A couple of examples for manual inspection.
    examples = []
    for role in ('train', 'test'):
        for s in out[role]:
            if len(s['nodes']) >= 3:
                examples.append({'split': role, 'task_id': s['task_id'],
                                 'label': s['label'],
                                 'node_types': [n['type'] for n in s['nodes']],
                                 'subtasks': [n['features']['subtask'][:300]
                                              for n in s['nodes'][:4]]})
                break
    manifest['examples'] = examples
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for ex in examples:
        print('\n--- example', ex['split'], ex['task_id'], 'label=', ex['label'])
        print('types:', ex['node_types'])
        for t in ex['subtasks']:
            print('  node:', t.replace('\n', '\\n')[:220])
    print(f'\nwrote {out_dir}/{{train,dev,test}}.jsonl + manifest.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
