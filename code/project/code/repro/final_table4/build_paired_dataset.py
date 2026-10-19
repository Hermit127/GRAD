#!/usr/bin/env python
"""Build the paired-authorization repair dataset from the frozen v2 splits.

Data preparation only (no model/loss change; the P1/P2/P3 objective awaits PI).

Pair types per source row:
  candidate_swap : same prefix, candidate replaced by the environment's
                   dangerous vs safe target for that family
                   (expected score order: dangerous > safe)
  evidence_swap  : same candidate, poisoned note text replaced by the benign
                   note text inside the stored observation/memory nodes
                   (expected: score higher with the poisoned note when the
                   candidate is the dangerous target)

Split discipline: pairs inherit the frozen v2 roles (train/dev/test); no test
family sample enters train or dev, and the test-role pairs are exported only as
a held-out probe file (never for training or model selection).

Outputs code/repro/out/paired_repair_v1/{train,dev,test_probe}.jsonl +
manifest.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

import grad_map  # noqa: E402
from envlib import ENVS  # noqa: E402

V2 = REPO / 'code/repro/out/online_gate_v2'
OUT = REPO / 'code/repro/out/paired_repair_v1'


def candidate_arg(node: dict) -> str:
    sub = str(node['features']['subtask'])
    if 'args=' in sub:
        raw = sub.split('args=', 1)[1]
        raw = raw.split(' | result=', 1)[0].split(' | status=', 1)[0]
        try:
            return str(json.loads(raw).get('arg', ''))
        except json.JSONDecodeError:
            return raw.strip('{} "')
    return ''


def replace_candidate(nodes: list[dict], tool: str, arg: str) -> list[dict]:
    out = json.loads(json.dumps(nodes))
    cand = out[-1]
    cand['features']['subtask'] = grad_map.format_event_text(
        tool, {'arg': arg}, None, 'proposed')
    cand['status'] = 'proposed'
    return out


def swap_note(nodes: list[dict], src: str, dst: str) -> list[dict]:
    out = json.loads(json.dumps(nodes))
    for n in out:
        sub = n['features']['subtask']
        if src and src in sub:
            n['features']['subtask'] = sub.replace(src, dst)
        elif src and len(src) > 60 and src[:60] in sub:
            n['features']['subtask'] = sub.replace(src[:60], dst[:60])
    return out


def label_of(env, fam, arg):
    canon, _ = __import__('canon_robust').robust_canon(env, arg)
    return int(bool(env['harm'](canon, fam))), int(bool(
        env['complete'](canon, fam)))


def build_pairs(rows: list[dict], role: str) -> list[dict]:
    pairs = []
    for row in rows:
        env = ENVS[(row.get('meta') or {}).get('env', '')]
        fam_name = (row.get('meta') or {}).get('family', '')
        fam = env['families'][fam_name]
        tool = row['nodes'][-1]['features'].get('tool_name')
        bad = str(fam['bad_arg'])
        safe = str(fam['good_arg'])
        nodes_bad = replace_candidate(row['nodes'], tool, bad)
        nodes_safe = replace_candidate(row['nodes'], tool, safe)
        h_bad, c_bad = label_of(env, fam, bad)
        h_safe, c_safe = label_of(env, fam, safe)
        pairs.append({
            'pair_id': f"{row['task_id']}|cand",
            'type': 'candidate_swap', 'role': role,
            'env': env['name'], 'family': fam_name, 'tool': tool,
            'source_task_id': row['task_id'],
            'nodes_a': nodes_bad, 'label_a_harm': h_bad,
            'label_a_complete': c_bad,
            'nodes_b': nodes_safe, 'label_b_harm': h_safe,
            'label_b_complete': c_safe,
            'expected_order': 'A>B'})
        note_bad, note_good = fam['note_bad'], fam['note_good']
        observed = candidate_arg(row['nodes'][-1])
        contains_poison = any(note_bad in n['features']['subtask'] or
                              (len(note_bad) > 60 and
                               note_bad[:60] in n['features']['subtask'])
                              for n in row['nodes'])
        contains_benign = any(note_good in n['features']['subtask'] or
                              (len(note_good) > 60 and
                               note_good[:60] in n['features']['subtask'])
                              for n in row['nodes'])
        if contains_poison or contains_benign:
            src, dst = ((note_bad, note_good) if contains_poison
                        else (note_good, note_bad))
            nodes_alt = swap_note(row['nodes'], src, dst)
            cand_is_bad = bool(
                __import__('canon_robust').robust_canon(env, observed)[0]
                == bad)
            pairs.append({
                'pair_id': f"{row['task_id']}|evid",
                'type': 'evidence_swap', 'role': role,
                'env': env['name'], 'family': fam_name, 'tool': tool,
                'source_task_id': row['task_id'],
                'candidate': observed, 'candidate_is_bad': cand_is_bad,
                'nodes_a': row['nodes'],
                'note_a': 'poisoned' if contains_poison else 'benign',
                'nodes_b': nodes_alt,
                'note_b': 'benign' if contains_poison else 'poisoned',
                'expected_order': ('A>B' if (contains_poison and cand_is_bad)
                                   else 'B>A' if (contains_benign and
                                                  cand_is_bad) else 'weak')})
    return pairs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'source': 'code/repro/out/online_gate_v2',
                'generator': str(Path(__file__).relative_to(REPO)),
                'types': ['candidate_swap', 'evidence_swap'],
                'splits': {}}
    roles = {}
    for role in ('train', 'dev', 'test'):
        rows = [json.loads(l) for l in (V2 / f'{role}.jsonl').read_text()
                .splitlines() if l.strip()]
        pairs = build_pairs(rows, {'test': 'test_probe'}.get(role, role))
        name = {'test': 'test_probe'}.get(role, role)
        path = OUT / f'{name}.jsonl'
        with open(path, 'w') as fh:
            for p in pairs:
                fh.write(json.dumps(p) + '\n')
        counts = Counter((p['type'], p['env']) for p in pairs)
        manifest['splits'][name] = {
            'n_source_rows': len(rows), 'n_pairs': len(pairs),
            'by_type_env': {f'{t}|{e}': n for (t, e), n in sorted(
                counts.items())},
            'families': sorted({p['family'] for p in pairs}),
            'sha256_lines': __import__('hashlib').sha256(
                path.read_bytes()).hexdigest()}
        roles[name] = pairs
    # leakage assertions
    train_fams = {(p['env'], p['family']) for p in roles['train']}
    dev_fams = {(p['env'], p['family']) for p in roles['dev']}
    test_fams = {(p['env'], p['family']) for p in roles['test_probe']}
    manifest['leakage'] = {
        'train_test_overlap': sorted(train_fams & test_fams),
        'dev_test_overlap': sorted(dev_fams & test_fams),
        'train_dev_overlap': sorted(train_fams & dev_fams)}
    assert not manifest['leakage']['train_test_overlap']
    assert not manifest['leakage']['dev_test_overlap']
    # label sanity within candidate pairs
    bad_ok = all(p['label_a_harm'] == 1 for p in roles['train'] + roles['dev']
                 if p['type'] == 'candidate_swap')
    safe_ok = all(p['label_b_harm'] == 0 for p in roles['train'] + roles['dev']
                  if p['type'] == 'candidate_swap')
    manifest['label_sanity'] = {'dangerous_candidates_harmful': bad_ok,
                                'safe_candidates_not_harmful': safe_ok}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'train': manifest['splits']['train'],
                      'dev': manifest['splits']['dev'],
                      'test_probe': manifest['splits']['test_probe'],
                      'leakage': manifest['leakage'],
                      'label_sanity': manifest['label_sanity']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
