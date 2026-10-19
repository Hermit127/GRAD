#!/usr/bin/env python3
"""Historical overlap audit for the confirmation case set (zero API).

Compares every new family text (task 1/2, both payloads, authorization) against
the development history: the dev closed-loop families (``envlib``) and the gate
dataset nodes (``online_gate_v2/{train,dev,test}.jsonl``). Reports token
5-gram Jaccard nearest neighbours, exact target-argument reuse, and the shared
scaffolding inventory. Writes ``LOGS/final_table4_indep/isolation_audit.{json,md}``.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

import envlib as dev  # noqa: E402
import envlib_conf  # noqa: E402

OUT = REPO / 'LOGS/final_table4_indep'
GATE = REPO / 'code/repro/out/online_gate_v2'
NGRAM = 5


def norm(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()


def shingles(text: str) -> set:
    toks = norm(text).split()
    return {tuple(toks[i:i + NGRAM]) for i in range(max(0, len(toks) - NGRAM
                                                          + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def all_strings(obj) -> list[str]:
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out += all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            out += all_strings(v)
    return out


def history_texts() -> list[tuple[str, str]]:
    """(source_id, text) history entries."""
    hist = []
    for env_name, env in dev.ENVS.items():
        for fam_name, fam in env['families'].items():
            for field in ('s1', 's2', 'note_bad', 'note_good', 'auth_text'):
                hist.append((f'dev:{env_name}:{fam_name}:{field}',
                             str(fam[field])))
    for split in ('train', 'dev', 'test'):
        p = GATE / f'{split}.jsonl'
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            for j, s in enumerate(all_strings(row.get('nodes', []))):
                hist.append((f'gate:{split}:{i}:node{j}', s))
    return hist


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / 'cases_manifest.json').read_text())
    hist = history_texts()
    hist_sh = [(sid, shingles(t)) for sid, t in hist]

    per_field = {'s1': [], 's2': [], 'payload': []}
    arg_hits = []
    nearest = {}
    for c in manifest['cases']:
        fields = {'s1': c['s1'], 's2': c['s2'], 'payload': c['payload']}
        for fname, text in fields.items():
            sh = shingles(text)
            best_sid, best_sim = '', 0.0
            for sid, hs in hist_sh:
                sim = jaccard(sh, hs)
                if sim > best_sim:
                    best_sid, best_sim = sid, sim
            per_field[fname].append(best_sim)
            key = f"{fname}"
            if best_sim > nearest.get(key, (0, ''))[0]:
                nearest[key] = (best_sim, best_sid)
        for label, arg in (('bad', c['detection_target']),
                           ('good', c['compliant_target'])):
            for sid, text in hist:
                if len(arg) >= 4 and arg.lower() in str(text).lower():
                    arg_hits.append({'case': c['case_id'], 'label': label,
                                     'arg': arg, 'history_source': sid})

    def stats(xs):
        xs = sorted(xs)
        return {'n': len(xs), 'max': round(xs[-1], 4),
                'p95': round(xs[int(0.95 * len(xs)) - 1], 4),
                'mean': round(sum(xs) / len(xs), 4),
                'n_ge_0_5': sum(1 for x in xs if x >= 0.5),
                'n_ge_0_7': sum(1 for x in xs if x >= 0.7)}

    report = {
        'n_history_texts': len(hist),
        'gate_rows_read': {
            split: sum(1 for _ in open(GATE / f'{split}.jsonl'))
            for split in ('train', 'dev', 'test')
            if (GATE / f'{split}.jsonl').exists()},
        'similarity': {k: stats(v) for k, v in per_field.items()},
        'nearest_overall': {k: {'sim': round(v[0], 4), 'source': v[1]}
                            for k, v in nearest.items()},
        'exact_target_arg_reuse': arg_hits,
        'shared_scaffolding': [
            'tool grammar (search/scan/lookup/click/recall/remember + action '
            'tool) and system-prompt template SYSTEM_GENERIC',
            'policy semantics and evaluator functions (verify/harm/complete) '
            'reused by design; only instance tables and payload texts are new',
            'task template skeletons are paraphrases that keep the scripted-'
            'mock anchors (looking for/repo/invoice/share, ASIN/to/ACCT/with)',
        ],
    }
    (OUT / 'isolation_audit.json').write_text(json.dumps(report, indent=1))

    gate_desc = ', '.join(f"{k}:{v}" for k, v in
                          report['gate_rows_read'].items())
    lines = ['# Confirmation set: historical overlap audit\n',
             f"History texts compared: {len(hist)} (dev families + gate "
             f"dataset nodes {gate_desc}).\n",
             '## 5-gram Jaccard vs history (per new-case field)\n',
             '| field | n | mean | p95 | max | ≥0.5 | ≥0.7 |',
             '|---|---|---|---|---|---|---|']
    for k, v in report['similarity'].items():
        lines.append(f"| {k} | {v['n']} | {v['mean']} | {v['p95']} | "
                     f"{v['max']} | {v['n_ge_0_5']} | {v['n_ge_0_7']} |")
    lines += ['', 'Nearest history per field: ' + '; '.join(
        f"{k} → {v['sim']} ({v['source']})"
        for k, v in report['nearest_overall'].items()) + '.', '',
        f"Exact target-argument reuse events: {len(arg_hits)}.", '',
        '## Shared scaffolding (declared)', '']
    lines += [f'- {s}' for s in report['shared_scaffolding']]
    (OUT / 'isolation_audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: report[k] for k in
                      ('similarity', 'nearest_overall',
                       'exact_target_arg_reuse')}, indent=1)[:2000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
