#!/usr/bin/env python3
"""Aggregate the unified GRAD-512 strict ablation (dev or formal).

Direct paired statistics between every pair of interest (not just vs naive):
snap|centroid vs replay / stage0|centroid / snap|all, with seed-cluster
bootstrap CIs; per-task ACC; F_T. Works for both dev and --report-test runs.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import statistics as st
from pathlib import Path

import numpy as np

COMPARISONS = [
    ('snap|centroid', 'replay|-'),
    ('snap|centroid', 'stage0|centroid'),
    ('snap|centroid', 'snap|all'),
    ('snap|oracle', 'snap|centroid'),
    ('stage0|centroid', 'replay|-'),
    ('replay|-', 'naive|-'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tex', default='')
    ap.add_argument('--prefix', default='')
    ap.add_argument('--caption', default='')
    args = ap.parse_args()
    per_key = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(Path(args.root) / '*' / 'results.json'))):
        r = json.load(open(f))
        name = Path(f).parent.name
        if args.prefix and not name.startswith(args.prefix):
            continue
        seed = int(name.split('_')[-1])
        for key, v in r.items():
            if key == 'meta' or 'macro_acc' not in v:
                continue
            am = v['acc_matrix']
            n = max(int(k.split('|')[0]) for k in am)
            final = {k.split('|')[1]: x for k, x in am.items()
                     if k.startswith(f'{n}|')}
            per_key[key].setdefault(seed, []).append(
                {'acc': v['macro_acc'], 'ft': v.get('forgetting'),
                 'per_task': final, 'seconds': v.get('seconds')})
    seeds = {k: {s: {'acc': st.mean(x['acc'] for x in v),
                     'ft': st.mean(x['ft'] for x in v if x['ft'] is not None),
                     'per_task': {t: st.mean(x['per_task'][t] for x in v)
                                  for t in v[0]['per_task']}}
                 for s, v in d.items()} for k, d in per_key.items()}

    def paired(a, b, field, n_boot=2000):
        common = sorted(set(seeds[a]) & set(seeds[b]))
        if len(common) < 2:
            return None
        if field == 'acc':
            diffs = [seeds[a][s]['acc'] - seeds[b][s]['acc'] for s in common]
        elif field == 'ft':
            diffs = [seeds[a][s]['ft'] - seeds[b][s]['ft'] for s in common]
        else:  # per-task
            diffs = [seeds[a][s]['per_task'][field]
                     - seeds[b][s]['per_task'][field] for s in common]
        rng = random.Random(7)
        boot = sorted(st.mean([diffs[rng.randrange(len(diffs))] for _ in diffs])
                      for _ in range(n_boot))
        return {'n_seeds': len(common), 'mean': st.mean(diffs),
                'ci95': [boot[50], boot[-51]]}

    summary = {}
    for key, d in seeds.items():
        summary[key] = {
            'n_seeds': len(d),
            'macro_acc': st.mean(v['acc'] for v in d.values()),
            'ft': st.mean(v['ft'] for v in d.values()),
            'per_task': {t: st.mean(v['per_task'][t] for v in d.values())
                         for t in next(iter(d.values()))['per_task']}}
    task_keys = sorted({t for d in seeds.values() for s in d.values()
                        for t in s['per_task']})
    report = {'summary': summary, 'pairwise': {}, 'tasks': task_keys}
    for a, b in COMPARISONS:
        if a in seeds and b in seeds:
            rec = {'acc': paired(a, b, 'acc'), 'ft': paired(a, b, 'ft')}
            for t in task_keys:
                rec[t] = paired(a, b, t)
            report['pairwise'][f'{a} - {b}'] = rec
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    L = ['# Unified GRAD-512 strict ablation', '',
         '| arm | seeds | ACC | F_T | ' + ' | '.join(task_keys) + ' |',
         '|---|---|' + '---|' * (3 + len(task_keys))]
    for key, v in sorted(summary.items(), key=lambda kv: -kv[1]['macro_acc']):
        L.append(f"| {key} | {v['n_seeds']} | {v['macro_acc']:.4f} | "
                 f"{v['ft']:.4f} | "
                 + ' | '.join(f"{v['per_task'][t]:.4f}"
                              for t in task_keys) + ' |')
    L += ['', '## Direct paired comparisons (bootstrap 95% CI)', '',
          '| comparison | dACC [CI] | dFT [CI] | '
          + ' | '.join(task_keys) + ' |',
          '|---|---|' + '---|' * (2 + len(task_keys))]
    for name, v in report['pairwise'].items():
        def f(x, field='mean'):
            if not x:
                return '--'
            if field == 'ci':
                return f"[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]"
            return f"{x['mean']:+.4f}"
        L.append(f"| {name} | {f(v['acc'])} {f(v['acc'],'ci') if v['acc'] else ''} | "
                 f"{f(v['ft'])} {f(v['ft'],'ci') if v['ft'] else ''} | "
                 + ' | '.join(f"{f(v[t])} {f(v[t],'ci') if v[t] else ''}"
                              for t in task_keys) + ' |')
    (out.parent / (out.stem + '.md')).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    if args.tex:
        labels = {
            'naive|-': 'Sequential FT',
            'replay|-': 'Replay',
            'stage0|centroid': 'Experts (stage-0 backbone) + routing',
            'stage0|oracle': 'Experts (stage-0 backbone) + task oracle',
            'stage0|all': 'Experts (stage-0 backbone) + all',
            'snap|centroid': 'Antibody experts (replay snapshots) + routing',
            'snap|oracle': 'Antibody experts (replay snapshots) + task oracle',
            'snap|all': 'Antibody experts (replay snapshots) + all'}
        order = ['naive|-', 'replay|-', 'snap|centroid', 'snap|oracle',
                 'snap|all', 'stage0|centroid', 'stage0|oracle', 'stage0|all']
        lines = ['% Auto-generated by real_cl_unified512_agg.py',
                 '\\begin{table}[t]', '\\centering',
                 '\\caption{' + (args.caption or
                 'Antibody continual-learning ablation, unified GRAD-512, '
                 'strict sequential protocol.') + '}',
                 '\\label{tab:u512-antibody}',
                 '\\resizebox{\\linewidth}{!}{%',
                 '\\begin{tabular}{lcccc}', '\\toprule',
                 'Method & ACC & $F_T$ & ' + ' / '.join(task_keys) + ' & '
                 '$\\Delta$ACC vs Replay \\\\', '\\midrule']
        for key in order:
            if key not in summary:
                continue
            v = summary[key]
            delta = '--'
            if key != 'replay|-':
                pv = report['pairwise'].get(f'{key} - replay|-')
                rev = report['pairwise'].get(f'replay|- - {key}')
                if pv and pv.get('acc'):
                    delta = (f"{pv['acc']['mean']:+.3f} "
                             f"[{pv['acc']['ci95'][0]:+.3f},"
                             f"{pv['acc']['ci95'][1]:+.3f}]")
                elif rev and rev.get('acc'):
                    lo, hi = -rev['acc']['ci95'][1], -rev['acc']['ci95'][0]
                    delta = (f"{-rev['acc']['mean']:+.3f} "
                             f"[{lo:+.3f},{hi:+.3f}]")
            lines.append(
                f"{labels[key]} & {v['macro_acc']:.3f} & {v['ft']:.3f} & "
                + ' / '.join(f"{v['per_task'][t]:.3f}"
                             for t in task_keys)
                + f' & {delta} \\\\')
        lines += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
        Path(args.tex).write_text('\n'.join(lines) + '\n')
        print('tex saved', args.tex)


if __name__ == '__main__':
    raise SystemExit(main())
