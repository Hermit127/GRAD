#!/usr/bin/env python3
"""Formal (test) aggregation for the decomp/mcp/mrt antibody ablation.

Reads:
- baselines: code/repro/out/real_cl/dmm/formal_{order}_{seed}/results.json
  (naive / replay / specialists / specialists_all / specialists_oracle)
- E1: code/repro/out/real_cl/dmm_experts/formal_{order}_{seed}/results.json
  (routers centroid / oracle / all)
Test metrics (runs were executed with --report-test). Emits seed-clustered
means, paired deltas vs naive with bootstrap CIs, per-task ACC, F_T, and a
LaTeX ablation table.
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


def seed_of(name):
    return int(name.split('_')[-1])


def order_of(name):
    return 'reversed' if 'reversed' in name else 'forward'


def load(root):
    rows = collections.defaultdict(dict)
    for f in sorted(glob.glob(f'{root}/dmm/formal_*/results.json')):
        r = json.load(open(f))
        name = Path(f).parent.name
        for arm in ('naive', 'replay', 'specialists', 'specialists_all',
                    'specialists_oracle'):
            v = r.get(arm)
            if not v or 'acc_matrix' not in v:
                continue
            am = v['acc_matrix']
            n = max(int(k.split('|')[0]) for k in am)
            final = {k.split('|')[1]: x for k, x in am.items()
                     if k.startswith(f'{n}|')}
            rows[f'baseline/{arm}'][(seed_of(name), order_of(name))] = {
                'acc': st.mean(final.values()), 'ft': v.get('forgetting'),
                'per_task': final}
    for f in sorted(glob.glob(f'{root}/dmm_experts/formal_*/results.json')):
        r = json.load(open(f))
        name = Path(f).parent.name
        for router, v in r.items():
            if router == 'meta' or not isinstance(v, dict) or 'acc_matrix' not in v:
                continue
            am = v['acc_matrix']
            n = max(int(k.split('|')[0]) for k in am)
            final = {k.split('|')[1]: x for k, x in am.items()
                     if k.startswith(f'{n}|')}
            rows[f'E1/{router}'][(seed_of(name), order_of(name))] = {
                'acc': v['macro_acc'], 'ft': v.get('forgetting'),
                'per_task': final}
    return rows


def per_seed(rows, key):
    d = collections.defaultdict(list)
    for (seed, _), v in rows[key].items():
        d[seed].append(v)
    return {s: {'acc': st.mean(x['acc'] for x in v),
                'ft': st.mean(x['ft'] for x in v if x['ft'] is not None),
                'per_task': {k: st.mean(x['per_task'][k] for x in v)
                             for k in v[0]['per_task']}}
            for s, v in d.items()}


def paired(rows, key, ref='baseline/naive', field='acc', n_boot=2000):
    a, b = per_seed(rows, key), per_seed(rows, ref)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    diffs = [a[s][field] - b[s][field] for s in seeds]
    rng = random.Random(7)
    boot = sorted(st.mean([diffs[rng.randrange(len(diffs))] for _ in diffs])
                  for _ in range(n_boot))
    return {'n_seeds': len(seeds), 'mean': st.mean(diffs),
            'ci95': [boot[50], boot[-51]],
            'per_seed': {str(s): a[s][field] - b[s][field] for s in seeds}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='code/repro/out/real_cl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tex', default='')
    args = ap.parse_args()
    rows = load(args.root)
    report = {}
    for key in sorted(rows):
        ps = per_seed(rows, key)
        if not ps:
            continue
        report[key] = {
            'n_seeds': len(ps),
            'macro_acc': st.mean(v['acc'] for v in ps.values()),
            'ft': st.mean(v['ft'] for v in ps.values()),
            'per_task': {t: st.mean(v['per_task'][t] for v in ps.values())
                         for t in next(iter(ps.values()))['per_task']},
            'delta_acc_vs_naive': paired(rows, key, field='acc'),
            'delta_ft_vs_naive': paired(rows, key, field='ft')}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    L = ['# decomp/mcp/mrt antibody ablation (formal, test, 5 seeds x 2 orders)', '',
         '| method | ACC | F_T | decomp | mcp | mrt | dACC vs naive [95% CI] | dFT |',
         '|---|---|---|---|---|---|---|---|']
    for key, v in sorted(report.items(), key=lambda kv: -kv[1]['macro_acc']):
        d = v['delta_acc_vs_naive']
        L.append(f"| {key} | {v['macro_acc']:.4f} | {v['ft']:.4f} | "
                 + ' | '.join(f"{v['per_task'][t]:.4f}"
                              for t in ('decomp', 'mcp', 'mrt')) + ' | '
                 + (f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]"
                    if d else '--') + ' | '
                 + (f"{v['delta_ft_vs_naive']['mean']:+.4f}"
                    if v['delta_ft_vs_naive'] else '--') + ' |')
    (out.parent / (out.stem + '.md')).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    if args.tex:
        # focused ablation table: baselines + antibody variants + diagnostics
        order = ['baseline/naive', 'baseline/replay', 'E1/centroid', 'E1/oracle',
                 'E1/all', 'baseline/specialists', 'baseline/specialists_all',
                 'baseline/specialists_oracle']
        labels = {
            'baseline/naive': 'Sequential FT',
            'baseline/replay': 'Replay',
            'baseline/specialists': 'Experts (stage-0 backbone) + routing',
            'baseline/specialists_all': 'Experts (stage-0) + all',
            'baseline/specialists_oracle': 'Experts (stage-0) + task oracle',
            'E1/centroid': 'Antibody experts (replay backbone) + routing',
            'E1/oracle': 'Antibody experts (replay backbone) + task oracle',
            'E1/all': 'Antibody experts (replay backbone) + all',
        }
        lines = ['% Auto-generated by real_cl_dmm_formal_agg.py',
                 '\\begin{table}[t]', '\\centering',
                 '\\caption{Antibody continual-learning ablation on '
                 'decomp/mcp/mrt (test split, 5 seeds $\\times$ 2 orders).}',
                 '\\label{tab:dmm-antibody}', '\\resizebox{\\linewidth}{!}{%',
                 '\\begin{tabular}{lcccc}', '\\toprule',
                 'Method & ACC & $F_T$ & decomp / mcp / mrt & '
                 '$\\Delta$ACC vs Sequential FT \\\\', '\\midrule']
        for key in order:
            if key not in report:
                continue
            v = report[key]
            d = v['delta_acc_vs_naive']
            delta = '--' if not d else f"{d['mean']:+.3f} " \
                f"[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"
            lines.append(
                f"{labels[key]} & {v['macro_acc']:.3f} & {v['ft']:.3f} & "
                + ' / '.join(f"{v['per_task'][t]:.3f}"
                             for t in ('decomp', 'mcp', 'mrt'))
                + f' & {delta} \\\\')
        lines += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
        Path(args.tex).write_text('\n'.join(lines) + '\n')
        print('tex saved', args.tex)


if __name__ == '__main__':
    raise SystemExit(main())
