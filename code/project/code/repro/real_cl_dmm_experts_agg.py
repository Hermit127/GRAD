#!/usr/bin/env python3
"""Aggregate the dmm expert-capability upgrades vs the dmm ablation baselines.

Zero API. Reads code/repro/out/real_cl/dmm_experts/*/results.json (E1/E2) and
code/repro/out/real_cl/dmm/dev_*/results.json (naive/replay/ewc/specialists),
then reports macro ACC / F_T per (mode, router) and paired per-seed deltas vs
naive (orders averaged).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics as st
from pathlib import Path


def seed_key(run_name):
    return int(run_name.split('_')[-1])


def order_of(run_name):
    return 'reversed' if 'reversed' in run_name else 'forward'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='code/repro/out/real_cl')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    # baselines
    for f in sorted(glob.glob(str(root / 'dmm' / 'dev_*' / 'results.json'))):
        r = json.load(open(f))
        name = Path(f).parent.name
        for arm in ('naive', 'replay', 'ewc', 'specialists', 'specialists_all',
                    'specialists_oracle'):
            v = r.get(arm)
            if not v or 'acc_matrix' not in v:
                continue
            am = v['acc_matrix']
            n = max(int(k.split('|')[0]) for k in am)
            acc = st.mean([x for k, x in am.items() if k.startswith(f'{n}|')])
            rows[f'baseline/{arm}'][(seed_key(name), order_of(name))].append(
                {'acc': acc, 'ft': v.get('forgetting')})
    # E1/E2
    for f in sorted(glob.glob(str(root / 'dmm_experts' / '*' / 'results.json'))):
        r = json.load(open(f))
        name = Path(f).parent.name
        for router, v in r.items():
            if router == 'meta' or not isinstance(v, dict) or 'macro_acc' not in v:
                continue
            rows[f'{r["meta"]["mode"]}/{router}'][
                (seed_key(name), order_of(name))].append(
                {'acc': v['macro_acc'], 'ft': v.get('forgetting')})
    # aggregate per seed (average orders) and paired vs naive
    def per_seed(key):
        d = collections.defaultdict(list)
        for (seed, _order), vals in rows[key].items():
            for v in vals:
                d[seed].append(v)
        return {s: {'acc': st.mean(x['acc'] for x in v),
                    'ft': st.mean(x['ft'] for x in v if x['ft'] is not None)}
                for s, v in d.items()}
    table = {}
    for key in sorted(rows):
        ps = per_seed(key)
        table[key] = {
            'n_seeds': len(ps),
            'macro_acc': st.mean(v['acc'] for v in ps.values()),
            'ft': st.mean(v['ft'] for v in ps.values() if v['ft'] is not None)
            if any(v['ft'] is not None for v in ps.values()) else None,
            'per_seed': {str(k): {'acc': round(v['acc'], 4),
                                  'ft': None if v['ft'] is None
                                  else round(v['ft'], 4)}
                         for k, v in sorted(ps.items())}}
    naive = per_seed('baseline/naive')
    for key in table:
        ps = per_seed(key)
        common = sorted(set(naive) & set(ps))
        if key != 'baseline/naive' and common:
            da = [ps[s]['acc'] - naive[s]['acc'] for s in common]
            df = [ps[s]['ft'] - naive[s]['ft'] for s in common
                  if ps[s]['ft'] is not None and naive[s]['ft'] is not None]
            table[key]['delta_acc_vs_naive'] = st.mean(da)
            table[key]['delta_ft_vs_naive'] = st.mean(df) if df else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    L = ['# dmm expert-capability upgrades (dev; vs ablation baselines)', '',
         '| method | seeds | macro ACC | F_T | dACC vs naive | dFT vs naive |',
         '|---|---|---|---|---|---|']
    for key, v in sorted(table.items(), key=lambda kv: -kv[1]['macro_acc']):
        L.append(f"| {key} | {v['n_seeds']} | {v['macro_acc']:.4f} | "
                 f"{'--' if v['ft'] is None else format(v['ft'], '.4f')} | "
                 f"{'--' if 'delta_acc_vs_naive' not in v else format(v['delta_acc_vs_naive'], '+.4f')} | "
                 f"{'--' if v.get('delta_ft_vs_naive') is None else format(v['delta_ft_vs_naive'], '+.4f')} |")
    (out.parent / (out.stem + '.md')).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    raise SystemExit(main())
