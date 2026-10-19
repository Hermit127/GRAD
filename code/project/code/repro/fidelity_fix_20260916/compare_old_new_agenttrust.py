#!/usr/bin/env python3
"""Acceptance comparison: old snapshot adapter vs fixed adapter (same split).

Runs both implementations on one frozen fix2 unit and reports per-graph
differences plus verdict distributions, so the change is explainable
(no silent metric shifts).

Usage: python compare_old_new_agenttrust.py --bench mg --fold 1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PKG = REPO / 'code/repro/static_big_table_v1'
FIX2 = REPO / 'code/repro/out/static_big_table_v1/fix2_20260910'
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

import groups  # noqa: E402
from common import write_json, utc_now  # noqa: E402

import agenttrust_fixed  # noqa: E402


def unit_roles(payload, fold):
    graphs = groups.load_benchmark(payload['benchmark'])
    by_group = {}
    for i, g in enumerate(graphs):
        by_group.setdefault(g['_group'], []).append(i)
    gr = payload['group_role'] if fold is None else payload['folds'][fold]['group_role']
    roles = {r: [] for r in ('train', 'dev', 'gate_fit', 'threshold_cal', 'test')}
    for gid, role in gr.items():
        roles[role].extend(by_group[gid])
    return graphs, roles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default='mg')
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    payload = json.load(open(FIX2 / 'split_ids' / f'{args.bench}.json'))
    graphs, roles = unit_roles(payload, None if args.fold < 0 else args.fold)
    test = [graphs[i] for i in roles['test']]
    if args.limit:
        test = test[:args.limit]
    labels = [int(g['label']) for g in test]

    sys.path.insert(0, str(PKG / 'snapshot'))
    for rel in ('ref/AgentTrust/src', 'ref/AgentTrust'):
        sys.path.insert(0, str(REPO / rel))
    from compare_all_baselines import baseline_agenttrust  # old adapter
    old = baseline_agenttrust(test)

    preds, ledgers = agenttrust_fixed.predict(test, 'prod')
    new_strict = preds['agenttrust_prod_strict']
    new_alarm = preds['agenttrust_prod_alarm']

    def conf(p):
        tp = sum(1 for x, y in zip(p, labels) if x and y)
        fp = sum(1 for x, y in zip(p, labels) if x and not y)
        fn = sum(1 for x, y in zip(p, labels) if not x and y)
        tn = sum(1 for x, y in zip(p, labels) if not x and not y)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'tpr': tp / (tp + fn) if tp + fn else None,
                'fpr': fp / (fp + tn) if fp + tn else None}

    verdicts = Counter()
    for d in ledgers:
        verdicts.update(d['verdict_counts'])
    changed = []
    for i, (g, o, s, a, d) in enumerate(zip(test, old, new_strict, new_alarm, ledgers)):
        if o != s:
            changed.append({'sample': str(g.get('task_id')), 'label': int(g['label']),
                            'old': int(o), 'new_strict': int(s), 'new_alarm': int(a),
                            'verdicts': d['verdict_counts'],
                            'first_strict_step': d['first_flagged_step_strict']})
    out = {
        'bench': args.bench, 'fold': args.fold, 'n': len(test),
        'n_pos': sum(labels), 'at': utc_now(),
        'old_confusion': conf(old),
        'new_strict_confusion': conf(new_strict),
        'new_alarm_confusion': conf(new_alarm),
        'verdict_totals': dict(verdicts),
        'n_changed': len(changed),
        'changed': changed[:50],
    }
    path = HERE / f'acceptance_old_vs_new_agenttrust_{args.bench}_f{args.fold}.json'
    write_json(path, out)
    print(json.dumps({k: out[k] for k in
                      ('n', 'n_pos', 'old_confusion', 'new_strict_confusion',
                       'new_alarm_confusion', 'verdict_totals', 'n_changed')},
                     ensure_ascii=False, indent=2))
    print('wrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
