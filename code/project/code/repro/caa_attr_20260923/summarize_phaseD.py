#!/usr/bin/env python3
"""Compact summary of the Phase D route-swap report (per arm, over runs)."""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']
ORDERS = {
    'fwd': STAGES,
    'rev': list(reversed(STAGES)),
    'trs': ['tool_poisoning', 'return_replay', 'shadow_confusion'],
    'rts': ['return_replay', 'tool_poisoning', 'shadow_confusion'],
    'str': ['shadow_confusion', 'tool_poisoning', 'return_replay'],
    'srt': ['shadow_confusion', 'return_replay', 'tool_poisoning'],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    args = ap.parse_args()
    d = json.load(open(args.report))
    for arm in ('snap', 'stage0'):
        agg = collections.defaultdict(list)
        ranks = collections.defaultdict(list)
        best = collections.defaultdict(list)
        reg = collections.defaultdict(list)
        mats = collections.defaultdict(lambda: collections.defaultdict(list))
        newest = collections.defaultdict(list)
        for run, arms in d['runs'].items():
            if arm not in arms:
                continue
            order = ORDERS[run.rsplit('_', 1)[0]]
            newest_stage = order[-1]
            newest_expert = order[-1]
            for s, v in arms[arm]['stages'].items():
                agg[(s, 'cent')].append(v['centroid_acc'])
                agg[(s, 'rand')].append(v['random_acc_exp'])
                agg[(s, 'all')].append(v['all_acc'])
                agg[(s, 'orc')].append(v['oracle_acc'])
                ranks[s].append(v['mean_rank_of_selected'])
                best[s].append(v['selected_best_frac'])
                reg[s].append(v['mean_regret'])
            for e, ms in arms[arm]['expert_stage_matrix'].items():
                for s, v in ms.items():
                    mats[e][s].append(v)
                    if s == newest_stage:
                        rel = 'newest' if e == newest_expert else (
                            'old' if order.index(e) == 0 else 'middle')
                        newest[rel].append(v)
        print(f'== {arm} (n_runs={len(agg[("tool_poisoning", "cent")])})')
        for s in STAGES:
            print(f'  {s}: centroid={np.mean(agg[(s, "cent")]):.3f} '
                  f'random_exp={np.mean(agg[(s, "rand")]):.3f} '
                  f'cent-rand={np.mean(agg[(s, "cent")]) - np.mean(agg[(s, "rand")]):+.3f} '
                  f'all={np.mean(agg[(s, "all")]):.3f} '
                  f'oracle={np.mean(agg[(s, "orc")]):.3f} | '
                  f'rank={np.mean(ranks[s]):.2f} '
                  f'selbest={np.mean(best[s]):.3f} '
                  f'regret={np.mean(reg[s]):.3f}')
        print('  expert x stage acc:')
        for e in sorted(mats):
            print(f'    {e}: ' + ' '.join(
                f'{s[:6]}={np.mean(mats[e][s]):.3f}' for s in STAGES))
        print('  accuracy on the newest (final) task, by expert vintage:')
        for rel in ('newest', 'middle', 'old'):
            if newest[rel]:
                print(f'    {rel}: {np.mean(newest[rel]):.3f} '
                      f'(n={len(newest[rel])})')
    for arm in ('snap', 'stage0'):
        c = d['pooled']['controls_centroid_minus_random'].get(arm)
        if not c:
            continue
        print(f'\n== pooled controls ({arm})')
        print('raw mean delta = {:+.4f} (n_samples={}, sources={})'.format(
            c['raw_mean_delta'], c['n_samples'], c['source_composition']))
        for n, v in c['ols'].items():
            print(f"  {n}: {v['beta']:+.4f} +- {v['se_cluster']:.4f} "
                  f"(cluster SE)")
        print('  strata:')
        for k, v in d['pooled']['strata_centroid_minus_random'][arm].items():
            print(f"    {k}: {v['delta']:+.4f} "
                  f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] n={v['n']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
