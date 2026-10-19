#!/usr/bin/env python3
"""Aggregate Phase C training runs into the comparison table (zero API)."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('.')
D = ROOT / 'LOGS/final_table2/grad_pairs_train_20260918'


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (round(statistics.mean(vals), 4),
            round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4))


def arm_key(r):
    a = r['args']
    return f"C3_lambda{a['lam']}" if a['arm'] == 'C3' else a['arm']


def metrics_of(r):
    ev = r['eval']
    return {
        'orig_test_auroc': ev['orig_test']['auroc'],
        'orig_test_f1': ev['orig_test']['f1_at_0.5'],
        'pair_dev_rank': ev['pair_dev']['ranking_accuracy'],
        'pair_dev_auroc': ev['pair_dev']['auroc'],
        'pair_dev_tpr': ev['pair_dev']['at_dev_fpr10_threshold']['tpr'],
        'pair_dev_fpr': ev['pair_dev']['at_dev_fpr10_threshold']['fpr'],
        'best_epoch': r.get('best_epoch', 0),
    }


def main() -> int:
    c0 = json.loads((D / 'C0_seed1.json').read_text())
    runs = [json.loads(p.read_text()) for p in sorted(D.glob('C*.json'))
            if not p.name.startswith('C0')]
    groups = defaultdict(list)
    for r in runs:
        groups[arm_key(r)].append(r)

    rows = [{'arm': 'C0', 'seeds': 1, **{k: stat([v]) for k, v in metrics_of(c0).items()}}]
    for arm, rs in sorted(groups.items()):
        ms = [metrics_of(r) for r in rs]
        rows.append({'arm': arm, 'seeds': len(rs),
                     **{k: stat([m[k] for m in ms]) for k in ms[0]}})

    md = ['# Phase C results (mean over seeds; training-free C0 reference)', '',
          '| arm | seeds | orig_test AUROC | orig_test F1@.5 | pair-dev rank acc | '
          'pair-dev AUROC | pair-dev TPR@FPR10 | pair-dev FPR@FPR10 | best epoch |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        def f(k):
            v = r[k]
            return '--' if not v or v[0] is None else (
                f'{v[0]:.3f}' + (f' [{v[1]:.3f}]' if r['seeds'] > 1 else ''))
        md.append(f"| {r['arm']} | {r['seeds']} | {f('orig_test_auroc')} | "
                  f"{f('orig_test_f1')} | {f('pair_dev_rank')} | {f('pair_dev_auroc')} | "
                  f"{f('pair_dev_tpr')} | {f('pair_dev_fpr')} | {f('best_epoch')} |")
    tex = [r'\begin{tabular}{lccccc}', r'\toprule',
           r'Arm & Test AUROC & Pair rank acc & Pair AUROC & Pair TPR@FPR10 & '
           r'Pair FPR@FPR10 \\', r'\midrule']
    for r in rows:
        def g(k):
            v = r[k]
            return '--' if not v or v[0] is None else f'{v[0]:.3f}'
        tex.append(f"{r['arm']} & {g('orig_test_auroc')} & {g('pair_dev_rank')} & "
                   f"{g('pair_dev_auroc')} & {g('pair_dev_tpr')} & {g('pair_dev_fpr')} \\\\")
    tex += [r'\bottomrule', r'\end{tabular}']
    (D / 'phaseC_results.md').write_text('\n'.join(md) + '\n')
    (D / 'phaseC_results.tex').write_text('\n'.join(tex) + '\n')
    (D / 'phaseC_results.json').write_text(json.dumps(rows, indent=1))
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
