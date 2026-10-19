#!/usr/bin/env python3
"""Unified-FPR operating-point audit + MCP/MRT confusion audit (zero API).

Sources:
  - GRAD: code/repro/out/unified_gnn/runs/*.json (per fold/seed: dev-selected
    threshold at dev FPR<=10% and the TEST metrics at that threshold, with
    per-fold n_pos/n_neg) + unified_aggregate.json.
  - NeuroTaint: raw_results/nt_op_point_v2.json (dev scoring pass with the
    pre-registered strict tie rule; supersedes nt_dev_calibration.json whose
    threshold order statistic was inverted) and the existing tpr_at_fpr10
    diagnostics from neurotaint_fixed_full_20260916_v3.
  - Sequential Monitor: native 0.5 threshold from the LSM JSONs.
  - Shortcut probes: node-count AUROC on each unified test split.

Output: raw_results/operating_point_audit.json and .md
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FF = REPO / 'LOGS/final_five_tables'
RAW = FF / 'raw_results'
RUNS = REPO / 'code/repro/out/unified_gnn/runs'
AGG = REPO / 'code/repro/out/unified_gnn/unified_aggregate.json'


def load_grad():
    out = {}
    for p in sorted(RUNS.glob('*_s21*.json')):
        name = p.stem.rsplit('_s', 1)[0]
        bench = name.split('_')[0] if not name.startswith('mrt') else 'mrt'
        d = json.loads(p.read_text())
        tr = (d.get('threshold_reports') or {}).get('fpr_10') or {}
        test = tr.get('test') or {}
        val = tr.get('val') or {}
        if not test:
            continue
        out.setdefault(bench, []).append({
            'unit': name, 'seed': p.stem.rsplit('_s', 1)[1],
            'dev_threshold': val.get('threshold'),
            'dev_fpr_at_threshold': val.get('fpr'),
            'test_tpr': test.get('recall'), 'test_fpr': test.get('fpr'),
            'test_auroc': (d.get('test_graph') or {}).get('auroc'),
        })
    return out


def mrt_lengths():
    folds = {}
    for task in ('american_airlines_2', 'api_key_calendar_agendas_2',
                 'bank_transfer_2', 'expenditure_accounting',
                 'spam_filter_update'):
        base = REPO / 'results/mrt_cross_task_v1' / f'fold_{task}'
        row = {}
        for split in ('train', 'calibration', 'admission'):
            gs = [json.loads(l) for l in open(base / f'{split}.jsonl') if l.strip()]
            row[split] = {'n': len(gs),
                          'avg_nodes': round(sum(len(g['nodes']) for g in gs) / len(gs), 1)}
        folds[task] = row
    return folds


def main():
    agg = json.loads(AGG.read_text())
    grad = load_grad()
    nt = json.loads((RAW / 'nt_op_point_v2.json').read_text())
    report = {
        'protocol': {
            'unified_dev_fpr_target': 0.10,
            'rule': 'continuous-score methods: threshold selected on the DEV '
                    'split at FPR<=10%, then the TEST TPR/FPR at that threshold '
                    'is reported (no test-set tuning); binary rule methods are '
                    'reported at their native operating point (separate table).',
        },
        'grad': {}, 'neurotaint': {}, 'sequential_monitor': {},
        'mcp_mrt_fold_audit': {}, 'mrt_length_shift': mrt_lengths(),
    }
    import re
    for bench, units in grad.items():
        tp = fp = fn = tn = 0
        folds_by_key = {(str(pf.get('fold')), str(pf.get('seed'))): pf
                        for pf in agg['benchmarks'].get(bench, {}).get('per_fold', [])}
        for u in units:
            # unit names: decomp / mcp_f0 / mg_f2 / minja_f1 / mrt_<task>
            m = re.search(r'_f(\d+)$', u['unit'])
            if bench == 'mrt':
                fold_key = u['unit'][len('mrt_'):]
            elif m:
                fold_key = m.group(1)
            else:
                fold_key = '0'
            pf = folds_by_key.get((fold_key, u['seed']))
            if pf is None:
                continue
            n_pos, n_neg = pf.get('n_pos'), pf.get('n_neg')
            if n_pos is None:
                continue
            tpr, fpr = u['test_tpr'], u['test_fpr']
            tp += round(tpr * n_pos); fn += round((1 - tpr) * n_pos)
            fp += round(fpr * n_neg); tn += round((1 - fpr) * n_neg)
        if tp + fn:
            report['grad'][bench] = {
                'pooled_test_tpr': tp / (tp + fn),
                'pooled_test_fpr': fp / (fp + tn) if fp + tn else None,
                'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
                'per_unit': units,
            }
    for bench, d in nt['datasets'].items():
        thr_vals = [f['dev_threshold_fpr10'] for f in d['folds']]
        pooled = d['pooled_at_dev_threshold']
        native = d.get('pooled_test_native_score_gt_0') or {}
        report['neurotaint'][bench] = {
            'dev_thresholds': thr_vals,
            'degenerate_dev_calibration': d['degenerate_dev_calibration'],
            'pooled_test_at_dev_threshold': pooled,
            'pooled_test_native_score_gt_0': native,
            'note': d.get('note'),
            'folds': d['folds'],
        }
    report['neurotaint_protocol'] = nt['protocol']
    for name, path in (('official_test_cumulative', 'LOGS/lsm_official_test_cumulative_20260916.json'),
                       ('official_test_separate', 'LOGS/lsm_official_test_separate_20260916.json'),
                       ('decomp_fix2_cumulative', 'LOGS/lsm_decomp_fix2_cumulative_20260916.json')):
        p = REPO / path
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        m = d.get('metrics', {})
        report['sequential_monitor'][name] = {
            'n_tasks': d.get('n_tasks'), 'auroc': m.get('auroc'),
            'at_0.5': m.get('at_0.5'),
            'note': 'native fixed threshold 0.5 (official protocol); no dev-calibrated op point recorded',
        }
    # MCP/MRT per-fold audit from the aggregate
    for bench in ('mcp', 'mrt'):
        report['mcp_mrt_fold_audit'][bench] = agg['benchmarks'][bench]
    out = RAW / 'operating_point_audit.json'
    out.write_text(json.dumps(report, indent=1))
    print('wrote', out)
    md = ['# Operating-point audit (unified dev FPR<=10%)', '']
    md.append('## GRAD pooled test TPR/FPR at the dev-selected threshold')
    md.append('')
    md.append('| dataset | pooled test TPR | pooled test FPR | TP/FP/FN/TN |')
    md.append('|---|---|---|---|')
    for bench, d in report['grad'].items():
        c = d['confusion']
        md.append(f"| {bench} | {d['pooled_test_tpr']:.3f} | {d['pooled_test_fpr']:.3f} | "
                  f"{c['tp']}/{c['fp']}/{c['fn']}/{c['tn']} |")
    md += ['', '## NeuroTaint (propagation component; dev FPR<=10% threshold, '
           'strict > rule, ties negative)', '',
           '| dataset | pooled test TPR | pooled test FPR | TP/FP/FN/TN | native >0 TPR/FPR | degenerate | note |',
           '|---|---|---|---|---|---|---|']
    for bench, d in report['neurotaint'].items():
        p = d['pooled_test_at_dev_threshold']
        n = d.get('pooled_test_native_score_gt_0') or {}
        md.append(f"| {bench} | {p['tpr']:.3f} | {p['fpr']:.3f} | "
                  f"{p['tp']}/{p['fp']}/{p['fn']}/{p['tn']} | "
                  f"{n.get('tpr', float('nan')):.3f}/{n.get('fpr', float('nan')):.3f} | "
                  f"{'Y' if d['degenerate_dev_calibration'] else 'N'} | {d['note']} |")
    md += ['', '## MRT covariate shift (avg nodes: admission differs from calibration/train in both directions)', '']
    md.append('| fold | train | calibration | admission |')
    md.append('|---|---|---|---|')
    for f, row in report['mrt_length_shift'].items():
        md.append(f"| {f} | {row['train']['avg_nodes']} | {row['calibration']['avg_nodes']} | {row['admission']['avg_nodes']} |")
    (out.with_suffix('.md')).write_text('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
