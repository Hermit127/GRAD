#!/usr/bin/env python3
"""NeuroTaint dev-calibration: unified-FPR operating points (zero API).

For every fix2 bench fold (and MRT cross-task folds) score the DEV and TEST
graphs with the fixed NeuroTaint primary variant, select the threshold at the
unified dev FPR target (<=10%), and report the resulting TEST TPR/FPR plus
AUROC. Output: LOGS/final_five_tables/raw_results/nt_dev_calibration.json/md.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/fidelity_fix_20260916'))
sys.path.insert(0, str(REPO / 'code/repro/neurotaint_v2'))
sys.path.insert(0, str(REPO / 'code/repro'))

import neurotaint_fixed as fx  # noqa: E402
from neurotaint import get_model  # noqa: E402
from run_neurotaint import roles_for, FIX2, MRT, auroc  # noqa: E402

VAR = {'restricted': True, 'lcs_unit': 'char', 'proxy': False}
TARGET_FPR = 0.10
BENCHES = ('decomp', 'mg', 'minja', 'mcp')
MRT_FOLDS = ('american_airlines_2', 'api_key_calendar_agendas_2',
             'bank_transfer_2', 'expenditure_accounting', 'spam_filter_update')


def score_graphs(graphs, model):
    out = []
    for g in graphs:
        r = fx.audit_fixed(g, model, restricted=VAR['restricted'],
                           use_causal_proxy=VAR['proxy'],
                           lcs_unit=VAR['lcs_unit'])
        out.append((float(r['score']), int(g['label'])))
    return out


def threshold_at_fpr(dev, target):
    neg = sorted(s for s, y in dev if y == 0)
    if not neg:
        return None
    import math as _m
    k = _m.ceil(target * len(neg)) - 1
    k = max(0, min(k, len(neg) - 1))
    return neg[k]


def test_metrics(test, thr):
    tp = sum(1 for s, y in test if y == 1 and s >= thr)
    fp = sum(1 for s, y in test if y == 0 and s >= thr)
    fn = sum(1 for s, y in test if y == 1 and s < thr)
    tn = sum(1 for s, y in test if y == 0 and s < thr)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'tpr': tp / (tp + fn) if tp + fn else None,
            'fpr': fp / (fp + tn) if fp + tn else None}


def main():
    model = get_model()
    report = {'variant': VAR, 'target_dev_fpr': TARGET_FPR, 'datasets': {}}
    for bench in BENCHES:
        payload = json.loads((FIX2 / 'split_ids' / f'{bench}.json').read_text())
        n = 1 if payload['mode'] == 'single' else len(payload['folds'])
        folds = []
        for f in range(n):
            roles = roles_for(bench, payload, f)
            dev = roles['dev']
            test = roles['test']
            dev_s = score_graphs(dev, model)
            test_s = score_graphs(test, model)
            thr = threshold_at_fpr(dev_s, TARGET_FPR)
            tm = test_metrics(test_s, thr) if thr is not None else None
            folds.append({'fold': f, 'n_dev': len(dev), 'n_test': len(test),
                          'dev_threshold_fpr10': thr,
                          'dev_auroc': auroc([s for s, _ in dev_s],
                                             [y for _, y in dev_s]),
                          'test_auroc': auroc([s for s, _ in test_s],
                                              [y for _, y in test_s]),
                          'test_at_dev_threshold': tm})
        report['datasets'][bench] = {'folds': folds}
    # MRT: dev = calibration.jsonl, test = admission.jsonl
    mrt_folds = []
    for task in MRT_FOLDS:
        dev = [json.loads(l) for l in
               open(MRT / f'fold_{task}' / 'calibration.jsonl') if l.strip()]
        test = [json.loads(l) for l in
                open(MRT / f'fold_{task}' / 'admission.jsonl') if l.strip()]
        dev_s = score_graphs(dev, model)
        test_s = score_graphs(test, model)
        thr = threshold_at_fpr(dev_s, TARGET_FPR)
        tm = test_metrics(test_s, thr) if thr is not None else None
        mrt_folds.append({'fold': task, 'n_dev': len(dev), 'n_test': len(test),
                          'dev_threshold_fpr10': thr,
                          'dev_auroc': auroc([s for s, _ in dev_s], [y for _, y in dev_s]),
                          'test_auroc': auroc([s for s, _ in test_s], [y for _, y in test_s]),
                          'test_at_dev_threshold': tm})
    report['datasets']['mrt'] = {'folds': mrt_folds}

    out = REPO / 'LOGS/final_five_tables/raw_results/nt_dev_calibration.json'
    out.write_text(json.dumps(report, indent=1))
    md = ['# NeuroTaint dev-calibrated operating points (unified dev FPR<=10%)',
          '', '| dataset | fold | n_dev | n_test | dev AUROC | test AUROC | test TPR | test FPR |',
          '|---|---|---|---|---|---|---|---|']
    for bench, d in report['datasets'].items():
        for f in d['folds']:
            tm = f['test_at_dev_threshold'] or {}
            md.append(f"| {bench} | {f['fold']} | {f['n_dev']} | {f['n_test']} | "
                      f"{'-' if f['dev_auroc'] is None else round(f['dev_auroc'],3)} | "
                      f"{'-' if f['test_auroc'] is None else round(f['test_auroc'],3)} | "
                      f"{'-' if tm.get('tpr') is None else round(tm['tpr'],3)} | "
                      f"{'-' if tm.get('fpr') is None else round(tm['fpr'],3)} |")
    (out.with_suffix('.md')).write_text('\n'.join(md) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
