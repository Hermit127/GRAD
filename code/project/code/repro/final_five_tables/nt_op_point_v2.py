#!/usr/bin/env python3
"""NeuroTaint operating points v2 (propagation component, zero API).

Supersedes ``nt_dev_calibration.py`` whose threshold order statistic was
inverted: it took the (floor(target*n_neg))-th *smallest* dev-negative score
and then flagged ``score >= t``, which realizes a dev FPR of ~0.9 instead of
the stated target 0.10 (see LOGS/final_five_tables/raw_results/
nt_dev_calibration.md; old outputs are kept, not overwritten).

Pre-registered rules used here (written before looking at any test metric):

* Threshold selection (dev only): with ``m = floor(target * n_neg)``, the
  selected threshold ``t*`` is the (m+1)-th largest dev-negative score.  Every
  observed score can serve as a candidate; this is the smallest candidate that
  satisfies the target under the decision rule below.
* Decision rule / tie handling (fixed before execution, identical on dev and
  test): a unit is positive iff ``score > t*`` -- ties at the threshold are
  negative.  With ties this guarantees the realized dev FPR <= target and,
  when the whole dev score vector is tied (the observed degeneration), it
  produces the all-negative dev prediction instead of the all-positive
  artifact of ``score >= t``.
* Degenerate calibration is reported, never hidden: if every dev score is
  identical the calibration carries no operating point; ``t*`` equals that
  value, the strict rule flags nothing on dev, and the *test* TPR/FPR at the
  same ``t*`` is still reported with a degeneration flag (no test-set
  threshold selection is ever performed).
* Native rule (secondary, pre-determined by the paper's thresholds): positive
  iff ``score > 0``, i.e. stage != none.  The score returned by
  ``neurotaint_fixed.audit_fixed`` already only records evidence that passed
  the paper tier thresholds (theta_str=0.15 / theta_sem=0.60/0.85), so
  ``score > 0`` is exactly "at least one tier produced evidence".

Datasets / splits (unchanged from the archived runs):
  decomp/mg/minja/mcp : fix2 split_ids dev/test via ``roles_for``
  mrt                 : calibration.jsonl (dev) / admission.jsonl (test)
  injecagent/asb/agentharm : fix2_splits_x4 dev/test (family-disjoint; test
                        identical to the archived x4 test used for AUROC)

Outputs:
  raw_results/nt_op_point_v2.json
  raw_results/nt_op_point_v2.md
  raw_results/nt_op_point_v2_scores.jsonl   (per-unit scores, provenance)

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python \
    code/repro/final_five_tables/nt_op_point_v2.py
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
RAW = REPO / 'LOGS/final_five_tables/raw_results'
X4 = REPO / 'code/repro/out/unified_gnn/fix2_splits_x4'
NT_V3 = (REPO / 'code/repro/out/baseline_fidelity_fix3_20260916/neurotaint_fixed/'
         'neurotaint_fixed_full_20260916_v3.json')
NT_X4 = (REPO / 'code/repro/out/baseline_fidelity_fix3_20260916/neurotaint_fixed/'
         'neurotaint_fixed_full_20260916_x4_20260916.json')
ARCHIVED_AUROC = {
    'decomp': NT_V3, 'mg': NT_V3, 'minja': NT_V3, 'mcp': NT_V3, 'mrt': NT_V3,
    'injecagent': NT_X4, 'asb': NT_X4, 'agentharm': NT_X4,
}
BENCHES = ('decomp', 'mg', 'minja', 'mcp')
MRT_FOLDS = ('american_airlines_2', 'api_key_calendar_agendas_2',
             'bank_transfer_2', 'expenditure_accounting', 'spam_filter_update')
X4_NAMES = ('injecagent', 'asb', 'agentharm')

PROTOCOL = {
    'variant': VAR,
    'target_dev_fpr': TARGET_FPR,
    'threshold_selection': (
        'dev only; m = floor(target*n_neg); t* = (m+1)-th largest dev-negative '
        'score (smallest candidate score satisfying the target under the strict '
        'decision rule)'),
    'decision_rule': (
        'positive iff score > t* (strict; ties at the threshold are negative); '
        'same rule on dev and test; no test-set threshold selection'),
    'degenerate_rule': (
        'if all dev scores are identical, t* equals that value and the strict '
        'rule flags nothing on dev; test TPR/FPR at the same t* is still '
        'reported with degenerate_dev_calibration=true (never left blank, never '
        'tuned on test)'),
    'native_rule': (
        'positive iff score > 0 (stage != none): the paper tier thresholds are '
        'applied inside audit_fixed, so score>0 means at least one evidence '
        'stage fired; reported as the secondary pre-determined operating point'),
    'supersedes': (
        'nt_dev_calibration.json/md (order-statistic bug: selected the '
        'floor(target*n_neg)-th smallest dev-negative under score>=t, realized '
        'dev FPR ~0.9); old files kept for the record'),
}


def score_graphs(graphs, model):
    out = []
    for g in graphs:
        r = fx.audit_fixed(g, model, restricted=VAR['restricted'],
                           use_causal_proxy=VAR['proxy'],
                           lcs_unit=VAR['lcs_unit'])
        out.append({
            'score': float(r['score']), 'label': int(g['label']),
            'stage': r['stage'],
            'task_id': g.get('task_id'),
            'group': g.get('_group') or g.get('family'),
        })
    return out


def select_threshold(dev):
    """(m+1)-th largest dev-negative score; None if no negatives."""
    neg = sorted(float(s['score']) for s in dev if s['label'] == 0)
    if not neg:
        return None
    m = int(math.floor(TARGET_FPR * len(neg)))
    neg_desc = neg[::-1]
    if m >= len(neg):
        return neg_desc[-1]
    return neg_desc[m]


def metrics(pairs, thr, strict=True):
    def pos(s):
        return s > thr if strict else s >= thr
    tp = sum(1 for s, y in pairs if y == 1 and pos(s))
    fp = sum(1 for s, y in pairs if y == 0 and pos(s))
    fn = sum(1 for s, y in pairs if y == 1 and not pos(s))
    tn = sum(1 for s, y in pairs if y == 0 and not pos(s))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'tpr': tp / (tp + fn) if tp + fn else None,
            'fpr': fp / (fp + tn) if fp + tn else None}


def _num(v):
    return None if v is None or math.isnan(v) else float(v)


def fold_entry(dev, test, model, auoc_ref, name, fold):
    dev_s = score_graphs(dev, model)
    test_s = score_graphs(test, model)
    thr = select_threshold(dev_s)
    tm = metrics([(s['score'], s['label']) for s in test_s], thr) \
        if thr is not None else None
    dm = metrics([(s['score'], s['label']) for s in dev_s], thr) \
        if thr is not None else None
    native_test = metrics([(s['score'], s['label']) for s in test_s], 0.0)
    native_dev = metrics([(s['score'], s['label']) for s in dev_s], 0.0)
    test_scores = [s['score'] for s in test_s]
    labels = [s['label'] for s in test_s]
    test_auroc = _num(auroc(test_scores, labels))
    dev_scores = [s['score'] for s in dev_s]
    dev_labels = [s['label'] for s in dev_s]
    dev_auroc = _num(auroc(dev_scores, dev_labels))
    all_tied = len(set(dev_scores)) == 1 if dev_scores else True
    entry = {
        'dataset': name, 'fold': fold,
        'n_dev': len(dev_s), 'n_dev_pos': sum(1 for s in dev_s if s['label'] == 1),
        'n_dev_neg': sum(1 for s in dev_s if s['label'] == 0),
        'n_test': len(test_s), 'n_test_pos': sum(1 for s in test_s if s['label'] == 1),
        'n_test_neg': sum(1 for s in test_s if s['label'] == 0),
        'dev_auroc': dev_auroc, 'test_auroc': test_auroc,
        'auroc_matches_archive': (
            auoc_ref is not None and test_auroc is not None
            and round(test_auroc, 4) == round(auoc_ref, 4)),
        'archive_test_auroc': auoc_ref,
        'dev_threshold_fpr10': thr,
        'dev_all_scores_tied': all_tied,
        'dev_neg_tied_at_zero': sum(1 for s in dev_s
                                    if s['label'] == 0 and s['score'] == 0.0),
        'dev_scores_max': max(dev_scores) if dev_scores else None,
        'dev_at_threshold': dm,
        'test_at_dev_threshold': tm,
        'native_dev': native_dev,
        'native_test': native_test,
        'degenerate_dev_calibration': bool(all_tied),
        'stage_hist_test': _hist(test_s),
    }
    return entry, dev_s, test_s


def _hist(rows):
    hist = {}
    for r in rows:
        hist[r['stage']] = hist.get(r['stage'], 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def pool(folds, key):
    """Sum TP/FP/FN/TN across folds at each fold's own threshold."""
    tp = sum(f[key]['tp'] for f in folds if f.get(key))
    fp = sum(f[key]['fp'] for f in folds if f.get(key))
    fn = sum(f[key]['fn'] for f in folds if f.get(key))
    tn = sum(f[key]['tn'] for f in folds if f.get(key))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'tpr': tp / (tp + fn) if tp + fn else None,
            'fpr': fp / (fp + tn) if fp + tn else None,
            'n_pos': tp + fn, 'n_neg': fp + tn}


def main():
    model = get_model()
    v3 = json.loads(NT_V3.read_text())
    x4 = json.loads(NT_X4.read_text())
    report = {'protocol': PROTOCOL, 'datasets': {}}
    score_dump = []
    for bench in BENCHES:
        payload = json.loads((FIX2 / 'split_ids' / f'{bench}.json').read_text())
        n = 1 if payload['mode'] == 'single' else len(payload['folds'])
        ref_folds = (v3['datasets'][bench]['variants']['primary'].get('auroc_per_fold')
                     if bench in v3['datasets'] else None) or []
        folds = []
        for f in range(n):
            roles = roles_for(bench, payload, f)
            ref = ref_folds[f] if f < len(ref_folds) else None
            entry, dev_s, test_s = fold_entry(roles['dev'], roles['test'],
                                              model, ref, bench, f)
            folds.append(entry)
            score_dump += _dump(bench, f, 'dev', dev_s) + _dump(bench, f, 'test', test_s)
        report['datasets'][bench] = _dataset_block(folds)
    ref_folds = (v3['datasets']['mrt']['variants']['primary'].get('auroc_per_fold')
                 if 'mrt' in v3['datasets'] else None) or []
    for i, task in enumerate(MRT_FOLDS):
        dev = [json.loads(line) for line in
               open(MRT / f'fold_{task}' / 'calibration.jsonl') if line.strip()]
        test = [json.loads(line) for line in
                open(MRT / f'fold_{task}' / 'admission.jsonl') if line.strip()]
        ref = ref_folds[i] if i < len(ref_folds) else None
        entry, dev_s, test_s = fold_entry(dev, test, model, ref, 'mrt', task)
        report['datasets'].setdefault('mrt', {'folds': []})['folds'].append(entry)
        score_dump += _dump('mrt', task, 'dev', dev_s) + _dump('mrt', task, 'test', test_s)
    report['datasets']['mrt'] = _dataset_block(report['datasets']['mrt']['folds'])
    for name in X4_NAMES:
        dev = _load_jsonl(X4 / name / 'dev.jsonl')
        test = _load_jsonl(X4 / name / 'test.jsonl')
        ref = (x4['datasets'][name]['variants']['primary'].get('auroc_per_fold')
               if name in x4['datasets'] else None) or [None]
        entry, dev_s, test_s = fold_entry(dev, test, model, ref[0], name, 0)
        report['datasets'][name] = _dataset_block([entry])
        score_dump += _dump(name, 0, 'dev', dev_s) + _dump(name, 0, 'test', test_s)

    out = RAW / 'nt_op_point_v2.json'
    out.write_text(json.dumps(report, indent=1))
    (RAW / 'nt_op_point_v2_scores.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in score_dump) + '\n')
    _write_md(report)
    print('wrote', out, f'({len(score_dump)} scored units)')


def _dump(name, fold, split, rows):
    return [{'dataset': name, 'fold': fold, 'split': split,
             'task_id': r['task_id'], 'group': r['group'],
             'score': r['score'], 'label': r['label'], 'stage': r['stage']}
            for r in rows]


def _load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def _dataset_block(folds):
    block = {
        'folds': folds,
        'pooled_at_dev_threshold': pool(folds, 'test_at_dev_threshold'),
        'pooled_test_native_score_gt_0': pool(folds, 'native_test'),
        'degenerate_dev_calibration': all(f['degenerate_dev_calibration'] for f in folds),
        'auroc_matches_archive': all(f['auroc_matches_archive'] for f in folds),
    }
    n_deg = sum(1 for f in folds if f['degenerate_dev_calibration'])
    if n_deg == len(folds):
        block['note'] = ('dev-FPR calibration degenerate in every fold (all dev '
                         'scores tie): the strict rule predicts nothing on dev; '
                         'test TPR/FPR at the same threshold is reported as-is')
    elif n_deg:
        block['note'] = f'dev-FPR calibration degenerate in {n_deg}/{len(folds)} folds'
    else:
        block['note'] = 'dev calibration non-degenerate'
    return block


def _write_md(report):
    md = ['# NeuroTaint operating points v2 (propagation component, zero API)', '',
          '> Rules are pre-registered in `code/repro/final_five_tables/'
          'nt_op_point_v2.py` (header). Supersedes `nt_dev_calibration.md` '
          '(inverted order statistic). Old files are kept.', '',
          '## Per fold', '',
          '| dataset | fold | n_dev | n_test | dev AUROC | test AUROC | dev t= | dev TPR/FPR@t | test TPR/FPR@t | test TPR/FPR (native >0) | degenerate |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, d in report['datasets'].items():
        for f in d['folds']:
            dm = f['dev_at_threshold'] or {}
            tm = f['test_at_dev_threshold'] or {}
            nt = f['native_test']
            md.append(
                f"| {name} | {f['fold']} | {f['n_dev']} | {f['n_test']} | "
                f"{_r(f['dev_auroc'])} | {_r(f['test_auroc'])} | "
                f"{_r(f['dev_threshold_fpr10'])} | "
                f"{_ff(dm)} | {_ff(tm)} | {_ff(nt)} | "
                f"{'Y' if f['degenerate_dev_calibration'] else 'N'} |")
    md += ['', '## Pooled at each fold\'s dev threshold', '',
           '| dataset | test TPR | test FPR | TP/FP/FN/TN | native >0 TPR/FPR | degenerate | note |',
           '|---|---|---|---|---|---|---|']
    for name, d in report['datasets'].items():
        p = d['pooled_at_dev_threshold']
        n = d['pooled_test_native_score_gt_0']
        md.append(f"| {name} | {_r(p['tpr'])} | {_r(p['fpr'])} | "
                  f"{p['tp']}/{p['fp']}/{p['fn']}/{p['tn']} | "
                  f"{_r(n['tpr'])}/{_r(n['fpr'])} | "
                  f"{'Y' if d['degenerate_dev_calibration'] else 'N'} | {d['note']} |")
    md += ['', 'AUROC columns must match the archived v3/x4 JSON values '
           '(`auroc_matches_archive` in the JSON). The native `score > 0` point '
           'coincides with the dev-selected point when the selected threshold is 0.',
           '']
    (RAW / 'nt_op_point_v2.md').write_text('\n'.join(md))


def _r(v):
    if v is None:
        return '-'
    return f'{v:.4f}' if abs(v) < 1 and v != 0 else f'{v:.4f}'


def _ff(m):
    if not m or m.get('tpr') is None or m.get('fpr') is None:
        return '-'
    return f"{m['tpr']:.3f}/{m['fpr']:.3f}"


if __name__ == '__main__':
    main()
