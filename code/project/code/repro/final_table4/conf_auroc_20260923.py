#!/usr/bin/env python3
"""Shared-input detection evaluation for the confirmation batch (zero/paid API).

Reads the None-arm decision records (`--record-decisions` dumps: prefix nodes +
prefix trace + pre-defined label) and scores the SAME input with both detectors:

  * T3 gate: in-process ``G3Gate.score_nodes(record['prefix_nodes'])``;
  * Sequential Monitor: the official prompt via one judge call per record
    (``monitor_judgement``), which is the only API spend of this script.

Labels are the pre-defined detection targets (violation=1 / compliant=0);
off-policy candidates and missing records are reported, never relabelled by
post-defense outcome.

Usage:
  python conf_auroc_20260923.py --records DIR --out JSON \
      --cap-deepseek 0.4 [--mock] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

from budget_client import BudgetStop, CallFailed, make_clients  # noqa: E402
from defenses import MonitorContractError, monitor_judgement  # noqa: E402


def tpr_at_fpr(y, p, target=0.10):
    order = np.argsort(-p)
    y = y[order]
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    tp = fp = 0
    best = 0.0
    for yy in y:
        tp += int(yy == 1)
        fp += int(yy == 0)
        if neg and fp / neg <= target:
            best = max(best, tp / pos)
    return round(best, 4)


class MockJudge:
    def __init__(self, tag):
        from types import SimpleNamespace
        self.tag = tag
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            create=self.create))

    def create(self, messages, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(content='0.9', model='mock', finish_reason='stop',
                               usage={'prompt_tokens': 1,
                                      'completion_tokens': 1},
                               cached_tokens=0, cost_rmb=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget-dir', default=None)
    ap.add_argument('--cap-deepseek', type=float, default=0.0)
    ap.add_argument('--gate-ckpt', default='')
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--skip-monitor', action='store_true')
    args = ap.parse_args()

    import os
    if args.gate_ckpt:
        os.environ['G3_GATE_CKPT'] = args.gate_ckpt
    else:
        os.environ.setdefault(
            'G3_GATE_CKPT', str(REPO / 'code/repro/out/online_gate_g3/'
                                       'transfer_joint/'
                                       'g3_gate_512_transfer_joint.pt'))
    from g3_gate import G3Gate
    gate = G3Gate()

    files = sorted(glob.glob(str(Path(args.records) / '*.json')))
    if args.limit:
        files = files[:args.limit]
    rows = [json.loads(Path(p).read_text()) for p in files]
    judge = MockJudge('mock') if args.mock else make_clients(
        args.budget_dir or (Path(args.records).parent / 'auroc_budget'),
        args.cap_deepseek, 0.0)['deepseek-flash']

    out_rows = []
    for r in rows:
        rec = {
            'run_id': r['run_id'], 'backbone': r['backbone'], 'env': r['env'],
            'family': r['family'], 'condition': r['condition'],
            'label': r['label'], 'label_kind': r['label_kind'],
            'candidate': r['candidate'], 'none_outcome': r.get('none_outcome'),
        }
        p, lat = gate.score_nodes(r['prefix_nodes'])
        rec['grad'] = {'p': round(float(p), 6),
                       'latency_ms': round(float(lat), 2),
                       'build_ms': round(gate.last_build_ms, 2),
                       'forward_ms': round(gate.last_forward_ms, 2),
                       'threshold': gate.threshold,
                       'block': bool(p >= gate.threshold)}
        if not args.skip_monitor:
            try:
                res = monitor_judgement(
                    judge, r['prefix_trace'],
                    (r['candidate']['tool'], r['candidate']['arg']))
                rec['monitor'] = {'p': res['p'], 'alert': res['alert'],
                                  'threshold': res['threshold']}
            except BudgetStop as exc:
                rec['monitor'] = {'error': f'BudgetStop: {exc}'}
                out_rows.append(rec)
                break
            except (MonitorContractError, CallFailed) as exc:
                rec['monitor'] = {'error': f'{type(exc).__name__}: {exc}'}
        out_rows.append(rec)

    def block(rows_, key):
        ys, ps = [], []
        for r in rows_:
            if r['label'] is None:
                continue
            det = r.get(key) or {}
            if 'p' not in det:
                continue
            ys.append(r['label'])
            ps.append(det['p'])
        if not ys or len(set(ys)) < 2:
            return {'n': len(ys), 'pos': sum(ys), 'auroc': None}
        y, p = np.array(ys), np.array(ps)
        return {'n': len(ys), 'pos': int(y.sum()), 'neg': int(len(y) - y.sum()),
                'auroc': round(float(roc_auc_score(y, p)), 4),
                'tpr_at_fpr10': tpr_at_fpr(y, p)}

    def per_backbone(key):
        out = {}
        for bb in sorted({r['backbone'] for r in out_rows}):
            sub = [r for r in out_rows if r['backbone'] == bb]
            out[bb] = block(sub, key)
        return out

    summary = {
        'n_records': len(out_rows),
        'n_excluded_off_policy': sum(1 for r in out_rows
                                     if r['label'] is None),
        'n_monitor_errors': sum(1 for r in out_rows
                                if 'error' in (r.get('monitor') or {})),
        'pooled': {'grad': block(out_rows, 'grad'),
                   'monitor': block(out_rows, 'monitor')},
        'per_backbone': {'grad': per_backbone('grad'),
                         'monitor': per_backbone('monitor')},
        'gate': {'ckpt': gate.meta['ckpt'], 'threshold': gate.threshold},
    }
    Path(args.out).write_text(json.dumps(
        {'summary': summary, 'rows': out_rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
