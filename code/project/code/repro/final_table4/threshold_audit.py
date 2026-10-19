#!/usr/bin/env python
"""Threshold provenance + score-distribution audit for the frozen GRAD v2 gate.

Zero API. Answers, in order:
  1. where the threshold came from (dev split, selection rule, checkpoint);
  2. whether raw scores separate benign/malicious on the GRAD dev/test splits;
  3. how the score distribution shifts on the two Table 4 backbones
     (diagnostic only; closed-loop metrics are NOT recomputed here);
  4. train/deploy input-format identity, encoder window, label direction,
     evaluation mode and repeat-inference determinism.

Outputs LOGS/final_table4/threshold_audit_20260918/:
  scores_v2_dev.jsonl, scores_v2_test.jsonl, scores_table4.jsonl,
  summary.json, report.md (plot added separately from the same JSON).

Usage: python code/repro/final_table4/threshold_audit.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

from defenses import GRAD_V2_CKPT, GradGateV2  # noqa: E402
from envlib import ENVS, valid_authorization  # noqa: E402
from grad_nodes import v10_trace_nodes  # noqa: E402

OUT = REPO / 'LOGS/final_table4/threshold_audit_20260918'
V2 = REPO / 'code/repro/out/online_gate_v2'
SOURCE_FILES = [
    'v10_main_shop.json', 'v10_main_workspace.json', 'v10_main_finance.json',
    'v10_mainb_shop.json', 'v10_mainb_workspace.json', 'v10_mainb_finance.json',
    'v10_conf_shop.json', 'v10_conf_workspace.json', 'v10_conf_finance.json',
    'v10_confb_shop.json', 'v10_confb_workspace.json', 'v10_confb_finance.json',
    'v10_cloud.json', 'v10_cloud_v2.json']


def q(vals, x):
    if not vals:
        return None
    vals = sorted(vals)
    i = min(len(vals) - 1, max(0, int(round(x * (len(vals) - 1)))))
    return vals[i]


def dist(vals):
    if not vals:
        return {}
    logits = []
    for p in vals:
        pc = min(max(p, 1e-9), 1 - 1e-9)
        logits.append(math.log(pc / (1 - pc)))
    return {
        'n': len(vals),
        'min': min(vals), 'p05': q(vals, 0.05), 'p25': q(vals, 0.25),
        'median': statistics.median(vals), 'p75': q(vals, 0.75),
        'p95': q(vals, 0.95), 'max': max(vals),
        'logit_min': min(logits), 'logit_median': statistics.median(logits),
        'logit_max': max(logits),
    }


def auroc(labels, scores):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rpos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def score_split(role, gate):
    rows = []
    for line in (V2 / f'{role}.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        p, lat = gate.gate.score_nodes(rec['nodes'])
        lens = [len(n['features']['subtask']) for n in rec['nodes']]
        rows.append({'task_id': rec['task_id'], 'label': rec['label'],
                     'p': p, 'latency_ms': lat, 'n_nodes': len(rec['nodes']),
                     'over_window': sum(1 for x in lens if x > 192),
                     'max_node_chars': max(lens)})
    return rows


def table4_scores(gate):
    runs = []
    for batch in ('pilot', 'main'):
        p = REPO / f'LOGS/final_table4/{batch}/results.json'
        if p.exists():
            runs += json.loads(p.read_text())['runs']
    labels_path = REPO / 'LOGS/final_table4/main/labels_corrected.json'
    labels = (json.loads(labels_path.read_text())['labels']
              if labels_path.exists() else {})
    rows = []
    for run in runs:
        env = ENVS[run['env']]
        trace = run.get('trace') or {}
        events = trace.get('events', [])
        cand = None
        prefix = None
        for idx, ev in enumerate(events):
            if ev.get('tool') == env['action']:
                cand = (ev['tool'], str(ev['arg']))
                prefix = dict(trace)
                prefix['events'] = events[:idx]
                break
        if cand is None:
            for d in run.get('decisions', []):
                if d.get('tool') == env['action']:
                    cand = (d['tool'], str(d['arg']))
                    break
        if cand is None:
            continue
        if valid_authorization(trace, env, env['action'], cand[1]):
            continue
        nodes = v10_trace_nodes(prefix if prefix is not None else trace,
                                cand)
        p, lat = gate.gate.score_nodes(nodes)
        lens = [len(n['features']['subtask']) for n in nodes]
        lab = labels.get(run['run_id'])
        rows.append({
            'run_id': run['run_id'], 'batch': batch,
            'backbone': run['backbone'], 'env': run['env'],
            'family': run['family'], 'condition': run['condition'],
            'arm': run['arm'], 'p': p, 'latency_ms': lat,
            'n_nodes': len(nodes),
            'over_window': sum(1 for x in lens if x > 192),
            'harm_old': run['independent_harm'],
            'harm_a1': (lab or {}).get('new_harm'),
            'error': run.get('error', '')})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = GradGateV2()
    import torch
    summary = {'checkpoint': str(GRAD_V2_CKPT),
               'ckpt_sha256': GradGateV2.ckpt_sha256(),
               'threshold': gate.threshold,
               'threshold_unrounded': float(gate.threshold),
               'eval_mode': not gate.gate.model.training,
               'bn_eval': all(not m.training for m in gate.gate.model.modules()
                              if isinstance(m, torch.nn.BatchNorm1d)),
               'selection_rule': ('train_gnn_v2.threshold_for_target_fpr: '
                                  'highest dev score with FPR<=10%, '
                                  'ties by recall/F1/precision; stored in '
                                  'checkpoint threshold_reports.fpr_10'),
               'train_cfg': gate.meta.get('train_cfg')}

    # ---- 1. threshold provenance on the frozen dev split
    dev = score_split('dev', gate)
    test = score_split('test', gate)
    for name, rows in (('dev', dev), ('test', test)):
        (OUT / f'scores_v2_{name}.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in rows) + '\n')
    dev_lab = [r['label'] for r in dev]
    dev_p = [r['p'] for r in dev]
    test_lab = [r['label'] for r in test]
    test_p = [r['p'] for r in test]
    thr = gate.threshold
    dev_metrics = {
        'n': len(dev), 'pos': sum(dev_lab), 'neg': len(dev_lab) - sum(dev_lab),
        'auroc': auroc(dev_lab, dev_p),
        'tp': sum(1 for y, p in zip(dev_lab, dev_p) if y and p >= thr),
        'fn': sum(1 for y, p in zip(dev_lab, dev_p) if y and p < thr),
        'fp': sum(1 for y, p in zip(dev_lab, dev_p) if not y and p >= thr),
        'tn': sum(1 for y, p in zip(dev_lab, dev_p) if not y and p < thr),
    }
    test_metrics = {
        'n': len(test), 'pos': sum(test_lab), 'neg': len(test_lab) - sum(test_lab),
        'auroc': auroc(test_lab, test_p),
        'tp': sum(1 for y, p in zip(test_lab, test_p) if y and p >= thr),
        'fn': sum(1 for y, p in zip(test_lab, test_p) if y and p < thr),
        'fp': sum(1 for y, p in zip(test_lab, test_p) if not y and p >= thr),
        'tn': sum(1 for y, p in zip(test_lab, test_p) if not y and p < thr),
    }
    pos_dev = [p for y, p in zip(dev_lab, dev_p) if y]
    neg_dev = [p for y, p in zip(dev_lab, dev_p) if not y]
    summary['v2_dev'] = {**dev_metrics,
                         'dist_pos': dist(pos_dev), 'dist_neg': dist(neg_dev),
                         'threshold_margin_to_max_neg': (
                             thr - max(neg_dev) if neg_dev else None),
                         'threshold_margin_to_min_pos': (
                             min(pos_dev) - thr if pos_dev else None),
                         'n_neg_at_or_above': sum(1 for p in neg_dev
                                                  if p >= thr),
                         'n_pos_below': sum(1 for p in pos_dev if p < thr)}
    summary['v2_test'] = {**test_metrics,
                          'dist_pos': dist([p for y, p in zip(test_lab, test_p)
                                            if y]),
                          'dist_neg': dist([p for y, p in zip(test_lab, test_p)
                                            if not y])}

    # ---- 2. Table 4 stored-trace scores (diagnostic only)
    t4 = table4_scores(gate)
    (OUT / 'scores_table4.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in t4) + '\n')
    by = defaultdict(list)
    for r in t4:
        by[(r['batch'], r['backbone'], r['condition'], r['arm'])].append(r)
    cells = {}
    for k, rows in sorted(by.items()):
        key = '|'.join(map(str, k))
        ps = [r['p'] for r in rows]
        harm = [r['p'] for r in rows if r.get('harm_a1')]
        safe = [r['p'] for r in rows if r.get('harm_a1') is False]
        cells[key] = {
            'n': len(rows), 'dist_all': dist(ps),
            'dist_harm': dist(harm), 'dist_safe': dist(safe),
            'n_at_or_above_threshold': sum(1 for p in ps if p >= thr),
            'n_within_1e-3': sum(1 for p in ps if abs(p - thr) < 1e-3),
            'n_within_1e-5': sum(1 for p in ps if abs(p - thr) < 1e-5),
            'over_window_nodes': sum(r['over_window'] for r in rows),
            'nodes_total': sum(r['n_nodes'] for r in rows),
        }
    summary['table4_diagnostic_only'] = {
        'note': ('Scores recomputed on stored trajectories. Changing a '
                 'threshold changes the closed loop; these are sensitivity '
                 'diagnostics, NOT new ASR/BU/UA results.'),
        'cells': cells}
    summary['window'] = {
        'v2_dev_over_window': sum(r['over_window'] for r in dev),
        'v2_dev_nodes': sum(r['n_nodes'] for r in dev),
        'v2_test_over_window': sum(r['over_window'] for r in test),
        'v2_test_nodes': sum(r['n_nodes'] for r in test),
        'table4_over_window': sum(r['over_window'] for r in t4),
        'table4_nodes': sum(r['n_nodes'] for r in t4),
    }

    # ---- 3. label direction against source traces (duplicate-aware)
    direction = {'checked': 0, 'mismatch': 0, 'duplicate_task_ids': 0,
                 'examples': []}
    truth_by_task: dict[str, list[int]] = defaultdict(list)
    for name in SOURCE_FILES:
        p = REPO / 'LOGS' / name
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for rec in data.get('runs', []):
            if rec.get('arm') != 'none' or rec.get('error'):
                continue
            env = ENVS[rec['env']]
            trace = rec['trace']
            events = trace.get('events', [])
            idx = None
            for i, ev in enumerate(events):
                if ev.get('tool') == env['action']:
                    idx = i
            if idx is None:
                continue
            fam = env['families'][rec['family']]
            tool, arg = events[idx]['tool'], events[idx]['arg']
            if valid_authorization(trace, env, tool, arg):
                continue
            truth = int(bool(env['harm'](env['canon'](arg), fam)))
            truth_by_task[f"{rec['run_id']}|{idx}"].append(truth)
    seen = set()
    for role in ('train', 'dev', 'test'):
        path = V2 / f'{role}.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row['task_id']
            if tid not in truth_by_task:
                continue
            direction['checked'] += 1
            truths = truth_by_task[tid]
            if len(truths) > 1:
                direction['duplicate_task_ids'] += 1
            if row['label'] not in truths:
                direction['mismatch'] += 1
                if len(direction['examples']) < 5:
                    direction['examples'].append(
                        {'task_id': tid, 'label_stored': row['label'],
                         'label_truth_candidates': truths})
    summary['label_direction'] = direction

    # ---- 4. repeat inference determinism + batch invariance
    sample = json.loads((V2 / 'test.jsonl').read_text().splitlines()[0])
    p1, _ = gate.gate.score_nodes(sample['nodes'])
    p2, _ = gate.gate.score_nodes(sample['nodes'])
    summary['repeat_inference'] = {'p1': p1, 'p2': p2,
                                   'max_abs_delta': abs(p1 - p2)}
    batch_nodes = [json.loads(l)['nodes'] for l in
                   (V2 / 'test.jsonl').read_text().splitlines()[:4]
                   if l.strip()]
    singles = [gate.gate.score_nodes(n)[0] for n in batch_nodes]
    alls = gate.gate.score_nodes(batch_nodes[0])[0]  # placeholder shape note
    summary['batch_invariance_note'] = (
        'GnnGate scores one graph per call; BN eval mode makes the result '
        f'batch independent. Single scores: {[round(x, 6) for x in singles]}; '
        f'first={round(alls, 6)}')

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2,
                                                 ensure_ascii=False))
    print(json.dumps({
        'threshold': summary['threshold'],
        'v2_dev': {k: summary['v2_dev'][k] for k in
                   ('n', 'auroc', 'fp', 'tn', 'fn', 'tp',
                    'threshold_margin_to_max_neg',
                    'threshold_margin_to_min_pos')},
        'v2_test': {k: summary['v2_test'][k] for k in
                    ('n', 'auroc', 'fp', 'tn', 'fn', 'tp')},
        'label_direction': direction,
        'repeat': summary['repeat_inference'],
        'window': summary['window'],
        'n_table4_scored': len(t4),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
