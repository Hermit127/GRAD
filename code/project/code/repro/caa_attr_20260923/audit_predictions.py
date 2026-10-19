#!/usr/bin/env python3
"""Audit B: recompute per-stage label direction / AUROC / balanced accuracy /
majority baseline from the saved raw predictions (zero training).

Panels:
- caa_formal (this round, CausalCharGNN, v3 matched stream, 6 orders x 5 seeds)
- g3_formal/fam_* (final G3-512, v2 family stream, 2 orders x 5 seeds)
- g3_formal/dmm_* (final G3-512, decomp/mcp/mrt)

Reports, per arm and task:
- final-stage n / pos / neg / majority baseline / accuracy / balanced accuracy
  (threshold 0.5, as deployed) / AUROC as-is (+ reversed AUROC as a diagnostic
  only; test scores are never flipped for any reported metric);
- "just after learning" vs "end of stream" from the stored accuracy matrix,
  with task forgetting;
- for the expert arms, the full routing decomposition from the saved expert
  matrices (centroid vs task-oracle vs label-best expert, route hit, and how
  much of the routing loss is recoverable by a different expert).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get('GRAD_BASE', '.'))
CAA_DEFAULT = BASE / 'code/repro/out/real_cl/caa_formal'
G3_DEFAULT = BASE / 'code/repro/out/real_cl/g3_formal'


def metrics(ys, ps):
    ys = np.asarray(ys)
    ps = np.asarray(ps)
    pred = (ps >= 0.5).astype(int)
    pos = int((ys == 1).sum())
    neg = int((ys == 0).sum())
    n = len(ys)
    tp = int(((pred == 1) & (ys == 1)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum())
    tn = int(((pred == 0) & (ys == 0)).sum())
    fp = int(((pred == 1) & (ys == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    out = {'n': n, 'pos': pos, 'neg': neg,
           'majority_acc': max(pos, neg) / max(n, 1),
           'acc': (tp + tn) / max(n, 1),
           'bal_acc': 0.5 * (tpr + tnr),
           'tpr': tpr, 'tnr': tnr}
    if len(set(ys.tolist())) > 1:
        a = float(roc_auc_score(ys, ps))
        out['auroc_as_is'] = a
        out['auroc_reversed_diagnostic_only'] = 1.0 - a
    else:
        out['auroc_as_is'] = None
        out['auroc_reversed_diagnostic_only'] = None
    return out


def order_of(run_name):
    if run_name.startswith('fwd') or run_name.startswith('dmm_forward') or \
            run_name.startswith('fam_forward'):
        return 'forward'
    if run_name.startswith('rev') or run_name.startswith('dmm_reversed') or \
            run_name.startswith('fam_reversed'):
        return 'reversed'
    return run_name.split('_')[0]


def audit_scores_runs(root, pattern='*'):
    rep = {}
    for rd in sorted(glob.glob(str(Path(root) / pattern))):
        run = Path(rd).name
        sp = Path(rd) / 'scores.jsonl'
        rp = Path(rd) / 'results.json'
        if not sp.exists() or not rp.exists():
            continue
        rows = collections.defaultdict(lambda: collections.defaultdict(list))
        for line in open(sp):
            r = json.loads(line)
            if r['stage'] != 3:
                continue
            rows[r['arm']][r['domain']].append(r)
        res = json.loads(rp.read_text())
        entry = {'arms': {}}
        for arm, doms in sorted(rows.items()):
            entry['arms'][arm] = {'final': {}, 'acc_matrix': {}}
            for dom, rs in sorted(doms.items()):
                entry['arms'][arm]['final'][dom] = metrics(
                    [r['label'] for r in rs], [float(r['score']) for r in rs])
            if arm in res:
                entry['arms'][arm]['acc_matrix'] = res[arm].get(
                    'acc_matrix', {})
                entry['arms'][arm]['macro_acc'] = res[arm].get('macro_acc')
                entry['arms'][arm]['forgetting'] = res[arm].get('forgetting')
        # just-learned vs end (all tasks in this run's order)
        order = res.get('meta', {}).get('order') or []
        for arm, a in entry['arms'].items():
            jl = {}
            for t, task in enumerate(order):
                key_jl = f'{t}|{task}'
                key_end = f'{len(order) - 1}|{task}'
                vals = [a['acc_matrix'].get(f'{j}|{task}')
                        for j in range(t, len(order))]
                vals = [v for v in vals if v is not None]
                acc_jl = a['acc_matrix'].get(key_jl)
                acc_end = a['acc_matrix'].get(key_end)
                if acc_jl is None or acc_end is None:
                    continue
                jl[task] = {
                    'just_learned_acc': acc_jl, 'end_acc': acc_end,
                    'best_acc': max(vals) if vals else None,
                    'forgetting': (max(vals) - acc_end) if vals else None}
            a['just_learned'] = jl
        rep[run] = entry
    return rep


def audit_expert_runs(root, pattern='*'):
    """snap / stage0 pool decomposition from expert_scores.jsonl."""
    rep = {}
    for rd in sorted(glob.glob(str(Path(root) / pattern))):
        run = Path(rd).name
        fp = Path(rd) / 'expert_scores.jsonl'
        if not fp.exists():
            continue
        recs = [json.loads(l) for l in open(fp)]
        if not recs:
            continue
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in recs:
            per[r['arm']][r['stage']].append(r)
        entry = {}
        for arm, tasks in sorted(per.items()):
            tr = {}
            for task, rs in sorted(tasks.items()):
                experts = sorted(rs[0]['p'].keys())
                ys = np.array([int(r['label']) for r in rs])
                P = np.array([[float(r['p'][e]) for e in experts]
                              for r in rs])
                sel = [r['centroid_selected'] if r['centroid_selected']
                       in experts else experts[0] for r in rs]
                cen = np.array([float(r['p'][e]) for r, e in zip(rs, sel)])
                orc = np.array([float(r['p'][task]) for r in rs])
                allp = P.mean(1)
                best_idx = np.argmax(
                    np.stack([(P[:, j] >= 0.5) == ys for j in range(P.shape[1])],
                             axis=1), axis=1)
                best = P[np.arange(len(rs)), best_idx]
                cm = (cen >= 0.5) == ys
                om = (orc >= 0.5) == ys
                bm = (best >= 0.5) == ys
                wrong = ~cm
                rec_any = float(np.mean(
                    [bool(((P[i] >= 0.5) == ys[i]).any()) for i in
                     range(len(rs))]))
                true_ok = float(np.mean(
                    [bool(((P[i][experts.index(task)] >= 0.5) == ys[i]))
                     for i in range(len(rs))]))
                tr[task] = {
                    'n': len(rs), 'experts': experts,
                    'route_hit_true_stage': float(np.mean(
                        [1.0 if s == task else 0.0 for s in sel])),
                    'centroid': metrics(ys, cen),
                    'oracle_task_id': metrics(ys, orc),
                    'label_best_expert': metrics(ys, best),
                    'all_mean': metrics(ys, allp),
                    'expert_metrics': {
                        e: metrics(ys, P[:, j])
                        for j, e in enumerate(experts)},
                    'wrong_centroid_frac': float(wrong.mean()),
                    'wrong_centroid_any_expert_correct_frac': rec_any,
                    'true_stage_expert_acc': true_ok,
                    'oracle_minus_centroid_acc':
                        float(om.mean() - cm.mean()),
                    'best_minus_centroid_acc':
                        float(bm.mean() - cm.mean()),
                }
            entry[arm] = tr
        rep[run] = entry
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--caa', default=str(CAA_DEFAULT))
    ap.add_argument('--g3', default=str(G3_DEFAULT))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    report = {'caa_final': audit_scores_runs(args.caa),
              'g3_fam_final': audit_scores_runs(args.g3, 'fam_*'),
              'g3_dmm_final': audit_scores_runs(args.g3, 'dmm_*'),
              'caa_expert_pools': audit_expert_runs(args.caa)}
    # aggregates over runs
    def agg(panel):
        out = collections.defaultdict(list)
        for run, entry in report[panel].items():
            for arm, a in entry['arms'].items():
                for task, m in a['final'].items():
                    out[(arm, task)].append(m)
        res = {}
        for k, ms in out.items():
            res['|'.join(k)] = {
                'runs': len(ms),
                'acc_mean': float(np.mean([m['acc'] for m in ms])),
                'bal_acc_mean': float(np.mean([m['bal_acc'] for m in ms])),
                'majority_mean': float(np.mean([m['majority_acc']
                                                for m in ms])),
                'auroc_mean': float(np.mean([m['auroc_as_is'] for m in ms
                                             if m['auroc_as_is'] is not None])),
                'auroc_below_half_runs': int(sum(
                    1 for m in ms if m['auroc_as_is'] is not None
                    and m['auroc_as_is'] < 0.5)),
                'n_mean': float(np.mean([m['n'] for m in ms]))}
        return res
    report['aggregates'] = {p: agg(p) for p in
                            ('caa_final', 'g3_fam_final', 'g3_dmm_final')}
    # just-learned aggregates (accuracy only, from acc_matrix)
    for panel, label in (('caa_final', 'caa'), ('g3_fam_final', 'g3_fam')):
        jl = collections.defaultdict(list)
        for run, entry in report[panel].items():
            for arm, a in entry['arms'].items():
                for task, v in a.get('just_learned', {}).items():
                    jl[(arm, task)].append(v)
        report.setdefault('just_learned', {})[label] = {
            '|'.join(k): {
                'runs': len(vs),
                'just_learned_mean': float(np.mean(
                    [v['just_learned_acc'] for v in vs])),
                'end_mean': float(np.mean([v['end_acc'] for v in vs])),
                'best_mean': float(np.mean([v['best_acc'] for v in vs
                                            if v['best_acc'] is not None])),
                'forgetting_mean': float(np.mean(
                    [v['forgetting'] for v in vs
                     if v['forgetting'] is not None]))}
            for k, vs in jl.items()}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'audit_predictions.json').write_text(json.dumps(report, indent=1))
    L = ['# Raw-prediction audit: direction / AUROC / balanced accuracy / '
         'majority baselines (zero training)', '',
         'All metrics at threshold 0.5, computed from saved scores. AUROC is '
         'reported as-is; the reversed AUROC is stored in the JSON as a '
         'diagnostic only and is never used as the reported metric.', '']
    for panel, title in (('caa_final', 'This round (CAA, v3 matched stream)'),
                         ('g3_fam_final', 'Final G3-512, fam panel (v2 stream)'),
                         ('g3_dmm_final', 'Final G3-512, dmm panel')):
        L += [f'## {title}', '',
              '| arm / task | runs | n | acc | bal-acc | majority | AUROC '
              '(as-is) | runs with AUROC<0.5 |', '|---|---|---|---|---|---|---|---|']
        for k, v in sorted(report['aggregates'][panel].items()):
            L.append(f"| {k} | {v['runs']} | {v['n_mean']:.0f} | "
                     f"{v['acc_mean']:.3f} | {v['bal_acc_mean']:.3f} | "
                     f"{v['majority_mean']:.3f} | {v['auroc_mean']:.3f} | "
                     f"{v['auroc_below_half_runs']} |")
        L.append('')
    L += ['## Just-learned vs end of stream (accuracy; from acc_matrix)', '',
          '| panel | arm / task | just learned | end | best | forgetting |',
          '|---|---|---|---|---|---|']
    for label, vals in report.get('just_learned', {}).items():
        for k, v in sorted(vals.items()):
            L.append(f"| {label} | {k} | {v['just_learned_mean']:.3f} | "
                     f"{v['end_mean']:.3f} | {v['best_mean']:.3f} | "
                     f"{v['forgetting_mean']:+.3f} |")
    L += ['', '## Expert-pool routing decomposition (this round, snap/stage0)',
          '', 'per run/task details in the JSON; aggregate below', '']
    pool = collections.defaultdict(list)
    for run, arms in report['caa_expert_pools'].items():
        for arm, tasks in arms.items():
            for task, v in tasks.items():
                pool[(arm, task)].append(v)
    L += ['| arm / task | route hit | centroid acc | oracle acc | best-expert '
          'acc | all acc | wrong & any-expert-right | oracle-centroid |',
          '|---|---|---|---|---|---|---|---|']
    for k, vs in sorted(pool.items()):
        L.append(f"| {k} | "
                 f"{np.mean([v['route_hit_true_stage'] for v in vs]):.3f} | "
                 f"{np.mean([v['centroid']['acc'] for v in vs]):.3f} | "
                 f"{np.mean([v['oracle_task_id']['acc'] for v in vs]):.3f} | "
                 f"{np.mean([v['label_best_expert']['acc'] for v in vs]):.3f} | "
                 f"{np.mean([v['all_mean']['acc'] for v in vs]):.3f} | "
                 f"{np.mean([v['wrong_centroid_any_expert_correct_frac'] for v in vs]):.3f} | "
                 f"{np.mean([v['oracle_minus_centroid_acc'] for v in vs]):+.3f} |")
    (outdir / 'audit_predictions.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
