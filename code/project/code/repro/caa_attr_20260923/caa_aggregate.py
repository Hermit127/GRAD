#!/usr/bin/env python3
"""CAA attribution Phase E: aggregate the formal matched-stream runs.

Reads `code/repro/out/real_cl/caa_formal/*/results.json` (+ `scores.jsonl`
for per-task AUROC) and reports, per arm/router:

- stage-by-task accuracy matrix, final macro ACC, F_T (mean across runs);
- primary paired deltas (CAA - Replay / - Replay-matched / - Stage-0,
  centroid - random/all pool contrast, Replay-matched - Replay) with two-way
  cluster bootstrap (resampling seeds and task orders);
- per-task AUROC at the final stage from the saved per-sample test scores;
- resource accounting (params, update budget, train seconds).

No training, no new data.
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
STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']
ORDERS = ['fwd', 'rev', 'trs', 'rts', 'str', 'srt']
SEEDS = [211, 212, 213, 214, 215]


def twoway_boot(by, seeds, orders, n=2000, seed=20260923):
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n):
        ss = rng.choice(seeds, size=len(seeds), replace=True)
        oo = rng.choice(orders, size=len(orders), replace=True)
        vals = [by[(o, s)] for o in oo for s in ss if (o, s) in by]
        if vals:
            deltas.append(float(np.mean(vals)))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def twoway_var(by, seeds, orders):
    arr = np.array([[by.get((o, s), np.nan) for s in seeds]
                    for o in orders])
    ok = ~np.isnan(arr)
    if ok.sum() < 4:
        return None
    grand = arr[ok].mean()
    seed_means = np.array([np.nanmean(arr[:, i]) for i in range(len(seeds))])
    order_means = np.array([np.nanmean(arr[i, :]) for i in range(len(orders))])
    seed_var = float(np.nanvar(seed_means))
    order_var = float(np.nanvar(order_means))
    resid = arr - seed_means[None, :] - order_means[:, None] + grand
    return {'grand': float(grand), 'seed_mean_std': float(np.sqrt(seed_var)),
            'order_mean_std': float(np.sqrt(order_var)),
            'resid_std': float(np.nanstd(resid)),
            'n': int(ok.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.runs) if args.runs else \
        BASE / 'code/repro/out/real_cl/caa_formal'
    report = {'runs': {}, 'arms': {}}
    per_run = {}   # (order, seed) -> {armkey: {macro, ft, per_stage}}
    auroc_run = collections.defaultdict(dict)
    for rd in sorted(glob.glob(str(root / '*'))):
        name = Path(rd).name
        if '_' not in name:
            continue
        order, seed = name.rsplit('_', 1)
        seed = int(seed)
        fp = Path(rd) / 'results.json'
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        meta = d.pop('meta')
        run = {}
        for key, v in d.items():
            run[key] = {'macro_acc': v['macro_acc'],
                        'ft': v['forgetting'],
                        'acc_matrix': v['acc_matrix'],
                        'seconds': v.get('seconds'),
                        'backbone_params': v.get('backbone_params'),
                        'expert_params': v.get('expert_params'),
                        'updates_total': v.get('updates_total'),
                        'snapshot_bytes': v.get('snapshot_bytes')}
        per_run[(order, seed)] = run
        # AUROC per arm/router at the final stage
        sp = Path(rd) / 'scores.jsonl'
        if sp.exists():
            rows = collections.defaultdict(list)
            for line in open(sp):
                r = json.loads(line)
                if r['stage'] == len(STAGES):
                    rows[(r['arm'], r['domain'])].append(r)
            for (arm, dom), rs in rows.items():
                ys = [r['label'] for r in rs]
                if len(set(ys)) < 2:
                    continue
                auroc_run[(order, seed)][(arm, dom)] = float(
                    roc_auc_score(ys, [float(r['score']) for r in rs]))
    report['runs'] = {f'{o}_{s}': v for (o, s), v in
                      sorted(per_run.items())}
    # per-arm summaries
    arms = sorted({a for run in per_run.values() for a in run})
    for arm in arms:
        macros, fts, secs, bps, eps, upts = [], [], [], [], [], []
        matrices = collections.defaultdict(list)
        for run in per_run.values():
            if arm not in run:
                continue
            v = run[arm]
            macros.append(v['macro_acc'])
            if v['ft'] is not None:
                fts.append(v['ft'])
            if v['seconds'] is not None:
                secs.append(v['seconds'])
            if v['backbone_params'] is not None:
                bps.append(v['backbone_params'])
            if v['expert_params'] is not None:
                eps.append(v['expert_params'])
            if v['updates_total'] is not None:
                upts.append(v['updates_total'])
            for k, acc in v['acc_matrix'].items():
                matrices[k].append(acc)
        report['arms'][arm] = {
            'n_runs': len(macros),
            'macro_acc_mean': float(np.mean(macros)),
            'macro_acc_std': float(np.std(macros, ddof=1)),
            'ft_mean': float(np.mean(fts)) if fts else None,
            'ft_std': float(np.std(fts, ddof=1)) if len(fts) > 1 else None,
            'stage_by_task': {k: {'mean': float(np.mean(v)),
                                  'std': float(np.std(v, ddof=1)),
                                  'n': len(v)}
                              for k, v in sorted(matrices.items())},
            'seconds_mean': float(np.mean(secs)) if secs else None,
            'backbone_params': bps[0] if bps else None,
            'expert_params': eps[0] if eps else None,
            'updates_total': upts[0] if upts else None,
        }
    # AUROC summary
    auroc_sum = {}
    keys = sorted({k for run in auroc_run.values() for k in run})
    for arm, dom in keys:
        vals = [run[(arm, dom)] for run in auroc_run.values()
                if (arm, dom) in run]
        auroc_sum[f'{arm}|{dom}'] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            'n': len(vals)}
    report['auroc_final_stage'] = auroc_sum
    # primary paired deltas
    pairs = [('snap|centroid', 'replay|-'),
             ('snap|centroid', 'replay_matched|-'),
             ('snap|centroid', 'stage0|centroid'),
             ('replay_matched|-', 'replay|-'),
             ('stage0|centroid', 'replay|-'),
             ('snap|all', 'snap|centroid'),
             ('snap|oracle', 'snap|centroid'),
             ('stage0|centroid', 'stage0|all')]
    deltas = {}
    for a, b in pairs:
        for metric in ('macro_acc', 'ft'):
            by = {}
            for (o, s), run in per_run.items():
                if a in run and b in run and run[a][metric] is not None \
                        and run[b][metric] is not None:
                    by[(o, s)] = run[a][metric] - run[b][metric]
            if not by:
                continue
            lo, hi = twoway_boot(by, SEEDS, ORDERS)
            key = f'{a} - {b}'
            deltas.setdefault(key, {})[metric] = {
                'mean': float(np.mean(list(by.values()))),
                'ci95_twoway': [lo, hi],
                'n_runs': len(by),
                'var': twoway_var(by, SEEDS, ORDERS)}
    report['primary_deltas'] = deltas
    # final-stage per-task deltas for the primary contrasts (all runs)
    def final_acc(run, arm, stage):
        if arm not in run:
            return None
        return run[arm]['acc_matrix'].get(f'{len(STAGES) - 1}|{stage}')

    def ft_task(run, arm, stage):
        if arm not in run:
            return None
        vals = []
        for j in range(0, len(STAGES)):
            v = run[arm]['acc_matrix'].get(f'{j}|{stage}')
            if v is not None:
                vals.append(v)
        f = run[arm]['acc_matrix'].get(f'{len(STAGES) - 1}|{stage}')
        if not vals or f is None:
            return None
        return max(vals) - f

    stage_deltas = {}
    for a, b in [('snap|centroid', 'replay|-'),
                 ('snap|centroid', 'replay_matched|-'),
                 ('snap|centroid', 'stage0|centroid'),
                 ('stage0|centroid', 'replay|-')]:
        for stage in STAGES:
            for label, fn in (('final_acc', final_acc),
                              ('ft_task', ft_task)):
                by = {}
                for (o, s), run in per_run.items():
                    ca, cb = fn(run, a, stage), fn(run, b, stage)
                    if ca is not None and cb is not None:
                        by[(o, s)] = ca - cb
                if not by:
                    continue
                lo, hi = twoway_boot(by, SEEDS, ORDERS)
                stage_deltas[f'{a} - {b} :: {stage} :: {label}'] = {
                    'mean': float(np.mean(list(by.values()))),
                    'ci95_twoway': [lo, hi], 'n_runs': len(by)}
    report['stage_deltas'] = stage_deltas
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'caa_formal_aggregate.json').write_text(
        json.dumps(report, indent=1))
    # markdown
    L = ['# CAA attribution formal results (matched stream v3)', '',
         '6 orders x 5 seeds = 30 runs; arms naive/replay/replay_matched/'
         'stage0/snap; routers centroid/oracle/all; test split.', '',
         '## Arm summary', '',
         '| arm | macro ACC | F_T | seconds | updates | params(bb+exp) |',
         '|---|---|---|---|---|---|']
    for arm, v in sorted(report['arms'].items()):
        L.append(f"| {arm} | {v['macro_acc_mean']:.4f}"
                 f"+-{v['macro_acc_std']:.4f} | "
                 f"{v['ft_mean']:.4f} | "
                 f"{v['seconds_mean']:.0f} | {v['updates_total']} | "
                 f"{v['backbone_params']}+{v['expert_params'] or 0} |")
    L += ['', '## Stage-by-task accuracy matrix (mean over 30 runs; '
          'columns = training stage index)', '',
          '| arm | ' + ' | '.join(
              [f's{i}-{s}' for i, s in enumerate(STAGES)]) + ' |',
          '|---|---|---|---|']
    for arm, v in sorted(report['arms'].items()):
        cells = []
        for j in range(len(STAGES)):
            row = [v['stage_by_task'].get(f'{j}|{s}', {}).get('mean')
                   for s in STAGES]
            cells.append(' / '.join(
                f'{x:.3f}' if x is not None else '-' for x in row))
        L.append(f"| {arm} | " + ' | '.join(cells) + ' |')
    L += ['', '## Primary paired deltas (two-way cluster bootstrap 95% CI)', '',
          '| comparison | metric | mean | 95% CI | n runs |', '|---|---|---|---|---|']
    for k, v in sorted(report['primary_deltas'].items()):
        for metric, m in v.items():
            L.append(f"| {k} | {metric} | {m['mean']:+.4f} | "
                     f"[{m['ci95_twoway'][0]:+.4f},{m['ci95_twoway'][1]:+.4f}]"
                     f" | {m['n_runs']} |")
    L += ['', '## Per-task deltas (final accuracy and task forgetting)', '',
          '| comparison | mean | 95% CI |', '|---|---|---|']
    for k, v in sorted(report['stage_deltas'].items()):
        L.append(f"| {k} | {v['mean']:+.4f} | "
                 f"[{v['ci95_twoway'][0]:+.4f},{v['ci95_twoway'][1]:+.4f}] |")
    L += ['', '## Per-task AUROC at final stage', '',
          '| arm (router) x task | AUROC mean+-sd | n |', '|---|---|---|']
    for k, v in sorted(report['auroc_final_stage'].items()):
        sd = f"+-{v['std']:.3f}" if v['std'] is not None else ''
        L.append(f"| {k} | {v['mean']:.3f}{sd} | {v['n']} |")
    (outdir / 'caa_formal_aggregate.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:60]))
    print('saved', outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
