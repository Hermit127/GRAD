#!/usr/bin/env python3
"""Zero-training family-level recheck of the existing family-stream panels.

Motivation (see diag_direction_20260923.md): on the family-isolated streams
every test sample belongs to a label-pure family, so the sample-level metrics
weight families by size. This script recomputes the panel endpoints with the
family as the unit (unweighted), for:

- CAA / CausalCharGNN-512 on the v2 family stream (`fam_formal`, 10 runs)
- final G3-512 on the v2 family stream (`g3_formal/fam_*`, 10 runs)
- this round on the v3 matched stream (`caa_formal`, 30 runs) for reference

Family-level metrics per run/arm/task: accuracy at threshold 0.5 over
pure-label families, balanced accuracy, AUROC as-is; macro = equal task
weights. Paired deltas are bootstrapped over seeds x orders. No scores are
flipped anywhere.
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
V2 = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
V3 = BASE / 'LOGS/real_cl_caa_attribution_20260923/stream_matched'
STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']
PANELS = {
    'caa_v2_fam': {'root': BASE / 'code/repro/out/real_cl/fam_formal',
                   'pattern': '*', 'stream': V2, 'orders': 2},
    'g3_v2_fam': {'root': BASE / 'code/repro/out/real_cl/g3_formal',
                  'pattern': 'fam_*', 'stream': V2, 'orders': 2},
    'caa_v3_fam': {'root': BASE / 'code/repro/out/real_cl/caa_formal',
                   'pattern': '*', 'stream': V3, 'orders': 6},
}
PAIRS = [('snap|centroid', 'replay|-'),
         ('stage0|centroid', 'replay|-'),
         ('snap|centroid', 'stage0|centroid'),
         ('snap|oracle', 'snap|centroid')]


def load_family_map(stream):
    out = {}
    for line in open(Path(stream) / 'test.jsonl'):
        if line.strip():
            g = json.loads(line)
            out[g['task_id']] = g['family_key']
    return out


def fam_metrics(rows, fam_map):
    """rows: list of {task_id, label, score}; returns family-level metrics."""
    by = collections.defaultdict(list)
    for r in rows:
        by[fam_map[r['task_id']]].append((int(r['label']), float(r['score'])))
    labels, scores, mixed = [], [], 0
    for f, vs in by.items():
        ys = [v[0] for v in vs]
        if len(set(ys)) > 1:
            mixed += 1
            continue
        labels.append(ys[0])
        scores.append(float(np.mean([v[1] for v in vs])))
    labels = np.array(labels)
    scores = np.array(scores)
    out = {'n_families_pure': len(labels), 'n_families_mixed': mixed}
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        out.update({'family_acc': None, 'family_bal_acc': None,
                    'family_auroc': None})
        return out
    pred = (scores >= 0.5).astype(int)
    pos = labels == 1
    out['family_acc'] = float((pred == labels).mean())
    tpr = float((pred[pos] == 1).mean())
    tnr = float((pred[~pos] == 0).mean())
    out['family_bal_acc'] = 0.5 * (tpr + tnr)
    out['family_auroc'] = float(roc_auc_score(labels, scores))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {'panels': {}}
    for pname, cfg in PANELS.items():
        fam_map = load_family_map(cfg['stream'])
        per_run = {}
        for rd in sorted(glob.glob(str(cfg['root'] / cfg['pattern']))):
            run = Path(rd).name
            if pname == 'g3_v2_fam':
                run = run.replace('fam_', '')
            sp = Path(rd) / 'scores.jsonl'
            if not sp.exists():
                continue
            rows = collections.defaultdict(lambda: collections.defaultdict(list))
            for line in open(sp):
                r = json.loads(line)
                if r['stage'] != 3 or r['domain'] not in STAGES:
                    continue
                rows[r['arm']][r['domain']].append(r)
            entry = {}
            for arm, doms in rows.items():
                task_rep = {}
                fam_accs = []
                for dom, rs in doms.items():
                    m = fam_metrics(rs, fam_map)
                    task_rep[dom] = m
                    if m['family_acc'] is not None:
                        fam_accs.append(m['family_acc'])
                task_rep['macro_family_acc'] = float(np.mean(fam_accs)) \
                    if fam_accs else None
                entry[arm] = task_rep
            per_run[run] = entry
        report['panels'][pname] = {'runs': per_run,
                                   'orders': cfg['orders']}
    # aggregate + paired deltas
    agg = {}
    for pname, pr in report['panels'].items():
        arms = sorted({a for run in pr['runs'].values() for a in run})
        a_rep = {}
        for arm in arms:
            vals = [run[arm]['macro_family_acc']
                    for run in pr['runs'].values()
                    if arm in run and run[arm]['macro_family_acc'] is not None]
            per_task = {}
            for task in STAGES:
                tv = [run[arm][task]['family_acc'] for run in pr['runs'].values()
                      if arm in run and run[arm].get(task, {}).get(
                          'family_acc') is not None]
                av = [run[arm][task]['family_auroc'] for run in pr['runs'].values()
                      if arm in run and run[arm].get(task, {}).get(
                          'family_auroc') is not None]
                per_task[task] = {
                    'family_acc_mean': float(np.mean(tv)) if tv else None,
                    'family_auroc_mean': float(np.mean(av)) if av else None,
                    'n': len(tv)}
            a_rep[arm] = {
                'macro_family_acc_mean': float(np.mean(vals)) if vals else None,
                'macro_family_acc_sd': float(np.std(vals, ddof=1))
                if len(vals) > 1 else None,
                'per_task': per_task, 'n_runs': len(vals)}
        # paired deltas with run-level bootstrap (and seed/order split)
        deltas = {}
        for a, b in PAIRS:
            by = {}
            for run, entry in pr['runs'].items():
                if a in entry and b in entry and \
                        entry[a]['macro_family_acc'] is not None and \
                        entry[b]['macro_family_acc'] is not None:
                    by[run] = entry[a]['macro_family_acc'] - \
                        entry[b]['macro_family_acc']
            if not by:
                continue
            keys = sorted(by)
            rng = np.random.RandomState(20260923)
            bs = [float(np.mean([by[keys[i]] for i in
                                 rng.randint(0, len(keys), len(keys))]))
                  for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            deltas[f'{a} - {b}'] = {
                'mean': float(np.mean(list(by.values()))),
                'ci95_run_bootstrap': [float(lo), float(hi)],
                'per_run': by, 'n_runs': len(by)}
        agg[pname] = {'arms': a_rep, 'deltas': deltas}
    report['aggregates'] = agg
    (outdir / 'family_level_recheck.json').write_text(
        json.dumps(report, indent=1))
    L = ['# Family-level recheck of existing family-stream panels '
         '(zero training)', '',
         'Unit = label-pure family (mixed families excluded; count in JSON). '
         'Macro = equal task weights. AUROC as-is, never flipped. Same data '
         'and runs as the original panels; only the aggregation unit and the '
         'task weighting change.', '']
    for pname, a in agg.items():
        L += [f'## {pname}', '',
              '| arm | macro family ACC | per task (family ACC / family AUC) |',
              '|---|---|---|']
        for arm, v in sorted(a['arms'].items()):
            cells = '; '.join(
                f"{t[:6]} {v['per_task'][t]['family_acc_mean']:.3f}/"
                f"{v['per_task'][t]['family_auroc_mean']:.3f}"
                if v['per_task'][t]['family_acc_mean'] is not None else
                f"{t[:6]} n/a" for t in STAGES)
            L.append(f"| {arm} | {v['macro_family_acc_mean']:.4f} "
                     f"+- {v['macro_family_acc_sd']:.4f} | {cells} |")
        L += ['', '| comparison | mean | 95% run-bootstrap CI |',
              '|---|---|---|']
        for k, v in sorted(a['deltas'].items()):
            L.append(f"| {k} | {v['mean']:+.4f} | "
                     f"[{v['ci95_run_bootstrap'][0]:+.4f},"
                     f"{v['ci95_run_bootstrap'][1]:+.4f}] |")
        L.append('')
    (outdir / 'family_level_recheck.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
