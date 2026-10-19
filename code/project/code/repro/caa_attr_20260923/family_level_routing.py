#!/usr/bin/env python3
"""Family-level recheck of the routing-selection effect (zero training).

Same saved expert pools as Phase D (`caa_formal/*/expert_scores.jsonl`), but
the unit is the label-pure family: per-family mean correctness for centroid,
random-expectation, all-expert and task-ID oracle routing, then family
accuracy at 0.5, macro over tasks; paired centroid-random / centroid-all
deltas bootstrapped over runs. Never flips any score.
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

BASE = Path(os.environ.get('GRAD_BASE', '.'))
V3 = BASE / 'LOGS/real_cl_caa_attribution_20260923/stream_matched'
STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']


def fam_map():
    out = {}
    for line in open(V3 / 'test.jsonl'):
        if line.strip():
            g = json.loads(line)
            out[g['task_id']] = g['family_key']
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',
                    default=str(BASE / 'code/repro/out/real_cl/caa_formal'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fmap = fam_map()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    per_run = {}
    for rd in sorted(glob.glob(str(Path(args.root) / '*'))):
        fp = Path(rd) / 'expert_scores.jsonl'
        if not fp.exists():
            continue
        run = Path(rd).name
        recs = collections.defaultdict(lambda: collections.defaultdict(list))
        for line in open(fp):
            r = json.loads(line)
            recs[r['arm']][r['stage']].append(r)
        entry = {}
        for arm, tasks in recs.items():
            task_rep = {}
            fam_vals = collections.defaultdict(list)
            for task, rs in tasks.items():
                for r in rs:
                    experts = sorted(r['p'])
                    sel = r['centroid_selected'] if r[
                        'centroid_selected'] in experts else experts[0]
                    y = int(r['label'])
                    cen = float((r['p'][sel] >= 0.5) == y)
                    allp = float(np.mean([r['p'][e] for e in experts]))
                    rnd = float(np.mean([
                        (r['p'][e] >= 0.5) == y for e in experts]))
                    orc = float((r['p'][task] >= 0.5) == y)
                    best = float(any((r['p'][e] >= 0.5) == y
                                     for e in experts))
                    f = fmap[r['task_id']]
                    fam_vals[(task, f)].append((y, cen, rnd, allp >= 0.5,
                                                orc, best))
                task_rep[task] = {}
            # family-level accuracy per task (pure families only)
            fams_by_task = collections.defaultdict(
                lambda: collections.defaultdict(list))
            for (task, f), vs in fam_vals.items():
                fams_by_task[task][f] = vs
            for task, fams in fams_by_task.items():
                vals = {'centroid': [], 'random': [], 'all': [], 'oracle': [],
                        'best_label': []}
                for f, vs in fams.items():
                    ys = [v[0] for v in vs]
                    if len(set(ys)) > 1:
                        continue
                    for i, name in enumerate(
                            ('centroid', 'random', 'all', 'oracle',
                             'best_label'), start=1):
                        vals[name].append(float(np.mean([v[i] for v in vs])))
                task_rep[task] = {k: (float(np.mean(v)) if v else None)
                                  for k, v in vals.items()}
            entry[arm] = task_rep
        per_run[run] = entry
    # aggregate: macro over tasks, paired deltas over runs
    agg = {}
    for arm in ('snap', 'stage0'):
        variants = collections.defaultdict(list)
        pairs = {'centroid-random': [], 'centroid-all': [],
                 'centroid-oracle': []}
        for run, e in sorted(per_run.items()):
            if arm not in e:
                continue
            for v in ('centroid', 'random', 'all', 'oracle'):
                vals = [e[arm][t][v] for t in STAGES
                        if e[arm].get(t, {}).get(v) is not None]
                if vals:
                    variants[v].append(float(np.mean(vals)))
            if all(e[arm].get(t, {}).get('centroid') is not None and
                   e[arm].get(t, {}).get('random') is not None
                   for t in STAGES):
                cm = np.mean([e[arm][t]['centroid'] for t in STAGES])
                rm = np.mean([e[arm][t]['random'] for t in STAGES])
                am = np.mean([e[arm][t]['all'] for t in STAGES])
                om = np.mean([e[arm][t]['oracle'] for t in STAGES])
                pairs['centroid-random'].append(cm - rm)
                pairs['centroid-all'].append(cm - am)
                pairs['centroid-oracle'].append(cm - om)
        rep = {'macro_family_acc': {k: float(np.mean(v)) if v else None
                                    for k, v in variants.items()}}
        for name, ds in pairs.items():
            ds = np.array(ds)
            if len(ds) < 2:
                rep[name] = None
                continue
            rng = np.random.RandomState(20260923)
            bs = [rng.choice(ds, len(ds), replace=True).mean()
                  for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rep[name] = {'mean': float(ds.mean()),
                         'ci95': [float(lo), float(hi)], 'n_runs': len(ds)}
        agg[arm] = rep
    report = {'per_run': per_run, 'aggregate': agg, 'unit': 'label-pure family'}
    (outdir / 'family_level_routing.json').write_text(
        json.dumps(report, indent=1))
    L = ['# Family-level routing recheck (zero training, v3 pools)', '',
         'Per-family mean correctness, then family accuracy; macro over tasks;'
         ' paired deltas with run bootstrap 95% CI.', '']
    for arm, rep in agg.items():
        L += [f'## {arm}', '',
              '| variant | macro family ACC |', '|---|---|']
        for k, v in rep['macro_family_acc'].items():
            L.append(f"| {k} | {v:.4f} |" if v is not None else f'| {k} | - |')
        L += ['', '| delta | mean | 95% CI |', '|---|---|---|']
        for k in ('centroid-random', 'centroid-all', 'centroid-oracle'):
            v = rep.get(k)
            if v:
                L.append(f"| {k} | {v['mean']:+.4f} | "
                         f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] |")
        L.append('')
    (outdir / 'family_level_routing.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
