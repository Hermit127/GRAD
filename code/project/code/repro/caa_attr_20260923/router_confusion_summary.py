#!/usr/bin/env python3
"""CAA attribution Phase A3: centroid router stage hit rate + confusion matrix.

Zero-training, deterministic. Rebuilds the frozen centroid router from the
stream train split (all three stages seen, as at the final training stage) and
scores every dev/test sample:

- confusion matrix (rows: true stage, cols: expert routed to);
- per-stage hit rate, macro hit, uniform-chance 1/3 and majority-expert
  baseline; family-cluster bootstrap CI for macro hit;
- nearest/second-nearest centroid distance margin per sample;
- cross-check against the router assignments already recorded in the saved
  fam_diag expert_scores.jsonl (must match exactly; confirms determinism).

Also aggregates the recorded per-run stage hit rates / selection matrices from
`LOGS/real_cl_family_confirm_20260921/fam_diag_analysis.json` if present.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('GRAD_BASE', '.'))
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(BASE / 'code/repro'))

import real_cl_p1 as p1  # noqa: E402

STREAM = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']


def load_stream():
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for ph in ('train', 'dev', 'test'):
        for line in open(STREAM / f'{ph}.jsonl'):
            if line.strip():
                g = json.loads(line)
                out[g['stage']][ph].append(g)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--diag', default='')
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    splits = load_stream()
    router = p1.Router()
    for s in STAGES:
        router.add_task(s, splits[s]['train'])
    report = {'centroids': {s: [round(float(x), 6) for x in c]
                            for s, c in zip(router.tasks, router.centroids)}}
    rng = np.random.RandomState(20260923)
    for ph in ('dev', 'test'):
        conf = np.zeros((3, 3), dtype=int)
        fam_of, stage_of, assign_of, margin_of = [], [], [], []
        s2 = {s: i for i, s in enumerate(STAGES)}
        for s in STAGES:
            for g in splits[s][ph]:
                v = p1.signature(g)
                d = np.array([np.linalg.norm(v - c) for c in router.centroids])
                order = np.argsort(d)
                a = STAGES[int(order[0])]
                conf[s2[s], s2[a]] += 1
                fam_of.append(g['family_key'])
                stage_of.append(s)
                assign_of.append(a)
                margin_of.append(float(d[order[1]] - d[order[0]]))
        hit = float(np.trace(conf) / conf.sum())
        # family cluster bootstrap for macro hit
        fams = sorted(set(fam_of))
        by_fam = collections.defaultdict(list)
        for i, f in enumerate(fam_of):
            by_fam[f].append(i)
        vals = []
        for _ in range(2000):
            picks = rng.choice(len(fams), size=len(fams), replace=True)
            idx = [i for p in picks for i in by_fam[fams[p]]]
            vals.append(np.mean([stage_of[i] == assign_of[i] for i in idx]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        maj = max(conf.sum(1)) / conf.sum()
        report[ph] = {
            'n': int(conf.sum()),
            'confusion': conf.tolist(),
            'row_majority': maj,
            'chance_uniform': 1.0 / 3.0,
            'macro_hit': hit,
            'macro_hit_ci95': [float(lo), float(hi)],
            'per_stage_hit': {s: float(conf[i, i] / conf[i].sum())
                              for i, s in enumerate(STAGES)},
            'margin_median': float(np.median(margin_of)),
            'margin_q10_q90': [float(np.percentile(margin_of, 10)),
                               float(np.percentile(margin_of, 90))],
        }
        with open(Path(args.out) / f'router_assign_{ph}.jsonl', 'w') as f:
            for i in range(len(fam_of)):
                f.write(json.dumps({
                    'phase': ph, 'family': fam_of[i], 'stage': stage_of[i],
                    'assigned': assign_of[i],
                    'margin': margin_of[i]}) + '\n')
    # cross-check against saved assignments
    xc = []
    import glob
    for rd in sorted(glob.glob(str(BASE / 'code/repro/out/real_cl/fam_diag/'
                                      '*'))):
        p = Path(rd) / 'expert_scores.jsonl'
        if not p.exists():
            continue
        seen = set()
        mism = 0
        for line in open(p):
            r = json.loads(line)
            key = (r['stage'], r['task_id'])
            if key in seen:
                continue
            seen.add(key)
            g = None
            for gg in splits[r['stage']]['dev']:
                if gg['task_id'] == r['task_id']:
                    g = gg
                    break
            if g is None:
                continue
            v = p1.signature(g)
            d = [np.linalg.norm(v - c) for c in router.centroids]
            if STAGES[int(np.argmin(d))] != r['centroid_selected']:
                mism += 1
        xc.append({'run': Path(rd).name, 'n': len(seen), 'mismatches': mism})
    report['crosscheck_vs_saved'] = xc
    if args.diag and Path(args.diag).exists():
        d = json.loads(Path(args.diag).read_text())
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for run, arms in d.items():
            for arm, stages in arms.items():
                for s, v in stages.items():
                    agg[arm][s].append({'hit': v['router_hit'],
                                        'sel': v['sel_matrix']})
        report['saved_runs'] = {
            arm: {s: {'hit_mean': float(np.mean([x['hit'] for x in xs])),
                      'hit_std': float(np.std([x['hit'] for x in xs])),
                      'n_runs': len(xs),
                      'sel_matrix_example': xs[0]['sel']}
                  for s, xs in stages.items()}
            for arm, stages in agg.items()}
    outdir = Path(args.out)
    (outdir / 'router_confusion.json').write_text(json.dumps(report, indent=1))
    L = ['# Centroid router: stage hit rate and confusion matrix (zero training)',
         '', 'Router: nearest centroid over the frozen observable signature '
         '(node-type histogram + n_nodes/50 + n_edges/50); centroids from the '
         'stream train split (all three stages seen). Deterministic.', '']
    for ph in ('dev', 'test'):
        v = report[ph]
        L += [f'## {ph} (n={v["n"]})', '',
              '| true \\ routed | ' + ' | '.join(STAGES) + ' | hit |',
              '|---|---|---|---|---|']
        for i, s in enumerate(STAGES):
            row = ' | '.join(str(x) for x in v['confusion'][i])
            L.append(f'| {s} | {row} | {v["per_stage_hit"][s]:.3f} |')
        L += ['', f'macro hit={v["macro_hit"]:.3f} '
              f'[{v["macro_hit_ci95"][0]:.3f},{v["macro_hit_ci95"][1]:.3f}], '
              f'uniform chance=0.333, majority={v["row_majority"]:.3f}; '
              f'margin median={v["margin_median"]:.3f} '
              f'[q10={v["margin_q10_q90"][0]:.3f}, '
              f'q90={v["margin_q10_q90"][1]:.3f}]', '']
    if 'saved_runs' in report:
        L += ['## Recorded per-run hit rates (fam_diag, 6 runs)', '',
              '| arm | stage | hit mean+-sd | runs |', '|---|---|---|---|']
        for arm, stages in sorted(report['saved_runs'].items()):
            for s, v in sorted(stages.items()):
                L.append(f'| {arm} | {s} | {v["hit_mean"]:.3f}'
                         f'+-{v["hit_std"]:.3f} | {v["n_runs"]} |')
    L += ['', 'Cross-check vs saved expert_scores.jsonl: '
          + json.dumps(report['crosscheck_vs_saved'])]
    (outdir / 'router_confusion.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
