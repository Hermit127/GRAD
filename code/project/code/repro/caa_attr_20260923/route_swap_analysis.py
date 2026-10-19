#!/usr/bin/env python3
"""CAA attribution Phase D: routing swaps on the frozen expert pools.

Zero training. For every formal run and both expert arms (`snap`, `stage0`),
reads the saved per-sample expert matrix (`expert_scores.jsonl`, test split)
and evaluates, on the *same* pool:

- centroid (deployed), random (analytic + 1000 seeded draws), all-expert
  averaging, task-ID / label-oracle (diagnostic only);
- expert-by-stage accuracy matrix (does the newest expert win its own task?);
- selected-expert loss rank / regret (does centroid pick lower-loss experts?);
- covariate controls: per-sample centroid-minus-random-expectation deltas
  regressed on source, graph size, edge count, text length and stage with
  family-clustered SE, plus per-stratum bootstrap deltas.

Outputs a combined JSON + Markdown under LOGS/.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('GRAD_BASE', '.'))
STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']


def bce(p, y):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def acc1(p, y):
    return float(((p >= 0.5) == y))


def source_group(g):
    md = g.get('metadata') or {}
    if str(md.get('source', '')) == 'mcp_safetybench' and \
       str(md.get('graph_origin', '')) == 'runtime_trace':
        return 'runtime'
    return 'generated'


def node_text(n):
    return str((n.get('features') or {}).get('subtask', '')) or \
        str(n.get('type', ''))


def load_stream(stream_dir):
    meta = {}
    for line in open(Path(stream_dir) / 'test.jsonl'):
        if line.strip():
            g = json.loads(line)
            types = collections.Counter(str(n.get('type', 'other'))
                                        for n in g['nodes'])
            meta[g['task_id']] = {
                'stage': g['stage'], 'family': g['family_key'],
                'label': int(g['label']), 'source': source_group(g),
                'n_nodes': len(g['nodes']), 'n_edges': len(g.get('edges', [])),
                'text_len': sum(len(node_text(n)) for n in g['nodes']),
                'dom_type': types.most_common(1)[0][0] if types else 'other'}
    return meta


def ols_cluster(X, y, clusters):
    """OLS with one-way cluster-robust (HC0) covariance."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in sorted(set(clusters)):
        idx = [i for i, cc in enumerate(clusters) if cc == c]
        Xg, eg = X[idx], resid[idx]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='')
    ap.add_argument('--stream', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=1000)
    args = ap.parse_args()
    runs_root = Path(args.runs) if args.runs else \
        BASE / 'code/repro/out/real_cl/caa_formal'
    stream_dir = Path(args.stream) if args.stream else \
        BASE / 'LOGS/real_cl_caa_attribution_20260923/stream_matched'
    meta = load_stream(stream_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {'runs': {}, 'pooled': {}}
    records = []
    for rd in sorted(glob.glob(str(runs_root / '*'))):
        run = Path(rd).name
        fp = Path(rd) / 'expert_scores.jsonl'
        if not fp.exists():
            continue
        recs = [json.loads(l) for l in open(fp)]
        if not recs:
            continue
        run_rep = {}
        for arm in sorted({r['arm'] for r in recs}):
            rows = [r for r in recs if r['arm'] == arm]
            experts = sorted(rows[0]['p'].keys())
            rng = np.random.RandomState(20260000 + int(hashlib.sha1(
                run.encode()).hexdigest()[:6], 16) % 10000)
            per_stage = collections.defaultdict(list)
            for r in rows:
                g = meta[r['task_id']]
                per_stage[g['stage']].append(r)
            stage_rep = {}
            for s, rs in sorted(per_stage.items()):
                ys = np.array([meta[r['task_id']]['label'] for r in rs])
                P = np.array([[r['p'][e] for e in experts] for r in rs])
                sel = [r['centroid_selected'] for r in rs]
                sel = [e if e in experts else experts[0] for e in sel]
                cen = np.array([r['p'][e] for r, e in zip(rs, sel)])
                allp = P.mean(1)
                orc = np.array([r['p'][meta[r['task_id']]['stage']]
                                for r in rs])
                correct = (P >= 0.5) == ys[:, None]
                cen_c = (cen >= 0.5) == ys
                all_c = (allp >= 0.5) == ys
                orc_c = (orc >= 0.5) == ys
                # random: analytic expectation + seeded draws
                exp_rand = correct.mean(1).mean()
                draws = []
                for _ in range(args.draws):
                    j = rng.randint(0, len(experts), size=len(ys))
                    draws.append(float(correct[np.arange(len(ys)), j].mean()))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                # losses / ranks
                ranks, regrets, sel_best = [], [], []
                for i, r in enumerate(rs):
                    losses = np.array([bce(r['p'][e], int(r['label']))
                                       for e in experts])
                    ranks.append(1 + int((np.argsort(np.argsort(
                        losses))[experts.index(sel[i])])))
                    regrets.append(float(losses[experts.index(sel[i])]
                                         - losses.min()))
                    sel_best.append(int(losses[experts.index(sel[i])]
                                        == losses.min()))
                stage_rep[s] = {
                    'n': len(rs), 'experts': experts,
                    'centroid_acc': float(cen_c.mean()),
                    'random_acc_exp': float(exp_rand),
                    'random_acc_draw_mean': float(np.mean(draws)),
                    'random_acc_ci95': [float(lo), float(hi)],
                    'all_acc': float(all_c.mean()),
                    'oracle_acc': float(orc_c.mean()),
                    'sel_expert_counts': dict(collections.Counter(sel)),
                    'mean_rank_of_selected': float(np.mean(ranks)),
                    'selected_best_frac': float(np.mean(sel_best)),
                    'mean_regret': float(np.mean(regrets)),
                    'expert_stage_acc': {
                        e: float(correct[:, j].mean())
                        for j, e in enumerate(experts)},
                }
                for i, r in enumerate(rs):
                    records.append({
                        'run': run, 'arm': arm, 'task_id': r['task_id'],
                        'stage': s, 'family': meta[r['task_id']]['family'],
                        'label': int(r['label']),
                        'source': meta[r['task_id']]['source'],
                        'n_nodes': meta[r['task_id']]['n_nodes'],
                        'n_edges': meta[r['task_id']]['n_edges'],
                        'text_len': meta[r['task_id']]['text_len'],
                        'dom_type': meta[r['task_id']]['dom_type'],
                        'centroid_selected': sel[i],
                        'correct_centroid': float(cen_c[i]),
                        'correct_random_exp': float(correct[i].mean()),
                        'correct_all': float(all_c[i]),
                        'correct_oracle': float(orc_c[i]),
                        'expert_scores': {e: float(P[i, j])
                                          for j, e in enumerate(experts)},
                    })
            # AUCs via sklearn on pooled stage scores
            from sklearn.metrics import roc_auc_score
            for s, rs in sorted(per_stage.items()):
                ys = [meta[r['task_id']]['label'] for r in rs]
                if len(set(ys)) < 2:
                    continue
                for name, key, fn in (
                        ('centroid', 'centroid_auc',
                         lambda r: r['p'][r['centroid_selected']]),
                        ('all', 'all_auc', lambda r: np.mean(
                            list(r['p'].values()))),
                        ('oracle', 'oracle_auc',
                         lambda r: r['p'][meta[r['task_id']]['stage']])):
                    sc = [fn(r) for r in rs]
                    stage_rep[s][key] = float(roc_auc_score(ys, sc))
            # expert x stage acc matrix
            mat = {e: {s: stage_rep[s]['expert_stage_acc'][e]
                       for s in stage_rep} for e in experts}
            run_rep[arm] = {'stages': stage_rep, 'expert_stage_matrix': mat}
        report['runs'][run] = run_rep
    # ---- pooled controls (family-clustered difference regression) ----
    # d_i = centroid - random-expectation correctness; covariates centred so
    # the intercept is the adjusted routing gain at mean covariates.
    boot = np.random.RandomState(7)

    def strat_delta(rows):
        d = np.array([r['correct_centroid'] - r['correct_random_exp']
                      for r in rows])
        fams = sorted(set(r['family'] for r in rows))
        by = collections.defaultdict(list)
        for i, r in enumerate(rows):
            by[r['family']].append(i)
        vals = []
        for _ in range(2000):
            picks = boot.choice(len(fams), len(fams), replace=True)
            idx = [i for p in picks for i in by[fams[p]]]
            vals.append(d[idx].mean())
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {'n': len(rows), 'delta': float(d.mean()),
                'ci95': [float(lo), float(hi)]}

    controls = {}
    strata_all = {}
    for c_arm in ('snap', 'stage0'):
        recs = [r for r in records if r['arm'] == c_arm]
        if not recs:
            continue
        d = np.array([r['correct_centroid'] - r['correct_random_exp']
                      for r in recs])
        nn = np.array([math.log1p(r['n_nodes']) for r in recs])
        ne = np.array([math.log1p(r['n_edges']) for r in recs])
        tx = np.array([math.log1p(r['text_len']) for r in recs])
        zn = (nn - nn.mean()) / max(nn.std(), 1e-9)
        ze = (ne - ne.mean()) / max(ne.std(), 1e-9)
        zt = (tx - tx.mean()) / max(tx.std(), 1e-9)
        stg = np.array([[1.0 if r['stage'] == s else 0.0
                         for s in STAGES[1:]] for r in recs])
        X = np.column_stack([np.ones(len(recs)), zn, ze, zt, stg])
        clusters = [f"{r['run']}|{r['family']}" for r in recs]
        beta, se = ols_cluster(X, d, clusters)
        names = ['adjusted_gain', 'z_log1p_nodes', 'z_log1p_edges',
                 'z_log1p_text'] + [f'stage_{s}' for s in STAGES[1:]]
        controls[c_arm] = {
            'n_samples': len(recs), 'raw_mean_delta': float(d.mean()),
            'ols': {n: {'beta': float(b), 'se_cluster': float(s)}
                    for n, b, s in zip(names, beta, se)},
            'cluster': 'run|family',
            'source_composition': dict(collections.Counter(
                r['source'] for r in recs)),
            'dominant_types': dict(collections.Counter(
                r['dom_type'] for r in recs))}
        strata_rep = {}
        for axis, keyf in (
                ('source', lambda r: r['source']),
                ('nodes', lambda r: f"n{r['n_nodes'] if r['n_nodes'] < 6 else '6+'}"),
                ('dominant_type', lambda r: r['dom_type'])):
            groups = collections.defaultdict(list)
            for r in recs:
                groups[keyf(r)].append(r)
            for key, rows in sorted(groups.items()):
                strata_rep[f'{axis}:{key}'] = strat_delta(rows)
        strata_all[c_arm] = strata_rep
    report['pooled']['controls_centroid_minus_random'] = controls
    report['pooled']['strata_centroid_minus_random'] = strata_all
    # save
    with open(outdir / 'route_swap_records.jsonl', 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    (outdir / 'route_swap_report.json').write_text(
        json.dumps(report, indent=1))
    # markdown
    L = ['# Routing swaps on the frozen expert pools (zero training, test)',
         '', 'Variants on the same pool: centroid (deployed), random '
         '(expected + 1000 draws), all-expert mean score, task-ID/label '
         'oracle (diagnostic only, not an upper bound).', '']
    for run, arms in report['runs'].items():
        for arm, rep in arms.items():
            L += [f'## {run} — {arm}', '',
                  '| stage | n | centroid | random(exp) | random 95% | all | '
                  'oracle | centroid AUC |', '|---|---|---|---|---|---|---|---|']
            for s in STAGES:
                v = rep['stages'].get(s)
                if not v:
                    continue
                L.append(f"| {s} | {v['n']} | {v['centroid_acc']:.3f} | "
                         f"{v['random_acc_exp']:.3f} | "
                         f"[{v['random_acc_ci95'][0]:.3f},"
                         f"{v['random_acc_ci95'][1]:.3f}] | "
                         f"{v['all_acc']:.3f} | {v['oracle_acc']:.3f} | "
                         f"{v['centroid_auc']:.3f} |")
            L += ['', 'expert x stage accuracy matrix:', '',
                  '| expert \\ stage | ' + ' | '.join(STAGES) + ' |',
                  '|---|---|---|---|']
            for e, m in rep['expert_stage_matrix'].items():
                L.append(f"| {e} | " + ' | '.join(
                    f"{m[s]:.3f}" for s in STAGES) + ' |')
            L += ['', 'selected-expert loss rank: ' + json.dumps({
                s: {k: rep['stages'][s][k] for k in
                    ('mean_rank_of_selected', 'selected_best_frac',
                     'mean_regret')} for s in STAGES}), '']
    L += ['## Pooled controls (snap arm, all runs)', '',
          json.dumps(report['pooled'], indent=1)]
    (outdir / 'route_swap_report.md').write_text('\n'.join(L) + '\n')
    print('saved', outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
