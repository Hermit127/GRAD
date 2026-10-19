#!/usr/bin/env python3
"""Zero-training endpoint recheck of the final G3 cross-dataset (dmm) panel.

Uses the existing per-example predictions (`g3_formal/dmm_*/scores.jsonl`) and
recomputes, side by side, keeping the original definition first and adding
equal-unit endpoints without post-hoc selection:

1. `sample_acc`        - original: mean over tasks of sample accuracy;
2. `pooled_sample_acc` - all samples pooled;
3. `family_acc`        - mean over tasks of mean over independent units
   (family/base case) of unit accuracy (unit = mean correctness of samples).

Family definitions (verified against the raw sources):
- decomp: base case = numeric case id grouping harmful/benign/mix variants;
- mcp: metadata.pair_id for generated pairs, task-template for runtime;
- mrt: scenario (`metadata.task_name`), the attack/benign pairing unit.

Also reports: split-level family overlap (are the units actually isolated
across splits?), per-task deltas under each endpoint, paired statistics
(original seed bootstrap / two-way seed x order / family-cluster with run
resampling), and per-family contributions to snap - replay with large/small
family split. No score is flipped anywhere.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('GRAD_BASE', '.'))
DOMAINS = ['financial_analysis', 'web_search', 'location_navigation',
           'repository_management', 'browser_automation']
STAGES = ['decomp', 'mcp', 'mrt']
SEEDS = [211, 212, 213, 214, 215]


def domain_of(g):
    md = g.get('metadata', {}) or {}
    s = str(md.get('config', '')) + str(md.get('task', ''))
    for d in DOMAINS:
        if d in s:
            return d
    return 'other'


def build_family_maps():
    fam = {'decomp': {}, 'mcp': {}, 'mrt': {}}
    for f in ('ref/dataset/decomp_graphs_train_mixed.jsonl',
              'ref/dataset/decomp_graphs_test_mixed.jsonl'):
        for line in open(BASE / f):
            g = json.loads(line)
            m = re.match(r'^(harmful|benign)_(\d+)(_mix[0-9.]+_r[0-9.]+)?$',
                         g['task_id'])
            fam['decomp'][g['task_id']] = f'case_{m.group(2)}' if m \
                else g['task_id']
    files = ([f'ref/dataset/mcp_safetybench_defense_{s}.jsonl'
              for s in ('train', 'test', 'validation')] +
             [f for f in sorted(glob.glob(str(
                 BASE / 'ref/dataset/mcp_residual_matched_v2_merged/graphs/'
                        '*.jsonl'))) if 'generated_all' not in f] +
             sorted(glob.glob(str(BASE / 'ref/dataset/mcp_dgc_v0*/*.jsonl'))) +
             sorted(glob.glob(str(BASE / 'ref/dataset/mcp_dgc_v1_seed*/'
                                       '*.jsonl'))) +
             sorted(glob.glob(str(BASE / 'ref/dataset/'
                                       'mcp_coverage_deficit_*/*/*.jsonl'))) +
             sorted(glob.glob(str(BASE / 'ref/dataset/'
                                       'mcp_channel_gap_dgc_v2_seed49*/'
                                       '*.jsonl'))))
    seen = set()
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        for line in open(p):
            g = json.loads(line)
            h = json.dumps(g, sort_keys=True)
            if h in seen:
                continue
            seen.add(h)
            if domain_of(g) == 'other':
                continue
            md = g.get('metadata', {}) or {}
            pid = md.get('pair_id')
            task = str(md.get('task', ''))
            if pid:
                fam['mcp'][g['task_id']] = f'pair_{pid}'
            elif task:
                parts = task.replace('\\', '/').split('/')
                base = re.sub(r'\d+$', '', parts[-1].rsplit('.', 1)[0])
                fam['mcp'][g['task_id']] = 'task_' + '/'.join(
                    parts[:-1] + [base.rstrip('_')])
            else:
                fam['mcp'][g['task_id']] = g['task_id']
    mrt_p = (BASE / 'results/mrt_shared_adapter_scale360_v1/'
                    'stage_07_top1_360.jsonl')
    for line in open(mrt_p):
        g = json.loads(line)
        md = g.get('metadata', {}) or {}
        fam['mrt'][g['task_id']] = md.get('task_name') or re.sub(
            r'_\d+$', '', g['task_id'])
    return fam


def load_run(rd):
    rows = collections.defaultdict(dict)
    tid2dom = {}
    for line in open(Path(rd) / 'scores.jsonl'):
        r = json.loads(line)
        if r['stage'] != 3 or r['domain'] not in STAGES:
            continue
        rows[r['arm']][r['task_id']] = (int(r['label']), float(r['score']))
        tid2dom[r['task_id']] = r['domain']
    return rows, tid2dom


def endpoints(rows, fam, tid2dom):
    per_task = {}
    pooled = []
    for task in STAGES:
        rid = [(t, v) for t, v in rows.items()
               if tid2dom.get(t, t.split('::')[0]) == task]
        if not rid:
            continue
        cs = [float((v[1] >= 0.5) == v[0]) for _, v in rid]
        by_fam = collections.defaultdict(list)
        for t, v in rid:
            by_fam[fam[task].get(t, t)].append(
                float((v[1] >= 0.5) == v[0]))
        fam_accs = [float(np.mean(v)) for v in by_fam.values()]
        per_task[task] = {'sample_acc': float(np.mean(cs)),
                          'family_acc': float(np.mean(fam_accs)),
                          'n': len(rid), 'n_families': len(by_fam),
                          'family_accs': fam_accs}
        pooled += cs
    macro = {k: float(np.mean([v[k] for v in per_task.values()]))
             for k in ('sample_acc', 'family_acc')}
    macro['pooled_sample_acc'] = float(np.mean(pooled)) if pooled else None
    return macro, per_task


def split_family_overlap(fam):
    """Fraction of test samples whose family also occurs in train, per seed."""
    try:
        sys.path.insert(0, str(BASE / 'code'))
        sys.path.insert(0, str(BASE / 'code/repro'))
        import real_cl_s2 as s2
    except Exception as e:  # pragma: no cover
        return {'error': repr(e)}
    out = {}
    for seed in SEEDS:
        splits = s2.build_splits(seed, STAGES)
        per_task = {}
        for task in STAGES:
            tr_fams = {fam[task].get(g['task_id'], g['task_id'])
                       for g in splits[task]['train']}
            te = splits[task]['test']
            te_fams = [fam[task].get(g['task_id'], g['task_id']) for g in te]
            per_task[task] = {
                'n_test': len(te),
                'n_test_families': len(set(te_fams)),
                'test_samples_with_family_in_train': float(np.mean(
                    [f in tr_fams for f in te_fams])) if te_fams else None}
        out[seed] = per_task
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(
        BASE / 'code/repro/out/real_cl/g3_formal'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fam = build_family_maps()
    runs = {}
    for rd in sorted(glob.glob(str(Path(args.root) / 'dmm_*'))):
        name = Path(rd).name.replace('dmm_', '')
        order, seed = name.rsplit('_', 1)
        rows, tid2dom = load_run(rd)
        runs[(int(seed), order)] = {'rows': rows, 'tid2dom': tid2dom}
    out = {'family_definitions': {
        'decomp': 'case_{number}: harmful/benign/mix variants of one case',
        'mcp': 'metadata.pair_id (generated) or runtime task template',
        'mrt': 'metadata.task_name scenario (attack/benign pairing unit)'},
        'split_overlap': split_family_overlap(fam),
        'endpoints': {}, 'deltas': {}, 'task_deltas': {},
        'families': {}}
    run_ep = {}
    for (seed, order), r in runs.items():
        for arm, rows in r['rows'].items():
            macro, per_task = endpoints(rows, fam, r['tid2dom'])
            run_ep[(seed, order, arm)] = {'macro': macro, 'per_task': per_task}
    arms = sorted({a for (_, _, a) in run_ep})
    for arm in arms:
        vals = [v['macro'] for (s, o, a), v in run_ep.items() if a == arm]
        out['endpoints'][arm] = {
            'task_macro_sample': float(np.mean([v['sample_acc'] for v in vals])),
            'sample_pooled': float(np.mean([v['pooled_sample_acc']
                                            for v in vals])),
            'task_macro_family': float(np.mean([v['family_acc']
                                                for v in vals])),
            'n_runs': len(vals)}
    seeds = sorted({s for (s, o) in runs})
    orders = sorted({o for (s, o) in runs})
    pairs = [('snap|centroid', 'replay|-'), ('stage0|centroid', 'replay|-'),
             ('snap|oracle', 'snap|centroid')]

    def ci_seed(vals):
        rng = np.random.RandomState(7)
        arr = np.array(vals)
        bs = [rng.choice(arr, len(arr), replace=True).mean()
              for _ in range(2000)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    def ci_twoway(vals_by_run):
        rng = np.random.RandomState(11)
        bs = []
        for _ in range(2000):
            ss = [seeds[i] for i in rng.randint(0, len(seeds), len(seeds))]
            oo = [orders[i] for i in rng.randint(0, len(orders), len(orders))]
            ds = [vals_by_run[(s, o)] for s in ss for o in oo
                  if (s, o) in vals_by_run]
            if ds:
                bs.append(float(np.mean(ds)))
        return [float(np.percentile(bs, 2.5)),
                float(np.percentile(bs, 97.5))] if bs else None

    for a, b in pairs:
        rec = {}
        for key in ('sample_acc', 'family_acc', 'pooled_sample_acc'):
            per_seed = {}
            per_run = {}
            for (s, o, arm), v in run_ep.items():
                if arm != a or (s, o, b) not in run_ep:
                    continue
                d = v['macro'][key] - run_ep[(s, o, b)]['macro'][key]
                per_run[(s, o)] = d
                per_seed.setdefault(s, []).append(d)
            seed_means = {s: float(np.mean(v)) for s, v in per_seed.items()}
            keyrep = {'mean_seed_avg': float(np.mean(list(seed_means.values()))),
                      'ci95_seed_bootstrap_original_style': ci_seed(
                          list(seed_means.values())),
                      'ci95_twoway_run': ci_twoway(per_run),
                      'n_runs': len(per_run)}
            if key == 'sample_acc':
                # family-cluster bootstrap with run resampling
                rng = np.random.RandomState(13)
                keys = sorted(runs)
                draws = []
                for _ in range(1000):
                    picked = [keys[i] for i in
                              rng.randint(0, len(keys), len(keys))]
                    num, den = 0.0, 0
                    for (s, o) in picked:
                        ra = runs[(s, o)]['rows'].get(a, {})
                        rb = runs[(s, o)]['rows'].get(b, {})
                        tid2dom = runs[(s, o)]['tid2dom']
                        byf = collections.defaultdict(list)
                        for t in set(ra) & set(rb):
                            task = tid2dom.get(t, '')
                            byf[fam.get(task, {}).get(t, t)].append(
                                float((ra[t][1] >= 0.5) == ra[t][0]) -
                                float((rb[t][1] >= 0.5) == rb[t][0]))
                        fams = sorted(byf)
                        for i in rng.randint(0, len(fams), len(fams)):
                            v = byf[fams[i]]
                            num += float(np.sum(v))
                            den += len(v)
                    if den:
                        draws.append(num / den)
                keyrep['ci95_family_cluster_run_resampled'] = [
                    float(np.percentile(draws, 2.5)),
                    float(np.percentile(draws, 97.5))] if draws else None
            rec[key] = keyrep
        out['deltas'][f'{a} - {b}'] = rec
    # per-task deltas under sample vs family endpoints
    for a, b in pairs:
        task_rec = {}
        for task in STAGES:
            for key in ('sample_acc', 'family_acc'):
                per_seed = {}
                for (s, o, arm), v in run_ep.items():
                    if arm != a or (s, o, b) not in run_ep:
                        continue
                    va = v['per_task'].get(task, {}).get(key)
                    vb = run_ep[(s, o, b)]['per_task'].get(task, {}).get(key)
                    if va is None or vb is None:
                        continue
                    per_seed.setdefault(s, []).append(va - vb)
                means = {s: float(np.mean(v)) for s, v in per_seed.items()}
                task_rec[f'{task}|{key}'] = {
                    'mean': float(np.mean(list(means.values()))) if means
                    else None,
                    'ci95_seed_bootstrap': ci_seed(list(means.values()))
                    if means else None}
        out['task_deltas'][f'{a} - {b}'] = task_rec
    # family structure + contributions (per-run mass-balanced)
    for task in STAGES:
        run_stats = []
        fam_store = collections.defaultdict(lambda: {'sz': [], 'd': [],
                                                     'c': []})
        for (s, o), r in runs.items():
            A, B = r['rows'].get('snap|centroid', {}), \
                r['rows'].get('replay|-', {})
            byf = collections.defaultdict(list)
            for t in sorted(set(A) & set(B)):
                if r['tid2dom'].get(t, '') != task:
                    continue
                byf[fam[task].get(t, t)].append(
                    (float((A[t][1] >= 0.5) == A[t][0]),
                     float((B[t][1] >= 0.5) == B[t][0])))
            if not byf:
                continue
            n_r = sum(len(v) for v in byf.values())
            sizes = {f: len(v) for f, v in byf.items()}
            q75 = float(np.percentile(list(sizes.values()), 75))
            large = small = 0.0
            delta_excl = None
            order_fams = sorted(byf, key=lambda f: -sizes[f])
            kk = max(1, len(order_fams) // 20)
            keep = order_fams[kk:]
            num = den = 0.0
            for f, v in byf.items():
                d = float(np.mean([x[0] for x in v])) - \
                    float(np.mean([x[1] for x in v]))
                c = sizes[f] / n_r * d
                fam_store[f]['sz'].append(sizes[f])
                fam_store[f]['d'].append(d)
                fam_store[f]['c'].append(c)
                if sizes[f] >= q75:
                    large += c
                else:
                    small += c
            for f in keep:
                num += sizes[f] * fam_store[f]['d'][-1]
                den += sizes[f]
            delta_excl = num / den if den else None
            run_stats.append({'n': n_r, 'fams': len(byf),
                              'size_mean': float(np.mean(
                                  list(sizes.values()))),
                              'large': large, 'small': small,
                              'delta_excl': delta_excl})
        contrib = [(f, float(np.mean(v['sz'])), float(np.mean(v['d'])),
                    float(np.mean(v['c']))) for f, v in fam_store.items()]
        contrib.sort(key=lambda x: -abs(x[3]))
        out['families'][task] = {
            'n_samples_mean': float(np.mean([r['n'] for r in run_stats])),
            'n_families_mean': float(np.mean([r['fams'] for r in run_stats])),
            'size_mean': float(np.mean([r['size_mean'] for r in run_stats])),
            'size_max': float(max(c[1] for c in contrib)) if contrib else 0,
            'delta_family_equal': float(np.mean(
                [c[2] for c in contrib])) if contrib else None,
            'large_family_contribution_mean': float(np.mean(
                [r['large'] for r in run_stats])),
            'small_family_contribution_mean': float(np.mean(
                [r['small'] for r in run_stats])),
            'delta_excl_top5pct_families_mean': float(np.mean(
                [r['delta_excl'] for r in run_stats
                 if r['delta_excl'] is not None])),
            'top_contributors': [{'family': f, 'size': sz, 'delta': d,
                                  'contribution': c}
                                 for f, sz, d, c in contrib[:8]]}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'g3_dmm_endpoint_recheck.json').write_text(
        json.dumps(out, indent=1))
    L = ['# Final G3 cross-dataset (dmm) endpoints recheck (zero training)',
         '',
         'Original definition first; equal-unit endpoints added, not '
         'substituted. No post-hoc selection, no score flipping.', '',
         '## Endpoint levels by arm (mean over 10 runs)', '',
         '| arm | task-macro sample (original) | pooled sample | '
         'task-macro family |', '|---|---|---|---|']
    for arm, v in sorted(out['endpoints'].items()):
        L.append(f"| {arm} | {v['task_macro_sample']:.4f} | "
                 f"{v['sample_pooled']:.4f} | {v['task_macro_family']:.4f} |")
    L += ['', '## Paired deltas', '',
          '| comparison | endpoint | mean | CI seed-only (original) | '
          'CI seed x order | CI family-cluster |', '|---|---|---|---|---|---|']
    for k, rec in out['deltas'].items():
        for key in ('sample_acc', 'family_acc', 'pooled_sample_acc'):
            v = rec[key]

            def fmt(ci):
                return '--' if not ci else f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
            L.append(f"| {k} | {key} | {v['mean_seed_avg']:+.4f} | "
                     f"{fmt(v['ci95_seed_bootstrap_original_style'])} | "
                     f"{fmt(v['ci95_twoway_run'])} | "
                     f"{fmt(v.get('ci95_family_cluster_run_resampled'))} |")
    L += ['', '## Per-task deltas (snap+centroid - replay)', '',
          '| task | endpoint | mean | 95% CI (seed) |', '|---|---|---|---|']
    for k, rec in out['task_deltas'].items():
        if k != 'snap|centroid - replay|-':
            continue
        for tk, v in rec.items():
            if v['mean'] is None:
                continue
            L.append(f"| {tk.split('|')[0]} | {tk.split('|')[1]} | "
                     f"{v['mean']:+.4f} | "
                     f"[{v['ci95_seed_bootstrap'][0]:+.4f},"
                     f"{v['ci95_seed_bootstrap'][1]:+.4f}] |")
    L += ['', '## Split-level family overlap (fraction of test samples whose '
          'family also appears in train)', '',
          '| seed | decomp | mcp | mrt |', '|---|---|---|---|']
    if isinstance(out['split_overlap'], dict) and 'error' not in \
            out['split_overlap']:
        for seed, per in out['split_overlap'].items():
            L.append(f"| {seed} | "
                     f"{per['decomp']['test_samples_with_family_in_train']:.3f} | "
                     f"{per['mcp']['test_samples_with_family_in_train']:.3f} | "
                     f"{per['mrt']['test_samples_with_family_in_train']:.3f} |")
    L += ['', '## Family structure and snap-replay contributions', '',
          '| task | samples/run | families/run | mean size | family-equal '
          'delta | large-family contrib | small-family contrib | delta excl '
          'top 5% families |', '|---|---|---|---|---|---|---|---|']
    for task, v in out['families'].items():
        L.append(f"| {task} | {v['n_samples_mean']:.1f} | "
                 f"{v['n_families_mean']:.1f} | {v['size_mean']:.2f} | "
                 f"{v['delta_family_equal']:+.4f} | "
                 f"{v['large_family_contribution_mean']:+.4f} | "
                 f"{v['small_family_contribution_mean']:+.4f} | "
                 f"{v['delta_excl_top5pct_families_mean']:+.4f} |")
    L += ['', 'Top contributors per task:']
    for task, v in out['families'].items():
        L.append(f"- {task}: " + '; '.join(
            f"{c['family']}(n={c['size']:.1f},d={c['delta']:+.2f},"
            f"c={c['contribution']:+.3f})" for c in v['top_contributors'][:5]))
    (outdir / 'g3_dmm_endpoint_recheck.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
