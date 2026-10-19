#!/usr/bin/env python3
"""Zero-training diagnosis of the below-random score direction on the
family-isolated streams (per PI decision after the audit).

Questions:
1. Is the label direction of *visible* cues (signature, sizes, text length)
   learnable on train, and does it flip on held-out families? (data-level)
2. Do train/test splits differ in covariate-label relation (sign flip)?
3. Is the model's anti-correlation driven by a few test families?
   (leave-one-family-out AUROC from saved scores)
4. Does the pattern differ between v2 and v3, and between CausalCharGNN and
   the final G3? (no training, saved scores only)

Inputs: v2 stream, v3 matched stream, caa_formal scores, g3_formal fam scores.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(os.environ.get('GRAD_BASE', '.'))
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(BASE / 'code/repro'))
import real_cl_p1 as p1  # noqa: E402

V2 = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
V3 = BASE / 'LOGS/real_cl_caa_attribution_20260923/stream_matched'
CAA = BASE / 'code/repro/out/real_cl/caa_formal'
G3 = BASE / 'code/repro/out/real_cl/g3_formal'
STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']


def load_stream(root, phase_key):
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for ph in ('train', 'dev', 'test'):
        for line in open(Path(root) / f'{ph}.jsonl'):
            if line.strip():
                g = json.loads(line)
                g['_phase'] = g.get(phase_key) or ph
                rows[g['stage']][g['_phase']].append(g)
    return rows


def text_len(g):
    return sum(len(str((n.get('features') or {}).get('subtask', '')) or
                   str(n.get('type', ''))) for n in g['nodes'])


def feats(g, kind):
    if kind == 'signature':
        return p1.signature(g)
    if kind == 'sizes':
        return np.array([len(g['nodes']), len(g.get('edges', [])),
                         text_len(g)], dtype=float)
    raise ValueError(kind)


def probe(tr, te, kind, dev=None):
    X = np.stack([feats(g, kind) for g in tr])
    y = np.array([int(g['label']) for g in tr])
    Xt = np.stack([feats(g, kind) for g in te])
    yt = np.array([int(g['label']) for g in te])
    if len(set(y.tolist())) < 2:
        return None
    # grouped CV inside train (learnability)
    gk = GroupKFold(n_splits=min(5, len({g['family_key'] for g in tr})))
    fams = np.array([g['family_key'] for g in tr])
    aucs_cv = []
    perm = np.random.RandomState(0).permutation(len(y))
    for a, b in gk.split(X[perm], y[perm], fams[perm]):
        if len(set(y[perm][a].tolist())) < 2 or \
                len(set(y[perm][b].tolist())) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0))
        clf.fit(X[perm][a], y[perm][a])
        aucs_cv.append(roc_auc_score(y[perm][b],
                                     clf.predict_proba(X[perm][b])[:, 1]))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, C=1.0))
    clf.fit(X, y)
    out = {'train_cv_auroc': float(np.mean(aucs_cv)) if aucs_cv else None,
           'test_auroc': float(roc_auc_score(yt, clf.predict_proba(Xt)[:, 1]))
           if len(set(yt.tolist())) > 1 else None,
           'test_auroc_reversed_diagnostic': (
               float(1 - roc_auc_score(yt, clf.predict_proba(Xt)[:, 1]))
               if len(set(yt.tolist())) > 1 else None),
           'n_train': len(tr), 'n_test': len(te)}
    if dev:
        Xd = np.stack([feats(g, kind) for g in dev])
        yd = np.array([int(g['label']) for g in dev])
        out['dev_auroc'] = float(roc_auc_score(
            yd, clf.predict_proba(Xd)[:, 1])) if len(set(yd.tolist())) > 1 \
            else None
    return out


def corr_label_cov(rows):
    y = np.array([int(g['label']) for g in rows])
    out = {}
    for name, fn in (('n_nodes', lambda g: len(g['nodes'])),
                     ('n_edges', lambda g: len(g.get('edges', []))),
                     ('text_len', text_len)):
        x = np.array([fn(g) for g in rows], dtype=float)
        if len(set(y.tolist())) < 2 or x.std() == 0:
            out[name] = None
        else:
            out[name] = float(np.corrcoef(x, y)[0, 1])
    return out


def sd_flip(rows_tr, rows_te):
    y_tr = np.array([int(g['label']) for g in rows_tr])
    y_te = np.array([int(g['label']) for g in rows_te])
    out = {}
    for name, fn in (('n_nodes', lambda g: len(g['nodes'])),
                     ('n_edges', lambda g: len(g.get('edges', []))),
                     ('text_len', text_len)):
        xtr = np.array([fn(g) for g in rows_tr], dtype=float)
        xte = np.array([fn(g) for g in rows_te], dtype=float)
        ctr = float(np.corrcoef(xtr, y_tr)[0, 1]) if \
            len(set(y_tr.tolist())) > 1 and xtr.std() else None
        cte = float(np.corrcoef(xte, y_te)[0, 1]) if \
            len(set(y_te.tolist())) > 1 and xte.std() else None
        out[name] = {'corr_train': ctr, 'corr_test': cte}
    return out


def family_scores(rows, scores):
    by_fam = collections.defaultdict(list)
    for g in rows:
        if g['task_id'] in scores:
            by_fam[g['family_key']].append((int(g['label']),
                                            float(scores[g['task_id']])))
    out = {}
    for f, vs in by_fam.items():
        ys = [v[0] for v in vs]
        ss = [v[1] for v in vs]
        out[f] = {'n': len(vs), 'pos': sum(ys),
                  'mean_score_pos': float(np.mean([s for y, s in zip(ys, ss)
                                                   if y == 1])) if any(ys)
                  else None,
                  'mean_score_neg': float(np.mean([s for y, s in zip(ys, ss)
                                                   if y == 0])) if
                  any(1 - y for y in ys) else None}
    return out


def loo_auroc(rows, scores):
    pairs = [(int(g['label']), float(scores[g['task_id']]),
              g['family_key']) for g in rows if g['task_id'] in scores]
    if len({p[0] for p in pairs}) < 2:
        return None
    ys = np.array([p[0] for p in pairs])
    ss = np.array([p[1] for p in pairs])
    fams = [p[2] for p in pairs]
    base = roc_auc_score(ys, ss)
    changes = {}
    for f in sorted(set(fams)):
        keep = np.array([i for i, ff in enumerate(fams) if ff != f])
        if len({ys[i] for i in keep}) < 2:
            continue
        changes[f] = float(roc_auc_score(ys[keep], ss[keep]) - base)
    top = sorted(changes.items(), key=lambda kv: -abs(kv[1]))[:5]
    return {'baseline': float(base), 'loo_top': top,
            'max_abs_change': float(max(abs(v) for v in changes.values()))
            if changes else None}


def collect_scores(root, pattern, arm='naive|-'): 
    out = {}
    for rd in sorted(glob.glob(str(Path(root) / pattern))):
        run = Path(rd).name
        sp = Path(rd) / 'scores.jsonl'
        if not sp.exists():
            continue
        per_task = collections.defaultdict(dict)
        for line in open(sp):
            r = json.loads(line)
            if r['arm'] == arm and r['stage'] == 3:
                per_task[r['domain']].setdefault(r['task_id'], []).append(
                    float(r['score']))
        out[run] = per_task
    return out


def mean_scores_across_runs(runs, stage):
    acc = collections.defaultdict(list)
    for run, per_task in runs.items():
        for tid, ss in per_task.get(stage, {}).items():
            acc[tid].append(float(np.mean(ss)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def family_level(rows, scores):
    by = collections.defaultdict(list)
    for g in rows:
        if g['task_id'] in scores:
            by[g['family_key']].append((int(g['label']),
                                        scores[g['task_id']]))
    fam_rows = []
    pure = 0
    for f, vs in by.items():
        ys = [v[0] for v in vs]
        ss = [v[1] for v in vs]
        if len(set(ys)) == 1:
            pure += len(vs)
        fam_rows.append((float(np.mean(ys)), float(np.mean(ss)), f, len(vs),
                         sum(ys)))
    labs = np.array([r[0] for r in fam_rows])
    scs = np.array([r[1] for r in fam_rows])
    out = {'n_families': len(fam_rows),
           'pure_sample_frac': pure / max(sum(r[3] for r in fam_rows), 1)}
    if len(set((labs >= 0.5).tolist())) > 1:
        out['family_auroc_as_is'] = float(roc_auc_score(
            (labs >= 0.5).astype(int), scs))
    else:
        out['family_auroc_as_is'] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {}
    streams = {'v2': load_stream(V2, 'v2_phase'),
               'v3': load_stream(V3, 'v3_phase')}
    # ---- data-level direction ----
    data_rep = {}
    for sname, stream in streams.items():
        data_rep[sname] = {}
        for stage in STAGES:
            tr = stream[stage]['train']
            dv = stream[stage]['dev']
            te = stream[stage]['test']
            if not tr or not te:
                continue
            entry = {'n': {'train': len(tr), 'dev': len(dv), 'test': len(te)},
                     'pos_rate': {p: (sum(int(g['label']) for g in stream[stage][p])
                                      / max(len(stream[stage][p]), 1))
                                  for p in ('train', 'dev', 'test')},
                     'corr_train': corr_label_cov(tr),
                     'corr_test': corr_label_cov(te),
                     'corr_shift': sd_flip(tr, te)}
            for kind in ('signature', 'sizes'):
                entry[f'probe_{kind}'] = probe(tr, te, kind, dv)
            data_rep[sname][stage] = entry
    report['data_direction'] = data_rep
    # ---- model-level family analysis (naive arm scores; v3 CAA vs v2 G3) ----
    model_rep = {}
    score_sets = {
        'caa_v3': (collect_scores(CAA, '*', 'naive|-'), streams['v3']),
        'caa_v3_replay': (collect_scores(CAA, '*', 'replay|-'), streams['v3']),
        'g3_v2': (collect_scores(G3, 'fam_*', 'naive|-'), streams['v2']),
    }
    for tag, (runs, stream) in score_sets.items():
        model_rep[tag] = {}
        for stage in STAGES:
            te = stream[stage]['test']
            all_loo = []
            for run, per_task in runs.items():
                scores = {k: float(np.mean(v)) for k, v in
                          per_task.get(stage, {}).items()}
                if not scores:
                    continue
                loo = loo_auroc(te, scores)
                if loo:
                    all_loo.append(loo)
            if not all_loo:
                continue
            ms = mean_scores_across_runs(runs, stage)
            fl = family_level(te, ms)
            fam = family_scores(te, ms)
            flipped = 0
            total = 0
            for f, v in fam.items():
                if v['mean_score_pos'] is not None and \
                        v['mean_score_neg'] is not None:
                    total += 1
                    if v['mean_score_pos'] < v['mean_score_neg']:
                        flipped += 1
            model_rep[tag][stage] = {
                'n_runs': len(all_loo),
                'auroc_mean': float(np.mean([x['baseline'] for x in all_loo])),
                'family_auroc_as_is': fl['family_auroc_as_is'],
                'n_families_test': fl['n_families'],
                'pure_sample_frac': fl['pure_sample_frac'],
                'mixed_families_score_flip': [flipped, total],
                'loo_max_abs_change_mean': float(np.mean(
                    [x['max_abs_change'] for x in all_loo
                     if x['max_abs_change'] is not None])),
                'top_loo_example': all_loo[0]['loo_top']}
    report['model_families'] = model_rep
    (outdir / 'diag_direction.json').write_text(json.dumps(report, indent=1))
    L = ['# Zero-training direction diagnosis (v2 vs v3, CausalCharGNN vs G3)',
         '', '## Data-level: can visible cues predict the label?', '',
         '| stream | stage | probe | train-CV AUROC | test AUROC (as-is) |',
         '|---|---|---|---|---|']
    for sname, stages in data_rep.items():
        for stage, v in stages.items():
            for kind in ('signature', 'sizes'):
                p = v.get(f'probe_{kind}')
                if not p:
                    continue
                L.append(f"| {sname} | {stage} | {kind} | "
                         f"{p['train_cv_auroc']:.3f} | "
                         f"{p['test_auroc']:.3f} |")
    L += ['', '## Label-covariate correlation shift (train vs test)', '',
          '| stream | stage | covariate | corr(train) | corr(test) |',
          '|---|---|---|---|---|']
    for sname, stages in data_rep.items():
        for stage, v in stages.items():
            for cov, cv in v['corr_shift'].items():
                L.append(f"| {sname} | {stage} | {cov} | "
                         f"{cv['corr_train'] if cv['corr_train'] is None else round(cv['corr_train'], 3)} | "
                         f"{cv['corr_test'] if cv['corr_test'] is None else round(cv['corr_test'], 3)} |")
    L += ['', '## Model-score family analysis (scores averaged over runs)',
          '', '| panel | stage | runs | sample AUROC | family-level AUROC | '
          'test families | pure-sample frac | mixed families flipped | '
          'max LOO change |', '|---|---|---|---|---|---|---|---|---|']
    for tag, stages in model_rep.items():
        for stage, v in stages.items():
            fa = v['family_auroc_as_is']
            L.append(f"| {tag} | {stage} | {v['n_runs']} | "
                     f"{v['auroc_mean']:.3f} | "
                     f"{'n/a' if fa is None else f'{fa:.3f}'} | "
                     f"{v['n_families_test']} | "
                     f"{v['pure_sample_frac']:.2f} | "
                     f"{v['mixed_families_score_flip'][0]}/"
                     f"{v['mixed_families_score_flip'][1]} | "
                     f"{v['loo_max_abs_change_mean']:.3f} |")
    (outdir / 'diag_direction.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
