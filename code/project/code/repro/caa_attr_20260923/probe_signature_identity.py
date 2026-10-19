#!/usr/bin/env python3
"""CAA attribution Phase A1: signature-only identity probes (zero API, no GPUs).

Re-checks the anti-shortcut evidence of the family-isolated confirmation
stream (`LOGS/real_cl_family_confirm_20260921/stream`):

- targets: task stage (3 classes) and data source (runtime/generated);
- features: exactly the observable signature used by the deployed centroid
  router (`real_cl_p1.signature` = node-type share histogram + n_nodes/50 +
  n_edges/50); an extended raw-count variant is reported for completeness;
- probes: linear (StandardScaler + LogisticRegression) and nonlinear
  (RandomForest, MLP); all preprocessing is fit inside the training folds;
  no hyperparameter search is performed (defaults fixed a priori);
- CV: GroupKFold(5) by family (families never cross folds);
- held-out protocol: fit on stream train, evaluate on stream dev and test
  (never fit on test), predictions saved per sample;
- CIs: fold-level t interval + family-cluster bootstrap over pooled
  out-of-fold predictions.

Outputs: JSON report, predictions JSONL, Markdown summary.
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

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

STREAM = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']
PHASES = ['train', 'dev', 'test']


def load_rows():
    rows = []
    for ph in PHASES:
        for line in open(STREAM / f'{ph}.jsonl'):
            if line.strip():
                r = json.loads(line)
                r['_phase'] = ph
                rows.append(r)
    return rows


def source_group(g):
    md = g.get('metadata') or {}
    if str(md.get('source', '')) == 'mcp_safetybench' and \
       str(md.get('graph_origin', '')) == 'runtime_trace':
        return 'runtime'
    return 'generated'


def sig(g, node_types):
    v = [0.0] * len(node_types)
    idx = {t: i for i, t in enumerate(node_types)}
    for n in g['nodes']:
        t = str(n.get('type', 'other'))
        v[idx.get(t, idx['other'])] += 1
    n = max(len(g['nodes']), 1)
    return np.array([x / n for x in v] +
                    [len(g['nodes']) / 50.0, len(g.get('edges', [])) / 50.0])


def raw_feats(g, node_types):
    counts = collections.Counter(str(n.get('type', 'other')) for n in g['nodes'])
    v = [counts.get(t, 0) for t in node_types]
    txt = sum(len(str((n.get('features') or {}).get('subtask', '')) or
                  str(n.get('type', ''))) for n in g['nodes'])
    v += [len(g['nodes']), len(g.get('edges', [])), txt]
    return np.array(v, dtype=float)


def cluster_boot(y, pred, fam, n_boot=2000, seed=20260923):
    """Family-cluster bootstrap CI for balanced accuracy of saved predictions."""
    fams = sorted(set(fam))
    by_fam = collections.defaultdict(list)
    for i, f in enumerate(fam):
        by_fam[f].append(i)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        picks = rng.choice(len(fams), size=len(fams), replace=True)
        idx = [i for p in picks for i in by_fam[fams[p]]]
        yb = [y[i] for i in idx]
        pb = [pred[i] for i in idx]
        if len(set(yb)) < 2:
            continue
        vals.append(balanced_accuracy_score(yb, pb))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def models(seed):
    return {
        'linear': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0)),
        'rf': RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=8),
        'mlp': make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                          random_state=seed)),
    }


def evaluate_cv(X, y, fam, seed, out_records, tag, target_name):
    reps = {}
    perm = np.random.RandomState(seed).permutation(len(y))
    Xp, yp, famp = X[perm], np.asarray(y)[perm], np.asarray(fam)[perm]
    for name, mk in models(seed).items():
        gkf = GroupKFold(n_splits=5)
        preds = np.zeros(len(y), dtype=object)
        folds = []
        for k, (tr, te) in enumerate(gkf.split(Xp, yp, famp)):
            clf = mk
            clf.fit(Xp[tr], yp[tr])
            p = clf.predict(Xp[te])
            for i, pi in zip(te, p):
                gi = perm[i]
                preds[gi] = pi
                out_records.append({
                    'audit': 'group_cv', 'feature': tag,
                    'target': target_name, 'model': name, 'fold': k,
                    'family': fam[gi], 'task_id': task_ids[gi],
                    'phase': phases[gi], 'y': str(y[gi]), 'pred': str(pi)})
            folds.append(float(balanced_accuracy_score(yp[te], p)))
        pred_list = [preds[i] for i in range(len(y))]
        balacc = float(balanced_accuracy_score(y, pred_list))
        lo, hi, n_ok = cluster_boot(list(y), pred_list, list(fam), seed=seed)
        reps[name] = {'balacc': balacc, 'folds': [round(f, 4) for f in folds],
                      'fold_mean': float(np.mean(folds)),
                      'fold_std': float(np.std(folds, ddof=1)),
                      'cluster_ci95': [lo, hi],
                      'n_boot': n_ok}
    return reps


def main() -> int:
    global task_ids, phases
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20260923)
    args = ap.parse_args()
    rows = load_rows()
    node_types = sorted(set(
        str(n.get('type', 'other')) for r in rows for n in r['nodes'])
        | {'other'})
    fam = [r['family_key'] for r in rows]
    task_ids = [r['task_id'] for r in rows]
    phases = [r['_phase'] for r in rows]
    report = {'n_rows': len(rows), 'n_families': len(set(fam)),
              'node_types': node_types, 'protocol': {
                  'cv': 'GroupKFold(5, shuffle=True) by family',
                  'preprocess': 'fit inside training folds only',
                  'tuning': 'none; fixed a-priori defaults',
                  'heldout': 'fit on stream train, evaluate dev/test'},
              'probes': {}}
    records = []
    targets = {
        'stage': [r['stage'] for r in rows],
        'source': [source_group(r) for r in rows],
    }
    for tag in ('stage', 'source'):
        report['probes'][tag] = {}
        X_sig = np.stack([sig(r, node_types) for r in rows])
        X_raw = np.stack([raw_feats(r, node_types) for r in rows])
        for feat_name, X in (('signature', X_sig), ('raw_counts', X_raw)):
            rep = evaluate_cv(X, targets[tag], fam, args.seed, records,
                              f'{tag}|{feat_name}', tag)
            report['probes'][tag][feat_name] = rep
        # held-out train-fit -> dev/test eval, signature features only
        tr = np.array([r['_phase'] == 'train' for r in rows])
        for name, mk in models(args.seed).items():
            clf = mk
            clf.fit(X_sig[tr], [targets[tag][i] for i in range(len(rows))
                                if tr[i]])
            out = {}
            for ph in ('dev', 'test'):
                sel = np.array([r['_phase'] == ph for r in rows])
                y_ph = [targets[tag][i] for i in range(len(rows)) if sel[i]]
                p_ph = clf.predict(X_sig[sel])
                for i, pi in zip(np.where(sel)[0], p_ph):
                    records.append({
                        'audit': 'train_fit', 'feature': f'{tag}|signature',
                        'target': tag, 'model': name, 'fold': -1,
                        'family': fam[i], 'task_id': task_ids[i],
                        'phase': phases[i], 'y': str(targets[tag][i]),
                        'pred': str(pi)})
                out[ph] = {'n': int(sel.sum()),
                           'balacc': float(balanced_accuracy_score(y_ph, p_ph))}
            report['probes'][tag].setdefault('train_fit', {})[name] = out
    # stage prediction for dev/test fit-on-train is the deployable router analogue
    report['chance'] = {'stage': 1.0 / 3.0, 'source': 'majority prior'}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'probe_report.json').write_text(json.dumps(report, indent=1))
    with open(outdir / 'probe_predictions.jsonl', 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    L = ['# Signature-only identity probes (re-check, zero training)', '',
         f"rows={len(rows)} families={len(set(fam))}; "
         'GroupKFold(5) by family; preprocessing fit in-fold; no tuning.', '',
         '| target | features | model | balacc | folds (mean+-sd) | '
         'family-cluster 95% CI |', '|---|---|---|---|---|---|']
    for tag, feats in report['probes'].items():
        for fn, mrep in feats.items():
            if fn == 'train_fit':
                continue
            for name, v in mrep.items():
                L.append(f"| {tag} | {fn} | {name} | {v['balacc']:.3f} | "
                         f"{v['fold_mean']:.3f}+-{v['fold_std']:.3f} | "
                         f"[{v['cluster_ci95'][0]:.3f},{v['cluster_ci95'][1]:.3f}] |")
    L += ['', '## Fit on stream train -> held-out dev/test (signature)',
          '', '| target | model | dev balacc | test balacc |',
          '|---|---|---|---|']
    for tag, feats in report['probes'].items():
        for name, v in feats.get('train_fit', {}).items():
            L.append(f"| {tag} | {name} | {v['dev']['balacc']:.3f} | "
                     f"{v['test']['balacc']:.3f} |")
    (outdir / 'probe_report.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
