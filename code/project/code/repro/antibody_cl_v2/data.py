"""Pilot data access for antibody_cl_v2 (D1 MCP pilot, family/pair isolated).

Reads only train/router-dev/threshold-cal during pilot. The outer split is
never loaded here; its usage audit lives in audit.py.
"""
from __future__ import annotations

import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__import__('os').environ.get('GRAD_BASE',
                                         '.'))
D1 = BASE / 'code/repro/out/d1'
DOMAINS = ['financial_analysis', 'browser_automation', 'repository_management']
HIGH_PRIV = {'shell_exec', 'code_exec', 'file_write', 'file_delete',
             'system_config', 'account_access', 'tool_execute'}
EXTERNAL = {'prompt_inject', 'web_search', 'web_download', 'information_search',
            'file_read'}


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_splits():
    """domain -> {train, router_dev, threshold_cal} graphs (outer never read)."""
    out = {}
    for split in ('train', 'router_dev', 'threshold_cal'):
        for g in load_jsonl(D1 / f'{split}.jsonl'):
            out.setdefault(g['_domain'], {}).setdefault(split, []).append(g)
    for d in out:
        for s in ('train', 'router_dev', 'threshold_cal'):
            out[d].setdefault(s, [])
    return out


def split_audit(splits):
    audit = {'family_isolation': True, 'pair_isolation': True,
             'per_domain': {}}
    seen_fam = defaultdict(set)
    seen_pair = defaultdict(set)
    for d, sp in splits.items():
        audit['per_domain'][d] = {
            s: {'graphs': len(v), 'pairs': len({g['_pair'] for g in v}),
                'labels': dict(Counter(g['label'] for g in v))}
            for s, v in sp.items()}
        for s, v in sp.items():
            for g in v:
                seen_fam[(g['_domain'], g['_family'])].add(s)
                seen_pair[(g['_domain'], g['_pair'])].add(s)
    audit['family_isolation'] = all(len(v) == 1 for v in seen_fam.values())
    audit['pair_isolation'] = all(len(v) == 1 for v in seen_pair.values())
    audit['n_families'] = len(seen_fam)
    audit['n_pairs'] = len(seen_pair)
    # cross-domain family-name collisions are documented, not leakage by
    # themselves (the review requires (domain, family) isolation)
    names = defaultdict(set)
    for (dom, fam) in seen_fam:
        names[fam].add(dom)
    audit['cross_domain_family_name_collisions'] = sum(
        1 for v in names.values() if len(v) > 1)
    return audit


def stage_pool(splits, seen_domains, split='train'):
    return [g for d in seen_domains for g in splits[d][split]]


def current_pool(splits, domain, split='train'):
    return list(splits[domain][split])


def common_balanced(splits, seen_domains, n, rng):
    """Balanced by (domain, label) over seen train pools (with replacement)."""
    buckets = defaultdict(list)
    for d in seen_domains:
        for g in splits[d]['train']:
            buckets[(d, g['label'])].append(g)
    keys = sorted(buckets)
    out = []
    i = 0
    while len(out) < n and keys:
        k = keys[i % len(keys)]
        out.append(rng.choice(buckets[k]))
        i += 1
    return out


def observable_specialty(g):
    """Observable (label-free, domain-free) specialty scores in [0,1]."""
    types = Counter(n['type'] for n in g['nodes'])
    n = max(len(g['nodes']), 1)
    ext = sum(types[t] for t in EXTERNAL) / n
    priv = sum(types[t] for t in HIGH_PRIV) / n
    return {
        'semantic_induced': min(1.0, ext / 0.6),
        'long_chain': min(1.0, n / 12.0),
        'cross_session': None,  # no session ids in this pilot -> not evaluable
        'high_privilege': min(1.0, priv / 0.4),
        'oof_hard': 0.0,        # filled by data.oof_hard_scores
    }


def _featurize(g, buckets=64):
    types = Counter(n['type'] for n in g['nodes'])
    v = [types.get(t, 0) for t in sorted(HIGH_PRIV | EXTERNAL)]
    v.append(len(g['nodes']) / 10.0)
    hv = [0.0] * buckets
    for nd in g['nodes']:
        txt = str(nd.get('features', {}).get('subtask', '')).lower()
        for tok in txt.split():
            hv[zlib.crc32(tok.encode()) % buckets] += 1.0
    return v + hv


def oof_hard_scores(train_graphs, seed=42, folds=3):
    """Train-only task-family OOF difficulty |p-y| from a cheap linear model."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = np.asarray([_featurize(g) for g in train_graphs], dtype=float)
    y = np.asarray([g['label'] for g in train_graphs])
    groups = [g['_family'] for g in train_graphs]
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = 0.5
            continue
        m = LogisticRegression(max_iter=500)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    diff = np.abs(oof - y)
    lo, hi = np.quantile(diff, 0.7), diff.max()
    span = max(hi - lo, 1e-9)
    return {g['task_id']: float(np.clip((diff[i] - lo) / span, 0, 1))
            for i, g in enumerate(train_graphs)}


def oof_family_out_prob(train_graphs, seed=42, folds=3):
    """Raw family-out OOF probabilities p(y=1|x) from a task-family GroupKFold
    linear model. This is the CL-R1 interface: probabilities, not difficulty."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = np.asarray([_featurize(g) for g in train_graphs], dtype=float)
    y = np.asarray([g['label'] for g in train_graphs])
    groups = [g['_family'] for g in train_graphs]
    oof = np.zeros(len(y))
    fold_of = {}
    gkf = GroupKFold(n_splits=folds)
    for fi, (tr, te) in enumerate(gkf.split(X, y, groups)):
        for i in te:
            fold_of[groups[i]] = fi
        if len(set(y[tr])) < 2:
            oof[te] = 0.5
            continue
        m = LogisticRegression(max_iter=1000)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return {g['task_id']: float(oof[i]) for i, g in enumerate(train_graphs)}


def cl02_residual_weights(oof_prob, labels, lam=2.0):
    """CL-R1 explicit formula (pending PI confirmation as the official CL02
    definition).

    r_FN(x) = y * (1 - p_oof)      # missed positives
    r_FP(x) = (1 - y) * p_oof      # false alarms
    w_plus(x)  = 1 + lam * r_FN(x)   -> applied to the positive-residual head
    w_minus(x) = 1 + lam * r_FP(x)   -> applied to the negative-residual head
    Weights are not normalized here (caller normalizes to mean 1 per batch).
    """
    w_plus, w_minus, r_fn, r_fp = [], [], [], []
    for p_oof, y in zip(oof_prob, labels):
        y = float(y)
        fn = y * (1.0 - p_oof)
        fp = (1.0 - y) * p_oof
        r_fn.append(fn)
        r_fp.append(fp)
        w_plus.append(1.0 + lam * fn)
        w_minus.append(1.0 + lam * fp)
    return {'w_plus': w_plus, 'w_minus': w_minus, 'r_fn': r_fn, 'r_fp': r_fp}


def specialty_table(train_graphs, seed=42):
    oof = oof_hard_scores(train_graphs, seed=seed)
    out = {}
    for g in train_graphs:
        s = observable_specialty(g)
        s['oof_hard'] = oof[g['task_id']]
        out[g['task_id']] = s
    return out


def coverage(table, graphs, key):
    vals = [table[g['task_id']][key] for g in graphs
            if table[g['task_id']].get(key) is not None]
    if not vals:
        return {'status': 'not_evaluable', 'n': 0}
    vals = sorted(vals)
    import numpy as np
    return {'status': 'evaluable', 'n': len(vals),
            'mean': float(np.mean(vals)),
            'frac_ge_0.5': float(np.mean([v >= 0.5 for v in vals])),
            'p90': float(np.quantile(vals, 0.9))}
