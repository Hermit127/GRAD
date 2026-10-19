"""CL-R2/R3 corrected statistics for the CL pilot/confirmation reports.

Design (dual_review_20260810 CL-R2/R3):
  - Point estimates come from the raw per-example rows (never from bootstrap
    means).
  - One family resample per replicate is shared by all methods and seeds;
    each replicate produces one complete macro delta (seed/order are fixed
    repeated measurements averaged inside the replicate).
  - Descriptive CIs are percentile intervals of that bootstrap distribution.
  - Significance uses a paired cluster permutation test (family-level sign
    exchange), not the bootstrap proportion of positive deltas.
  - Non-inferiority uses a one-sided 95% upper bound on the macro FPR delta.
  - Holm correction is really applied across the declared comparison family.
  - Domains with fewer than 2 independent families are flagged; cross-family
    inference is not claimed for them.

Rows per method/seed/order are dictionaries with keys:
  domain, task_id, family, label, decision ('block'/'allow').
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DOMAINS = ['financial_analysis', 'browser_automation', 'repository_management']
from . import protocol as P
P_ORDERS = P.ORDERS
REQUIRED_ARTIFACTS = ['stage_lineage.json', 'training_ids.jsonl',
                      'budget_ledger.json', 'calibration.json',
                      'predictions.jsonl', 'metrics.json']
VALID_KEYS = {(211, 'forward'), (211, 'reverse'), (212, 'forward'),
              (212, 'reverse')}


def _final_stage(rows):
    """Final stage rows, last occurrence per (domain, task_id) wins.

    Predictions are appended round-by-round (CL09) and stage-by-stage, so the
    last occurrence is the final snapshot. Rows written before the explicit
    `round` field existed are handled the same way."""
    max_stage = max(r.get('stage', 0) for r in rows)
    out = {}
    for r in rows:
        if r.get('stage', 0) != max_stage:
            continue
        out[(r['domain'], r['task_id'])] = r
    return list(out.values())


def collect_dev_rows(run_root, methods=None, protocol='antibody_cl_v2_P2.2'):
    """Load final-stage router_dev rows from run dirs + validity metadata."""
    run_root = Path(run_root)
    rows = defaultdict(list)      # (method, seed, order) -> rows
    validity = {}
    for d in sorted(run_root.glob('*')):
        mp = d / 'metrics.json'
        pp = d / 'predictions.jsonl'
        if not mp.exists() or not pp.exists():
            continue
        m = json.load(open(mp))
        run = m['run']
        cfg = run.get('config') or {}
        strategy = m.get('strategy') or cfg.get('strategy')
        weight_mode = cfg.get('weight_mode')
        mode = m.get('mode') or cfg.get('mode')
        suffix = strategy or mode
        method = run['method']
        if weight_mode and run['method'] == 'CL02v2':
            method += f':{weight_mode}'
        elif suffix and run['method'] in ('CL04', 'CL05', 'CL07', 'CL10'):
            method += f':{suffix}'
        if methods and method.split(':')[0] not in methods and \
                method not in methods:
            continue
        key = (method, run['seed'], run['order'])
        if key in rows:
            validity.setdefault('duplicate_keys', []).append(list(key))
            continue
        pr = [json.loads(l) for l in open(pp)]
        pr = [r for r in pr if r.get('split') == 'router_dev']
        if not pr:
            validity.setdefault('missing_rows', []).append(list(key))
            continue
        # C-R22: duplicates on the full (stage, round, domain, task) key must
        # be detected BEFORE selecting the final round
        raw_cnt = Counter((r.get('stage'), r.get('round'), r['domain'],
                           r['task_id']) for r in pr)
        raw_dups = [list(x) for x, c in raw_cnt.items() if c > 1]
        final = _final_stage(pr)
        # manifest check: final router_dev IDs must match split_ids.json
        manifest_ok = True
        manifest_detail = {}
        sid_path = d / 'split_ids.json'
        if sid_path.exists():
            sid = json.load(open(sid_path))
            if 'splits' in sid:
                sid = sid['splits']
            for dom, bysplit in sid.items():
                want = sorted(bysplit.get('router_dev', []))
                got = sorted(r['task_id'] for r in final
                             if r['domain'] == dom)
                if want != got:
                    manifest_ok = False
                    manifest_detail[dom] = {'expected': len(want),
                                            'got': len(got)}
        else:
            manifest_ok = False
            manifest_detail['split_ids.json'] = 'missing'
        model_hash = run.get('model_hash')
        vocab_hash = run.get('vocab_hash')
        rows[key] = [{'domain': r['domain'], 'task_id': r['task_id'],
                      'family': r.get('family_id', r['task_id']),
                      'label': r['label'], 'decision': r['decision'],
                      'status': r.get('status', 'ok'),
                      'score': r.get('final_score')}
                     for r in final]
        validity[str(key)] = {
            'protocol_version': m.get('protocol_version',
                                      run.get('protocol_version')),
            'artifacts_ok': all((d / f).exists() for f in REQUIRED_ARTIFACTS),
            'n_rows': len(final),
            'raw_duplicates': raw_dups[:10],
            'n_raw_duplicates': len(raw_dups),
            'manifest_match': manifest_ok,
            'manifest_detail': manifest_detail,
            'model_hash_ok': bool(model_hash and len(str(model_hash)) >= 8),
            'vocab_hash_ok': bool(vocab_hash and len(str(vocab_hash)) >= 8)}
    return rows, validity


def collect_confirm_rows(confirm_json):
    """Rows from confirm_p22/confirmation.json {'rows': {method:{seed:[...]}}}."""
    d = json.load(open(confirm_json))['rows']
    rows = {}
    for method, seeds in d.items():
        for seed, rs in seeds.items():
            rows[(method, int(seed), 'forward')] = [
                {'domain': r['domain'], 'task_id': r['task_id'],
                 'family': r['family'], 'label': r['label'],
                 'decision': r['decision']} for r in rs]
    return rows, {}


DECISIONS = {'block', 'allow'}
OK_STATUS = {'ok'}


def integrity_problems(rows, keys):
    """Exact per-key ID multiset + value legality checks (review C1)."""
    problems = {'duplicate_ids': [], 'missing_ids': [], 'extra_ids': [],
                'illegal_values': []}
    per_key = {}
    for k in keys:
        cnt = Counter((r['domain'], r['task_id']) for r in rows[k])
        dups = [list(x) for x, c in cnt.items() if c > 1]
        if dups:
            problems['duplicate_ids'].append({'key': list(k),
                                              'examples': dups[:5]})
        per_key[k] = set(cnt)
        for r in rows[k]:
            if r.get('decision') not in DECISIONS:
                problems['illegal_values'].append(
                    {'key': list(k), 'task_id': r['task_id'],
                     'decision': r.get('decision')})
            if r.get('status', 'ok') not in OK_STATUS:
                problems['illegal_values'].append(
                    {'key': list(k), 'task_id': r['task_id'],
                     'status': r.get('status')})
            sc = r.get('score')
            if sc is None or not np.isfinite(sc):
                problems['illegal_values'].append(
                    {'key': list(k), 'task_id': r['task_id'], 'score': sc})
            if r.get('label') not in (0, 1):
                problems['illegal_values'].append(
                    {'key': list(k), 'task_id': r['task_id'],
                     'label': r.get('label')})
    if keys:
        ref = per_key[keys[0]]
        for k in keys[1:]:
            miss = sorted(ref - per_key[k])
            extra = sorted(per_key[k] - ref)
            if miss:
                problems['missing_ids'].append(
                    {'key': list(k), 'n': len(miss), 'examples':
                     [list(x) for x in miss[:5]]})
            if extra:
                problems['extra_ids'].append(
                    {'key': list(k), 'n': len(extra), 'examples':
                     [list(x) for x in extra[:5]]})
    problems['n_problems'] = sum(len(v) for v in problems.values())
    return problems


def _newest_domain(order_name):
    return P_ORDERS[order_name][-1]


def newest_domain_stats(rows, candidate, baseline, n_boot=10000, seed=42,
                        alpha=0.05):
    """Newest-domain TPR delta: real per-order domains, shared family draws.

    Object of each draw: resample families within the newest domain (shared
    across seeds and both methods), compute the per-seed TPR delta, average
    the seeds -> one complete draw. `per_order` estimates and the aggregate
    over orders are reported separately. >=10,000 valid draws are required;
    NaN / missing class / no valid draw never pass non-inferiority.
    """
    cm = {(k[1], k[2]): k for k in rows if k[0] == candidate}
    bm = {(k[1], k[2]): k for k in rows if k[0] == baseline}
    orders = sorted({k[1] for k in cm if k in bm})
    if not orders:
        return {'status': 'no_paired_keys'}

    def domain_tpr(dec, labels, idx):
        lab = labels[idx]
        pos = lab == 1
        if not pos.any():
            return None
        return dec[idx][pos].mean()

    per_order = {}
    order_draws = {}
    for order in orders:
        d = _newest_domain(order)
        seeds = [k[0] for k in cm if k[1] == order and k in bm]
        if not seeds:
            continue
        arr = {}
        ok = True
        for s in seeds:
            k = (s, order)
            ca = _domain_arrays({cm[k]: rows[cm[k]]}, [cm[k]])
            ba = _domain_arrays({bm[k]: rows[bm[k]]}, [bm[k]])
            if d not in ca or d not in ba:
                ok = False
                break
            a, b = ca[d], ba[d]
            arr[s] = {'a': a, 'b': b}
        if not ok:
            continue
        pvals = []
        for s in arr:
            tc = domain_tpr(arr[s]['a']['dec'][0], arr[s]['a']['labels'],
                            np.arange(len(arr[s]['a']['labels'])))
            tb = domain_tpr(arr[s]['b']['dec'][0], arr[s]['b']['labels'],
                            np.arange(len(arr[s]['b']['labels'])))
            if tc is not None and tb is not None:
                pvals.append(tc - tb)
        if not pvals:
            per_order[order] = {'status': 'degenerate_no_class',
                                'n_valid_draws': 0, 'point_dTPR': None,
                                'lower_95': None,
                                'non_inferior_1pp': False}
            continue
        point = float(np.mean(pvals))
        a0 = arr[seeds[0]]['a']
        fams = sorted(set(a0['fid']))
        by_fam = {f: np.where(a0['fid'] == f)[0] for f in fams}
        rng = np.random.default_rng(seed + 13)
        draws = []
        for _ in range(n_boot):
            pick = rng.choice(len(fams), size=len(fams), replace=True)
            idx = np.concatenate([by_fam[fams[p]] for p in pick])
            vals = []
            valid = True
            for s in arr:
                tc = domain_tpr(arr[s]['a']['dec'][0], arr[s]['a']['labels'],
                                idx)
                tb = domain_tpr(arr[s]['b']['dec'][0], arr[s]['b']['labels'],
                                idx)
                if tc is None or tb is None:
                    valid = False
                    break
                vals.append(tc - tb)
            if valid:
                draws.append(float(np.mean(vals)))
        if len(draws) < n_boot:
            per_order[order] = {'status': 'degenerate_no_class',
                                'n_valid_draws': len(draws),
                                'point_dTPR': point, 'lower_95': None,
                                'non_inferior_1pp': False}
            continue
        order_draws[order] = np.asarray(draws)
        per_order[order] = {
            'status': 'evaluable', 'newest_domain': d,
            'point_dTPR': point, 'n_valid_draws': len(draws),
            'lower_95': float(np.quantile(draws, alpha)),
            'non_inferior_1pp': bool(np.quantile(draws, alpha) >= -0.01)}
    if not order_draws:
        return {'status': 'degenerate_no_class', 'per_order': per_order,
                'non_inferior_1pp': False}
    # aggregate: average per-order draws in the same replicate index
    m = min(len(v) for v in order_draws.values())
    agg = np.mean(np.stack([order_draws[o][:m] for o in sorted(order_draws)]),
                  axis=0)
    point_agg = float(np.mean([per_order[o]['point_dTPR']
                               for o in sorted(order_draws)]))
    return {'status': 'evaluable',
            'newest_domain_per_order': {o: per_order[o]['newest_domain']
                                        for o in order_draws},
            'per_order': per_order,
            'point_dTPR': point_agg,
            'n_valid_draws': int(m),
            'lower_95': float(np.quantile(agg, alpha)),
            'non_inferior_1pp': bool(np.quantile(agg, alpha) >= -0.01),
            'one_sided_bound_note': f'{int((1-alpha)*100)}% one-sided lower '
                                    'bound from shared family draws'}


def _domain_arrays(rows, keys):
    """Per domain: labels, family ids, and decision matrix [n_keys, n]."""
    by_domain = defaultdict(lambda: defaultdict(dict))
    for ki, k in enumerate(keys):
        for r in rows[k]:
            by_domain[r['domain']][r['task_id']][ki] = r
    out = {}
    for d in DOMAINS:
        items = by_domain.get(d, {})
        tids = sorted(items)
        if not tids:
            continue
        labels = np.asarray([next(iter(items[t].values()))['label']
                             for t in tids])
        fams = [next(iter(items[t].values()))['family'] for t in tids]
        fam_idx = {f: i for i, f in enumerate(sorted(set(fams)))}
        fid = np.asarray([fam_idx[f] for f in fams])
        dec = np.zeros((len(keys), len(tids)))
        for ki, k in enumerate(keys):
            for ti, t in enumerate(tids):
                rr = items[t].get(ki)
                if rr is None:
                    raise ValueError(
                        f'missing prediction for key {k} task {t}; refusing '
                        f'to fill allow (review C1)')
                dec[ki, ti] = 1.0 if rr['decision'] == 'block' else 0.0
        out[d] = {'labels': labels, 'fid': fid, 'dec': dec,
                  'n_families': len(fam_idx), 'n': len(tids)}
    return out


def macro_tpr_fpr(dm, decision_index):
    """Macro TPR/FPR for one key index, given domain arrays."""
    tprs, fprs = [], []
    for d in DOMAINS:
        if d not in dm:
            continue
        a = dm[d]
        dec = a['dec'][decision_index]
        pos = a['labels'] == 1
        neg = ~pos
        tprs.append(dec[pos].mean() if pos.any() else np.nan)
        fprs.append(dec[neg].mean() if neg.any() else np.nan)
    return float(np.nanmean(tprs)), float(np.nanmean(fprs))


def _draw_indices(dm, rng):
    """One shared family resample per domain; returns folded index vectors."""
    out = {}
    for d, a in dm.items():
        n_fam = a['n_families']
        pick = rng.integers(0, n_fam, size=n_fam)
        # map sampled family slots back to row indices
        by_fam = defaultdict(list)
        for i, f in enumerate(a['fid']):
            by_fam[f].append(i)
        idx = np.concatenate([np.asarray(by_fam[p], dtype=int) for p in pick])
        out[d] = idx
    return out


def _metric_from_idx(a, dec_row, idx):
    lab = a['labels'][idx]
    dec = dec_row[idx]
    pos = lab == 1
    tpr = dec[pos].mean() if pos.any() else np.nan
    fpr = dec[~pos].mean() if (~pos).any() else np.nan
    return tpr, fpr


def paired_stats(rows, candidate, baseline, n_boot=10000, n_perm=10000,
                 seed=42, alpha=0.05):
    """Point estimates + shared-family bootstrap CI + permutation p-values.

    keys used = intersection of valid (seed,order) present for both methods;
    seed/order are fixed repeated measurements averaged inside each draw.
    """
    cand_keys = sorted([k for k in rows if k[0] == candidate])
    base_keys = sorted([k for k in rows if k[0] == baseline])
    base_map = {(k[1], k[2]): k for k in base_keys}
    pairs = [(k, base_map[(k[1], k[2])]) for k in cand_keys
             if (k[1], k[2]) in base_map]
    if not pairs:
        return {'status': 'no_paired_keys'}
    probs = integrity_problems(rows, [k for k, _ in pairs] +
                               [bk for _, bk in pairs])
    if probs['n_problems'] > 0:
        return {'status': 'invalid_rows', 'problems': probs}
    dm_c = _domain_arrays({k: rows[k] for k, _ in pairs},
                          [k for k, _ in pairs])
    dm_b = _domain_arrays({bk: rows[bk] for _, bk in pairs},
                          [bk for _, bk in pairs])
    # point estimates from raw samples
    pt_c = [macro_tpr_fpr(dm_c, i) for i in range(len(pairs))]
    pt_b = [macro_tpr_fpr(dm_b, i) for i in range(len(pairs))]
    obs_tpr = float(np.mean([c[0] - b[0] for c, b in zip(pt_c, pt_b)]))
    obs_fpr = float(np.mean([c[1] - b[1] for c, b in zip(pt_c, pt_b)]))
    # shared-family bootstrap
    rng = np.random.default_rng(seed)
    draws_t, draws_f, valid = [], [], 0
    dom_index = {}
    for d, a in dm_c.items():
        by_fam = defaultdict(list)
        for i, f in enumerate(a['fid']):
            by_fam[f].append(i)
        dom_index[d] = {f: np.asarray(v, dtype=int) for f, v in by_fam.items()}
    for _ in range(n_boot):
        idx_by_dom = {}
        ok = True
        for d in dm_c:
            fams = sorted(dom_index[d])
            pick = rng.choice(len(fams), size=len(fams), replace=True)
            idx = np.concatenate([dom_index[d][fams[p]] for p in pick])
            lab = dm_c[d]['labels'][idx]
            if not (lab == 1).any() or not (lab == 0).any():
                ok = False
                break
            idx_by_dom[d] = idx
        if not ok:
            continue
        valid += 1
        tc, fc, tb, fb = [], [], [], []
        for i in range(len(pairs)):
            tpr_c, fpr_c, tpr_b, fpr_b = [], [], [], []
            for d in dm_c:
                idx = idx_by_dom[d]
                t, f = _metric_from_idx(dm_c[d], dm_c[d]['dec'][i], idx)
                tpr_c.append(t); fpr_c.append(f)
                t, f = _metric_from_idx(dm_b[d], dm_b[d]['dec'][i], idx)
                tpr_b.append(t); fpr_b.append(f)
            tc.append(np.nanmean(tpr_c)); fc.append(np.nanmean(fpr_c))
            tb.append(np.nanmean(tpr_b)); fb.append(np.nanmean(fpr_b))
        draws_t.append(np.mean(np.asarray(tc) - np.asarray(tb)))
        draws_f.append(np.mean(np.asarray(fc) - np.asarray(fb)))
    dt = np.asarray(draws_t)
    df = np.asarray(draws_f)
    if len(dt) == 0 or len(df) == 0:
        return {'status': 'degenerate_no_valid_draws',
                'n_boot_valid': valid, 'n_paired_keys': len(pairs)}
    # paired cluster permutation (family-level sign exchange, shared per seed)
    rngp = np.random.default_rng(seed + 1)
    perm_t, perm_f = [], []
    total_fams = sum(dm_c[d]['n_families'] for d in dm_c)
    exhaustive = total_fams <= 12
    # precompute per-family sufficient stats per seed
    fam_stats = {}
    for ci, (k, bk) in enumerate(pairs):
        for d in dm_c:
            a = dm_c[d]
            for f in range(a['n_families']):
                m = a['fid'] == f
                pos = m & (a['labels'] == 1)
                neg = m & (a['labels'] == 0)
                fam_stats.setdefault((ci, d, f), {})
                fam_stats[(ci, d, f)]['tp_c'] = float(
                    a['dec'][ci][pos].sum()) if pos.any() else 0.0
                fam_stats[(ci, d, f)]['n_pos'] = float(pos.sum())
            b = dm_b[d]
            for f in range(b['n_families']):
                m = b['fid'] == f
                pos = m & (b['labels'] == 1)
                neg = m & (b['labels'] == 0)
                fam_stats[(ci, d, f)]['tp_b'] = float(
                    b['dec'][ci][pos].sum()) if pos.any() else 0.0
                fam_stats[(ci, d, f)]['fp_c'] = float(
                    a['dec'][ci][neg].sum()) if neg.any() else 0.0
                fam_stats[(ci, d, f)]['fp_b'] = float(
                    b['dec'][ci][neg].sum()) if neg.any() else 0.0
                fam_stats[(ci, d, f)]['n_neg'] = float(neg.sum())
    fam_counts = {(d, f): (dm_c[d]['fid'] == f).sum() for d in dm_c
                  for f in range(dm_c[d]['n_families'])}
    if exhaustive:
        import itertools
        doms = sorted(dm_c)
        counts = [dm_c[d]['n_families'] for d in doms]
        sign_iter = itertools.product([-1.0, 1.0], repeat=sum(counts))
    else:
        doms = sorted(dm_c)
        counts = [dm_c[d]['n_families'] for d in doms]
        sign_iter = None
    n_draws = (2 ** sum(counts)) if exhaustive else n_perm
    for draw_i in range(n_draws):
        if exhaustive:
            flat = next(sign_iter)
            signs = {}
            off = 0
            for d, c in zip(doms, counts):
                signs[d] = np.asarray(flat[off:off + c], dtype=float)
                off += c
        else:
            signs = {d: rngp.choice([-1.0, 1.0], size=dm_c[d]['n_families'])
                     for d in dm_c}
        t_pr, f_pr = [], []
        for ci in range(len(pairs)):
            tprs, fprs = [], []
            for d in dm_c:
                tp_diff = fp_diff = n_pos = n_neg = 0.0
                for f in range(dm_c[d]['n_families']):
                    s = signs[d][f]
                    st = fam_stats[(ci, d, f)]
                    tp_diff += s * (st['tp_c'] - st['tp_b'])
                    fp_diff += s * (st['fp_c'] - st['fp_b'])
                    n_pos += st['n_pos']
                    n_neg += st['n_neg']
                if n_pos:
                    tprs.append(tp_diff / n_pos)
                if n_neg:
                    fprs.append(fp_diff / n_neg)
            t_pr.append(np.mean(tprs)); f_pr.append(np.mean(fprs))
        perm_t.append(np.mean(t_pr)); perm_f.append(np.mean(f_pr))
    pt = np.asarray(perm_t)
    pf = np.asarray(perm_f)
    if exhaustive:
        # exact randomization p (identity included), no +1 correction
        p_tpr = float(np.mean(pt >= obs_tpr - 1e-15))
        p_fpr = float(np.mean(pf >= obs_fpr - 1e-15))
    else:
        # valid Monte-Carlo p with +1 correction
        p_tpr = float((1 + np.sum(pt >= obs_tpr)) / (1 + len(pt)))
        p_fpr = float((1 + np.sum(pf >= obs_fpr)) / (1 + len(pf)))
    return {
        'status': 'evaluable',
        'n_paired_keys': len(pairs),
        'point': {'dTPR': obs_tpr, 'dFPR': obs_fpr},
        'ci': {'dTPR_95': [float(np.quantile(dt, alpha / 2)),
                           float(np.quantile(dt, 1 - alpha / 2))],
               'dFPR_95': [float(np.quantile(df, alpha / 2)),
                           float(np.quantile(df, 1 - alpha / 2))]},
        'significance': {'perm_p_one_sided_tpr_gt_0': p_tpr,
                         'perm_p_one_sided_fpr_gt_0': p_fpr,
                         'n_perm': len(pt),
                         'perm_mode': 'exhaustive' if exhaustive
                                      else 'monte_carlo_plus1',
                         'exchangeability_note':
                             'family-level sign exchange assumes paired '
                             'families are exchangeable under the sharp null; '
                             'with <2 families per domain the test is '
                             'degenerate and flagged'},
        'non_inferiority': {
            'dFPR_upper_95': float(np.quantile(df, 0.95)),
            'pass_le_1pp': bool(np.quantile(df, 0.95) <= 0.01)},
        'bootstrap': {'n_boot_requested': n_boot, 'n_boot_valid': valid},
        'domains': {d: {'n_families': dm_c[d]['n_families'],
                        'n_rows': int(dm_c[d]['n']),
                        'cross_family_supported': dm_c[d]['n_families'] > 1}
                    for d in dm_c},
    }


def holm(pvals, alpha=0.05):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev_reject = {}, True
    for rank, (name, p) in enumerate(items):
        thresh = alpha / (m - rank)
        reject = prev_reject and (p <= thresh)
        out[name] = {'p': p, 'threshold': thresh, 'reject': bool(reject)}
        prev_reject = reject
    return out


def comparisons_table(rows, candidate, baselines, n_boot=10000, n_perm=10000,
                      seed=42):
    stats = {b: paired_stats(rows, candidate, b, n_boot=n_boot,
                             n_perm=n_perm, seed=seed) for b in baselines}
    pvals = {b: s['significance']['perm_p_one_sided_tpr_gt_0']
             for b, s in stats.items()
             if s.get('status') == 'evaluable'}
    return {'stats': stats, 'holm': holm(pvals) if pvals else {}}
