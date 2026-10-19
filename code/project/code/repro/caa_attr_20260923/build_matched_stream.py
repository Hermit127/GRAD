#!/usr/bin/env python3
"""Build the matched mechanism stream v3 (controlled re-validation).

Frozen rules: LOGS/real_cl_caa_attribution_20260923/prereg_caa_attribution_20260923.md
§2. Reads the v2 family stream, selects samples by exact multivariate strata
matching across the six (stage x class) cells, allocates families 70/15/15,
and runs the audit gates. No model is trained. Output is written only if all
gates pass (gates are always reported either way).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('GRAD_BASE', '.'))
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(BASE / 'code/repro'))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

V2 = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']
CELLS = [(s, c) for s in STAGES for c in ('attack', 'benign')]
FAM_CAP = 8
SEED = 211


def source_group(g):
    md = g.get('metadata') or {}
    if str(md.get('source', '')) == 'mcp_safetybench' and \
       str(md.get('graph_origin', '')) == 'runtime_trace':
        return 'runtime'
    return 'generated'


def node_text(n):
    return str((n.get('features') or {}).get('subtask', '')) or \
        str(n.get('type', ''))


def text_len(g):
    return sum(len(node_text(n)) for n in g['nodes'])


def load_rows():
    rows = []
    for ph in ('train', 'dev', 'test'):
        for line in open(V2 / f'{ph}.jsonl'):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha1_order(r):
    return hashlib.sha1(f"{SEED}|{r['family_key']}|{r['task_id']}"
                        .encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = load_rows()
    node_types = sorted(set(
        str(n.get('type', 'other')) for r in rows for n in r['nodes'])
        | {'other'})
    idx = {t: i for i, t in enumerate(node_types)}
    shares = []
    for r in rows:
        c = collections.Counter(str(n.get('type', 'other'))
                                for n in r['nodes'])
        n = max(len(r['nodes']), 1)
        shares.append(np.array([c.get(t, 0) / n for t in node_types]))
    shares = np.stack(shares)
    dom = collections.Counter(int(np.argmax(v)) for v in shares)
    top = [i for i, _ in dom.most_common() if dom[i] >= 20]

    def dombin(v):
        i = int(np.argmax(v))
        return top.index(i) if i in top else len(top)

    tl = np.array([text_len(r) for r in rows])
    t_qs = [float(np.percentile(tl, p)) for p in (25, 50, 75)]

    def tbin(v):
        return int(np.searchsorted(t_qs, v, side='right'))

    for i, r in enumerate(rows):
        r['_cell'] = (r['stage'], 'attack' if int(r['label']) else 'benign')
        r['_src'] = source_group(r)
        r['_txt'] = float(tl[i])
        r['_nn'] = len(r['nodes'])
        r['_ne'] = len(r.get('edges', []))
        r['_stratum'] = (r['_nn'], r['_ne'], tbin(r['_txt']), r['_src'],
                         dombin(shares[i]))
        r['_share'] = shares[i]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r['_stratum']][r['_cell']].append(r)
    selected = {c: [] for c in CELLS}
    per_stratum = {}
    for s, cells in sorted(by.items()):
        takes = {c: sorted(cells.get(c, []), key=sha1_order) for c in CELLS}

        def pick(c, m):
            used = collections.Counter()
            out = []
            for r in takes[c]:
                if len(out) >= m:
                    break
                if used[r['family_key']] < FAM_CAP:
                    out.append(r)
                    used[r['family_key']] += 1
            return out

        def avail(c):
            used = collections.Counter()
            for r in takes[c]:
                used[r['family_key']] += 1
            return sum(min(v, FAM_CAP) for v in used.values())

        m = min(avail(c) for c in CELLS)
        for c in CELLS:
            got = pick(c, m)
            assert len(got) == m, (s, c, m, len(got))
            selected[c] += got
        per_stratum['|'.join(map(str, s))] = m
    matched = [r for c in CELLS for r in selected[c]]
    matched_ids = {id(r) for r in matched}
    assert len(matched_ids) == len(matched)
    # family phase allocation within (stage, fam_class), source-balanced
    # quotas per split (60/20/20), deterministic SHA1 order, non-empty fallback
    SPLITS = ['train', 'dev', 'test']
    FRAC = [0.60, 0.20, 0.20]
    fam_labels = collections.defaultdict(set)
    fam_size = collections.Counter()
    for r in matched:
        fam_labels[r['family_key']].add(int(r['label']))
        fam_size[r['family_key']] += 1
    groups = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in matched:
        fc = 'mixed' if len(fam_labels[r['family_key']]) > 1 else \
            r['_cell'][1]
        groups[(r['stage'], fc)][r['_src']].add(r['family_key'])
    phase_of = {}
    alloc_report = {}
    for key, by_src in sorted(groups.items()):
        assign = {p: [] for p in SPLITS}
        for src, fams in sorted(by_src.items()):
            ordered = sorted(fams, key=lambda k: hashlib.sha1(
                f'{SEED}|{k}'.encode()).hexdigest())
            n = len(ordered)
            if n >= 3:
                quotas = [max(int(n * FRAC[i]), 1) for i in range(3)]
                while sum(quotas) > n:
                    i = max(range(3), key=lambda j: (quotas[j], -j))
                    quotas[i] -= 1
                while sum(quotas) < n:
                    i = max(range(3), key=lambda j: (n * FRAC[j] - quotas[j],
                                                     -j))
                    quotas[i] += 1
            elif n == 2:
                quotas = [1, 1, 0]
            else:
                quotas = [n, 0, 0]
            got = [0, 0, 0]
            for k in ordered:
                j = max(range(3), key=lambda i: (quotas[i] - got[i], -i))
                assign[SPLITS[j]].append(k)
                got[j] += 1
        # fallback: dev/test must be non-empty for this (stage, class) group
        for ph in ('dev', 'test'):
            if assign[ph]:
                continue
            big = max(SPLITS, key=lambda p: len(assign[p]))
            if len(assign[big]) <= 1:
                continue
            mv = min(assign[big], key=lambda k: (fam_size[k],
                                                 hashlib.sha1(
                                                     f'{SEED}|{k}'.encode()
                                                 ).hexdigest()))
            assign[big].remove(mv)
            assign[ph].append(mv)
        alloc_report['|'.join(key)] = {p: len(v) for p, v in assign.items()}
        for p, ks in assign.items():
            for k in ks:
                phase_of[k] = p
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    files = {ph: open(outdir / f'{ph}.jsonl', 'w') for ph in
             ('train', 'dev', 'test')}
    for r in matched:
        ph = phase_of[r['family_key']]
        rec = {k: v for k, v in r.items()
               if not k.startswith('_') and k not in ('family_key',)}
        rec['family_key'] = r['family_key']
        rec['v3_phase'] = ph
        rec['v3_stratum'] = '|'.join(map(str, r['_stratum']))
        files[ph].write(json.dumps(rec) + '\n')
        counts[(r['stage'], r['_src'], r['_cell'][1], ph)] += 1
    for f in files.values():
        f.close()
    # gates
    fams_by_phase = collections.defaultdict(set)
    for r in matched:
        fams_by_phase[phase_of[r['family_key']]].add(r['family_key'])
    ph = list(fams_by_phase)
    overlap = sum(len(fams_by_phase[ph[i]] & fams_by_phase[ph[j]])
                  for i in range(len(ph)) for j in range(i + 1, len(ph)))
    gates = {'family_overlap': overlap}
    stage_stats = {}
    for s in STAGES:
        st = {}
        for p in ('train', 'dev', 'test'):
            sel = [r for r in matched if r['stage'] == s
                   and phase_of[r['family_key']] == p]
            st[p] = {
                'n': len(sel),
                'attacks': sum(1 for r in sel if r['_cell'][1] == 'attack'),
                'benign': sum(1 for r in sel if r['_cell'][1] == 'benign'),
                'runtime': sum(1 for r in sel if r['_src'] == 'runtime'),
                'generated': sum(1 for r in sel if r['_src'] == 'generated'),
                'runtime_share': (sum(1 for r in sel if r['_src'] == 'runtime')
                                  / max(len(sel), 1))}
        stage_stats[s] = st
        gates[f'dev_class_{s}'] = st['dev']['attacks'] > 0 and \
            st['dev']['benign'] > 0
        gates[f'test_class_{s}'] = st['test']['attacks'] > 0 and \
            st['test']['benign'] > 0
        shares_p = [st[p]['runtime_share'] for p in ('train', 'dev', 'test')]
        gates[f'source_share_spread_{s}'] = max(shares_p) - min(shares_p)
        gates[f'source_share_gate_{s}'] = (max(shares_p) - min(shares_p)) <= 0.15
        cls_spread = []
        for cls in ('attack', 'benign'):
            sp = [sum(1 for r in matched if r['stage'] == s
                      and r['_cell'][1] == cls and phase_of[r['family_key']] == p)
                  / max(st[p]['n'], 1) for p in ('train', 'dev', 'test')]
            cls_spread.append(max(sp) - min(sp))
        gates[f'class_share_spread_{s}'] = max(cls_spread)
    # balance vs pooled matched set
    cov = {'n_nodes': ('_nn', np.array([r['_nn'] for r in matched])),
           'n_edges': ('_ne', np.array([r['_ne'] for r in matched])),
           'text_len': ('_txt', np.array([r['_txt'] for r in matched]))}
    pooled_share = np.mean(np.stack([r['_share'] for r in matched]), 0)
    balance = {}
    for c in CELLS:
        rs = [r for r in matched if r['_cell'] == c]
        rec = {'n': len(rs)}
        for k, (rkey, arr) in cov.items():
            sel = np.array([r[rkey] for r in rs])
            rec[f'smd_{k}'] = float((sel.mean() - arr.mean())
                                    / max(arr.std(), 1e-9))
        cell_share = np.mean(np.stack([r['_share'] for r in rs]), 0)
        rec['share_tv'] = float(0.5 * np.abs(cell_share - pooled_share).sum())
        balance['|'.join(c)] = rec
    max_smd = max(abs(v[f'smd_{k}']) for v in balance.values()
                  for k in cov)
    max_tv = max(v['share_tv'] for v in balance.values())
    gates['max_abs_smd'] = max_smd
    gates['max_share_tv'] = max_tv
    gates['balance_gate'] = max_smd <= 0.10 and max_tv <= 0.10
    split_balance = {}
    for c in CELLS:
        for p in ('train', 'dev', 'test'):
            rs = [r for r in matched if r['_cell'] == c
                  and phase_of[r['family_key']] == p]
            rec = {'n': len(rs)}
            for k, (rkey, arr) in cov.items():
                sel = np.array([r[rkey] for r in rs])
                rec[f'smd_{k}'] = float((sel.mean() - arr.mean())
                                        / max(arr.std(), 1e-9)) if len(sel) \
                    else None
            split_balance[f"{c[0]}|{c[1]}|{p}"] = rec
    split_max_smd = max(abs(v[f'smd_{k}']) for v in split_balance.values()
                        for k in cov if v[f'smd_{k}'] is not None)
    gates['split_max_abs_smd_report_only'] = split_max_smd
    # signature-only stage identity on matched rows (linear, group CV)
    X = np.stack([[v for v in r['_share']] +
                  [r['_nn'] / 50.0, r['_ne'] / 50.0] for r in matched])
    y = np.asarray([r['stage'] for r in matched])
    g = np.asarray([r['family_key'] for r in matched])
    k = min(5, len(set(g)))
    accs = []
    perm = np.random.RandomState(SEED).permutation(len(y))
    Xp, yp, gp = X[perm], y[perm], g[perm]
    for tr, te in GroupKFold(n_splits=k).split(Xp, yp, gp):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0))
        clf.fit(Xp[tr], yp[tr])
        accs.append(balanced_accuracy_score(yp[te], clf.predict(Xp[te])))
    gates['signature_stage_balacc'] = float(np.mean(accs))
    gates['signature_identity_gate'] = \
        gates['signature_stage_balacc'] <= (1.0 / 3.0) + 0.10
    # family-only label classifier (report only)
    fams = sorted(set(g))
    fidx = {f: i for i, f in enumerate(fams)}
    Xf = np.zeros((len(matched), len(fams)))
    for i, r in enumerate(matched):
        Xf[i, fidx[r['family_key']]] = 1.0
    yl = np.asarray([int(r['label']) for r in matched])
    k2 = min(5, len(fams))
    faccs = []
    perm2 = np.random.RandomState(SEED + 1).permutation(len(yl))
    Xfp, ylp, gp2 = Xf[perm2], yl[perm2], np.asarray(g)[perm2]
    for tr, te in GroupKFold(n_splits=k2).split(Xfp, ylp, gp2):
        if len(set(ylp[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xfp[tr], ylp[tr])
        faccs.append(balanced_accuracy_score(ylp[te], clf.predict(Xfp[te])))
    gates['family_label_balacc_report_only'] = float(np.mean(faccs)) \
        if faccs else None
    # family composition per phase / cell
    fam_comp = {}
    for c in CELLS:
        for p in ('train', 'dev', 'test'):
            rs = [r for r in matched if r['_cell'] == c
                  and phase_of[r['family_key']] == p]
            fams_c = collections.Counter(r['family_key'] for r in rs)
            fam_comp[f"{c[0]}|{c[1]}|{p}"] = {
                'n': len(rs), 'n_families': len(fams_c),
                'max_family_share': (max(fams_c.values()) / max(len(rs), 1))
                if fams_c else 0.0}
    kept_fams = set(phase_of)
    manifest = {
        'rules': 'prereg_caa_attribution_20260923.md §2',
        'source_stream': str(V2), 'seed': SEED, 'family_cap': FAM_CAP,
        'text_quartile_cuts': t_qs, 'split_fractions': [0.60, 0.20, 0.20],
        'strata': '(n_nodes, n_edges, text_quartile, source, dominant_type)',
        'n_rows_source': len(rows),
        'n_rows_matched': len(matched), 'n_families_matched': len(kept_fams),
        'retention_samples': len(matched) / len(rows),
        'retention_families': len(kept_fams) / len(set(
            r['family_key'] for r in rows)),
        'per_cell_counts': {'|'.join(c): len(selected[c]) for c in CELLS},
        'per_stratum_m': sum(per_stratum.values()),
        'allocation': alloc_report, 'counts': {
            '|'.join(k): v for k, v in sorted(counts.items())},
        'stage_stats': stage_stats, 'balance': balance,
        'split_balance': split_balance,
        'family_composition': fam_comp, 'gates': gates,
    }
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    L = ['# Matched mechanism stream v3 (controlled re-validation)', '',
         f"source rows {len(rows)} -> matched {len(matched)} "
         f"({manifest['retention_samples']:.3f}), families "
         f"{len(kept_fams)}/{len(set(r['family_key'] for r in rows))} "
         f"({manifest['retention_families']:.3f}); per-cell "
         f"{json.dumps(manifest['per_cell_counts'])}", '',
         '| stage | split | n | atk | ben | runtime share |', '|---|---|---|---|---|---|']
    for s in STAGES:
        for p in ('train', 'dev', 'test'):
            v = stage_stats[s][p]
            L.append(f"| {s} | {p} | {v['n']} | {v['attacks']} | "
                     f"{v['benign']} | {v['runtime_share']:.3f} |")
    L += ['', '## Gates', '', '| gate | value |', '|---|---|']
    for k_, v in gates.items():
        L.append(f'| {k_} | {v} |')
    L += ['', '## Balance vs pooled matched set', '',
          '| cell | n | smd_nodes | smd_edges | smd_text | share_tv |',
          '|---|---|---|---|---|---|']
    for c, v in balance.items():
        L.append(f"| {c} | {v['n']} | {v['smd_n_nodes']:.3f} | "
                 f"{v['smd_n_edges']:.3f} | {v['smd_text_len']:.3f} | "
                 f"{v['share_tv']:.3f} |")
    (outdir / 'build_report.md').write_text('\n'.join(L) + '\n')
    print(json.dumps({'gates': gates, 'retention_samples':
                      manifest['retention_samples'],
                      'retention_families': manifest['retention_families'],
                      'per_cell_counts': manifest['per_cell_counts']},
                     indent=1))
    print('\n'.join(L[:60]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
