"""Threshold selection, confusion metrics and grouped bootstrap (spec §7)."""
from __future__ import annotations

import math
import random
from collections import defaultdict


def select_threshold(scores: list[float], labels: list[int],
                     target_fpr: float = 0.10) -> dict:
    """§7 rule: candidates = unique finite score boundaries + (+inf).

    Among thresholds with FP <= floor(target_fpr * n_benign), maximize TPR;
    ties -> lower FPR, then higher threshold.  +inf is the report-nothing point.
    """
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)
             if not math.isnan(float(s))]
    if not pairs:
        return {'threshold': math.inf, 'status': 'no_scores',
                'budget': 0, 'fp': 0, 'tp': 0, 'n_benign': 0, 'n_pos': 0}
    labels_arr = [y for _, y in pairs]
    n_ben = sum(1 for y in labels_arr if y == 0)
    n_pos = sum(1 for y in labels_arr if y == 1)
    budget = math.floor(target_fpr * n_ben)
    if n_pos == 0:
        # Tiny-data adaptation (frozen rule, MINJA): benign-only calibration.
        # Most sensitive threshold with FP <= budget.
        neg = sorted(s for s, y in pairs if y == 0)
        k = min(budget, len(neg))
        if k >= len(neg):
            t = float('-inf')
        else:
            t = math.nextafter(neg[len(neg) - 1 - k], math.inf)
        fp = sum(1 for s in neg if s >= t)
        return {'threshold': float(t), 'status': 'ok_benign_only',
                'rule': 'benign_only_sensitive_v1', 'budget': budget,
                'fp': int(fp), 'tp': 0, 'n_benign': n_ben, 'n_pos': 0,
                'tpr': None, 'fpr': fp / n_ben if n_ben else float('nan'),
                'note': 'no positive calibration samples; most sensitive '
                        'threshold subject to the FP budget'}
    candidates = sorted({s for s, _ in pairs})
    candidates.append(math.inf)
    best = None
    for t in candidates:
        tp = sum(1 for s, y in pairs if y == 1 and s >= t)
        fp = sum(1 for s, y in pairs if y == 0 and s >= t)
        if fp > budget:
            continue
        cand = (tp, -fp, t)
        if best is None or cand > best[0]:
            best = (cand, t, tp, fp)
    if best is None:
        return {'threshold': math.inf, 'status': 'no_threshold_within_budget',
                'budget': budget, 'fp': 0, 'tp': 0, 'n_benign': n_ben,
                'n_pos': n_pos}
    _, t, tp, fp = best
    return {'threshold': float(t) if math.isfinite(t) else math.inf,
            'status': 'ok', 'budget': budget, 'fp': int(fp), 'tp': int(tp),
            'n_benign': n_ben, 'n_pos': n_pos,
            'tpr': tp / n_pos if n_pos else float('nan'),
            'fpr': fp / n_ben if n_ben else float('nan')}


def confusion(scores: list[float], labels: list[int], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for s, y in zip(scores, labels):
        if math.isnan(float(s)):
            continue
        pred = 1 if s >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 1 and pred == 0:
            fn += 1
        elif y == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'n_eval': n,
            'tpr': rec, 'fpr': fp / (fp + tn) if fp + tn else float('nan'),
            'precision': prec,
            'f1': 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def auroc(scores: list[float], labels: list[int]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)
             if not math.isnan(float(s))]
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1] == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def weighted_fold_auroc(fold_aucs: list[tuple[float, int, int]]) -> float:
    num = sum(a * p * n for a, p, n in fold_aucs
              if not math.isnan(a))
    den = sum(p * n for a, p, n in fold_aucs if not math.isnan(a))
    return num / den if den else float('nan')


def _resample_groups(records: list[dict], rng: random.Random) -> list[dict]:
    by_group = defaultdict(list)
    for r in records:
        by_group[r['group']].append(r)
    groups = sorted(by_group)
    picked = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
    out = []
    for gi, g in enumerate(picked):
        for r in by_group[g]:
            clone = dict(r)
            clone['_boot_group'] = gi
            out.append(clone)
    return out


def bootstrap_cv_auroc(records: list[dict], n_boot: int = 2000,
                       seed: int = 42) -> dict:
    """Grouped bootstrap within each outer fold, weighted by n_pos*n_neg."""
    folds = sorted({r['fold'] for r in records})
    per_fold = {f: [r for r in records if r['fold'] == f] for f in folds}
    weights = {}
    for f, rs in per_fold.items():
        p = sum(1 for r in rs if r['label'] == 1)
        n = len(rs) - p
        weights[f] = p * n
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        num = den = 0.0
        for f, rs in per_fold.items():
            sub = _resample_groups(rs, rng)
            a = auroc([r['score'] for r in sub], [r['label'] for r in sub])
            w = weights[f]
            if not math.isnan(a) and w:
                num += a * w
                den += w
        if den:
            vals.append(num / den)
    vals.sort()
    fold_aucs = [(auroc([r['score'] for r in rs], [r['label'] for r in rs]),
                  sum(1 for r in rs if r['label'] == 1),
                  len(rs) - sum(1 for r in rs if r['label'] == 1))
                 for rs in per_fold.values()]
    point = weighted_fold_auroc(fold_aucs)
    if not vals:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'n_valid': 0, 'point': point}
    return {'mean': sum(vals) / len(vals), 'lo': vals[int(0.025 * len(vals))],
            'hi': vals[min(int(0.975 * len(vals)), len(vals) - 1)],
            'n_valid': len(vals), 'point': point}


def paired_bootstrap_delta_multi(records_a_by_seed: list[list[dict]],
                                 records_b_by_seed: list[list[dict]],
                                 n_boot: int = 2000, seed: int = 42) -> dict:
    """Resample groups once per replicate; compute each seed's paired delta,
    then average across seeds (spec §7: no 3x independent-sample inflation)."""
    rng = random.Random(seed)
    by_seed = []
    for ra_list, rb_list in zip(records_a_by_seed, records_b_by_seed):
        map_b = {(r['group'], r['sample_id']): r for r in rb_list}
        by_group = defaultdict(list)
        for ra in ra_list:
            rb = map_b.get((ra['group'], ra['sample_id']))
            if rb is not None:
                by_group[ra['group']].append((ra, rb))
        by_seed.append(by_group)
    diffs = []
    for _ in range(n_boot):
        seed_deltas = []
        for by_group in by_seed:
            groups = sorted(by_group)
            a_s, a_y, b_s, b_y = [], [], [], []
            for _ in groups:
                g = groups[rng.randrange(len(groups))]
                for ra, rb in by_group[g]:
                    a_s.append(ra['score']); a_y.append(ra['label'])
                    b_s.append(rb['score']); b_y.append(rb['label'])
            aa = auroc(a_s, a_y)
            ab = auroc(b_s, b_y)
            if not math.isnan(aa) and not math.isnan(ab):
                seed_deltas.append(aa - ab)
        if seed_deltas:
            diffs.append(sum(seed_deltas) / len(seed_deltas))
    diffs.sort()
    if not diffs:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    return {'mean': sum(diffs) / len(diffs),
            'lo': diffs[int(0.025 * len(diffs))],
            'hi': diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)],
            'n_valid': len(diffs)}


def bootstrap_auroc(records: list[dict], n_boot: int = 2000,
                    seed: int = 42) -> dict:
    """records: {group, label, score}. Resample original groups with replacement."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        rs = _resample_groups(records, rng)
        a = auroc([r['score'] for r in rs], [r['label'] for r in rs])
        if not math.isnan(a):
            vals.append(a)
    vals.sort()
    if not vals:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'n_valid': 0}
    return {'mean': sum(vals) / len(vals), 'lo': vals[int(0.025 * len(vals))],
            'hi': vals[min(int(0.975 * len(vals)), len(vals) - 1)],
            'n_valid': len(vals), 'point': auroc(
                [r['score'] for r in records], [r['label'] for r in records])}


def paired_bootstrap_delta(records_a: list[dict], records_b: list[dict],
                           n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired by original group; both methods evaluated on the same resample."""
    key = lambda r: (r['group'], r.get('sample_id'))
    map_b = {key(r): r for r in records_b}
    joined = [(r, map_b[key(r)]) for r in records_a if key(r) in map_b]
    by_group = defaultdict(list)
    for ra, rb in joined:
        by_group[ra['group']].append((ra, rb))
    groups = sorted(by_group)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        a_scores, a_labels, b_scores, b_labels = [], [], [], []
        for _ in groups:
            g = groups[rng.randrange(len(groups))]
            for ra, rb in by_group[g]:
                a_scores.append(ra['score']); a_labels.append(ra['label'])
                b_scores.append(rb['score']); b_labels.append(rb['label'])
        aa = auroc(a_scores, a_labels)
        ab = auroc(b_scores, b_labels)
        if not math.isnan(aa) and not math.isnan(ab):
            diffs.append(aa - ab)
    diffs.sort()
    if not diffs:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    return {'mean': sum(diffs) / len(diffs),
            'lo': diffs[int(0.025 * len(diffs))],
            'hi': diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)],
            'n_valid': len(diffs),
            'point': auroc([r['score'] for r in records_a],
                           [r['label'] for r in records_a]) -
                     auroc([r['score'] for r in records_b],
                           [r['label'] for r in records_b])}
