#!/usr/bin/env python
"""V10 confirmation analysis (unseen families).

Reads LOGS/v10_conf_<env>.json; uses the explicit `phase` field when present,
otherwise falls back to the preregistered plan positions:
  P1 = first 72 runs  (2 fam x 3 cond x 4 arms x 3 reps)
  P2 = next 24 runs   (2 fam x attack x {flat,seq,relation,verify} x 3 reps;
                        seq is a recency-only control)
  P3 = last 8 runs    (reasoner subset)
McNemar reports both one-sided and two-sided exact tails.
Writes LOGS/v10_confirmation_analysis.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import sys
REPO = Path('.')
PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'conf'
ENVS = ('shop', 'workspace', 'finance')


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def mcnemar(runs, arm_a, arm_b, cond, metric):
    key = {}
    for r in runs:
        if r['condition'] != cond or r['error']:
            continue
        key[(r['family'], r['repeat'], r['arm'])] = bool(r[metric])
    b = c = 0
    for k in key:
        if k[2] != arm_a:
            continue
        other = (k[0], k[1], arm_b)
        if other not in key:
            continue
        a, x = key[k], key[other]
        b += int(a and not x)
        c += int(not a and x)
    n = b + c
    if n == 0:
        return {'b': 0, 'c': 0, 'p_one_sided': None, 'p_two_sided': None,
                'p': None}
    k = min(b, c)
    one = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {'b': b, 'c': c, 'p_one_sided': one,
            'p_two_sided': min(1.0, 2 * one), 'p': one}


def summarize(runs):
    out = {}
    for cond in ('attack_instruction', 'benign_instruction',
                 'authorized_override'):
        out[cond] = {}
        arms = sorted({r['arm'] for r in runs if r['condition'] == cond})
        for arm in arms:
            rs = [r for r in runs if r['condition'] == cond and
                  r['arm'] == arm and not r['error']]
            n = len(rs)
            harm = sum(r['independent_harm'] for r in rs)
            comp = sum(r['independent_completion'] for r in rs)
            out[cond][arm] = {
                'n': n, 'harm': harm,
                'harm_rate': round(harm / n, 4) if n else None,
                'harm_wilson': wilson(harm, n),
                'completion': comp,
                'completion_rate': round(comp / n, 4) if n else None,
                'completion_wilson': wilson(comp, n)}
    return out


def main() -> int:
    out = {'phases': {}, 'per_env': {}, 'pooled': {}, 'mcnemar': {}}
    p1_all, p2_all, p3_all = [], [], []
    for env in ENVS:
        d = json.loads((REPO / f'LOGS/v10_{PREFIX}_{env}.json').read_text())
        runs = d['runs']
        if runs and runs[0].get('phase'):
            p1 = [r for r in runs if r.get('phase') == 'P1']
            p2 = [r for r in runs if r.get('phase') == 'P2']
            p3 = [r for r in runs if r.get('phase') == 'P3']
        else:
            p1, p2, p3 = runs[:72], runs[72:96], runs[96:]
        out['phases'][env] = {'p1': len(p1), 'p2': len(p2), 'p3': len(p3),
                              'rmb': d['ledger']['rmb']}
        out['per_env'][env] = {'P1': summarize(p1), 'P2': summarize(p2),
                               'P3': summarize(p3)}
        p1_all += p1
        p2_all += p2
        p3_all += p3
    out['pooled']['P1'] = summarize(p1_all)
    out['pooled']['P2'] = summarize(p2_all)
    out['pooled']['P3'] = summarize(p3_all)
    for other in ('none', 'strong', 'verify'):
        out['mcnemar'][f'attack_harm|{other}->verify_replan'] = mcnemar(
            p1_all, other, 'verify_replan', 'attack_instruction',
            'independent_harm')
        out['mcnemar'][f'benign_completion|{other}->verify_replan'] = mcnemar(
            p1_all, other, 'verify_replan', 'benign_instruction',
            'independent_completion')
    (REPO / f'LOGS/v10_{PREFIX}_analysis.json').write_text(
        json.dumps(out, indent=1))
    print(json.dumps({'phases': out['phases'],
                      'pooled_P1_attack': out['pooled']['P1'][
                          'attack_instruction'],
                      'pooled_P1_benign': out['pooled']['P1'][
                          'benign_instruction'],
                      'pooled_P2_attack': out['pooled']['P2'][
                          'attack_instruction'],
                      'mcnemar': out['mcnemar']}, indent=1)[:3500])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
