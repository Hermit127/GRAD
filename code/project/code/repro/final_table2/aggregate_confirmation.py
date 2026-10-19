#!/usr/bin/env python3
"""Aggregate the frozen confirmation batch (None arm) - no API, read-only.

Reports per suite and overall: ASR/BU/UA with Wilson 95% CIs, payload delivery
rate, error count, distinct successful tasks, and breakdowns by family,
injection task and vector type. Main table keeps every frozen case; a clearly
labelled conditional table restricts to cases that failed in the None arm.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def vector_of(case):
    return ','.join(sorted(case.get('injections', {}).keys())) or 'official'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    results = json.loads(Path(args.results).read_text())
    cases = {(c['suite'], c['user_task'], c['injection_task'] or '-',
              c.get('strategy', '')): c for c in manifest['cases']}
    runs = results['runs']
    by_key = {}
    for r in runs:
        key = (r['suite'], r['user_task'], r['injection_task'] or '-',
               r.get('strategy', ''))
        by_key[key] = r

    suites = sorted({c['suite'] for c in manifest['cases']})
    out_rows = []

    def block(rs):
        atk = [r for r in rs if r.get('injection_task')]
        ben = [r for r in rs if not r.get('injection_task')]
        k = sum(1 for r in atk if r.get('security') is True)
        bu = sum(1 for r in ben if r.get('utility') is True)
        ua = sum(1 for r in atk if r.get('utility') is True)
        deliv = sum(1 for r in atk if r.get('payload_delivered_in_run'))
        err = sum(1 for r in rs if r.get('error'))
        tasks = {(r['suite'], r['user_task']) for r in atk
                 if r.get('security') is True}
        return {
            'attack_n': len(atk), 'attack_success': k,
            'asr': round(k / len(atk), 4) if atk else None,
            'asr_ci95': wilson(k, len(atk)),
            'benign_n': len(ben), 'bu': bu,
            'bu_rate': round(bu / len(ben), 4) if ben else None,
            'bu_ci95': wilson(bu, len(ben)),
            'ua': ua,
            'ua_rate': round(ua / len(atk), 4) if atk else None,
            'ua_ci95': wilson(ua, len(atk)),
            'delivery_rate': round(deliv / len(atk), 4) if atk else None,
            'errors': err, 'successful_tasks': len(tasks),
        }

    for suite in suites + ['ALL']:
        rs = [r for r in runs
              if (suite == 'ALL' or r['suite'] == suite)]
        expected = [c for c in manifest['cases']
                    if suite == 'ALL' or c['suite'] == suite]
        b = block(rs)
        b['suite'] = suite
        b['planned_runs'] = len(expected)
        b['completed_runs'] = len(rs)
        out_rows.append(b)

    families = defaultdict(list)
    for key, r in by_key.items():
        c = cases.get(key)
        if c and c.get('injection_task'):
            families[c.get('family', '?')].append(r)
    family_rows = {f: block(rs) for f, rs in sorted(families.items())}

    inj_rows = defaultdict(list)
    for key, r in by_key.items():
        c = cases.get(key)
        if c and c.get('injection_task'):
            inj_rows[(c['suite'], c['injection_task'])].append(r)
    per_inj = {f'{s}|{i}': block(rs) for (s, i), rs in sorted(inj_rows.items())}

    succ = [r['case_id'] for r in runs
            if r.get('security') is True and r.get('injection_task')]

    md = ['# Confirmation batch 1 (None arm) - aggregation', '',
          f"manifest: `{Path(args.manifest).name}`  ",
          f"results: `{Path(args.results).name}`  ",
          f"runs completed: {len(runs)} / {len(manifest['cases'])}", '',
          '| suite | planned | completed | attack n | ASR | ASR 95% CI | '
          'benign n | BU | BU 95% CI | UA | delivery | errors | succ tasks |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for b in out_rows:
        md.append(
            f"| {b['suite']} | {b['planned_runs']} | {b['completed_runs']} | "
            f"{b['attack_n']} | {b['asr']} | {b['asr_ci95']} | {b['benign_n']} "
            f"| {b['bu_rate']} | {b['bu_ci95']} | {b['ua_rate']} | "
            f"{b['delivery_rate']} | {b['errors']} | {b['successful_tasks']} |")
    md += ['', '## By family', '',
           '| family | attack n | ASR | ASR 95% CI | delivery | errors |',
           '|---|---|---|---|---|---|']
    for f, b in family_rows.items():
        md.append(f"| {f} | {b['attack_n']} | {b['asr']} | {b['asr_ci95']} | "
                  f"{b['delivery_rate']} | {b['errors']} |")
    md += ['', '## By suite x injection task', '',
           '| cell | attack n | ASR | ASR 95% CI |', '|---|---|---|---|']
    for cell, b in per_inj.items():
        md.append(f"| {cell} | {b['attack_n']} | {b['asr']} | {b['asr_ci95']} |")
    md += ['', '## Successful cases', '', *(succ or ['(none)']), '']
    if succ:
        md += ['> Conditional analysis only: the main table keeps all frozen '
               'cases. The subset below consists of cases attacked successfully '
               'in the None arm and must be labelled as such.', '']
        md += ['| case |', '|---|'] + [f'| {s} |' for s in succ]

    Path(args.out).write_text('\n'.join(md) + '\n')
    json_out = {'manifest': Path(args.manifest).name,
                'results': Path(args.results).name,
                'rows': out_rows, 'families': family_rows,
                'per_injection': per_inj, 'successes': succ}
    Path(args.out.replace('.md', '.json')).write_text(
        json.dumps(json_out, indent=1, ensure_ascii=False))
    print('\n'.join(md[:12]))
    print(f'-> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
