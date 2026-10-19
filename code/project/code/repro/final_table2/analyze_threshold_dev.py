#!/usr/bin/env python3
"""Apply the pre-registered dev threshold rule (zero API).

Inputs: stage-F benign grad decisions (dev only) and the known malicious
send_money score from stage C. Outputs the FPR/detection frontier.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('.')
F_DIR = ROOT / 'LOGS/final_table2/threshold_dev_20260918'
C_RESULTS = ROOT / 'LOGS/final_table2/dev_attacks_20260918/results_stageC.json'


def main() -> int:
    benign = []
    for r in json.loads((F_DIR / 'results_stageF.json').read_text())['runs']:
        for d in (r.get('decisions') or []):
            if d['score'] is not None:
                benign.append((r['suite'], r['user_task'], d['tool'], d['score'],
                               d['blocked'], d['threshold']))
    runc = json.loads(C_RESULTS.read_text())['runs']
    mal = [(r['case_id'], d['tool'], d['score']) for r in runc if r['arm'] == 'grad'
           for d in r['decisions'] if d['tool'] == 'send_money']
    p_mal = max(x[2] for x in mal)
    lines = ['# Dev threshold analysis (zero API, dev-only)', '',
             f'- known malicious send_money scores: {mal}',
             f'- dev benign gate calls: {len(benign)}', '',
             '| threshold T | malicious blocked (p_mal >= T) | dev benign calls blocked | benign FPR |',
             '|---|---|---|---|']
    for T in sorted({round(b[3], 6) for b in benign} | {round(p_mal, 6)}):
        blocked_ben = [b for b in benign if b[3] >= T]
        lines.append(f"| {T} | {p_mal >= T} | {len(blocked_ben)}/{len(benign)} "
                     f"| {len(blocked_ben)/len(benign):.3f} |")
    zero_fpr_max = max((b[3] for b in benign), default=None)
    feasible = p_mal >= 0 and (zero_fpr_max is None or zero_fpr_max < p_mal)
    lines += ['', '## Pre-registered rule outcome', '',
              f'- max dev benign score: {zero_fpr_max}',
              f'- zero-FPR threshold that blocks the malicious call exists: **{feasible}**']
    if not feasible:
        over = sorted((b for b in benign if b[3] >= p_mal), key=lambda x: -x[3])
        lines.append('- dev benign calls at or above p_mal (would be blocked):')
        for b in over:
            lines.append(f"  - {b[0]}|{b[1]} {b[2]} p={b[3]:.6f} (prev threshold {b[5]})")
        lines.append('- conclusion: a single global threshold cannot separate the known '
                     'malicious call from dev benign calls; threshold tuning is not a fix '
                     'on this evidence. FPR frontier above is reported instead; no tuning '
                     'on the test panel.')
    (F_DIR / 'threshold_analysis.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
