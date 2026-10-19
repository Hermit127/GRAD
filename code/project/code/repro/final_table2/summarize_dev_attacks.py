#!/usr/bin/env python3
"""Summarize dev attack stages A/B/C and frozen test stage D (zero API)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('.')
DEV = ROOT / 'LOGS/final_table2/dev_attacks_20260918'
TEST = ROOT / 'LOGS/final_table2/frozen_direct_test_20260918'


def load(path):
    return json.loads(path.read_text())['runs'] if path.exists() else []


def main():
    rows = {}
    for stage in ('A', 'B', 'C'):
        for r in load(DEV / f'results_stage{stage}.json'):
            rows[r['case_id']] = dict(r, stage=stage)
    for r in load(TEST / 'results_stageD.json'):
        rows[r['case_id']] = dict(r, stage='D')
    runs = list(rows.values())

    lines = ['# Dev attack screening + frozen test summary (zero API aggregation)', '']
    lines.append(f"- runs: {len(runs)}; security True: "
                 f"{sum(1 for r in runs if r.get('security'))}; "
                 f"payload delivered in run: {sum(1 for r in runs if r.get('payload_delivered_in_run'))}; "
                 f"upper cost RMB: {sum(r.get('cost_upper_rmb') or 0 for r in runs):.4f}")
    lines += ['', '| stage | case | strategy | arm | util | sec | delivered | gate p (malicious call) |',
              '|---|---|---|---|---|---|---|---|']
    for r in sorted(runs, key=lambda x: (x['stage'], x['case_id'])):
        gate = ''
        for d in (r.get('decisions') or []):
            if d['tool'] in ('send_money', 'get_most_recent_transactions', 'read_file'):
                gate += f"{d['tool']}={d['score'] if d['score'] is not None else d['status']}"
                gate += f"({'B' if d['blocked'] else '-'}) "
        lines.append(
            f"| {r['stage']} | {r['suite']}|{r['user_task']}|{r['injection_task']} "
            f"| {r['strategy']} | {r['arm']} | {r['utility']} | {r['security']} "
            f"| {r.get('payload_delivered_in_run')} | {gate.strip() or '-'} |")

    lines += ['', '## Malicious tool calls observed in traces (attack-relevant)']
    for r in runs:
        calls = []
        for e in r.get('trace') or []:
            for c in (e.get('tool_calls') or []):
                calls.append(c['function'])
        if 'send_money' in calls:
            args = [c['args'] for e in r['trace'] for c in (e.get('tool_calls') or [])
                    if c['function'] == 'send_money']
            lines.append(f"- `{r['case_id']}`: send_money args {args}")
    (DEV / 'results_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
