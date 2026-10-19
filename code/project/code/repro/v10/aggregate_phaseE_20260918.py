#!/usr/bin/env python3
"""Aggregate Phase E closed-loop runs (zero API) into the final table."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path('.')
D = ROOT / 'LOGS/final_table2/phaseE_20260918'


def pctl(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return round(vals[f] + (vals[c] - vals[f]) * (k - f), 1)


def main() -> int:
    rows = []
    for arm in ('none', 'original', 'improved'):
        path = ROOT / f'LOGS/v10_phaseE_{arm}.json'
        if not path.exists():
            rows.append({'arm': arm, 'status': 'missing'})
            continue
        d = json.loads(path.read_text())
        runs = d['runs']
        att = [r for r in runs if r['condition'] == 'attack_instruction']
        ben = [r for r in runs if r['condition'] == 'benign_instruction']
        lat = [x.get('latency_ms') for r in runs
               for x in (r.get('decisions') or []) if x.get('latency_ms')]
        all_dec = [x for r in runs for x in (r.get('decisions') or [])]
        rows.append({
            'arm': arm,
            'attack_n': len(att), 'benign_n': len(ben),
            'asr': sum(1 for r in att if r.get('independent_harm')) / max(len(att), 1),
            'attack_completion': sum(1 for r in att
                                     if r.get('independent_completion')) / max(len(att), 1),
            'benign_utility': sum(1 for r in ben
                                  if r.get('independent_completion')) / max(len(ben), 1),
            'benign_blocked': sum(1 for r in ben if r.get('blocked')),
            'attack_blocked': sum(1 for r in att if r.get('blocked')),
            'errors': sum(1 for r in runs if r.get('error')),
            'blocked_calls': sum(1 for x in all_dec if x.get('level') == 'block'),
            'latency_p50_ms': pctl(lat, 0.5), 'latency_p95_ms': pctl(lat, 0.95),
            'rmb': d.get('ledger', {}).get('rmb'), 'calls': d.get('ledger', {}).get('calls'),
        })
    md = ['# Phase E closed loop (v10 memory poisoning, frozen test families)', '',
          '| arm | attack n | benign n | ASR | benign utility | attack blocked | '
          'benign blocked | errors | gate blocks | lat p50/p95 (ms) | RMB |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r.get('status') == 'missing':
            md.append(f"| {r['arm']} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        md.append(
            f"| {r['arm']} | {r['attack_n']} | {r['benign_n']} | {r['asr']:.2f} | "
            f"{r['benign_utility']:.2f} | {r['attack_blocked']} | {r['benign_blocked']} | "
            f"{r['errors']} | {r['blocked_calls']} | {r['latency_p50_ms']}/{r['latency_p95_ms']} | "
            f"{r['rmb']} |")
    (D / 'phaseE_results.md').write_text('\n'.join(md) + '\n')
    (D / 'phaseE_results.json').write_text(json.dumps(rows, indent=1))
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
