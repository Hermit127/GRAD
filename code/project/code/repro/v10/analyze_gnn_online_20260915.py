"""Aggregate the GNN online closed-loop batch into the runtime table."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))

from build_gate_dataset_20260915 import family_roles  # noqa: E402
from envlib import ENVS  # noqa: E402


def mcnemar(pairs: list[tuple[int, int]]) -> dict:
    b = sum(1 for x, y in pairs if x == 1 and y == 0)
    c = sum(1 for x, y in pairs if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'p': 1.0}
    # exact two-sided binomial
    from math import comb
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {'b': b, 'c': c, 'p': min(1.0, p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.src).read_text())
    runs = data['runs']
    roles = {env: family_roles(env, list(spec['families']))
             for env, spec in ENVS.items()}
    for r in runs:
        r['_role'] = roles[r['env']].get(r['family'], 'train')

    def summ(rs: list[dict]) -> dict:
        if not rs:
            return {'n': 0}
        return {
            'n': len(rs),
            'harm': sum(1 for r in rs if r['independent_harm']),
            'completion': sum(1 for r in rs if r['independent_completion']),
            'harm_rate': round(sum(1 for r in rs if r['independent_harm']) / len(rs), 4),
            'completion_rate': round(sum(1 for r in rs if r['independent_completion']) / len(rs), 4),
            'blocked': sum(1 for r in rs if r['blocked']),
            'errors': sum(1 for r in rs if r['error']),
        }

    table = {}
    lat = []
    for r in runs:
        for d in r.get('decisions', []):
            if 'latency_ms' in d:
                lat.append(d['latency_ms'])
    for arm in sorted({r['arm'] for r in runs}):
        table[arm] = {}
        for cond in ('attack_instruction', 'benign_instruction', 'authorized_override'):
            rs = [r for r in runs if r['arm'] == arm and r['condition'] == cond]
            table[arm][cond] = summ(rs)
            for role in ('train_fam', 'unseen_fam'):
                rs2 = [r for r in rs if (r['_role'] == 'train') == (role == 'train_fam')]
                table[arm][f'{cond}|{role}'] = summ(rs2)

    # paired harm: ours vs none / strong on attack runs
    def paired(arm_a: str, arm_b: str) -> dict:
        by: dict[tuple, dict] = {}
        for r in runs:
            if r['condition'] != 'attack_instruction':
                continue
            key = (r['env'], r['family'], r.get('repeat', 0))
            by.setdefault(key, {})[r['arm']] = r['independent_harm']
        pairs = [(1 if d.get(arm_b) else 0, 1 if d.get(arm_a) else 0)
                 for d in by.values() if arm_a in d and arm_b in d]
        return {'n_pairs': len(pairs), **mcnemar(pairs)}

    out = {
        'src': args.src, 'table': table,
        'latency_ms': ({'n': len(lat), 'mean': round(sum(lat) / len(lat), 2),
                        'p50': round(sorted(lat)[len(lat) // 2], 2),
                        'p95': round(sorted(lat)[int(len(lat) * 0.95)], 2)}
                       if lat else None),
        'mcnemar_attack_harm': {
            'gnn_replan_vs_none': paired('gnn_replan', 'none'),
            'gnn_replan_vs_strong': paired('gnn_replan', 'strong'),
            'gnn_vs_none': paired('gnn', 'none'),
        },
    }
    Path(args.out_json).write_text(json.dumps(out, indent=1))

    lines = ['# GNN 在线闭环聚合（2026-09-15）', '',
             '| 臂 | 条件 | n | harm | comp | blocked |', '|---|---|---|---|---|---|']
    for arm in sorted(table):
        for cond in ('attack_instruction', 'benign_instruction', 'authorized_override'):
            s = table[arm][cond]
            lines.append(f"| {arm} | {cond} | {s['n']} | {s['harm']} | {s['completion']} | {s['blocked']} |")
    lines += ['', f"延迟(ms)={out['latency_ms']}", '',
              f"McNemar attack harm: {json.dumps(out['mcnemar_attack_harm'])}"]
    Path(args.out_md).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
