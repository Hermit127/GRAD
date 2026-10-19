#!/usr/bin/env python3
"""Inspect which nodes drive fixed-adapter flags on one MRT fold.

Prints, for every trace where old != new (or all flagged traces with --all),
the triggering node text and the official policy evidence, so metric moves
are explainable from execution traces.

Usage: python inspect_agenttrust_flags.py --fold api_key_calendar_agendas_2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PKG = REPO / 'code/repro/static_big_table_v1'
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))
for rel in ('ref/AgentTrust/src', 'ref/AgentTrust'):
    sys.path.insert(0, str(REPO / rel))

import agenttrust_fixed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='api_key_calendar_agendas_2')
    ap.add_argument('--limit', type=int, default=6)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--label', type=int, default=None)
    args = ap.parse_args()
    p = REPO / 'results/mrt_cross_task_v1' / f'fold_{args.fold}' / 'admission.jsonl'
    graphs = [json.loads(line) for line in open(p) if line.strip()]

    TrustInterceptor, _, Action, ActionType, _ = agenttrust_fixed._import_official()
    policy = agenttrust_fixed.build_policy('prod')
    n_shown = 0
    for g in graphs:
        if args.label is not None and int(g['label']) != args.label:
            continue
        interceptor = TrustInterceptor(policy=policy)
        flagged = []
        for i, node in enumerate(g['nodes']):
            try:
                action = agenttrust_fixed._action_from_node(
                    Action, ActionType, node, g.get('task_id', ''))
                report = interceptor.verify(action)
            except Exception as exc:
                flagged.append((i, node, 'EXC', str(exc), []))
                continue
            if report.verdict.value != 'allow':
                flagged.append((i, node, report.verdict.value,
                                report.explanation[:220],
                                list(report.policy_violations)[:3]))
        if not flagged:
            continue
        n_shown += 1
        print(f"\n=== {g.get('task_id')} label={g['label']} nodes={len(g['nodes'])} ===\n")
        for i, node, verdict, expl, viol in flagged[:4]:
            text = str((node.get('features') or {}).get('subtask', ''))
            print(f'  node#{i} type={node.get("type")} verdict={verdict}')
            print(f'    text[:300]={text[:300]!r}')
            print(f'    expl={expl!r}')
            for v in viol:
                print(f'    viol={v[:200]}')
        if n_shown >= args.limit:
            break
    print(f'\nshown {n_shown} flagged traces')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
