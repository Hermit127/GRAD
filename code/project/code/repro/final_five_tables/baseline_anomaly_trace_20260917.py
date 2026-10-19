#!/usr/bin/env python3
"""Zero-API per-sample decision trace for the anomalous Table 1 baseline cells.

Companion to LOGS/baseline_anomaly_trace_20260917.md. Reads existing prediction
artifacts only; writes nothing. Run:

    python code/repro/final_five_tables/baseline_anomaly_trace_20260917.py
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIX3 = REPO / 'code/repro/out/baseline_fidelity_fix3_20260916'


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scope_rate(rows: list[dict], key: str) -> str:
    n = len(rows)
    pos = sum(int(r.get(key, 0)) for r in rows)
    if 'label' not in rows[0]:
        return f'n={n:4d} pred1={pos:4d} ({pos / n:.2f}) labels=metrics.json'
    tp = sum(1 for r in rows if r.get('label') == 1 and r.get(key) == 1)
    npos = sum(1 for r in rows if r.get('label') == 1)
    fp = pos - tp
    nneg = n - npos
    return (f'n={n:4d} pred1={pos:4d} ({pos / n:.2f}) '
            f'TPR={tp / max(npos, 1):.2f} FPR={fp / max(nneg, 1):.2f}')


def trace_agentarmor() -> None:
    base = FIX3 / 'f3_agentarmor_20260916/agentarmor'
    print('== AgentArmor (mechanism comparator: assumed registry) ==')
    for scope in sorted(os.listdir(base)):
        path = base / scope / 'predictions.jsonl'
        if not path.exists():
            continue
        rows = read_jsonl(path)
        rules = collections.Counter()
        for r in rows:
            for v in r.get('violations', []):
                rules[v['rule']] += 1
        print(f'  {scope:30s} {scope_rate(rows, "pred")} '
              f'rules={dict(rules.most_common(2))}')


def trace_aegis() -> None:
    base = FIX3 / 'f4_aegis_official_20260916/aegis_official'
    print('== AEGIS (official TS classifier + text-fallback adapter) ==')
    for scope in sorted(os.listdir(base)):
        path = base / scope / 'predictions.jsonl'
        if not path.exists():
            continue
        rows = read_jsonl(path)
        cats = collections.Counter()
        srcs = collections.Counter()
        for r in rows:
            for v in r.get('violations', []):
                cats[v.get('category', '?')] += 1
                srcs[v.get('source', '?')] += 1
        print(f'  {scope:30s} {scope_rate(rows, "pred_block")} '
              f'cats={dict(cats.most_common(3))} src={dict(srcs)}')


def trace_agenttrust() -> None:
    base = FIX3 / 'f1_agenttrust_prod_20260916/agenttrust_prod'
    print('== AgentTrust+RiskChain (rules-only, chain-inclusive) ==')
    for scope in sorted(os.listdir(base)):
        path = base / scope / 'predictions.jsonl'
        if not path.exists():
            continue
        rows = read_jsonl(path)
        alert = [r for r in rows
                 if (r.get('n_warn', 0) + r.get('n_block', 0)
                     + r.get('n_review', 0)) > 0 or r.get('n_chain_alerts', 0) > 0]
        tp = sum(1 for r in alert if r.get('label') == 1)
        npos = sum(1 for r in rows if r.get('label') == 1)
        nneg = len(rows) - npos
        fp = len(alert) - tp
        print(f'  {scope:30s} n={len(rows):4d} alert={len(alert):4d} '
              f'({len(alert) / len(rows):.2f}) TPR={tp / max(npos, 1):.2f} '
              f'FPR={fp / max(nneg, 1):.2f}')


def trace_lsm() -> None:
    print('== Sequential Monitor (DeepSeek; official prompt) ==')
    for name in ('lsm_decomp_fix2_cumulative_20260916',
                 'lsm_official_test_cumulative_20260916',
                 'lsm_official_test_separate_20260916'):
        path = REPO / 'LOGS' / f'{name}.json'
        if not path.exists():
            continue
        d = json.load(open(path))
        probs = d.get('per_task_probs', {})
        hard = sum(1 for v in probs.values() if max(v) in (0.0, 1.0))
        benign = [(k, v) for k, v in probs.items() if k.startswith('benign')]
        max_fp = [k for k, v in benign if max(v) >= 0.5]
        first_only = [k for k, v in benign
                      if v[0] >= 0.5 and all((p or 0) < 0.5 for p in v[1:])]
        m = d.get('metrics', {}).get('at_0.5', {})
        print(f'  {name:42s} n={d.get("n_tasks")} at0.5 TP{m.get("tp")}/FP'
              f'{m.get("fp")}/TN{m.get("tn")}/FN{m.get("fn")} '
              f'benign_FP_official={m.get("fp")}/{len(benign)} '
              f'max_based={len(max_fp)} first_subtask_only={first_only} '
              f'hard0/1={hard}/{len(probs)}')
    print('  note: official confusion counts triggers at subtask index > 1 only;')
    print('        AUROC max-score includes index 1 (harness caveat, recorded 21 vs 22).')


def main() -> None:
    trace_agentarmor()
    print()
    trace_aegis()
    print()
    trace_agenttrust()
    print()
    trace_lsm()


if __name__ == '__main__':
    main()
