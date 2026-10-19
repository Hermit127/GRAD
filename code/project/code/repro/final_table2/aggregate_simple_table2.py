#!/usr/bin/env python3
"""Aggregate the simple Table 2 panels (unified AgentDojo v1.2 / 0.1.35).

Inputs (frozen list, raw results of the 2026-09-17 simple run):
  simple_none_sanitizer.json  (arms none, sanitizer)
  simple_grad.json            (arm grad)
  simple_drift.json           (arm drift)
  simple_camel.json           (arm camel_policy)
  simple_prudentia.json       (arm prudentia)

Outputs: simple_table2.json / simple_table2.md / table2_main.tex +
cell_provenance.json entries. Dedup follows code/repro/final_table2/retry_policy.py
(all attempts kept; deterministic last-success selection).
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/final_table2'))
from retry_policy import resolve_cells  # noqa: E402

DIR = REPO / 'LOGS/final_table2/agentdojo-v1.2-unified-20260917'
RAW = DIR / 'raw_results'
PANELS = {
    'none': 'simple_none_sanitizer.json',
    'sanitizer': 'simple_none_sanitizer.json',
    'grad': 'simple_grad.json',
    'drift': 'simple_drift.json',
    'camel': 'simple_camel.json',
    'prudentia': 'simple_prudentia.json',
}
ARM_LABEL = {'none': 'None', 'sanitizer': 'Sanitizer', 'grad': 'GRAD',
             'drift': 'DRIFT', 'camel': 'CaMeL (policy)',
             'prudentia': 'PRUDENTIA (optimistic)'}


def latency_of(r: dict) -> list[float]:
    # The legacy grad_latency_ms field duplicates decisions[].latency_ms.
    if r.get('decisions'):
        return [float(d['latency_ms']) + float(d.get('build_latency_ms', 0))
                for d in r['decisions'] if d.get('latency_ms') is not None]
    out = []
    for x in (r.get('defense_latency_ms') or []):
        if isinstance(x, (int, float)):
            out.append(float(x))
    g = r.get('grad_latency_ms')
    if isinstance(g, list):
        out.extend(float(x) for x in g if isinstance(x, (int, float)))
    elif isinstance(g, (int, float)):
        out.append(float(g))
    return out


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=DIR,
                        help='Use a new directory to preserve historical tables.')
    out_dir = parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = {}
    for arm, fname in PANELS.items():
        path = RAW / fname
        d = json.loads(path.read_text())
        recs = []
        for r in d.get('runs', []):
            if r.get('arm') != (arm if arm != 'camel' else 'camel_policy'):
                continue
            recs.append({
                'suite': r.get('suite'), 'user_task': r.get('user_task'),
                'condition': r.get('condition'), 'arm': arm,
                'injection_task': r.get('injection_task') or '',
                'utility': r.get('utility'), 'security': r.get('security'),
                'error': r.get('error'), 'origin': fname,
                'latency': latency_of(r),
                **{k: r.get(k, d.get(k)) for k in (
                    'runner_contract', 'benchmark_version', 'model',
                    'agentdojo_version')},
            })
        panels[arm] = recs

    table = {}
    prov = {}
    for arm, recs in panels.items():
        resolved, log = resolve_cells(recs)
        if any(v['protocol_mismatch'] for v in resolved.values()):
            raise ValueError(f'protocol mismatch within {arm}; refusing aggregation')
        chosen = [v['chosen'] for v in resolved.values() if v['chosen']]
        atk = [r for r in chosen if r['condition'] == 'attack']
        ben = [r for r in chosen if r['condition'] == 'benign']
        n_atk = sum(k[2] == 'attack' for k in resolved)
        n_ben = sum(k[2] == 'benign' for k in resolved)
        lat = [x for r in chosen for x in r['latency']]
        fails = sum(1 for v in resolved.values() if v['status'] != 'ok')
        table[arm] = {
            'label': ARM_LABEL[arm],
            'n_cases': len(resolved),
            'n_attack': n_atk, 'n_benign': n_ben,
            'tool_asr': f"{sum(1 for r in atk if r['security'])}/{n_atk}",
            'attack_outcome_unknown': n_atk - len(atk),
            'bu': f"{sum(1 for r in ben if r['utility'])}/{n_ben}",
            'ua': f"{sum(1 for r in atk if r['utility'])}/{n_atk}",
            'final_failures': fails,
            'attempts': len(recs),
            'latency_ms': ({'p50': pct(lat, .5), 'p95': pct(lat, .95),
                            'n': len(lat)} if lat else None),
            'error_types': {},
        }
        prov[arm] = [{'case': list(k), 'origin': v['chosen']['origin']
                      if v['chosen'] else None, 'status': v['status'],
                      'n_attempts': v['n_attempts']} for k, v in resolved.items()]
    (out_dir / 'simple_table2.json').write_text(json.dumps(table, indent=1))
    (out_dir / 'cell_provenance.json').write_text(json.dumps(prov, indent=1))

    lines = ['| Method | tool ASR | BU | UA | final failures | attempts | Lat p50/p95 (ms) |',
             '|---|---|---|---|---|---|---|']
    for arm, t in table.items():
        lt = t['latency_ms']
        lines.append(f"| {t['label']} | {t['tool_asr']} | {t['bu']} | {t['ua']} "
                     f"| {t['final_failures']} | {t['attempts']} "
                     f"| {lt['p50']}/{lt['p95']} |" if lt else
                     f"| {t['label']} | {t['tool_asr']} | {t['bu']} | {t['ua']} "
                     f"| {t['final_failures']} | {t['attempts']} | n/a |")
    md = '\n'.join(lines) + '\n'
    (out_dir / 'simple_table2.md').write_text(md)

    tex = [r'\begin{table}[t]', r'\centering',
           r'\caption{AgentDojo v1.2 closed-loop panel (simple table, '
           r'package 0.1.35; CaMeL runner pinned to 0.1.34 with task data '
           r'verified byte-identical). 4 suites $\times$ 2 user tasks; '
           r'benign + first injection per suite; tool-side ASR is 0 for every '
           r'arm in this evaluated subset; '
           r'IPIGuard is a native v1.1.2 panel (appendix). Latency = recorded '
           r'overhead per call/decision (GRAD includes prefix construction); '
           r'not full-trajectory overhead. Cross-version equivalence remains '
           r'unverified; this is a diagnostic panel, not a matched ranking.}',
           r'\label{tab:table2-simple}', r'\small',
           r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{lccccc}', r'\toprule',
           r'Arm & tool ASR $\downarrow$ & BU $\uparrow$ & UA $\uparrow$ '
           r'& fin.\ fail. $\downarrow$ & Lat p50/p95 (ms) \\', r'\midrule']
    for arm, t in table.items():
        lt = t['latency_ms']
        lat = f"{lt['p50']}/{lt['p95']}" if lt else (r'N/A' if arm == 'none' else r'--')
        row = (f"{t['label']} & {t['tool_asr']} & {t['bu']} & {t['ua']} "
               f"& {t['final_failures']} & {lat} \\\\")
        if arm == 'none':
            row = r'\rowcolor{gray!15}' + row
        tex.append(row)
    tex += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    (out_dir / 'table2_main.tex').write_text('\n'.join(tex))
    print(md)
    print('wrote simple_table2.json/md, cell_provenance.json, table2_main.tex')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
