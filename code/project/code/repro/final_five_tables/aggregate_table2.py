#!/usr/bin/env python3
"""Table 2 (AgentDojo closed-loop) aggregation with mandatory dedup.

Reads every panel file, dedupes by (suite,user_task,condition,arm,injection_task)
keeping the FIRST non-error record, and emits per-panel/per-arm:
  tool-side ASR (attack cells excluding injection_task_6),
  text-output ASR (injection_task_6 only, travel),
  benign utility (BU), utility under attack (UA), error count/rate,
  defense latency p50/p95 (per call/decision where recorded),
  method-specific counters (sanitizer was_safe, GRAD blocks, DRIFT
  planner/isolator/interpreter errors, PRUDENTIA HITL/noncompliant).

Usage: python aggregate_tables.py table2
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FF = REPO / 'LOGS/final_five_tables'
RAW = FF / 'raw_results'
RAW.mkdir(parents=True, exist_ok=True)

PANEL_FILES = {
    'v1.2_first_injection': [
        REPO / 'LOGS/agentdojo_sanitizer_b1_none_sanitizer_20260916.json',
        REPO / 'LOGS/final_five_tables/grad_slack_workspace_first_20260916.json',
        REPO / 'LOGS/final_five_tables/grad_banking_full_20260916.json',
        REPO / 'LOGS/final_five_tables/grad_travel_full_20260916.json',
        REPO / 'LOGS/final_five_tables/camel_full_20260916.json',
    ],
    'v1.2_full_injection_banking': [
        REPO / 'LOGS/agentdojo_sanitizer_b1travel_20260916.json',
        REPO / 'LOGS/final_five_tables/grad_banking_full_20260916.json',
        REPO / 'LOGS/final_five_tables/drift_banking_travel_20260916.json',
        REPO / 'LOGS/final_five_tables/prudentia_banking_20260916.json',
    ],
    'v1.2_full_injection_travel': [
        REPO / 'LOGS/agentdojo_sanitizer_b1travel_20260916.json',
        REPO / 'LOGS/final_five_tables/grad_travel_full_20260916.json',
        REPO / 'LOGS/final_five_tables/drift_banking_travel_20260916.json',
    ],
    'v1.1.2_ipiguard': [
        REPO / 'LOGS/ipiguard_results_fixed_20260916.json',
        REPO / 'LOGS/final_five_tables/grad_ipiguard_20260916.json',
    ],
}

# which suites are part of each panel's intended grid
PANEL_SUITES = {
    'v1.2_first_injection': {'banking', 'slack', 'travel', 'workspace'},
    'v1.2_full_injection_banking': {'banking'},
    'v1.2_full_injection_travel': {'travel'},
    'v1.1.2_ipiguard': {'banking', 'travel'},
}


def _norm(r: dict) -> dict:
    """Normalize a run record from any runner."""
    suite = r.get('suite')
    uid = r.get('user_task')
    cond = r.get('condition')
    arm = r.get('arm')
    iid = r.get('injection_task') or ''
    if suite is None or uid is None:  # prudentia task_key fallback
        parts = (r.get('task_key') or '').split('|')
        if len(parts) >= 6:
            suite, uid, iid, cond, arm = parts[0], parts[1], parts[2], parts[3], parts[4]
    lat = list(r.get('defense_latency_ms') or [])
    for d in (r.get('decisions') or []):
        if d.get('latency_ms') is not None:
            lat.append(d['latency_ms'])
    return {
        'suite': suite, 'user_task': uid, 'condition': cond, 'arm': arm,
        'injection_task': iid, 'utility': r.get('utility'),
        'security': r.get('security'), 'error': r.get('error'),
        'latency': lat, 'raw': r,
    }


def _pct(values, p):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 1)


def load_panel(paths, suites, only_injections=None):
    """Load one panel.

    Returns (records, error_records, file_meta). Errors are filtered with the
    SAME suite/injection rules as the successful records, and each record keeps
    `origin` (file name) so that native pairings never depend on file order.
    """
    recs = {}
    error_recs = []
    file_meta = {}
    for path in paths:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        file_meta[path.name] = {
            'model': d.get('model'),
            'agentdojo_version': d.get('agentdojo_version'),
            'benchmark_version': d.get('benchmark_version'),
            'runner_contract': d.get('runner_contract') or d.get('hook_version'),
        }
        for r in d.get('runs', []):
            if r.get('suite') not in suites:
                continue
            n = _norm(r)
            if only_injections is not None:
                iid = n['injection_task']
                if n['condition'] == 'attack' and iid not in only_injections.get(n['suite'], set()):
                    continue
            if n['arm'] in ('optimistic', 'pessimistic') or (n['arm'] and n['arm'].startswith('prudentia')):
                base = n['arm'].split('|')[0]
                n['arm'] = 'prudentia' if base.startswith('prudentia') else base
            n['origin'] = path.name
            if n.get('error'):
                error_recs.append(n)
                continue
            key = (n['suite'], n['user_task'], n['condition'], n['arm'], n['injection_task'])
            recs.setdefault(key, n)  # first error-free record wins (per cell)
    return recs, error_recs, file_meta


def metrics(recs):
    by_arm = {}
    arms = sorted({r['arm'] for r in recs.values() if r['arm']})

    def is_text(r):
        # injection_task_6 is a pure-text-output attack ONLY in the travel suite
        return r['suite'] == 'travel' and r['injection_task'] == 'injection_task_6'

    for arm in arms:
        rs = [r for r in recs.values() if r['arm'] == arm]
        atk = [r for r in rs if r['condition'] == 'attack']
        tool = [r for r in atk if not is_text(r)]
        text = [r for r in atk if is_text(r)]
        ben = [r for r in rs if r['condition'] == 'benign']
        lat = [x for r in rs for x in r['latency']]
        hits = sum(1 for r in tool if r['security'])
        hits_t = sum(1 for r in text if r['security'])
        by_arm[arm] = {
            'n_attack_tool': len(tool), 'tool_asr': f'{hits}/{len(tool)}',
            'n_attack_text': len(text), 'text_asr': f'{hits_t}/{len(text)}',
            'bu': f"{sum(1 for r in ben if r['utility'])}/{len(ben)}",
            'ua': f"{sum(1 for r in atk if r['utility'])}/{len(atk)}",
            'n': len(rs),
            'latency_ms': ({'p50': _pct(lat, .5), 'p95': _pct(lat, .95),
                            'n_calls': len(lat)} if lat else None),
        }
        method_counters = {}
        for r in rs:
            raw = r['raw']
            for f in ('blocked_counts', 'hitl_requests', 'hitl_approvals',
                      'hitl_endorsements', 'noncompliant_calls',
                      'planner_calls', 'isolator_masks', 'interpreter_errors',
                      'tool_errors'):
                v = raw.get(f)
                if v:
                    method_counters[f] = method_counters.get(f, 0) + v
        if method_counters:
            by_arm[arm]['counters'] = method_counters
    return by_arm


def error_bookkeeping(recs, error_recs):
    """Per-arm error accounting with retry/vs-final distinction.

    attempt_errors   : filtered error records for this arm
    cells_retried    : cells that have >=1 error AND a final error-free record
    final_failures   : cells that only ever produced error records
    recovered        : == cells_retried (successfully completed after >=1 error)
    """
    by_arm = {}
    for e in error_recs:
        arm = e['arm']
        d = by_arm.setdefault(arm, {'attempt_errors': 0, 'cells_retried': 0,
                                    'final_failures': 0, 'error_types': {}})
        d['attempt_errors'] += 1
        et = str(e.get('error') or 'unknown').split(':')[0]
        d['error_types'][et] = d['error_types'].get(et, 0) + 1
    for e in error_recs:
        arm = e['arm']
        key = (e['suite'], e['user_task'], e['condition'], e['arm'], e['injection_task'])
        d = by_arm[arm]
        if key in recs:
            d['cells_retried'] += 1
        else:
            d['final_failures'] += 1
    for arm, d in by_arm.items():
        d['recovered'] = d['cells_retried']
    return by_arm


def table2():
    out = {'table': 'table2_agentdojo_closed_loop', 'panels': {}}
    first_inj = {}
    b1 = REPO / 'LOGS/agentdojo_sanitizer_b1_none_sanitizer_20260916.json'
    if b1.exists():
        for r in json.loads(b1.read_text()).get('runs', []):
            if r.get('condition') == 'attack' and not r.get('error'):
                first_inj.setdefault(r['suite'], set()).add(r.get('injection_task'))
    for panel, paths in PANEL_FILES.items():
        only = first_inj if panel == 'v1.2_first_injection' else None
        recs, errs, meta = load_panel(paths, PANEL_SUITES[panel],
                                      only_injections=only)
        # native None per harness file (computed from that file alone; NO
        # cross-file merging, so each runner's own paired baseline is visible)
        none_by_origin = {}
        for path in paths:
            if not path.exists():
                continue
            sub, _, _ = load_panel([path], PANEL_SUITES[panel],
                                   only_injections=only)
            m = metrics(sub).get('none')
            if m:
                none_by_origin[path.name] = m
        out['panels'][panel] = {
            'files': [str(p.relative_to(REPO)) for p in paths if p.exists()],
            'files_missing': [str(p.relative_to(REPO)) for p in paths if not p.exists()],
            'file_meta': meta,
            'error_bookkeeping': error_bookkeeping(recs, errs),
            'unique_cells': len(recs),
            'first_injection_filter': ({k: sorted(v) for k, v in first_inj.items()}
                                       if only is not None else None),
            'none_by_origin': none_by_origin,
            'metrics': metrics(recs),
        }
    (RAW / 'table2.json').write_text(json.dumps(out, indent=1))

    md = ['# Table 2 数据（自动生成；去重=首个非错误记录）', '']
    for panel, d in out['panels'].items():
        md.append(f"## {panel}")
        md.append(f"files: {', '.join(d['files']) or '(none)'}")
        if d['files_missing']:
            md.append(f"missing: {', '.join(d['files_missing'])}")
        md.append('')
        md.append('| arm | n | toolASR | textASR | BU | UA | latency p50/p95 ms | counters |')
        md.append('|---|---|---|---|---|---|---|---|')
        for arm, m in d['metrics'].items():
            lat = m['latency_ms']
            lat_s = f"{lat['p50']}/{lat['p95']}" if lat else '--'
            md.append(f"| {arm} | {m['n']} | {m['tool_asr']} | {m['text_asr']} | {m['bu']} | {m['ua']} | {lat_s} | {m.get('counters','')} |")
        md.append('')
        eb = d.get('error_bookkeeping') or {}
        if eb:
            md.append('Error bookkeeping (filtered to the panel):')
            md.append('')
            md.append('| arm | attempts failed | retried+recovered | final failures | types |')
            md.append('|---|---|---|---|---|')
            for arm, e in eb.items():
                md.append(f"| {arm} | {e['attempt_errors']} | {e['recovered']} | {e['final_failures']} | {e['error_types']} |")
            md.append('')
        if d.get('none_by_origin'):
            md.append('Native None per harness (no merging):')
            md.append('')
            md.append('| origin | BU | UA | tool ASR |')
            md.append('|---|---|---|---|')
            for origin, m in d['none_by_origin'].items():
                if m:
                    md.append(f"| {origin} | {m['bu']} | {m['ua']} | {m['tool_asr']} |")
            md.append('')
    (RAW / 'table2.md').write_text('\n'.join(md) + '\n')
    print('table2 written:', RAW / 'table2.json')
    return out


if __name__ == '__main__':
    table2()
