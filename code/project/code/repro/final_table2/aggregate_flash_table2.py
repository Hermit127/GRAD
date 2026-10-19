#!/usr/bin/env python3
"""Aggregate the frozen Flash matched panel into the PI 7-row Table 2.

Source of truth: LOGS/final_table2/flash_matched_20260917/results.json
(54 attempts: 18 cases x 3 arms; failures kept in denominators).
Rows without a matched Flash implementation stay `--` (no historical model
numbers are merged). Outputs: flash_table2.{json,md}, table2_main.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('.')
D = ROOT / 'LOGS/final_table2/flash_matched_20260917'
AUTHGRAPH_D = ROOT / 'LOGS/final_table2/authgraph_online_20260917'
AUTHGRAPH_ROW = 'AuthGraph (exploratory)'
ROWS = ['None', 'Sanitizer', 'IPIGuard', 'CaMeL', 'DRIFT',
        'Security-aware planning', 'GRAD (Ours)', AUTHGRAPH_ROW]
ARM_OF = {'None': 'none', 'Sanitizer': 'sanitizer', 'GRAD (Ours)': 'grad',
          AUTHGRAPH_ROW: 'authgraph'}
MISSING = {
    'IPIGuard': 'no matched Flash implementation (official code vendored on agentdojo 0.1.23/v1.1.2)',
    'CaMeL': 'not yet rerun on deepseek-flash in this batch',
    'DRIFT': 'not yet rerun on deepseek-flash in this batch',
    'Security-aware planning': 'not yet rerun on deepseek-flash in this batch',
}


def load_runs() -> list[dict]:
    runs = json.loads((D / 'results.json').read_text())['runs']
    ag_path = AUTHGRAPH_D / 'results.json'
    if ag_path.exists():
        runs += json.loads(ag_path.read_text())['runs']
    return runs


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 1)


def main() -> int:
    r = {'runs': load_runs()}
    runs = r['runs']
    table = {}
    for label in ROWS:
        arm = ARM_OF.get(label)
        if arm is None:
            table[label] = {'status': '--', 'reason': MISSING[label]}
            continue
        rs = [x for x in runs if x['arm'] == arm]
        ok = [x for x in rs if not x.get('error')]
        atk = [x for x in ok if x['scope'] in ('tool', 'text')]
        ben = [x for x in rs if x['scope'] == 'benign']
        tool = [x for x in atk if x['scope'] == 'tool']
        text = [x for x in atk if x['scope'] == 'text']
        fails = len(rs) - len(ok)
        lat = []
        for x in ok:
            if x.get('defense_module_ms') is not None:
                lat.append(x['defense_module_ms'])
            else:
                lat += [d.get('latency_ms', 0) + d.get('build_latency_ms', 0)
                        for d in (x.get('decisions') or [])]
        table[label] = {
            'status': 'ok',
            'n_attempts': len(rs), 'n_final_failures': fails,
            'n_degraded': sum(1 for x in ok if x.get('degraded')),
            'tool_asr': f"{sum(1 for x in tool if x['security'])}/{len(tool)}",
            'text_asr': f"{sum(1 for x in text if x['security'])}/{len(text)}",
            'bu': f"{sum(1 for x in ben if x['utility'])}/{len(ben)}",
            'ua': f"{sum(1 for x in atk if x['utility'])}/{len(atk)}",
            'latency_ms': ({'p50': pct(lat, .5), 'p95': pct(lat, .95),
                            'n': len(lat)} if lat else None),
        }
    (D / 'flash_table2.json').write_text(json.dumps(table, indent=1))

    md = ['| Method | ASR (tool) | ASR (text) | BU | UA | Errors | Lat p50/p95 (ms) |',
          '|---|---|---|---|---|---|---|']
    for label in ROWS:
        t = table[label]
        if t['status'] == '--':
            md.append(f'| {label} | -- | -- | -- | -- | -- | -- |')
        else:
            lt = t['latency_ms']
            md.append(f"| {label} | {t['tool_asr']} | {t['text_asr']} | {t['bu']} "
                      f"| {t['ua']} | {t['n_final_failures']}/{t['n_attempts']} "
                      f"| {lt['p50']}/{lt['p95']} |" if lt else
                      f"| {label} | {t['tool_asr']} | {t['text_asr']} | {t['bu']} "
                      f"| {t['ua']} | {t['n_final_failures']}/{t['n_attempts']} | -- |")
    md += ['',
           'Notes:',
           '- BU denominator = all benign runs per arm (8); failures stay in the '
           'denominator and errored/failed benign runs count as utility failures. '
           'None/GRAD 6/8, Sanitizer 5/8, AuthGraph 8/8.',
           '- Latency = incremental defense-module wall time summed per trajectory '
           '(GRAD: build + score; AuthGraph: Planner + Checker LLM judgments + replans); '
           'agent regeneration is not isolated. This is NOT end-to-end defense overhead; '
           'the end-to-end figure is still pending.',
           '- Seven baseline rows: None / Sanitizer / IPIGuard / CaMeL / DRIFT / '
           'Security-aware planning / GRAD. Four rows still have no matched-Flash '
           'implementation and stay -- (no historical-model numbers merged). AuthGraph '
           'is an exploratory auxiliary online adaptation outside this panel and does '
           'not substitute for any missing row.',
           '- Panel positioning: small-scale usability/utility check (completion, false '
           'blocks, module cost). Tool-side 0/8 under one fixed ImportantInstructions '
           'template is NOT security-effectiveness evidence; text-side 2/2 for none and '
           'GRAD is a coverage gap, not an attack-difficulty result.',
           '- Coverage boundary: GRAD checks tool calls before execution only; there is '
           'no final-text-output checkpoint. The two text-side misses are outside GRAD '
           'coverage and are not explained away; output-side checking is a method '
           'extension requiring separate implementation and validation.',
           '- AuthGraph row = prefix-only pre-execution check (`authgraph_online_20260917`, '
           'manifest v1.1); blocked calls are not executed and the agent continues. '
           'Errors are fail-open and recorded: 2/18 runs degraded (1 Planner empty-output, '
           '1 replan JSON parse), 0 final failures.',
           '- AuthGraph tool blocks (11 across runs) stopped every tool-side attack; the '
           'text-output attack (travel injection_task_6) is outside the tool-call threat '
           'surface and still succeeded 2/2, same as none/GRAD.']
    (D / 'flash_table2.md').write_text('\n'.join(md) + '\n')

    tex = [r'\begin{table}[t]', r'\centering',
           r'\caption{AgentDojo v1.2 closed loop (package 0.1.35), '
           r'deepseek-flash, non-thinking. Frozen subset: 4 suites $\times$ '
           r'numeric user\_task\_3/4; benign + first tool injection; travel '
           r'text attack (injection\_task\_6) separate. Seven baseline rows; '
           r'failures stay in every denominator (BU over all benign runs, '
           r'errored or failed benign runs count as utility failures). Rows '
           r'without a matched Flash run show \textendash{}\textendash{}: IPIGuard, '
           r'CaMeL, DRIFT and Security-aware planning are still missing and no '
           r'historical-model numbers are merged. AuthGraph is an exploratory '
           r'auxiliary adaptation outside the seven-row panel and does not '
           r'substitute for any missing row. This panel is a small-scale '
           r'usability and utility check (completion, false blocks, module '
           r'cost): the tool-side $0/8$ under a single fixed template is not '
           r'security-effectiveness evidence. Coverage boundary: GRAD checks '
           r'tool calls before execution only, with no final-text-output '
           r'checkpoint; the text-side injection outcomes (none and GRAD '
           r'$2/2$) are outside GRAD coverage and are not explained away. '
           r'Lat$_{\mathrm{def}}$ = incremental '
           r'defense-module wall time summed per trajectory (GRAD: build + score; '
           r'AuthGraph: Planner + Checker LLM judgments + replans); agent '
           r'regeneration is not isolated and this is \emph{not} end-to-end '
           r'defense overhead (end-to-end still pending). AuthGraph errors are '
           r'fail-open and recorded: 2/18 degraded, 0 final failures.}',
           r'\label{tab:table2-flash}', r'\small',
           r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{lcccccc}', r'\toprule',
           r'Method & ASR$_{\mathrm{tool}}\downarrow$ & '
           r'ASR$_{\mathrm{text}}\downarrow$ & BU$\uparrow$ & UA$\uparrow$ '
           r'& Error$\downarrow$ & Lat$_{\mathrm{def}}$ p50/p95 (ms) \\', r'\midrule']
    for label in ROWS:
        t = table[label]
        if t['status'] == '--':
            row = f'{label} & -- & -- & -- & -- & -- & -- \\\\'
        else:
            lt = t['latency_ms']
            lat = f"{lt['p50']}/{lt['p95']}" if lt else '--'
            row = (f"{label} & {t['tool_asr']} & {t['text_asr']} & {t['bu']} "
                   f"& {t['ua']} & {t['n_final_failures']}/{t['n_attempts']} "
                   f"& {lat} \\\\")
        if label == 'None':
            row = r'\rowcolor{gray!15}' + row
        tex.append(row)
    tex += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    (D / 'table2_main.tex').write_text('\n'.join(tex))
    print('\n'.join(md))
    print('budget_upper_rmb:', json.loads((D / 'budget.json').read_text())['charged_upper_rmb'])
    ag_budget = AUTHGRAPH_D / 'budget.json'
    if ag_budget.exists():
        print('authgraph_budget_upper_rmb:',
              json.loads(ag_budget.read_text())['charged_upper_rmb'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
