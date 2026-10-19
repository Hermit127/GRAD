#!/usr/bin/env python
"""Aggregate the information-matched sequential-monitor runs (zero API).

Reads:
  LOGS/lsm_infomatch_{mcp,mg,minja,injecagent,asb,agentharm}_cumulative_20260917.json
  LOGS/lsm_decomp_fix2_cumulative_20260916.json            (native Decomp reference)
  LOGS/lsm_infomatch_mrt_probe_20260917.json               (blocked-cost evidence)

Writes:
  LOGS/final_five_tables/raw_results/lsm_infomatch.json
  LOGS/final_five_tables/raw_results/lsm_infomatch.md
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOGS = REPO / 'LOGS'
RAW = REPO / 'LOGS/final_five_tables/raw_results'

CORPORA = [
    ('mcp', 'MCP', 'LOGS/lsm_infomatch_mcp_cumulative_20260917.json'),
    ('mg', 'MG', 'LOGS/lsm_infomatch_mg_cumulative_20260917.json'),
    ('minja', 'MINJA', 'LOGS/lsm_infomatch_minja_cumulative_20260917.json'),
    ('injecagent', 'InjecAgent', 'LOGS/lsm_infomatch_injecagent_cumulative_20260917.json'),
    ('asb', 'ASB', 'LOGS/lsm_infomatch_asb_cumulative_20260917.json'),
    ('agentharm', 'AgentHarm', 'LOGS/lsm_infomatch_agentharm_cumulative_20260917.json'),
]


def load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def cell(rel: str, adapted: bool) -> dict:
    d = load(rel)
    m = d['metrics']
    c = m['at_0.5']
    return {
        'auroc': m['auroc'],
        'tpr_at_0.5': c['tp'] / (c['tp'] + c['fn']) if c['tp'] + c['fn'] else None,
        'fpr_at_0.5': c['fp'] / (c['fp'] + c['tn']) if c['fp'] + c['tn'] else None,
        'confusion_at_0.5': {k: c[k] for k in ('tp', 'fp', 'tn', 'fn',
                                               'too_late', 'never_triggered')},
        'n_tasks': d['n_tasks'], 'n_recorded': d.get('n_recorded'),
        'errors': d['errors'], 'batch_calls': d['batch_calls'],
        'batch_rmb': d['batch_rmb'],
        'model': d['model'], 'prompt': d['prompt'], 'eval_mode': d['eval_mode'],
        'official_prompts_sha256': d['official_prompts_sha256'],
        'adapted': adapted, 'source': rel,
    }


def main() -> int:
    out = {'table': 'table5_panelA_information_matched_monitor',
           'identity': 'Sequential Monitor (ICLR\'26) -- information-matched '
                       'adaptation (declared): same visible events and sources, '
                       'official prompt BINARY_BASELINE, cumulative prefixes',
           'prereg': 'LOGS/lsm_infomatch_prereg_20260917.md',
           'runner': 'code/repro/lsm_monitor_infomatch_20260917.py',
           'model': 'deepseek-chat (paper uses GPT-4o/o3-mini/Llama; disclosed)',
           'cells': {}, 'deviations': {
               'mcp': 'graph-level harm_index is frozen at 1 (converter default) '
                      'and inconsistent with node harm_step; first node harm_step used',
               'minja/injecagent/asb/agentharm': 'no harm annotation; AUROC uses max '
                      'over all prefixes instead of up-to-harm-index (declared)',
               'mg': 'harm_index=1 for harmful traces, so with the official i>1 '
                     'earliest-flag rule any flag is "too late" by construction; '
                     'AUROC for harmful uses prefix 1 only',
               'mrt': 'blocked-cost: full-history cumulative needs ~¥199 (probe-backed); '
                      'not run in this batch'}}
    for key, label, rel in CORPORA:
        out['cells'][label] = cell(rel, adapted=True)
    decomp = cell('LOGS/lsm_decomp_fix2_cumulative_20260916.json', adapted=False)
    decomp['note'] = 'native setting (official subtask lists); rerun 2026-09-16'
    out['cells']['Decomp'] = decomp
    if (LOGS / 'lsm_infomatch_mrt_probe_20260917.json').exists():
        out['mrt_blocked_cost'] = load('LOGS/lsm_infomatch_mrt_probe_20260917.json')

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / 'lsm_infomatch.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))

    md = ['# Information-matched Sequential Monitor (declared adaptation)', '',
          'Source JSONs under `LOGS/`; prereg `LOGS/lsm_infomatch_prereg_20260917.md`. '
          'Monitor model deepseek-chat, official prompt BINARY_BASELINE, cumulative prefixes.',
          '', '| corpus | n | AUROC | TPR@0.5 | FPR@0.5 | TP/FP/TN/FN | errors | ¥ |',
          '|---|---|---|---|---|---|---|---|']
    order = ['Decomp', 'MCP', 'MG', 'MINJA', 'InjecAgent', 'ASB', 'AgentHarm']
    for label in order:
        c = out['cells'][label]
        conf = c['confusion_at_0.5']
        tpr = '--' if c['tpr_at_0.5'] is None else f"{c['tpr_at_0.5']:.3f}"
        fpr = '--' if c['fpr_at_0.5'] is None else f"{c['fpr_at_0.5']:.3f}"
        md.append(f"| {label}{'' if not c['adapted'] else ' (adapted)'} | "
                  f"{c['n_tasks']} | {c['auroc']:.4f} | {tpr} | {fpr} | "
                  f"{conf['tp']}/{conf['fp']}/{conf['tn']}/{conf['fn']} | "
                  f"{c['errors']} | {c['batch_rmb']:.4f} |")
    md += ['', 'Deviations/blockers:', '']
    for k, v in out['deviations'].items():
        md.append(f'- {k}: {v}')
    (RAW / 'lsm_infomatch.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({'written': [str(RAW / 'lsm_infomatch.json'),
                                  str(RAW / 'lsm_infomatch.md')],
                      'cells': {k: {'auroc': v['auroc'], 'n': v['n_tasks']}
                                for k, v in out['cells'].items()}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
