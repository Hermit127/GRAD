#!/usr/bin/env python3
"""Aggregate the five main paper tables from raw results (no manual numbers).

Sub-commands:
  table3   continuous-learning antibody experts (p16 in-domain + s2d cross-dataset)

Design rules (PI 2026-09-16 22:05):
  - every cell maps to raw result files, configs and aggregation scripts;
  - denominators / n_runs / paired units are attached to each value;
  - protocol consistency checks fail loudly instead of silently filling values.

Outputs (per sub-command):
  LOGS/final_five_tables/raw_results/<table>.json     machine-readable cells
  LOGS/final_five_tables/raw_results/<table>.md       human-readable draft
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FF = REPO / 'LOGS/final_five_tables'
RAW = FF / 'raw_results'
RAW.mkdir(parents=True, exist_ok=True)

P16 = REPO / 'code/repro/out/real_cl/p16_aggregate.json'
S2D = REPO / 'code/repro/out/real_cl/s2d_aggregate_v2_20260916.json'

# Table 3 row -> source arm (both panels use the same arm names)
TABLE3_ROWS = [
    ('Sequential fine-tuning', 'naive'),
    ('Replay', 'replay'),
    ('EWC', 'ewc'),
    ('GRAD antibody experts', 'specialists'),
    ('Joint training (offline ref.)', 'joint'),
]


def _cell(summary: dict, arm: str, key: str):
    a = summary.get(arm)
    if not a or a.get(key) is None:
        return None
    v = a[key]
    return {'mean': v[0], 'std': v[1]}


def _ci(summary: dict, arm: str, key: str):
    a = summary.get(arm) or {}
    v = a.get(key)
    return v


def _paired(summary: dict, arm: str):
    a = summary.get(arm) or {}
    return a.get('paired_vs_naive') or {}


def table3() -> dict:
    p16 = json.loads(P16.read_text())
    s2d = json.loads(S2D.read_text())
    assert p16['n_runs'] == 40, f"p16 n_runs={p16['n_runs']} != 40"
    assert s2d['n_runs'] == 10, f"s2d n_runs={s2d['n_runs']} != 10"

    rows = {}
    for label, arm in TABLE3_ROWS:
        rows[label] = {
            'arm': arm,
            'in_domain': {
                'final_macro_acc': _cell(p16['summary'], arm, 'macro_acc'),
                'final_macro_acc_ci': _ci(p16['summary'], arm, 'macro_acc_ci'),
                'forgetting': _cell(p16['summary'], arm, 'forgetting'),
                'forgetting_ci': _ci(p16['summary'], arm, 'forgetting_ci'),
                'paired_vs_sequential': _paired(p16['summary'], arm),
            },
            'cross_dataset': {
                'final_macro_acc': _cell(s2d['summary'], arm, 'macro_acc'),
                'final_macro_acc_ci': _ci(s2d['summary'], arm, 'macro_acc_ci'),
                'forgetting': _cell(s2d['summary'], arm, 'forgetting'),
                'forgetting_ci': _ci(s2d['summary'], arm, 'forgetting_ci'),
                'paired_vs_sequential': _paired(s2d['summary'], arm),
            },
        }

    out = {
        'table': 'table3_antibody_continual_learning',
        'protocol': {
            'in_domain': {
                'task_flow': 'MCP 5 domains',
                'n_runs': p16['n_runs'],
                'units': '20 seeds x 2 orders',
                'forgetting_definition': 'F_T=(1/T)*sum_j [max_{t=j..T} a_tj - a_Tj]',
                'source': str(P16.relative_to(REPO)),
                'report': 'LOGS/real_cl_p16_fixed_results_20260915.md',
            },
            'cross_dataset': {
                'task_flow': 'mcp -> mrt -> decomp -> mg',
                'n_runs': s2d['n_runs'],
                'units': '5 seeds x 2 orders',
                'forgetting_definition': 'F_T=(1/T)*sum_j [max_{t=j..T} a_tj - a_Tj]',
                'source': str(S2D.relative_to(REPO)),
                'report': 'LOGS/real_cl_s2d_fixed_results_20260915.md',
                'aggregation_script': 'code/repro/real_cl_p15_aggregate.py',
            },
            'notes': [
                'Joint training is a finite-budget offline reference, not an upper bound.',
                'MINJA is NOT part of the cross-dataset flow.',
                'Replay keeps the best cross-dataset ACC; retained as-is (no suppression).',
                'Oracle routing is a diagnostic reference only (panel B/Table 5).',
            ],
        },
        'rows': rows,
    }
    (RAW / 'table3.json').write_text(json.dumps(out, indent=1))

    md = ['# Table 3 数据（自动生成）', '',
          '| 行 | A. MCP in-domain ACC | A. Forgetting | B. Cross-dataset ACC | B. Forgetting |',
          '|---|---|---|---|---|']
    for label, arm in TABLE3_ROWS:
        r = rows[label]
        def f(panel, key):
            c = r[panel][key]
            return f"{c['mean']:.3f}±{c['std']:.3f}" if c else 'NR'
        md.append(f"| {label} | {f('in_domain','final_macro_acc')} | {f('in_domain','forgetting')} "
                  f"| {f('cross_dataset','final_macro_acc')} | {f('cross_dataset','forgetting')} |")
    md += ['', '配对（vs Sequential fine-tuning，Wilcoxon 双侧，含 95% bootstrap CI，见 JSON）：', '']
    for label, arm in TABLE3_ROWS:
        p = rows[label]['in_domain']['paired_vs_sequential']
        q = rows[label]['cross_dataset']['paired_vs_sequential']
        if arm == 'naive':
            continue
        md.append(f"- {label}: in-domain ΔACC={p.get('delta_macro_acc'):+.3f} "
                  f"CI{[round(x,3) for x in p.get('delta_macro_acc_ci',[])]} p={p.get('p_macro_acc'):.4g}, "
                  f"ΔF={p.get('delta_forgetting'):+.3f} p={p.get('p_forgetting'):.4g}; "
                  f"cross ΔACC={q.get('delta_macro_acc'):+.3f} "
                  f"CI{[round(x,3) for x in q.get('delta_macro_acc_ci',[])]} p={q.get('p_macro_acc'):.4g}, "
                  f"ΔF={q.get('delta_forgetting'):+.3f} p={q.get('p_forgetting'):.4g}")
    (RAW / 'table3.md').write_text('\n'.join(md) + '\n')
    print(f'table3 written: {RAW / "table3.json"}')
    return out


def table5() -> dict:
    """Panel B (antibody experts & routing) is fully backed by p16 raw runs.

    Panel A (semantic-structural ablations) requires retrained variants and is
    emitted with status pending (see completion matrix).
    """
    p16 = json.loads(P16.read_text())
    assert p16['n_runs'] == 40

    panel_b_rows = [
        ('Shared detector: sequential fine-tuning', 'naive'),
        ('Experts + random routing', 'specialists_random'),
        ('Experts + all-expert aggregation', 'specialists_all'),
        ('Experts + proposed routing', 'specialists'),
        ('Experts + oracle routing', 'specialists_oracle'),
    ]
    rows = {}
    for label, arm in panel_b_rows:
        a = p16['summary'].get(arm) or {}
        router = a.get('router')
        rows[label] = {
            'arm': arm,
            'final_macro_acc': _cell(p16['summary'], arm, 'macro_acc'),
            'final_macro_acc_ci': _ci(p16['summary'], arm, 'macro_acc_ci'),
            'forgetting': _cell(p16['summary'], arm, 'forgetting'),
            'forgetting_ci': _ci(p16['summary'], arm, 'forgetting_ci'),
            'router_hit': ({'mean': router[0], 'std': router[1]}
                           if (router and router[0] is not None) else None),
            'paired_vs_shared': _paired(p16['summary'], arm),
        }

    out = {
        'table': 'table5_ablations',
        'panel_a_semantic_structural': {
            'status': 'pending',
            'reason': 'variant models must be retrained on the unified backbone '
                      '(code/repro/final_five_tables/train_gnn_ablate.py); '
                      'GPU host gpu-server currently unreachable',
            'rows': [
                'Information-matched Monitor', 'GUARDIAN-adapted',
                'GRAD: text only', 'GRAD: structure only',
                'GRAD: chronological edges only', 'GRAD: without relation types',
                'GRAD: without keyword features', 'GRAD: full joint encoder',
            ],
            'columns': ['Decomp', 'MCP', 'MRT'],
            'metric': 'AUROC',
            'full_model_reference': {
                'source': 'code/repro/out/unified_gnn/unified_aggregate.json',
                'note': 'reused only if the ablation protocol (same splits/'
                        'seeds/budget) matches; verified in train_gnn_ablate.py',
            },
        },
        'panel_b_antibody_routing': {
            'protocol': {
                'task_flow': 'MCP 5 domains',
                'n_runs': p16['n_runs'],
                'units': '20 seeds x 2 orders',
                'source': str(P16.relative_to(REPO)),
                'oracle_note': 'diagnostic reference only, not a deployable method',
                'cost_note': 'proposed routing evaluates 1 trained expert per '
                             'example (+router); all-expert aggregation evaluates '
                             'all experts; oracle assumes perfect routing.',
            },
            'rows': rows,
        },
    }
    (RAW / 'table5.json').write_text(json.dumps(out, indent=1))

    md = ['# Table 5 数据（自动生成）', '',
          '## Panel B. Antibody Experts and Routing（MCP in-domain）', '',
          '| 行 | Final ACC | Forgetting | router |', '|---|---|---|---|']
    for label, arm in panel_b_rows:
        r = rows[label]
        def f(c):
            return f"{c['mean']:.3f}±{c['std']:.3f}" if c else '--'
        rt = r['router_hit']
        router_s = f"{rt['mean']:.3f}±{rt['std']:.3f}" if rt else '--'
        md.append(f"| {label} | {f(r['final_macro_acc'])} | {f(r['forgetting'])} "
                  f"| {router_s} |")
    md += ['', '## Panel A. Semantic–Structural（pending）', '',
           '变体训练脚本 `train_gnn_ablate.py` 已计划；GPU 主机不可达 → pending，不填数字。']
    (RAW / 'table5.md').write_text('\n'.join(md) + '\n')
    print(f'table5 written: {RAW / "table5.json"}')
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: aggregate_tables.py <table3|table1|table2|table4|table5>')
        return 2
    cmd = sys.argv[1]
    if cmd == 'table3':
        table3()
        return 0
    if cmd == 'table5':
        table5()
        return 0
    raise SystemExit(f'{cmd}: not implemented yet')


if __name__ == '__main__':
    raise SystemExit(main())
