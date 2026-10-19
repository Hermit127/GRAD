"""Strict prefix replay for the pre-execution gate (offline, zero API).

For each unblocked (arm='none') run, replay the trajectory step by step through
the gate: at every step score the prefix graph with the proposed call as the
last node. Report (a) detection at action-decision points, (b) earliest
pre-execution crossing (lead in tool calls), (c) per-step scores.

NOTE: replay is NOT a closed loop -- blocking changes the subsequent trajectory.
This only measures the gate's pre-execution signal on the observed trajectory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))

from envlib import ENVS  # noqa: E402
from gate_graph import trace_nodes  # noqa: E402
from gnn_gate import GnnGate  # noqa: E402


def score_step(trace: dict, upto: int, candidate: tuple[str, str]) -> float:
    cut_seq = int(trace['events'][upto].get('seq', 0) or 0)
    prefix = {
        'events': trace['events'][:upto],
        'records': trace.get('records', []),
        'mem_events': [m for m in trace.get('mem_events', [])
                       if int(m.get('seq', 0) or 0) <= cut_seq],
    }
    nodes = trace_nodes(prefix, candidate=candidate)
    p, _ = GnnGate.get().score_nodes(nodes)
    return p


def replay_run(rec: dict) -> dict | None:
    env = ENVS[rec['env']]
    action = env['action']
    trace = rec.get('trace') or {}
    events = trace.get('events', [])
    action_idx = [i for i, e in enumerate(events) if e.get('tool') == action]
    if not action_idx:
        return None
    idx = action_idx[0]
    scores = []
    first_cross = None
    for i, ev in enumerate(events):
        p = score_step(trace, i, (ev['tool'], ev['arg']))
        scores.append({'i': i, 'tool': ev['tool'], 'p': round(p, 4)})
        if i < idx and first_cross is None and p >= GnnGate.get().threshold:
            first_cross = i
    final_p = score_step(trace, idx, (events[idx]['tool'], events[idx]['arg']))
    fam = env['families'][rec['family']]
    arg = events[idx]['arg']
    harm = bool(env['harm'](env['canon'](arg), fam)) if 'canon' in env \
        else bool(env['harm'](arg, fam))
    return {
        'run_id': rec['run_id'], 'env': rec['env'], 'family': rec['family'],
        'condition': rec['condition'], 'label': int(harm),
        'n_steps': len(events), 'action_idx': idx,
        'p_at_action': round(final_p, 4),
        'block_at_action': final_p >= GnnGate.get().threshold,
        'first_cross_idx': first_cross,
        'lead_steps': (idx - first_cross) if first_cross is not None else None,
        'scores': scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    rows = []
    for f in args.files:
        data = json.loads(Path(f).read_text())
        for rec in data.get('runs', []):
            if rec.get('arm') != 'none' or rec.get('error'):
                continue
            r = replay_run(rec)
            if r is not None:
                rows.append(r)

    thr = GnnGate.get().threshold
    tp = sum(1 for r in rows if r['label'] == 1 and r['block_at_action'])
    fn = sum(1 for r in rows if r['label'] == 1 and not r['block_at_action'])
    fp = sum(1 for r in rows if r['label'] == 0 and r['block_at_action'])
    tn = sum(1 for r in rows if r['label'] == 0 and not r['block_at_action'])
    leads = [r['lead_steps'] for r in rows if r['label'] == 1 and r['lead_steps']]
    out = {
        'threshold': thr, 'n': len(rows),
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'tpr': tp / (tp + fn) if tp + fn else None,
        'fpr': fp / (fp + tn) if fp + tn else None,
        'mean_lead_steps_when_crossed': (sum(leads) / len(leads)) if leads else None,
        'by_condition': {f'{c}|label{l}': n for (c, l), n in
                         Counter((r['condition'], r['label']) for r in rows).items()},
        'rows': rows,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    md = [f'# 前缀重放（严格前缀；threshold={thr}）', '',
          f'- n={len(rows)}；TP={tp} FN={fn} FP={fp} TN={tn}；'
          f'TPR={out["tpr"]}；FPR={out["fpr"]}',
          f'- 提前检出（首次越阈值→action）平均步数：{out["mean_lead_steps_when_crossed"]}',
          '- 注意：重放 ≠ 闭环；真实阻断会改变后续轨迹。']
    Path(args.out_md).write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
