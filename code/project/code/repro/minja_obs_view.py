#!/usr/bin/env python
"""Observation-inclusive MINJA trace view (new auxiliary method version v1).

The frozen fix2 MINJA view used action text only; the environment-injection
confirmation (EXP-2026W37-047) lives in the observations, so this view adds one
`tool_execute` node per recorded observation and chains action -> observation
with `data_flow` edges. The text channel then sees observation content; the
struct channel sees the extra nodes/edges. No label-like node type or edge is
introduced (no `prompt_inject`/attack words).

Used by the development pilot only; it does not modify the frozen fix2 harness
or its checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path('.')
PKG = REPO / 'code/repro/static_big_table_v1'
CONF = REPO / 'code/repro/out/minja_confirmation'
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO / 'code/repro'))
from channels import StructChannel, TextChannel, static_fuse  # noqa: E402
from gated_fusion import fit_gated, predict_gated  # noqa: E402


def action_type(action: str) -> str:
    a = str(action).lower()
    if a.startswith('search['):
        return 'web_search'
    if a.startswith('click['):
        return 'web_request'
    if 'buy' in a:
        return 'purchase'
    if a.startswith('think'):
        return 'agent_message'
    return 'agent_message'


OBS_TYPE = 'tool_execute'
MAX_OBS = 1200


def trace_to_graph_obs(rec: dict, obs: bool = True) -> dict:
    """rec: raw capture record (with trace entries action/observation)."""
    nodes, edges = [], []
    prev = None
    for i, t in enumerate(rec.get('trace', [])):
        aid = f'a{i}'
        nodes.append({'id': aid, 'type': action_type(t.get('action', '')),
                      'turn': i, 'features': {
                          'subtask': str(t.get('action', ''))[:200]}})
        if prev:
            edges.append({'source': prev, 'target': aid, 'type': 'data_flow'})
        prev = aid
        if obs and t.get('observation'):
            oid = f'o{i}'
            nodes.append({'id': oid, 'type': OBS_TYPE, 'turn': i,
                          'features': {
                              'subtask': str(t['observation'])[:MAX_OBS]}})
            edges.append({'source': aid, 'target': oid, 'type': 'data_flow'})
            prev = oid
    return {'task_id': rec['task_id'], 'label': int(rec.get('label', 0)),
            'nodes': nodes, 'edges': edges,
            'metadata': {'view': 'obs_inclusive_v1' if obs else 'action_only',
                         'condition': rec.get('condition')}}


def load_corpus() -> list[dict]:
    """All 156 confirmation records labelled by condition (task-group identity)."""
    manifest = json.load(open(CONF / 'manifest.json'))
    out = []
    for task in manifest['tasks']:
        for cond in manifest['conditions'][task['split']]:
            for rep in manifest['replicates']:
                p = CONF / 'traces' / f"{task['task_id']}__{cond}__r{rep}.json"
                rec = json.load(open(p))
                rec['label'] = int(cond != 'benign')
                rec['condition'] = cond
                rec['split'] = task['split']
                rec['group'] = task['task_id']
                out.append(rec)
    return out


def roc_tpr_at_fpr(scores, labels, budget=0.10) -> float:
    """ROC diagnostic TPR at the lowest achievable FPR <= budget."""
    import math
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)
             if s is not None and not math.isnan(float(s))]
    pos = sum(1 for _, y in pairs if y == 1)
    neg = sum(1 for _, y in pairs if y == 0)
    if not pos or not neg:
        return float('nan')
    ts = sorted({s for s, _ in pairs})
    best = 0.0
    for t in ts:
        fp = sum(1 for s, y in pairs if y == 0 and s >= t)
        if fp / neg > budget:
            continue
        tp = sum(1 for s, y in pairs if y == 1 and s >= t)
        best = max(best, tp / pos)
    return best
