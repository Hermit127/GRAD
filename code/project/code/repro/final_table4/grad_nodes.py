"""v10 trace -> GRAD v2 (format-unified) prefix nodes.

The GRAD gate used for Table 4 is the frozen checkpoint
``code/repro/out/online_gate_v2/gnn_gate_v2.pt``. Its training data was built by
``code/repro/v10/build_gate_dataset_full_20260917.py`` using
``grad_map.format_event_text`` plus the v10-specific single-argument wrapping and
memory-write nodes. This module re-implements the *identical* node text for the
live closed-loop runner; ``tests/test_grad_nodes_identity.py`` asserts byte
equality against the training builder on the historical no-gate runs.

Deployment-side node construction is prefix-only: the caller passes the trace as
it exists *before* the proposed call executes, so the candidate node can never
contain its own result.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/v10'))

import grad_map  # noqa: E402
from gate_graph import _node  # noqa: E402

VALID_STATUS = ('executed', 'proposed')


def tasks_text(trace: dict) -> str:
    """Join the run's user-task entries with ' | ' (same as the training builder)."""
    tasks = trace.get('tasks') or []
    parts = []
    for t in tasks:
        if isinstance(t, str):
            parts.append(t)
        elif isinstance(t, dict):
            parts.append(str(t.get('content') or t.get('text') or ''))
    return ' | '.join(p for p in parts if p)


def v10_trace_nodes(trace: dict, candidate: tuple[str, str]) -> list[dict]:
    """Full-format prefix nodes for a proposed v10 call (status=proposed).

    Prefix = every event currently in ``trace['events']`` (the runner appends an
    event only when it actually executes), so a blocked call is never part of any
    later prefix. Memory writes are materialized exactly as in training.
    """
    records = {(r['record_id'], r.get('version', 1)): r
               for r in trace.get('records', [])}
    writes_by_event: dict[str, list[dict]] = {}
    for m in trace.get('mem_events', []):
        if m.get('op') == 'write':
            writes_by_event.setdefault(
                str(m.get('source_event_id', '')), []).append(m)

    nodes: list[dict] = []
    task_text = tasks_text(trace)
    if task_text:
        nodes.append({
            'id': f'n{len(nodes)}', 'type': 'user_task', 'taint': 0,
            'turn': len(nodes), 'status': 'context',
            'features': {'subtask': grad_map.task_node_text(task_text),
                         'tool_name': 'user_task', 'command': ''},
        })
    for ev in trace.get('events', []):
        tool = ev.get('tool')
        if tool in (None, 'end'):
            continue
        arg = str(ev.get('arg', ''))
        obs = str(ev.get('observation', ''))
        node = _node(len(nodes), str(tool), arg)
        node['status'] = 'executed'
        node['features']['subtask'] = grad_map.format_event_text(
            str(tool), {'arg': arg}, obs, 'executed')
        nodes.append(node)
        for m in writes_by_event.get(str(ev.get('event_id', '')), []):
            rec = records.get((m['record_id'], m.get('version', 1)))
            content = str(rec.get('content', '')) if rec else ''
            mem_node = _node(len(nodes), 'remember', content[:200])
            mem_node['type'] = 'memory_write'
            mem_node['features']['source_kind'] = (
                rec.get('source_kind', '') if rec else '')
            nodes.append(mem_node)
    cand_node = _node(len(nodes), str(candidate[0]), str(candidate[1]))
    cand_node['status'] = 'proposed'
    cand_node['features']['subtask'] = grad_map.format_event_text(
        str(candidate[0]), {'arg': str(candidate[1])}, None, 'proposed')
    nodes.append(cand_node)
    return nodes


def node_texts(nodes: list[dict]) -> list[str]:
    return [str(n['features']['subtask']) for n in nodes]
