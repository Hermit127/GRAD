"""Faithful snapshot of the MINJA trace -> graph converter (S1 fix).

Source logic: code/repro/fill_table_d.py::minja_to_graphs (node/edge
construction preserved).  Fix per final-acceptance S1: the original instruction
identity is retained as graph-level metadata (`instruction` +
`instruction_key`) instead of being discarded; `task_id` keeps the row index but
is no longer used as the grouping identity.  The instruction does not enter node
features and therefore never enters model input.
"""
from __future__ import annotations

import json
import re


def normalize_instruction(text) -> str:
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def minja_to_graphs(path: str) -> list[dict]:
    d = json.load(open(path))
    out = []
    for i, t in enumerate(d):
        nodes, edges, prev = [], [], None
        for j, s in enumerate(t.get('steps', [])):
            nid = f'm{j}'
            nodes.append({'id': nid, 'type': s.get('type', 'tool_execute'), 'taint': 0,
                          'turn': j, 'features': {'subtask': str(s.get('action', ''))}})
            if prev:
                edges.append({'source': prev, 'target': nid, 'type': 'data_flow'})
            prev = nid
        instruction = str(t.get('instruction', ''))
        out.append({'task_id': f'minja_{i}', 'label': t['label'],
                    'instruction': instruction,
                    'instruction_key': normalize_instruction(instruction),
                    'nodes': nodes, 'edges': edges})
    return out
