"""Trace -> V2 graph nodes for the pre-execution gate (v10 mock environments).

`train_gnn_v2.graph_to_v2_data` rebuilds edges from node type/turn/text, so this
builder only has to emit ordered nodes with the right `type` and `features`.
Mock-env tools are mapped onto the 25-type space used by the unified backbone.
"""
from __future__ import annotations

TOOL_TYPE = {
    'search': 'web_search', 'scan': 'web_search', 'lookup': 'web_search',
    'click': 'web_request',
    'remember': 'memory_write', 'recall': 'memory_read',
    'buy': 'tool_execute', 'pay': 'account_access',
    'push': 'version_control', 'share': 'message_send',
}


def _node(idx: int, tool: str, arg: str) -> dict:
    return {
        'id': f'n{idx}',
        'type': TOOL_TYPE.get(tool, 'tool_execute'),
        'taint': 0,
        'turn': idx,
        'features': {
            'subtask': f'{tool} {arg}'.strip(),
            'tool_name': tool,
            'command': str(arg),
        },
    }


def trace_nodes(trace: dict, candidate: tuple[str, str] | None = None) -> list[dict]:
    """Executed events (minus `end`) + optional proposed call as the last node.

    Memory-write events are materialized as `memory_write` nodes carrying the
    record content, mirroring the MCP benchmark graph schema. Without them the
    graph would only see tool calls and lose the poisoned-note signal.
    """
    records = {(r['record_id'], r.get('version', 1)): r
               for r in trace.get('records', [])}
    writes_by_event: dict[str, list[dict]] = {}
    for m in trace.get('mem_events', []):
        if m.get('op') == 'write':
            writes_by_event.setdefault(str(m.get('source_event_id', '')), []).append(m)

    nodes = []
    for ev in trace.get('events', []):
        tool = ev.get('tool')
        if tool in (None, 'end'):
            continue
        nodes.append(_node(len(nodes), str(tool), str(ev.get('arg', ''))))
        for m in writes_by_event.get(str(ev.get('event_id', '')), []):
            rec = records.get((m['record_id'], m.get('version', 1)))
            content = str(rec.get('content', '')) if rec else ''
            node = _node(len(nodes), 'remember', content[:200])
            node['type'] = 'memory_write'
            node['features']['source_kind'] = rec.get('source_kind', '') if rec else ''
            nodes.append(node)
    if candidate is not None:
        nodes.append(_node(len(nodes), str(candidate[0]), str(candidate[1])))
    return nodes
