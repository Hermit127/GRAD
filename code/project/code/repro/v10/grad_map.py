"""AgentDojo tool -> node-type mapping and prefix-node builder for the GRAD gate."""
from __future__ import annotations

import json

TYPE_MAP = {
    'send_money': 'account_access', 'schedule_transaction': 'account_access',
    'update_scheduled_transaction': 'account_access',
    'send_email': 'message_send', 'send_direct_message': 'message_send',
    'post_message': 'social_media', 'post_webpage': 'social_media',
    'invite_user_to_slack': 'message_send', 'add_user_to_channel': 'tool_execute',
    'remove_user_from_channel': 'tool_execute',
    'create_calendar_event': 'document_create', 'reserve_hotel': 'tool_execute',
    'reserve_restaurant': 'tool_execute', 'book_flight': 'tool_execute',
    'create_file': 'file_write', 'append_to_file': 'file_write',
    'delete_file': 'file_delete', 'share_file': 'document_create',
    'update_password': 'system_config', 'update_user_info': 'system_config',
    'read_file': 'file_read', 'read_email': 'file_read', 'read_inbox': 'file_read',
    'read_channel_messages': 'file_read', 'read_document': 'file_read',
    'get_webpage': 'web_request', 'get_most_recent_transactions': 'database_query',
    'get_balance': 'database_query', 'get_iban': 'database_query',
    'get_user_info': 'database_query', 'get_channels': 'database_query',
    'get_users_in_channel': 'database_query', 'get_contacts': 'database_query',
    'get_draft': 'database_query', 'get_emails': 'file_read',
    'search_hotels': 'web_search', 'search_flights': 'web_search',
    'search_flights_by_city': 'web_search', 'search_calendar_events': 'web_search',
    'get_flight_information': 'web_search', 'get_hotels_prices': 'web_search',
    'get_restaurants_address': 'web_search', 'get_all_restaurants_in_city': 'web_search',
    'get_day_calendar_events': 'database_query', 'get_current_day': 'database_query',
}


def format_event_text(tool: str, args: dict | None, result: str | None = None,
                      status: str = 'executed') -> str:
    """Canonical node text. Shared by the training-data builder and the live hook.

    ``status`` is one of ``executed`` / ``blocked`` / ``proposed`` so the encoder
    can tell what actually happened to a prior action (PI 2026-09-17).
    """
    parts = [str(tool),
             'args=' + json.dumps(args or {}, ensure_ascii=False,
                                  sort_keys=True)]
    if result is not None:
        parts.append('result=' + str(result))
    parts.append('status=' + str(status))
    return ' | '.join(parts)


def task_node_text(user_task: str) -> str:
    return 'user_task | ' + str(user_task)


def trace_nodes_full(trace: list[dict], candidate: dict | None = None,
                     user_task: str | None = None) -> list[dict]:
    """Full-visibility node builder (Table 2 GRAD hook).

    Unlike :func:`trace_nodes` (110-char tool+args snippets), this includes:
      - an explicit user-task node (trusted context);
      - for every prior tool call: full JSON arguments, the full tool result
        text (no truncation at this layer) and the action status
        (executed / blocked / proposed);
      - the proposed candidate call with full arguments (status=proposed).

    Node text goes through :func:`format_event_text`, the same formatter the
    offline gate dataset builder uses, so training and deployment share one
    format. The GRAD encoder itself has a fixed per-node character window
    (``MAX_NODE_CHARS`` in train_gnn_v2); that model-side limit is disclosed
    separately and is not applied here.
    """
    nodes = []
    if user_task:
        nodes.append({'id': f'n{len(nodes)}', 'type': 'user_task', 'taint': 0,
                      'turn': len(nodes), 'status': 'context',
                      'features': {'subtask': task_node_text(user_task),
                                   'tool_name': 'user_task'}})
    for e in trace:
        status = str(e.get('status', 'executed'))
        nodes.append({'id': f'n{len(nodes)}', 'type': node_type(e['tool']),
                      'taint': 0, 'turn': len(nodes), 'status': status,
                      'features': {'subtask': format_event_text(
                          e['tool'], e.get('args', {}), e.get('result'),
                          status),
                          'tool_name': e['tool']}})
    if candidate is not None:
        nodes.append({'id': f'n{len(nodes)}', 'type': node_type(candidate['tool']),
                      'taint': 0, 'turn': len(nodes), 'status': 'proposed',
                      'features': {'subtask': format_event_text(
                          candidate['tool'], candidate.get('args', {}), None,
                          'proposed'),
                          'tool_name': candidate['tool']}})
    return nodes


def node_type(tool: str) -> str:
    if tool in TYPE_MAP:
        return TYPE_MAP[tool]
    n = tool.lower()
    if any(k in n for k in ('email', 'message', 'channel', 'slack', 'dm')):
        return 'message_send' if n.startswith(('send', 'post', 'invite', 'add', 'remove')) else 'file_read'
    if any(k in n for k in ('money', 'transaction', 'bank', 'account', 'iban', 'pay')):
        return 'account_access' if n.startswith(('send', 'schedule', 'transfer', 'pay')) else 'database_query'
    if any(k in n for k in ('web', 'url', 'hotel', 'flight', 'restaurant', 'search')):
        return 'web_search'
    if n.startswith(('read', 'get', 'list', 'find')):
        return 'file_read'
    if n.startswith(('send', 'post', 'create', 'update', 'delete', 'remove', 'add', 'share', 'invite')):
        return 'tool_execute'
    return 'tool_execute'


def trace_nodes(trace: list[dict], candidate: dict | None = None) -> list[dict]:
    nodes = []
    for e in trace:
        nodes.append({'id': f'n{len(nodes)}', 'type': node_type(e['tool']), 'taint': 0,
                      'turn': len(nodes),
                      'features': {'subtask': f"{e['tool']} {json.dumps(e.get('args', {}))[:110]}",
                                   'tool_name': e['tool']}})
    if candidate is not None:
        nodes.append({'id': f'n{len(nodes)}', 'type': node_type(candidate['tool']), 'taint': 0,
                      'turn': len(nodes),
                      'features': {'subtask': f"{candidate['tool']} {json.dumps(candidate.get('args', {}))[:110]}",
                                   'tool_name': candidate['tool']}})
    return nodes
