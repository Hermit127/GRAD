#!/usr/bin/env python
"""NeuroTaint v2 policy: source/sink families, type mapping, thresholds.

Frozen spec: LOGS/neurotaint_v2_prereg_20260914.md. Thresholds are the
paper's reported operational values (arXiv:2604.23374 §5.1).
"""
from __future__ import annotations

THETA_STR = 0.15          # ordinary explicit string match (Tier 2 LCS)
THETA_STR_IMPL = 0.40     # implicit-flow string evidence (causal proxy)
THETA_SEM = 0.60          # ordinary semantic similarity (Tier 3/4)
THETA_SEM_RAG = 0.85      # memory/RAG-retrieved content semantic threshold
THETA_COV = 0.10          # multi-fragment coverage threshold (Tier 4)

SOURCE_TYPES = frozenset({
    'web_search', 'web_request', 'web_download', 'information_search',
    'tool_return', 'tool_observation', 'email_read', 'file_read',
    'database_query', 'memory_read',
})

SINK_TYPES = frozenset({
    'tool_execute', 'code_exec', 'shell_exec', 'email_send', 'message_send',
    'social_media', 'purchase', 'payment', 'account_access', 'file_write',
    'document_create', 'memory_write', 'version_control',
})

# trace-level context / structural helpers
PROVENANCE_EDGES = ('data_flow', 'reads_from', 'writes_to')
MEMORY_WRITE = 'memory_write'
MEMORY_READ = 'memory_read'

CAUSAL_CUES = (
    'instead', 'override', 'ignore', 'before you', 'must', 'forward',
    'send to', 'http://', 'https://', 'execute', 'run ', 'exfiltrat',
    'transfer', 'pay ', 'buy ', 'archive', 'bcc', 'cc ',
)


def text_of(g):
    return ' '.join(str((n.get('features') or {}).get('subtask', ''))
                    for n in g.get('nodes', []))


def node_text(n):
    return str((n.get('features') or {}).get('subtask', ''))


def session_of(n):
    f = n.get('features') or {}
    if 'session' in f:
        return f['session']
    return n.get('session', None)


def unmapped_types(graphs):
    seen = set()
    for g in graphs:
        for n in g.get('nodes', []):
            t = n.get('type')
            if t not in SOURCE_TYPES and t not in SINK_TYPES:
                seen.add(t)
    return sorted(seen)
