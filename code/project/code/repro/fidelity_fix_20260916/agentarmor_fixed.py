#!/usr/bin/env python3
"""AgentArmor corrected implementation (audit 2026-09-16 fixes).

Paper: arXiv:2508.01249 (sections V-A..V-C, VI-E). Fixes vs
snapshot/agentarmor_repro.py:
  1. full dual-domain type system {confidentiality, integrity} with the
     paper's join semantics (conf most restrictive / max, int least
     restrictive / min), propagated forward with per-edge direction;
  2. first-class rule engine: the paper's published rule
     Forbid {Int < Mid} where Node = ToolName is implemented as a rule object
     applied to action nodes (the old code hardcoded one DANGEROUS_OPS check);
  3. dependency edges follow their direction vocabulary: data_flow / writes_to
     / causal_derivation / triggers forward, reads_from reversed; the old code
     only used data_flow / writes_to / reads_from and treated reads_from
     forward;
  4. Optional, explicitly-labeled confidentiality exfiltration rule (the
     paper does not publish its rule catalogue beyond the Forbid rule; the
     default run keeps only the published rule).

Documented assumptions (paper does not publish its registry, section V-B):
  - per-type initial {confidentiality, integrity} registry in NODE_REGISTRY;
  - ACTION_NODE_TYPES is the merged-node proxy for the paper's ToolName nodes
    (our re-hosted graphs merge ToolName/Tool/Param/Observation into one node).
"""
from __future__ import annotations

from collections import Counter

LOW, MID, HIGH = 0, 1, 2
DOMAIN_NAMES = {0: 'L', 1: 'M', 2: 'H'}

# registry: type -> (confidentiality, integrity); unlisted = (MID, MID)
NODE_REGISTRY = {
    'web_search': (LOW, LOW), 'web_download': (LOW, LOW),
    'web_request': (LOW, LOW), 'information_search': (LOW, LOW),
    'file_read': (MID, LOW), 'memory_read': (MID, LOW),
    'database_query': (MID, LOW), 'data_aggregate': (MID, LOW),
    'account_access': (HIGH, LOW),
    'tool_execute': (MID, HIGH), 'shell_exec': (MID, HIGH),
    'code_exec': (MID, HIGH), 'file_write': (MID, HIGH),
    'memory_write': (MID, HIGH), 'agent_spawn': (MID, HIGH),
    'version_control': (MID, HIGH),
    'email_send': (LOW, MID), 'message_send': (LOW, MID),
    'social_media': (LOW, MID), 'document_create': (LOW, MID),
}

ACTION_NODE_TYPES = frozenset({
    'tool_execute', 'shell_exec', 'code_exec', 'file_write', 'file_delete',
    'memory_write', 'agent_spawn', 'version_control', 'email_send',
    'message_send', 'social_media', 'document_create', 'web_request',
    'system_config', 'database_query', 'account_access', 'data_aggregate',
})

EGRESS_NODE_TYPES = frozenset({'email_send', 'message_send', 'social_media',
                               'web_request', 'document_create'})

EDGE_DIRECTION = {
    'data_flow': +1, 'writes_to': +1, 'causal_derivation': +1, 'triggers': +1,
    'reads_from': -1,
}


def initial_types(nodes):
    """Registry types where published/assumed; None = unannotated runtime node
    (paper: inferred purely from dependencies, not defaulted to Mid)."""
    conf, integ = [], []
    for n in nodes:
        entry = NODE_REGISTRY.get(n.get('type'))
        if entry is None:
            conf.append(None)
            integ.append(None)
        else:
            conf.append(entry[0])
            integ.append(entry[1])
    return conf, integ


def _join_value(cur, incoming, join):
    if incoming is None:
        return cur
    if cur is None:
        return incoming
    return join(cur, incoming)


def propagate(nodes, edges, values, join):
    id2i = {n['id']: i for i, n in enumerate(nodes)}
    changed = True
    while changed:
        changed = False
        for e in edges:
            d = EDGE_DIRECTION.get(e.get('type'), 0)
            s, t = id2i.get(e['source']), id2i.get(e['target'])
            if s is None or t is None or d == 0:
                continue
            src, tgt = (s, t) if d > 0 else (t, s)
            j = _join_value(values[tgt], values[src], join)
            if j != values[tgt]:
                values[tgt] = j
                changed = True
    return values


def conf_join(a, b):
    return max(a, b)


def int_join(a, b):
    return min(a, b)


def rule_forbid_low_integrity_action(idx, node, conf, integ, violations):
    if node.get('type') in ACTION_NODE_TYPES and integ[idx] is not None \
            and integ[idx] < MID:
        violations.append({
            'rule': 'Forbid{Int<Mid} where Node=ToolName',
            'node': node['id'], 'node_type': node.get('type'),
            'int': DOMAIN_NAMES[integ[idx]],
            'conf': DOMAIN_NAMES.get(conf[idx], '?'),
        })


def rule_conf_exfil_heuristic(idx, node, conf, integ, violations):
    if node.get('type') in EGRESS_NODE_TYPES and conf[idx] is not None \
            and conf[idx] > LOW:
        violations.append({
            'rule': 'HEURISTIC{Con>Low -> egress}',
            'node': node['id'], 'node_type': node.get('type'),
            'int': DOMAIN_NAMES.get(integ[idx], '?'),
            'conf': DOMAIN_NAMES[conf[idx]],
        })


def incoming_values(nodes, edges, values, join):
    """Per-node join of in-neighbor values only.

    Our re-hosted graphs merge the paper's ToolName/Tool/Param/Observation
    nodes, so a node's own registry value (e.g. an untrusted retrieval's LOW
    integrity) must not by itself violate the ToolName rule. The paper's
    create_trans example types the ToolName node from its arguments; the
    equivalent here is the incoming-edge join."""
    id2i = {n['id']: i for i, n in enumerate(nodes)}
    out = [None] * len(nodes)
    for e in edges:
        d = EDGE_DIRECTION.get(e.get('type'), 0)
        s, t = id2i.get(e['source']), id2i.get(e['target'])
        if s is None or t is None or d == 0:
            continue
        src, tgt = (s, t) if d > 0 else (t, s)
        out[tgt] = _join_value(out[tgt], values[src], join)
    return out


def inspect_graph(graph, conf_exfil=False):
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    conf, integ = initial_types(nodes)
    conf = propagate(nodes, edges, conf, conf_join)
    integ = propagate(nodes, edges, integ, int_join)
    conf_in = incoming_values(nodes, edges, conf, conf_join)
    integ_in = incoming_values(nodes, edges, integ, int_join)
    violations = []
    for idx, node in enumerate(nodes):
        rule_forbid_low_integrity_action(idx, node, conf_in, integ_in, violations)
        if conf_exfil:
            rule_conf_exfil_heuristic(idx, node, conf_in, integ_in, violations)
    return {
        'pred': int(len(violations) > 0),
        'n_violations': len(violations),
        'violations': violations,
        'types': {n['id']: {'conf': DOMAIN_NAMES.get(conf[i], '?'),
                            'int': DOMAIN_NAMES.get(integ[i], '?'),
                            'conf_in': DOMAIN_NAMES.get(conf_in[i], '?'),
                            'int_in': DOMAIN_NAMES.get(integ_in[i], '?')}
                  for i, n in enumerate(nodes)},
    }


def predict(graphs, conf_exfil=False):
    ledgers = [inspect_graph(g, conf_exfil=conf_exfil) for g in graphs]
    preds = [d['pred'] for d in ledgers]
    return preds, ledgers
