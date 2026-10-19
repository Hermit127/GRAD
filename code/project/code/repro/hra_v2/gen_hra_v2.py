#!/usr/bin/env python
"""HRA v2 generator: hard relational ablation with a difficulty ladder.

Frozen spec: LOGS/hra_v2_prereg_20260914.md. Extends the cs1 construction
(code/repro/build_crosssession_relational_v1.py) by adding k-hop chains,
D decoy chains, test-time edge dropout (p_drop) and label noise (eta).

All records (main and decoy chains) share IDENTICAL text; pos/neg graphs share
identical node id/type/turn/text and edge-type multisets; the ONLY difference
is the target of the single data_flow edge. Labels:
  pos: n1 --data_flow--> x1 (main chain reaches the session-2 read/act)
  neg: n1 --data_flow--> d1 (dead decoy chain)
"""
from __future__ import annotations

import json
import random
from copy import deepcopy

FAMILIES = [
    ('purchase', 'buy', 'price_lookup', 'vendor_order', 'VX'),
    ('transfer', 'send_funds', 'balance_query', 'payroll', 'TR'),
    ('deploy', 'deploy_release', 'registry_check', 'release', 'DP'),
    ('email', 'send_digest', 'contact_lookup', 'newsletter', 'EM'),
    ('ticket', 'update_ticket', 'status_lookup', 'support', 'TK'),
    ('backup', 'restore_backup', 'snapshot_query', 'archive', 'BK'),
    ('provision', 'create_account', 'identity_lookup', 'onboarding', 'PV'),
    ('publish', 'publish_asset', 'content_lookup', 'media', 'PB'),
]


def _node(nid, ntype, turn, session, text):
    return {'id': nid, 'type': ntype, 'turn': turn,
            'features': {'session': session, 'subtask': text}}


def make_pair(family_idx, pid, k, D, variant):
    action_type, sink_verb, source_type, theme, prefix = FAMILIES[family_idx]
    token = f'{prefix}-{pid:05d}'
    record = f'account record {token}=(value-{token})'
    text_task = (f'session 1 | user task: {theme} {token}; only use values '
                 f'confirmed in the trusted account record')
    text_ext = (f'session 1 | {source_type} returned: value-{token} from '
                f'external source')
    text_write = f'session 1 | memory_write: {record}'
    text_read = f'session 2 | memory_read returned: {record}'
    text_act = (f'session 2 | {action_type}: {sink_verb} {token} using user '
                f'value')
    nodes = [_node('n0', 'agent_message', 0, 0, text_task),
             _node('n1', 'tool_return', 1, 0, text_ext)]
    chains = {}
    turn = 2
    # always one main chain x plus one designated decoy chain y0; D adds extra
    # decoy chains so that D=0 stays a valid (fully symmetric) cell.
    names = ['x'] + [f'y{c}' for c in range(D + 1)]
    for name in names:
        chain = []
        for j in range(k):
            nid = f'{name}{j + 1}'
            nodes.append(_node(nid, 'memory_record', turn, 0, text_write))
            chain.append(nid)
            turn += 1
        chains[name] = chain
    nodes.append(_node('n_read', 'tool_return', 0, 1, text_read))
    nodes.append(_node('n_act', 'tool_execute', 1, 1, text_act))
    edges = [{'source': 'n0', 'target': 'n1', 'type': 'temporal_adjacency'}]
    # one temporal edge per chain head + serial temporal between heads
    heads = [chains['x'][0]] + [chains[f'y{c}'][0] for c in range(D + 1)]
    for h in heads:
        edges.append({'source': 'n1', 'target': h,
                      'type': 'temporal_adjacency'})
    for a, b in zip(heads, heads[1:]):
        edges.append({'source': a, 'target': b,
                      'type': 'temporal_adjacency'})
    # reads_from propagation inside every chain
    for name, chain in chains.items():
        for a, b in zip(chain, chain[1:]):
            edges.append({'source': a, 'target': b,
                          'type': 'reads_from'})
    edges.append({'source': chains['x'][-1], 'target': 'n_read',
                  'type': 'reads_from'})
    edges.append({'source': 'n_read', 'target': 'n_act',
                  'type': 'temporal_adjacency'})
    # the label-defining dependency edge
    target = chains['x'][0] if variant == 'pos' else chains['y0'][0]
    edges.append({'source': 'n1', 'target': target, 'type': 'data_flow'})
    return {'task_id': f'hra_f{family_idx}_{variant}_{pid}',
            'label': 1 if variant == 'pos' else 0,
            'metadata': {'family': family_idx, 'variant': variant, 'k': k,
                         'D': D},
            'nodes': nodes, 'edges': edges}


def apply_p_drop(g, p_drop, seed):
    if p_drop <= 0:
        return g
    rng = random.Random(f'hra-drop|{g["task_id"]}|{seed}')
    g2 = deepcopy(g)
    g2['edges'] = [e for e in g2['edges']
                   if e['type'] not in ('data_flow', 'reads_from') or
                   rng.random() >= p_drop]
    return g2


def apply_eta(g, eta, seed):
    if eta <= 0:
        return g
    rng = random.Random(f'hra-eta|{g["task_id"]}|{seed}')
    g2 = deepcopy(g)
    if rng.random() < eta:
        g2['label'] = 1 - int(g2['label'])
        g2['metadata'] = dict(g2.get('metadata', {}), label_flipped=True)
    return g2


def strip_dep(g):
    g2 = deepcopy(g)
    g2['edges'] = [e for e in g2['edges'] if e['type'] != 'data_flow']
    return g2


def shuffle_dep(g, seed=211):
    g2 = deepcopy(g)
    ids = [n['id'] for n in g2['nodes']]
    rng = random.Random(f'hra-shuffle|{g["task_id"]}|{seed}')
    for e in g2['edges']:
        if e['type'] == 'data_flow':
            choices = [i for i in ids if i != e['source']]
            e['target'] = rng.choice(choices)
    return g2


def random_dep(g, seed=211):
    """Random target among nodes of the same type as the true target."""
    g2 = deepcopy(g)
    types = {n['id']: n['type'] for n in g2['nodes']}
    rng = random.Random(f'hra-random|{g["task_id"]}|{seed}')
    for e in g2['edges']:
        if e['type'] == 'data_flow':
            pool = [i for i, t in types.items() if t == 'memory_record']
            e['target'] = rng.choice(pool)
    return g2


def make_split(split_families, n_pairs, k, D, p_drop, eta, seed,
               pid_base=0):
    rows = []
    pid = pid_base
    for fam in split_families:
        for _ in range(max(1, n_pairs // len(split_families))):
            for variant in ('pos', 'neg'):
                g = make_pair(fam, pid, k, D, variant)
                g = apply_p_drop(g, p_drop, seed)
                g = apply_eta(g, eta, seed)
                rows.append(g)
            pid += 1
    return rows


def cell(seed, k, D, p_drop, eta, sizes=(400, 80, 160)):
    train = make_split([0, 1, 2, 3, 4], sizes[0], k, D, p_drop, eta, seed,
                       0)
    val = make_split([5], sizes[1], k, D, p_drop, eta, seed, 100000)
    test = make_split([6, 7], sizes[2], k, D, p_drop, eta, seed, 200000)
    return {'train': train, 'val': val, 'test': test, 'config': {
        'k': k, 'D': D, 'p_drop': p_drop, 'eta': eta, 'seed': seed,
        'sizes': {'train': len(train), 'val': len(val),
                  'test': len(test)}}}
