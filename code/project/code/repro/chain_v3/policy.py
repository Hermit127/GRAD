"""Frozen decision policy + isomorphic relational-table relations (v3).

Policy scope: only purchases are risk-scored; other tools always pass.
Order of rules:
  1. explicit denial of the exact proposed ASIN -> block;
  2. explicit authorization of the exact ASIN -> allow;
  3. verified consumption (exact binding) of an untrusted-provenance record
     -> block; heuristic binding of untrusted -> block (recorded as heuristic);
  4. no/unknown binding but the argument matches an untrusted record's
     entities -> block (a missing reference is not evidence of no dependency);
  5. otherwise allow (utility preserved; no blanket blocking).
"""
from __future__ import annotations

from chain_v3 import capture as CAP
from chain_v3.auth import authorization_state


def decide(tool: str, arg: str, auth: dict, cons: dict) -> tuple:
    if tool != 'buy':
        return 0.0, 'not_purchase'
    if auth['state'] == 'denied' and auth['asin'] == str(arg).upper():
        return 1.0, 'explicit_denial'
    if auth['state'] == 'authorized' and auth['asin'] == str(arg).upper():
        return 0.0, 'explicit_authorization'
    if cons['untrusted_exact']:
        return 1.0, 'untrusted_exact_consumption'
    if cons['untrusted_heuristic']:
        return 1.0, 'untrusted_heuristic_consumption'
    if cons['coverage'] != 'exact' and cons['arg_matches_untrusted_record']:
        return 1.0, 'missing_or_unverified_binding'
    return 0.0, 'no_untrusted_consumption'


def relational_relations(trace: dict, proposed: dict) -> dict:
    """Pure relational joins (no graph library) over the raw tables."""
    seq = proposed['seq']
    records = trace.get('records', [])
    mem = [m for m in trace.get('mem_events', []) if m['seq'] < seq]
    writes = {(m['record_id'], m['version']): m['seq'] for m in mem
              if m['op'] in ('write', 'update')}
    reads = {}
    for m in mem:
        if m['op'] == 'read':
            reads.setdefault((m['record_id'], m['version']), []).append(m['seq'])
    triggers = []
    for src in proposed.get('param_sources', []):
        if src.get('kind') not in ('memory_handle', 'llm_claim',
                                   'free_text_guess'):
            continue
        key = (src.get('record_id'), src.get('version'))
        rec = next((r for r in records if (r['record_id'], r['version']) == key
                    and writes.get(key, 10 ** 9) < seq), None)
        if rec is None or key not in reads:
            continue
        w, r = writes[key], min(reads[key])
        if w < r < seq:
            triggers.append({'record_id': key[0], 'version': key[1],
                             'write_seq': w, 'read_seq': r,
                             'kind': src.get('kind')})
    return {'triggers': triggers,
            'trigger_refs': sorted({(t['record_id'], t['version'])
                                    for t in triggers})}


def graph_relations(graph: dict) -> dict:
    proposed = next((n for n in graph['nodes']
                     if (n.get('features') or {}).get('proposed')), None)
    refs = []
    if proposed is not None:
        for e in graph['edges']:
            if e['type'] == 'triggers' and e['target'] == proposed['id']:
                node = next(n for n in graph['nodes'] if n['id'] == e['source'])
                f = node.get('features') or {}
                refs.append((f.get('record_id'), f.get('version')))
    return {'trigger_refs': sorted(set(refs))}


def graph_decision(graph: dict, trace: dict, proposed: dict) -> tuple:
    auth = authorization_state(trace, trace.get('user_id', 'user'),
                               proposed.get('arg', ''), proposed['seq'])
    cons = CAP.consumption(trace, proposed)
    rel = graph_relations(graph)
    exact = {(r['record_id'], r['version']) for r in cons['exact']}
    heur = {(r['record_id'], r['version']) for r in cons['heuristic']}
    by_id = {(r['record_id'], r['version']): r for r in trace.get('records',
                                                                  [])}
    untrusted_exact = any(CAP.provenance_untrusted(
        trace.get('records', []),
        {'record_id': k[0], 'version': k[1]}) for k in rel['trigger_refs'])
    cons_g = dict(cons)
    cons_g['untrusted_exact'] = untrusted_exact
    cons_g['untrusted_heuristic'] = False
    cons_g['coverage'] = 'exact' if (
        rel['trigger_refs'] and set(rel['trigger_refs']) <= exact) else \
        cons['coverage']
    del by_id, heur
    return decide(proposed['tool'], proposed.get('arg', ''), auth, cons_g) + \
        (auth, rel)


def relational_decision(trace: dict, proposed: dict) -> tuple:
    auth = authorization_state(trace, trace.get('user_id', 'user'),
                               proposed.get('arg', ''), proposed['seq'])
    cons = CAP.consumption(trace, proposed)
    rel = relational_relations(trace, proposed)
    exact = {(r['record_id'], r['version']) for r in cons['exact']}
    cons_r = dict(cons)
    cons_r['untrusted_exact'] = any(
        CAP.provenance_untrusted(trace.get('records', []),
                                 {'record_id': k[0], 'version': k[1]})
        for k in rel['trigger_refs'])
    cons_r['coverage'] = 'exact' if (
        rel['trigger_refs'] and set(rel['trigger_refs']) <= exact) else \
        cons['coverage']
    del exact
    return decide(proposed['tool'], proposed.get('arg', ''), auth, cons_r) + \
        (auth, rel)
