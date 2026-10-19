"""Source-capture adapter: how a purchase argument binds to memory records.

Binding kinds (runtime-recorded, never inferred from task labels):
  * memory_handle: the agent/tool runtime bound the argument to a record
    handle; verified by exact record version + prior read + argument match.
  * llm_claim: the model claims record refs; verified the same way. Failed
    claims are errors and demote to unknown.
  * free_text_guess: substring/entity overlap only -> heuristic, never exact.
  * absent: no binding information -> unknown (not "no dependency").
"""
from __future__ import annotations

import re

ASIN_RE = re.compile(r'\bB[0-9][0-9A-Z]{3,11}\b')


def extract_entities(content: str) -> list:
    return sorted(set(ASIN_RE.findall(str(content).upper())))


def _visible(trace: dict, proposed: dict) -> dict:
    seq = proposed['seq']
    return {'tasks': [m for m in trace.get('tasks', [])
                      if m['issued_seq'] <= seq],
            'events': [e for e in trace.get('events', []) if e['seq'] < seq],
            'mem_events': [m for m in trace.get('mem_events', [])
                           if m['seq'] < seq],
            'records': trace.get('records', [])}


def _record_at(records, rid, version):
    return next((r for r in records
                 if r['record_id'] == rid and r['version'] == version), None)


def _read_before(mem_events, rid, version, proposed_seq):
    return any(m['op'] == 'read' and m['record_id'] == rid and
               m['version'] == version and m['seq'] < proposed_seq
               for m in mem_events)


def _arg_matches(arg, rec):
    a = str(arg).upper()
    if a in (rec.get('entities') or extract_entities(rec['content'])):
        return True
    return bool(re.search(rf'\b{re.escape(a)}\b', str(rec['content']).upper()))


def verify_binding(trace: dict, proposed: dict) -> dict:
    vis = _visible(trace, proposed)
    arg = proposed.get('arg', '')
    out = {'exact': [], 'heuristic': [], 'errors': [],
           'claimed': 0, 'coverage': 'unknown',
           'unknown_reason': 'no_binding_information'}
    for src in proposed.get('param_sources', []):
        kind = src.get('kind')
        rid, version = src.get('record_id'), src.get('version')
        rec = _record_at(vis['records'], rid, version) if rid is not None \
            else None
        if kind in ('memory_handle', 'llm_claim'):
            out['claimed'] += 1
            if rec is None:
                out['errors'].append({'kind': kind, 'ref': [rid, version],
                                      'error': 'unknown_record'})
                continue
            if not _read_before(vis['mem_events'], rid, version,
                                proposed['seq']):
                out['errors'].append({'kind': kind, 'ref': [rid, version],
                                      'error': 'no_prior_read'})
                continue
            if not _arg_matches(arg, rec):
                out['errors'].append({'kind': kind, 'ref': [rid, version],
                                      'error': 'arg_not_in_record'})
                continue
            out['exact'].append({'record_id': rid, 'version': version})
        elif kind == 'free_text_guess':
            if rec is None:
                out['errors'].append({'kind': kind, 'ref': [rid, version],
                                      'error': 'unknown_record'})
                continue
            if _arg_matches(arg, rec):
                out['heuristic'].append({'record_id': rid, 'version': version,
                                         'evidence': 'text_overlap'})
            else:
                out['errors'].append({'kind': kind, 'ref': [rid, version],
                                      'error': 'no_text_overlap'})
        else:
            out['errors'].append({'kind': str(kind), 'error': 'unknown_kind'})
    if out['exact']:
        out['coverage'] = 'exact'
    elif out['heuristic']:
        out['coverage'] = 'heuristic'
    elif out['errors']:
        out['coverage'] = 'unknown'
        out['unknown_reason'] = 'binding_verification_failed'
    return out


def provenance_untrusted(records, ref) -> bool:
    """Walk derived_from ancestry; unrequested environment observation is
    untrusted anywhere in the chain."""
    by_id = {(r['record_id'], r['version']): r for r in records}
    stack, seen = [(ref['record_id'], ref['version'])], set()
    while stack:
        key = stack.pop()
        if key in seen or key not in by_id:
            continue
        seen.add(key)
        rec = by_id[key]
        if rec.get('origin') == 'env_observation' and \
                rec.get('user_requested') is False:
            return True
        for parent in rec.get('derived_from', []):
            stack.append((parent['record_id'], parent['version']))
    return False


def consumption(trace: dict, proposed: dict) -> dict:
    """Verified consumption by exact/heuristic refs plus the unknown case."""
    vis = _visible(trace, proposed)
    ver = verify_binding(trace, proposed)
    exact_u = [r for r in ver['exact']
               if provenance_untrusted(vis['records'], r)]
    heur_u = [r for r in ver['heuristic']
              if provenance_untrusted(vis['records'], r)]
    arg = str(proposed.get('arg', '')).upper()
    arg_matches_untrusted = any(
        arg in (r.get('entities') or extract_entities(r['content'])) and
        provenance_untrusted(vis['records'],
                             {'record_id': r['record_id'],
                              'version': r['version']})
        for r in vis['records'])
    return {'exact': ver['exact'], 'heuristic': ver['heuristic'],
            'errors': ver['errors'], 'claimed': ver['claimed'],
            'coverage': ver['coverage'],
            'unknown_reason': ver['unknown_reason'],
            'untrusted_exact': bool(exact_u),
            'untrusted_heuristic': bool(heur_u),
            'arg_matches_untrusted_record': arg_matches_untrusted}
