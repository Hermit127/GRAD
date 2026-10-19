#!/usr/bin/env python3
"""Behavior acceptance tests for the corrected AEGIS implementation.

Run: python tests_aegis_fixed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aegis_fixed as aegis  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'[PASS] {name} {detail}')
    else:
        FAIL += 1
        print(f'[FAIL] {name} {detail}')


def node(nid, ntype, text, **feat):
    f = {'subtask': text}
    f.update(feat)
    return {'id': nid, 'type': ntype, 'features': f}


def main() -> int:
    check('shallow extraction works', aegis.extract_string_values(
        {'a': {'b': 'x'}}) == ['x'])
    deep = cur = {}
    for _ in range(40):
        cur['n'] = {}
        cur = cur['n']
    cur['leaf'] = 'x'
    check('depth overflow returns sentinel',
          aegis.SCAN_TRUNCATED in aegis.extract_string_values(deep))

    sqli = aegis.classify_tool_call('run_query', {'query': "x' OR '1'='1"})
    check('SQL injection signal fires', any(
        r['type'] == 'sql_injection' for r in sqli['risks']), sqli['risks'][:1])
    check('content category inference = database', sqli['category'] == 'database')

    trav = aegis.classify_tool_call('read_file', {'path': '../../etc/passwd'})
    check('path traversal + sensitive file fire',
          {r['type'] for r in trav['risks']} >= {'path_traversal', 'sensitive_file'},
          trav['risks'])

    benign = aegis.classify_tool_call('get_weather', {'city': 'Paris'})
    check('benign call has no risks', benign['risks'] == [], benign)

    pol = aegis.inspect_graph({'nodes': [node(
        'n0', 'database_query',
        'tool_name: run_query arguments: {"query": "DELETE FROM users"} response: ok')]})
    check('published policy + guardrail block destructive SQL',
          pol['pred_block'] == 1 and pol['top_risk'] in ('HIGH', 'CRITICAL'),
          pol['violations'][:1])

    benign_graph = aegis.inspect_graph({'nodes': [node(
        'n0', 'information_search', 'the weather in Paris is sunny')]})
    check('benign node -> allow', benign_graph['pred_block'] == 0,
          benign_graph)

    big = 'x' * 60000
    big_graph = aegis.inspect_graph({'nodes': [node('n0', 'file_write', big)]})
    check('oversized argument flagged (large_payload)',
          big_graph['pred_block'] == 1 and
          'large_payload:' in str(big_graph['violations']), big_graph['violations'][:1])

    raw = aegis.inspect_graph({'nodes': [node(
        'n0', 'tool_execute', 'ignored',
        raw_payload={'tool_name': 'fetch', 'arguments': {'url': 'http://x/y'}})]})
    check('raw_payload adapter used when present',
          raw['violations'] and 'plaintext_url:' in str(raw['violations']),
          raw['violations'][:1])

    check('explicit empty policy list does not enable the default policy',
          aegis.validate_policies({'category': 'database'},
                                  {'query': 'DELETE FROM users'}, []) == ([], []))
    mixed = {'id': 'mixed_paths', 'schema': {
        'allOf': [{'properties': {'x': {'type': 'integer'}}},
                  {'propertyNames': {'pattern': '^allowed$'}}]}}
    violations, failed = aegis.validate_policies(
        {'category': 'unknown'}, {'x': 'bad'}, [mixed])
    check('multiple schema errors remain reportable',
          len(violations) == 2 and failed == ['mixed_paths'])

    print(f'\n{PASS} passed, {FAIL} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
