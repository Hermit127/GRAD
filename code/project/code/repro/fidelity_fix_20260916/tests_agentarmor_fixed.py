#!/usr/bin/env python3
"""Behavior acceptance tests for the corrected AgentArmor implementation.

Run: python tests_agentarmor_fixed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agentarmor_fixed as aa  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'[PASS] {name} {detail}')
    else:
        FAIL += 1
        print(f'[FAIL] {name} {detail}')


def node(nid, ntype):
    return {'id': nid, 'type': ntype, 'features': {'subtask': ''}}


def graph(nodes, edges, label=0):
    return {'task_id': 't', 'label': label, 'nodes': nodes, 'edges': edges}


def main() -> int:
    direct = graph(
        [node('n0', 'web_search'), node('n1', 'shell_exec')],
        [{'source': 'n0', 'target': 'n1', 'type': 'data_flow'}], label=1)
    clean = graph(
        [node('n0', 'file_write'), node('n1', 'version_control')],
        [{'source': 'n0', 'target': 'n1', 'type': 'data_flow'}], label=0)
    reads_rev = graph(
        [node('n0', 'web_search'), node('n1', 'memory_write'),
         node('n2', 'file_read'), node('n3', 'code_exec')],
        [{'source': 'n0', 'target': 'n1', 'type': 'data_flow'},
         {'source': 'n2', 'target': 'n1', 'type': 'reads_from'},
         {'source': 'n2', 'target': 'n3', 'type': 'data_flow'}], label=1)
    multi = graph(
        [node('n0', 'web_search'), node('n1', 'account_access'),
         node('n2', 'shell_exec')],
        [{'source': 'n0', 'target': 'n2', 'type': 'data_flow'},
         {'source': 'n1', 'target': 'n2', 'type': 'data_flow'}], label=1)

    d = aa.inspect_graph(direct)
    check('untrusted -> shell_exec blocked (Forbid Int<Mid)', d['pred'] == 1,
          d['violations'][:1])
    check('source node itself is not an action violation',
          all(v['node'] != 'n0' for v in d['violations']))
    inherit = graph(
        [node('n0', 'web_search'), node('n1', 'mystery_action')],
        [{'source': 'n0', 'target': 'n1', 'type': 'data_flow'}], label=0)
    i = aa.inspect_graph(inherit)
    check('unannotated single-source node inherits exactly',
          i['types']['n1']['conf'] == 'L' and i['types']['n1']['int'] == 'L',
          i['types']['n1'])
    check('registry-typed node joins instead of inheriting',
          d['types']['n1']['conf'] == 'M' and d['types']['n1']['int'] == 'L',
          d['types']['n1'])

    c = aa.inspect_graph(clean)
    check('no untrusted ancestor -> allow', c['pred'] == 0,
          {k: v for k, v in c['types'].items()})

    r = aa.inspect_graph(reads_rev)
    check('reads_from is reversed: reader inherits the memory node types',
          r['types']['n2']['int'] == 'L', r['types'])

    m = aa.inspect_graph(multi)
    check('multi-source join: conf max, int min',
          m['types']['n2']['conf'] == 'H' and m['types']['n2']['int'] == 'L',
          m['types']['n2'])
    check('multi-source action blocked', m['pred'] == 1)

    e1 = aa.inspect_graph(
        graph([node('n0', 'account_access'), node('n1', 'email_send')],
              [{'source': 'n0', 'target': 'n1', 'type': 'data_flow'}], label=0),
        conf_exfil=False)
    e2 = aa.inspect_graph(
        graph([node('n0', 'account_access'), node('n1', 'email_send')],
              [{'source': 'n0', 'target': 'n1', 'type': 'data_flow'}], label=0),
        conf_exfil=True)
    check('confidentiality exfil rule only under explicit flag',
          any('HEURISTIC' in v['rule'] for v in e2['violations']) and
          not any('HEURISTIC' in v['rule'] for v in e1['violations']),
          (e1['violations'], e2['violations']))

    print(f'\n{PASS} passed, {FAIL} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
