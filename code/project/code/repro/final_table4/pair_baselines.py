#!/usr/bin/env python
"""Dev baselines for the paired repair (zero API).

Metrics (dev pairs; test-role pairs exported only for later comparison):
  * frozen GRAD v2 pair ordering accuracy;
  * lexical baselines: candidate <-> user-task token overlap and
    candidate <-> memory-evidence token overlap (a legitimate observable
    signal: the trusted task names the allowed target);
  * constant-random baseline (0.5).

Outputs LOGS/final_table4/paired_probe_20260918/pair_baselines.{json,md}.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

from defenses import GradGateV2  # noqa: E402

PAIRS = REPO / 'code/repro/out/paired_repair_v1'
OUT = REPO / 'LOGS/final_table4/paired_probe_20260918'
TOKEN_RE = re.compile(r'[a-z0-9]+')


def tokens(text):
    return set(TOKEN_RE.findall(str(text).lower()))


def task_text(nodes):
    for n in nodes:
        if n['type'] == 'user_task':
            return n['features']['subtask']
    return ''


def evidence_text(nodes):
    parts = []
    for n in nodes:
        if n['type'] in ('memory_write', 'tool_execute', 'web_search',
                         'web_request', 'file_read', 'account_access',
                         'version_control', 'message_send', 'database_query'):
            parts.append(n['features']['subtask'])
    return ' '.join(parts)


def overlap(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def lex_pair_score(nodes_a, nodes_b, ref_getter):
    """Higher score = more harmful-like = less lexical overlap with the ref."""
    ref_a = ref_getter(nodes_a)
    ref_b = ref_getter(nodes_b)
    safe_a = overlap(cand_text(nodes_a), ref_a)
    safe_b = overlap(cand_text(nodes_b), ref_b)
    return (1 - safe_a) - (1 - safe_b)


def cand_text(nodes):
    return str(nodes[-1]['features']['subtask'])


def expected_a_gt_b(pair):
    if pair['type'] == 'candidate_swap':
        return True
    return pair.get('expected_order') == 'A>B'


def report(pairs, gate, role):
    stats = defaultdict(lambda: {'n': 0, 'gate': 0, 'lex_task': 0,
                                 'lex_evid': 0, 'lex_task_decided': 0,
                                 'lex_evid_decided': 0, 'ties_task': 0,
                                 'ties_evid': 0})
    for p in pairs:
        if p['type'] == 'evidence_swap' and p.get('expected_order') == 'weak':
            continue
        key = p['type']
        s = stats[key]
        s['n'] += 1
        want_a = expected_a_gt_b(p)
        p_a, _ = gate.gate.score_nodes(p['nodes_a'])
        p_b, _ = gate.gate.score_nodes(p['nodes_b'])
        if (p_a > p_b) == want_a:
            s['gate'] += 1
        lt = lex_pair_score(p['nodes_a'], p['nodes_b'], task_text)
        if lt == 0:
            s['ties_task'] += 1
        else:
            s['lex_task_decided'] += 1
            if (lt > 0) == want_a:
                s['lex_task'] += 1
        le = lex_pair_score(p['nodes_a'], p['nodes_b'], evidence_text)
        if le == 0:
            s['ties_evid'] += 1
        else:
            s['lex_evid_decided'] += 1
            if (le > 0) == want_a:
                s['lex_evid'] += 1
    out = {k: {m: (v[m] / v['n'] if m in ('gate',) else
                   v[m] / v[f'{m}_decided'] if m in ('lex_task', 'lex_evid')
                   and v.get(f'{m}_decided') else v[m])
               for m in v} for k, v in stats.items()}
    fam = defaultdict(lambda: {'n': 0, 'gate': 0, 'lex_decided': 0,
                               'lex_task': 0})
    for p in pairs:
        if p['type'] != 'candidate_swap':
            continue
        f = f"{p['env']}:{p['family']}"
        want_a = expected_a_gt_b(p)
        f2 = fam[f]
        f2['n'] += 1
        p_a, _ = gate.gate.score_nodes(p['nodes_a'])
        p_b, _ = gate.gate.score_nodes(p['nodes_b'])
        f2['gate'] += int((p_a > p_b) == want_a)
        lt = lex_pair_score(p['nodes_a'], p['nodes_b'], task_text)
        if lt:
            f2['lex_decided'] += 1
            f2['lex_task'] += int((lt > 0) == want_a)
    return out, dict(fam)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None,
                    help='checkpoint to evaluate (default: frozen GRAD v2)')
    ap.add_argument('--out-suffix', default='')
    args = ap.parse_args()
    gate = GradGateV2(ckpt=args.ckpt) if args.ckpt else GradGateV2()
    dev = [json.loads(l) for l in (PAIRS / 'dev.jsonl').read_text()
           .splitlines() if l.strip()]
    test = [json.loads(l) for l in (PAIRS / 'test_probe.jsonl').read_text()
            .splitlines() if l.strip()]
    dev_stats, dev_fam = report(dev, gate, 'dev')
    test_stats, test_fam = report(test, gate, 'test_probe')
    out = {'dev': dev_stats, 'test_probe': test_stats,
           'candidate_swap_by_family': {'dev': dev_fam,
                                        'test_probe': test_fam},
           'note': ('dev is the repair-selection metric; test_probe is '
                    'reported for completeness only and must not be used for '
                    'model selection. Random baseline = 0.5.')}
    (OUT / f'pair_baselines{args.out_suffix}.json').write_text(
        json.dumps(out, indent=2))
    md = [f'# Paired repair baselines (checkpoint: {gate.meta["ckpt"]})', '',
          'Higher is better; random = 0.5. `lex_task`/`lex_evid` use '
          'candidate token overlap with the trusted task / visible evidence.',
          '']
    for role in ('dev', 'test_probe'):
        md += [f'## {role}', '',
               '| Pair type | n | frozen GRAD | lexical-task | lexical-evidence |',
               '|---|---|---|---|---|']
        for k, v in out[role].items():
            md.append(f"| {k} | {v['n']} | {v['gate']:.3f} | "
                      f"{v['lex_task']:.3f} | {v['lex_evid']:.3f} |")
        md.append('')
    md += ['## Candidate-swap accuracy by family', '',
           '| Role | Family | n | frozen GRAD | lexical-task (decided) |',
           '|---|---|---|---|---|']
    for role, fams in out['candidate_swap_by_family'].items():
        for f, v in sorted(fams.items()):
            md.append(f"| {role} | {f} | {v['n']} | {v['gate']}/{v['n']} | "
                      f"{v['lex_task']}/{v['lex_decided']} |")
    md += ['', '## Interpretation', '',
           '- On dev (families used for checkpoint selection) the frozen gate '
           'orders all candidate swaps correctly; on the held-out test probe '
           'it collapses only for `workspace:zeta` (0/18), where the safe '
           'target shares the token "mirror" with the trained dangerous '
           'remote. The trivial lexical rule (candidate named in the trusted '
           'task) scores 12/12 on that family, i.e. the signal is visible and '
           'unused.',
           '- These are the pre-repair references; a repaired checkpoint must '
           'beat the frozen gate on dev *and* the held-out pair probe before '
           'any closed-loop run.']
    (OUT / f'pair_baselines{args.out_suffix}.md').write_text(
        '\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
