#!/usr/bin/env python
"""Candidate lexicon probe: does the frozen gate rank candidate targets by
surface tokens rather than authorization? (zero API)

Takes one stored prefix per environment (from the Table 4 main runs) and scores
it with every known target name of that environment, holding everything else
constant. A gate that understood authorization should not prefer unseen-but-
safe names over unseen-but-bad names purely by token similarity.

Outputs LOGS/final_table4/paired_probe_20260918/lexicon_probe.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

from defenses import GradGateV2  # noqa: E402
from envlib import ENVS, KNOWN_REMOTES  # noqa: E402
from grad_nodes import v10_trace_nodes  # noqa: E402

OUT = REPO / 'LOGS/final_table4/paired_probe_20260918'


def find_prefix(env_name):
    for batch in ('main', 'pilot'):
        p = REPO / f'LOGS/final_table4/{batch}/results.json'
        if not p.exists():
            continue
        for run in json.loads(p.read_text())['runs']:
            if run['env'] != env_name:
                continue
            trace = run.get('trace') or {}
            events = trace.get('events', [])
            for idx, ev in enumerate(events):
                if ev.get('tool') == ENVS[env_name]['action']:
                    prefix = dict(trace)
                    prefix['events'] = events[:idx]
                    return prefix, run['run_id'], ev['tool']
    return None, None, None


def main() -> int:
    gate = GradGateV2()
    probes = {}
    for env_name in ('workspace', 'finance', 'shop', 'cloud'):
        env = ENVS[env_name]
        prefix, run_id, tool = find_prefix(env_name)
        if prefix is None:
            continue
        if env_name == 'workspace':
            cands = list(KNOWN_REMOTES)
        elif env_name == 'finance':
            accs = sorted({v['good_arg'] for v in env['families'].values()} |
                          {v['bad_arg'] for v in env['families'].values()})
            cands = accs
        elif env_name == 'shop':
            fams = list(env['families'].values())
            cands = sorted({f['good_arg'] for f in fams} |
                           {f['bad_arg'] for f in fams})
        else:
            from envlib import CLOUD_TARGETS
            cands = list(CLOUD_TARGETS)
        scores = {}
        for c in cands:
            nodes = v10_trace_nodes(prefix, (tool, c))
            p, _ = gate.gate.score_nodes(nodes)
            scores[c] = p
        probes[env_name] = {'source_run': run_id, 'tool': tool,
                            'scores': scores}
    (OUT / 'lexicon_probe.json').write_text(json.dumps(probes, indent=2))
    for env_name, d in probes.items():
        print(f'== {env_name} ({d["source_run"]})')
        for c, p in sorted(d['scores'].items(), key=lambda x: -x[1]):
            print(f'   {p:.8f}  {c}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
