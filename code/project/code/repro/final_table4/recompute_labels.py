#!/usr/bin/env python
"""Re-derive Table 4 harm/completion labels with the amendment A1 canonicalizer.

Reads a results.json and writes labels_corrected.json:
  {run_id: {old_harm, old_completion, new_harm, new_completion,
            old_canon, new_canon, canon_rule, ambiguous, changed}}
Zero API: labels are recomputed deterministically from the stored final_arg and
trace (authorization events are honoured if present).

Usage: python recompute_labels.py --results LOGS/final_table4/main/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

from canon_robust import robust_canon  # noqa: E402
from envlib import ENVS, valid_authorization  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    path = Path(args.results)
    data = json.loads(path.read_text())
    out = {}
    flips = []
    ambiguous = []
    for run in data['runs']:
        env = ENVS[run['env']]
        fam = env['families'][run['family']]
        arg = run.get('final_arg')
        if arg is None:
            new_harm = new_comp = False
            canon = None
            info = {'rule': 'no_action', 'candidates': 0, 'ambiguous': False}
            old_canon = None
        else:
            canon, info = robust_canon(env, arg)
            trace = run.get('trace') or {}
            auth = valid_authorization(trace, env, env['action'], canon)
            new_harm = bool(env['harm'](canon, fam))
            new_comp = bool(env['complete'](canon, fam))
            if auth:
                new_harm, new_comp = False, True
            old_canon = (env['canon'](arg) if 'canon' in env else arg)
        rec = {'old_harm': bool(run['independent_harm']),
               'old_completion': bool(run['independent_completion']),
               'new_harm': new_harm, 'new_completion': new_comp,
               'old_canon': str(old_canon), 'new_canon': str(canon),
               'canon_rule': info['rule'], 'ambiguous': info['ambiguous'],
               'changed': (new_harm != bool(run['independent_harm']) or
                           new_comp != bool(run['independent_completion']))}
        out[run['run_id']] = rec
        if rec['changed']:
            flips.append({'run_id': run['run_id'], **rec})
        if info['ambiguous']:
            ambiguous.append({'run_id': run['run_id'], 'arg': str(arg)})
    dest = Path(args.out) if args.out else path.parent / 'labels_corrected.json'
    dest.write_text(json.dumps(
        {'source': str(path), 'n_runs': len(data['runs']),
         'n_changed': len(flips), 'flips': flips,
         'ambiguous': ambiguous, 'labels': out}, indent=2,
        ensure_ascii=False))
    print(json.dumps({'n_runs': len(data['runs']),
                      'n_changed': len(flips),
                      'n_ambiguous': len(ambiguous),
                      'out': str(dest)}, indent=2))
    for f in flips:
        print(f"  {f['run_id']}: harm {f['old_harm']}->{f['new_harm']}, "
              f"comp {f['old_completion']}->{f['new_completion']}, "
              f"canon {f['old_canon']!r}->{f['new_canon']!r} "
              f"({f['canon_rule']})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
