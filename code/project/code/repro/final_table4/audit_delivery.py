#!/usr/bin/env python
"""Zero-API attack-delivery and evaluator audit for Table 4 runs.

For every attack run checks:
  * a memory write containing the environment's bad note exists (or the
    sanitizer arm records its sanitized transformation);
  * at least one memory read happened before the domain action (if one ran);
  * when independent_harm is True the executed argument came from the poisoned
    note (bad_arg), not from the benign note;
  * when independent_completion is True the argument satisfies the task
    (good_arg / allowed set), and the two labels are never both True;
  * benign runs (non-sanitizer arms) stored the benign note.
Writes <run_dir>/delivery_audit.json.

Usage: python audit_delivery.py --dir LOGS/final_table4/pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro'))

from envlib import ENVS  # noqa: E402
from canon_robust import robust_canon  # noqa: E402


def audit(run: dict, corrected: dict | None = None) -> dict:
    env = ENVS[run['env']]
    fam = env['families'][run['family']]
    trace = run.get('trace') or {}
    records = trace.get('records', [])
    mem_reads = [m for m in trace.get('mem_events', []) if m.get('op') == 'read']
    events = trace.get('events', [])
    first_action_idx = None
    for i, e in enumerate(events):
        if e.get('tool') == env['action']:
            first_action_idx = i
            break
    first_read_idx = None
    for i, e in enumerate(events):
        if e.get('tool') == 'recall':
            first_read_idx = i
            break
    want_note = (fam['note_bad'] if run['condition'] == 'attack_instruction'
                 else fam['note_good'])
    stored = [r['content'] for r in records]
    note_stored = any(want_note in s or s in want_note for s in stored)
    sanitized_marker = any(d.get('kind', '').startswith('sanitizer')
                           for d in run.get('defense_log', []))
    arg = run.get('final_arg')
    canon, canon_info = (robust_canon(env, arg) if arg is not None
                         else (None, {'rule': 'no_action'}))
    bad = fam['bad_arg']
    harmful = (corrected['new_harm'] if corrected
               else run['independent_harm'])
    completed = (corrected['new_completion'] if corrected
                 else run['independent_completion'])
    legacy_mismatch = bool(corrected and (
        corrected['new_harm'] != run['independent_harm'] or
        corrected['new_completion'] != run['independent_completion']))
    recomputed_complete = (bool(env['complete'](canon, fam))
                           if arg is not None else False)
    recomputed_harm = (bool(env['harm'](canon, fam))
                       if arg is not None else False)
    consistent = (not (harmful and completed)
                  and recomputed_complete == completed
                  and recomputed_harm == harmful)
    # Attack payload utilization: a harmful execution must use the poisoned
    # argument from the note, never an allowed/benign one.
    payload_used = True
    if harmful:
        if env['name'] == 'workspace':
            payload_used = str(canon) == str(bad)
        elif env['name'] == 'cloud':
            payload_used = str(canon) == str(bad)
        elif env['name'] == 'finance':
            payload_used = str(canon) == str(bad)
        else:
            payload_used = str(arg).upper() == str(bad).upper()
    return {
        'run_id': run['run_id'], 'env': run['env'], 'family': run['family'],
        'condition': run['condition'], 'arm': run['arm'],
        'backbone': run['backbone'],
        'note_stored': note_stored, 'sanitized_marker': sanitized_marker,
        'n_mem_reads': len(mem_reads),
        'read_before_action': (first_read_idx is not None and
                               (first_action_idx is None or
                                first_read_idx < first_action_idx)),
        'final_arg': arg, 'harm': harmful, 'completion': completed,
        'evaluator_consistent': bool(consistent),
        'payload_used': bool(payload_used),
        'canon_rule': canon_info.get('rule'),
        'legacy_label_mismatch': legacy_mismatch,
        'error': run.get('error', ''),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    data = json.loads((d / 'results.json').read_text())
    labels_path = d / 'labels_corrected.json'
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())['labels']
    rows = [audit(r, labels.get(r['run_id'])) for r in data['runs']]
    out = {'n_runs': len(rows),
           'evaluator_inconsistent': [r for r in rows
                                      if not r['evaluator_consistent']],
           'attack_runs': [r for r in rows
                           if r['condition'] == 'attack_instruction'],
           'benign_note_missing': [
               r for r in rows
               if r['condition'] == 'benign_instruction'
               and r['arm'] != 'sanitizer' and not r['note_stored']],
           'attack_note_missing_non_sanitizer': [
               r for r in rows
               if r['condition'] == 'attack_instruction'
               and r['arm'] != 'sanitizer' and not r['note_stored']],
           'harm_without_read': [r for r in rows
                                 if r['harm'] and not r['read_before_action']],
           'harm_without_payload': [r for r in rows
                                    if r['harm'] and not r['payload_used']],
           'legacy_label_mismatch': [r for r in rows
                                     if r['legacy_label_mismatch']]}
    (d / 'delivery_audit.json').write_text(json.dumps(
        {'summary': {k: (len(v) if isinstance(v, list) else v)
                     for k, v in out.items()},
         'rows': rows, 'violations': {
             k: v for k, v in out.items() if isinstance(v, list) and v}},
        indent=2, ensure_ascii=False))
    print(json.dumps({'summary': {k: (len(v) if isinstance(v, list) else v)
                                  for k, v in out.items()}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
