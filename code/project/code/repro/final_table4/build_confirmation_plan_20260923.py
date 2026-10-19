#!/usr/bin/env python3
"""Freeze the confirmation-case manifest and the 1,200-run plan (zero API).

Cases: 100 new families (``envlib_conf``, 25 per env) x 2 paired conditions
(attack_instruction, benign_instruction) = 200 cases. Plan: 200 cases x
3 defenses x 2 backbones x rep 0 = 1,200 runs, all arms on the identical case
list.

Success of the build is validated by tests (counts, semantics, mock anchors);
this script only materialises the frozen files and their hashes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

import envlib_conf  # noqa: E402

OUT = REPO / 'LOGS/final_table4_indep'
BACKBONES = ('deepseek-flash', 'glm-5.3-flash')
ARMS = ('none', 'monitor', 'grad')
CONDITIONS = ('attack_instruction', 'benign_instruction')


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    families = envlib_conf.all_families()
    cases = []
    for f in families:
        for cond in CONDITIONS:
            cid = f"{f['env']}:{f['family']}:{cond}"
            payload = (f['note_bad'] if cond == 'attack_instruction'
                       else f['note_good'])
            cases.append({
                'case_id': cid, 'env': f['env'], 'family': f['family'],
                'condition': cond, 'action': f['action'],
                'detection_target': str(f['bad_arg']),
                'compliant_target': str(f['good_arg']),
                'payload': payload,
                'payload_sha256': sha256_text(payload),
                's1': f['s1'], 's2': f['s2'],
            })
    manifest = {
        'name': 'table4_independent_confirmation',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'env_module': 'envlib_conf',
        'env_module_sha256': hashlib.sha256(
            (REPO / 'code/repro/v10/envlib_conf.py').read_bytes()).hexdigest(),
        'n_families': len(families),
        'n_cases': len(cases),
        'n_attack_cases': sum(1 for c in cases
                              if c['condition'] == 'attack_instruction'),
        'n_benign_cases': sum(1 for c in cases
                              if c['condition'] == 'benign_instruction'),
        'families_by_env': {e: sum(1 for f in families if f['env'] == e)
                            for e in ('shop', 'workspace', 'finance', 'cloud')},
        'cases': cases,
    }
    mp = OUT / 'cases_manifest.json'
    mp.write_text(json.dumps(manifest, indent=1))

    entries = []
    for c in cases:
        for arm in ARMS:
            for backbone in BACKBONES:
                entries.append({'backbone': backbone, 'env': c['env'],
                                'family': c['family'],
                                'condition': c['condition'], 'arm': arm,
                                'rep': 0})
    plan = {
        'name': 'table4_independent_confirmation',
        'kind': 'confirmation', 'phase': 'confirmation',
        'frozen_at_utc': manifest['frozen_at_utc'],
        'family_role': 'all 100 families are new; disjoint from dev env, '
                       'gate train/dev/test and all previous closed-loop cases',
        'reps': 1, 'entries': entries,
    }
    pp = OUT / 'plan_main.json'
    pp.write_text(json.dumps(plan, indent=1))

    print(json.dumps({
        'manifest': str(mp), 'manifest_sha256': hashlib.sha256(
            mp.read_bytes()).hexdigest(),
        'plan': str(pp), 'plan_sha256': hashlib.sha256(
            pp.read_bytes()).hexdigest(),
        'n_cases': len(cases), 'n_entries': len(entries)}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
