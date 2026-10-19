#!/usr/bin/env python3
"""Zero-API pre-flight for stage-H pre-authored attack payloads.

For every case in the manifest this script proves, before any paid request:

- the payload strings really enter the agent-readable context on the correct
  user path (present in a ground-truth tool output);
- which tool output carries each payload;
- the benchmark evaluator is sane for the pair (GT injection -> security True,
  GT benign user task -> security False);
- a sha256 of every payload for the freeze record.

Usage:
  python \
      code/repro/final_table2/verify_payloads_v2.py \
      --manifest LOGS/final_table2/attack_dev_v2_20260919/manifest_stageH.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/final_five_tables'))

from agentdojo.agent_pipeline.ground_truth_pipeline import (  # noqa: E402
    GroundTruthPipeline)
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402


def norm(text: str) -> str:
    for esc, raw in (('\\n', ' '), ("\\'", "'"), ('\\"', '"'), ('\\\\', '\\'),
                     ("''", "'")):
        text = text.replace(esc, raw)
    return ' '.join(text.split())


def delivery_map(suite, user_task, injections):
    """tool output -> payload keys whose text appears in it (GT path)."""
    env = suite.load_and_inject_default_environment(injections)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, messages, _ = GroundTruthPipeline(user_task).query(
        user_task.PROMPT, runtime, env)
    hits: dict[str, list[str]] = {}
    for msg in messages:
        if msg.get('role') != 'tool':
            continue
        content = norm(get_text_content_as_str(msg['content']))
        tool = msg['tool_call'].function
        for key, payload in injections.items():
            if norm(payload) and norm(payload) in content:
                hits.setdefault(tool, [])
                if key not in hits[tool]:
                    hits[tool].append(key)
    return hits


def evaluator_sane(suite, user_task, inj_task):
    env = suite.load_and_inject_default_environment({})
    _, sec_attack = suite.run_task_with_pipeline(
        GroundTruthPipeline(inj_task), user_task, inj_task, {},
        environment=env)
    env2 = suite.load_and_inject_default_environment({})
    _, sec_benign = suite.run_task_with_pipeline(
        GroundTruthPipeline(user_task), user_task, inj_task, {},
        environment=env2)
    return bool(sec_attack), bool(sec_benign)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    out = (Path(args.out) if args.out
           else manifest_path.parent / 'payload_verification.json')

    rows = []
    ok_all = True
    for case in manifest['cases']:
        suite = get_suite('v1.2', case['suite'])
        ut = suite.get_user_task_by_id(case['user_task'])
        inj = suite.get_injection_task_by_id(case['injection_task'])
        injections = case.get('injections') or {}
        if not injections:
            rows.append({
                'case_id': '|'.join([case['suite'], case['user_task'],
                                     case['injection_task'], case['strategy'],
                                     'none']),
                'skipped': 'registry strategy, no pre-authored payload',
                'ok': True,
            })
            print(f"{rows[-1]['case_id']:70s} skipped (registry strategy)",
                  flush=True)
            continue
        hits = delivery_map(suite, ut, injections)
        undelivered = [k for k in injections
                       if not any(k in v for v in hits.values())]
        sec_attack, sec_benign = evaluator_sane(suite, ut, inj)
        sane = sec_attack and not sec_benign
        payload_sha = {k: hashlib.sha256(v.encode()).hexdigest()[:16]
                       for k, v in injections.items()}
        delivered = not undelivered and bool(injections)
        ok = delivered and sane
        ok_all = ok_all and ok
        rows.append({
            'case_id': '|'.join([case['suite'], case['user_task'],
                                 case['injection_task'], case['strategy'],
                                 'none']),
            'delivered': delivered,
            'undelivered_payload_keys': undelivered,
            'delivery_tools': hits,
            'evaluator_attack_gt': sec_attack,
            'evaluator_benign_gt': sec_benign,
            'evaluator_sane': sane,
            'payload_sha256_16': payload_sha,
            'ok': ok,
        })
        print(f"{rows[-1]['case_id']:70s} delivered={delivered} "
              f"sane={sane} tools={list(hits)}", flush=True)

    report = {
        'protocol': manifest.get('protocol'),
        'stage': manifest.get('stage'),
        'manifest_sha256': hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
        'all_ok': ok_all,
        'cases': rows,
    }
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f'all_ok={ok_all} -> {out}', flush=True)
    return 0 if ok_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
