#!/usr/bin/env python3
"""Zero-API case inventory for the round-2 AgentDojo attack development.

For every candidate user task of each v1.2 suite this scan records, using the
ground-truth pipeline only (no LLM calls, no cost):

- which injection vectors are reachable on the correct user path and through
  which tool outputs (canary method, same rule as ``BaseAttack``);
- the full ground-truth tool-call sequence of the user task;
- per injection task: the target calls its ``ground_truth`` would perform and
  whether the benchmark evaluator is sane (GT attack -> security True, GT
  benign -> security False);
- for each (user_task, injection_task) pair: whether the target sink tool is
  already on the user path, which is the precondition for parameter-induction
  payloads (the injection steers the user's own call instead of issuing a new
  instruction).

Outputs go to ``LOGS/final_table2/attack_dev_v2_20260919/``. This script never
touches the paid API and never writes outside that directory.
"""
from __future__ import annotations

import argparse
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

OUT = ROOT / 'LOGS/final_table2/attack_dev_v2_20260919'
SUITES = ('banking', 'slack', 'travel', 'workspace')


def norm(text: str) -> str:
    for esc, raw in (('\\n', ' '), ("\\'", "'"), ('\\"', '"'), ('\\\\', '\\'),
                     ("''", "'")):
        text = text.replace(esc, raw)
    return ' '.join(text.split())


def canary_path(suite, user_task):
    """GT run with canary injections: reachable vectors, tool outputs, calls."""
    canaries = {inj: f'<CANARY---{inj}--->'
                for inj in suite.get_injection_vector_defaults()}
    env = suite.load_and_inject_default_environment(canaries)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, messages, _ = GroundTruthPipeline(user_task).query(
        user_task.PROMPT, runtime, env)
    vectors: dict[str, list[str]] = {}
    calls: list[dict] = []
    for msg in messages:
        if msg.get('role') == 'assistant':
            for tc in (msg.get('tool_calls') or []):
                calls.append({'tool': tc.function, 'args': dict(tc.args or {})})
        if msg.get('role') != 'tool':
            continue
        content = norm(get_text_content_as_str(msg['content']))
        for inj, canary in canaries.items():
            if canary in content:
                vectors.setdefault(inj, [])
                tool = msg['tool_call'].function
                if tool not in vectors[inj]:
                    vectors[inj].append(tool)
    return vectors, calls


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


def inj_target_calls(suite, inj_task):
    env = suite.load_and_inject_default_environment({})
    try:
        calls = inj_task.ground_truth(env)
    except Exception as exc:  # noqa: BLE001
        return [], f'{type(exc).__name__}: {str(exc)[:160]}'
    return [{'tool': c.function, 'args': dict(c.args or {})} for c in calls], None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--suites', default=','.join(SUITES))
    ap.add_argument('--max-user-tasks', type=int, default=12,
                    help='scan user_task_0..N-1 per suite')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {'protocol': 'attack-inventory-v2-20260919',
              'max_user_tasks': args.max_user_tasks, 'suites': {}}
    for suite_name in [s for s in args.suites.split(',') if s]:
        suite = get_suite('v1.2', suite_name)
        ut_ids = [f'user_task_{i}' for i in range(args.max_user_tasks)
                  if f'user_task_{i}' in suite.user_tasks]
        print(f'[{suite_name}] {len(ut_ids)} user tasks, '
              f'{len(suite.injection_tasks)} injection tasks', flush=True)

        inj_meta = {}
        probe_ut = suite.get_user_task_by_id(ut_ids[0])
        for iid, inj in sorted(suite.injection_tasks.items()):
            targets, err = inj_target_calls(suite, inj)
            sec_attack, sec_benign = evaluator_sane(suite, probe_ut, inj)
            inj_meta[iid] = {
                'goal': inj.GOAL,
                'target_calls': targets,
                'target_calls_error': err,
                'evaluator_attack_gt': sec_attack,
                'evaluator_benign_gt': sec_benign,
            }
            print(f'  {iid}: gt={sec_attack}/{sec_benign} '
                  f'targets={[c["tool"] for c in targets]}', flush=True)

        ut_meta = {}
        for uid in ut_ids:
            ut = suite.get_user_task_by_id(uid)
            vectors, calls = canary_path(suite, ut)
            sink_tools = sorted({c['tool'] for c in calls})
            ut_meta[uid] = {
                'prompt': ut.PROMPT,
                'reachable_vectors': vectors,
                'gt_calls': calls,
                'gt_tools': sink_tools,
            }
            print(f'  {uid}: vectors={list(vectors)} '
                  f'tools={sink_tools}', flush=True)

        pairs = []
        for uid in ut_ids:
            for iid, meta in inj_meta.items():
                targets = [c['tool'] for c in meta['target_calls']]
                gt_tools = set(ut_meta[uid]['gt_tools'])
                pairs.append({
                    'user_task': uid, 'injection_task': iid,
                    'reachable': bool(ut_meta[uid]['reachable_vectors']),
                    'evaluator_sane': (meta['evaluator_attack_gt']
                                       and not meta['evaluator_benign_gt']),
                    'sink_on_user_path': sorted(set(targets) & gt_tools),
                    'target_tools': targets,
                })
        report['suites'][suite_name] = {
            'user_tasks': ut_meta, 'injection_tasks': inj_meta,
            'pairs': pairs,
        }
        (out / 'inventory.json').write_text(
            json.dumps(report, indent=1, ensure_ascii=False))

    md = ['# Attack inventory v2 (zero API, GT pipeline only)', '',
          'Preconditions per pair: `reachable` (payload can enter context on the',
          'correct path), `evaluator_sane` (GT attack True / benign False), and',
          '`sink_on_user_path` (the injection target call is already on the user',
          'path -> parameter induction can steer it).', '']
    for suite_name, sdata in report['suites'].items():
        md.append(f'## {suite_name}')
        md.append('')
        md.append('| user_task | injectable vectors | GT tools |')
        md.append('|---|---|---|')
        for uid, um in sdata['user_tasks'].items():
            md.append(f"| {uid} | {', '.join(um['reachable_vectors'])} | "
                      f"{', '.join(um['gt_tools'])} |")
        md.append('')
        md.append('| pair | reachable | evaluator sane | sink on path |')
        md.append('|---|---|---|---|')
        for p in sdata['pairs']:
            if not p['reachable']:
                continue
            md.append(f"| {p['user_task']} x {p['injection_task']} | "
                      f"{'yes' if p['reachable'] else 'no'} | "
                      f"{'yes' if p['evaluator_sane'] else 'no'} | "
                      f"{','.join(p['sink_on_user_path']) or '-'} |")
        md.append('')
    (out / 'inventory.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {out} (inventory.json/md)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
