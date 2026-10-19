#!/usr/bin/env python3
"""Zero-API audit of attack delivery for the frozen Flash Table 2 panel.

For every panel case this checks three things the PI asked for:
1. payload delivery: the ImportantInstructions payload actually appears in the
   tool output the agent reads when executing the user task correctly;
2. goal reachability + evaluator sanity: the injection task ground truth is
   executable and the evaluator returns security=True on it (and False for a
   benign ground-truth run);
3. runtime cross-check: which tool calls the GRAD arm actually made, so we can
   see whether the delivering call was even issued under Flash.

No paid API is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/final_five_tables'))
import grad_agentdojo_hook as grad  # noqa: E402

from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline  # noqa: E402
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402

PANEL = ROOT / 'LOGS/final_table2/flash_matched_20260917'
AUTHGRAPH = ROOT / 'LOGS/final_table2/authgraph_online_20260917'
OUT = ROOT / 'LOGS/final_table2/attack_audit_20260917'


def load_recorded() -> dict:
    rec: dict[str, dict] = {}
    for path in (PANEL / 'results.json', AUTHGRAPH / 'results.json'):
        for run in json.loads(path.read_text())['runs']:
            base = run['case_id'].rsplit('|', 1)[0]
            rec.setdefault(base, {})[run['arm']] = {
                'utility': run.get('utility'), 'security': run.get('security'),
                'error': run.get('error'),
                'calls': [d['tool'] for d in (run.get('decisions') or [])],
            }
    return rec


def norm(text: str) -> str:
    for esc, raw in (('\\n', ' '), ("\\'", "'"), ('\\"', '"'), ('\\\\', '\\'),
                     ("''", "'")):
        text = text.replace(esc, raw)
    return ' '.join(text.split())


def gt_user_delivery(suite, ut, injections):
    env = suite.load_and_inject_default_environment(injections)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, messages, _ = GroundTruthPipeline(ut).query(ut.PROMPT, runtime, env)
    calls, delivered = [], []
    for msg in messages:
        if msg.get('role') != 'tool':
            continue
        tool = msg['tool_call'].function
        calls.append(tool)
        content = norm(get_text_content_as_str(msg['content']))
        for placeholder, payload in injections.items():
            if norm(payload) in content and tool not in [d['tool'] for d in delivered]:
                delivered.append({'tool': tool, 'placeholder': placeholder,
                                  'placeholder_is_user_task_candidate': True})
    return calls, delivered


def evaluator_sanity(suite, ut, inj, injections):
    env = suite.load_and_inject_default_environment(injections)
    _, security_attack = suite.run_task_with_pipeline(
        GroundTruthPipeline(inj), ut, inj, injections, environment=env)
    env2 = suite.load_and_inject_default_environment(injections)
    _, security_benign = suite.run_task_with_pipeline(
        GroundTruthPipeline(ut), ut, inj, injections, environment=env2)
    return bool(security_attack), bool(security_benign)


def main() -> int:
    manifest = json.loads((PANEL / 'manifest.json').read_text())
    recorded = load_recorded()
    rows = []
    for case in manifest['cases']:
        if not case['injection_task']:
            continue
        suite = get_suite('v1.2', case['suite'])
        ut = suite.get_user_task_by_id(case['user_task'])
        inj = suite.get_injection_task_by_id(case['injection_task'])
        attack = grad.make_attack_class()(suite, None)
        injections = attack.attack(ut, inj)
        calls, delivered = gt_user_delivery(suite, ut, injections)
        sec_attack, sec_benign = evaluator_sanity(suite, ut, inj, injections)
        rec = recorded.get('|'.join([case['suite'], case['user_task'],
                                     str(case['injection_task'])]), {})
        grad_calls = (rec.get('grad') or {}).get('calls', [])
        row = {
            'suite': case['suite'], 'user_task': case['user_task'],
            'injection_task': case['injection_task'], 'scope': case['scope'],
            'goal': inj.GOAL,
            'n_placeholders': len(injections),
            'placeholders': list(injections),
            'payload_chars': len(next(iter(injections.values()))) if injections else 0,
            'gt_user_calls': calls,
            'delivered_in_gt_path': delivered,
            'delivery_tools': sorted({d['tool'] for d in delivered}),
            'evaluator_security_on_injection_ground_truth': sec_attack,
            'evaluator_security_on_user_ground_truth': sec_benign,
            'grad_arm_calls': grad_calls,
            'delivery_call_observed_in_grad_arm': any(
                d['tool'] in grad_calls for d in delivered),
            'recorded': {arm: {k: v for k, v in vals.items() if k != 'calls'}
                         for arm, vals in rec.items()},
        }
        rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'attack_delivery_audit.json').write_text(
        json.dumps({'manifest': str(PANEL / 'manifest.json'), 'rows': rows}, indent=1))

    md = ['# Attack delivery audit (zero API) - frozen Flash Table 2 panel', '',
          '| case | scope | goal reachable / evaluator | delivered on GT path | '
          'delivery tool observed under GRAD arm | recorded security (none/grad/san/authgraph) |',
          '|---|---|---|---|---|---|']
    for r in rows:
        cid = f"{r['suite']}|{r['user_task']}|{r['injection_task']}"
        secs = []
        for arm in ('none', 'grad', 'sanitizer', 'authgraph'):
            v = (r['recorded'].get(arm) or {}).get('security')
            secs.append('--' if v is None else str(v))
        check = ('ok' if r['evaluator_security_on_injection_ground_truth']
                 else 'FAIL') + (' / benign-control ' +
                                 ('False' if not r['evaluator_security_on_user_ground_truth']
                                  else 'True(!)'))
        md.append(
            f"| {cid} | {r['scope']} | {check} | "
            f"{', '.join(r['delivery_tools']) or 'NONE'} | "
            f"{'yes' if r['delivery_call_observed_in_grad_arm'] else 'no'} | "
            f"{'/'.join(secs)} |")
    md += ['', 'Per-case goals:']
    for r in rows:
        md.append(f"- `{r['suite']}|{r['user_task']}|{r['injection_task']}`: {r['goal']}")
    (OUT / 'attack_delivery_audit.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
