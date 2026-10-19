#!/usr/bin/env python3
"""Zero-API feasibility scan for the independent dev attack grid.

For candidate dev user tasks and every injection task of each suite, check:
- which benchmark attack strategies produce injection payloads at all;
- whether each payload is actually delivered on the ground-truth user path;
- whether the injection-task evaluator is sane (GT -> security True, benign -> False);
- whether the tool_knowledge strategy has placeholder args to build its tool plan.

No paid API is used. Selection of the dev grid is frozen in the pre-registration
after this scan.
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
from agentdojo.attacks.attack_registry import ATTACKS  # noqa: E402
from agentdojo.attacks.important_instructions_attacks import ToolKnowledgeAttack  # noqa: E402
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402

OUT = ROOT / 'LOGS/final_table2/dev_attacks_20260918'
SUITES = ['banking', 'slack', 'travel', 'workspace']
DEV_USER_TASKS = ['user_task_0', 'user_task_1', 'user_task_2']
STRATEGIES = ['important_instructions_deepseek', 'direct', 'ignore_previous',
              'system_message', 'injecagent', 'tool_knowledge']


def norm(text: str) -> str:
    for esc, raw in (('\\n', ' '), ("\\'", "'"), ('\\"', '"'), ('\\\\', '\\'),
                     ("''", "'")):
        text = text.replace(esc, raw)
    return ' '.join(text.split())


class _DummyPipeline:
    name = 'gpt-4o-2024-05-13'


class DeepSeekToolKnowledge(ToolKnowledgeAttack):
    name = 'tool_knowledge_deepseek'

    def __init__(self, suite):
        super().__init__(suite, _DummyPipeline())
        self.model_name = 'DeepSeek'


def make_attack(strategy: str, suite):
    if strategy == 'important_instructions_deepseek':
        return grad.make_attack_class()(suite, None)
    if strategy == 'tool_knowledge':
        try:
            return DeepSeekToolKnowledge(suite)
        except Exception as exc:  # noqa: BLE001
            return exc
    return ATTACKS[strategy](suite, None)


def delivery_tools(suite, ut, injections):
    env = suite.load_and_inject_default_environment(injections)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, messages, _ = GroundTruthPipeline(ut).query(ut.PROMPT, runtime, env)
    tools = set()
    for msg in messages:
        if msg.get('role') != 'tool':
            continue
        content = norm(get_text_content_as_str(msg['content']))
        if any(norm(p) in content for p in injections.values()):
            tools.add(msg['tool_call'].function)
    return sorted(tools)


def evaluator_sane(suite, ut, inj):
    env = suite.load_and_inject_default_environment({})
    _, sec_attack = suite.run_task_with_pipeline(
        GroundTruthPipeline(inj), ut, inj, {}, environment=env)
    env2 = suite.load_and_inject_default_environment({})
    _, sec_benign = suite.run_task_with_pipeline(
        GroundTruthPipeline(ut), ut, inj, {}, environment=env2)
    return bool(sec_attack), bool(sec_benign)


def main() -> int:
    rows = []
    for suite_name in SUITES:
        suite = get_suite('v1.2', suite_name)
        for ut_id in DEV_USER_TASKS:
            if ut_id not in suite.user_tasks:
                continue
            ut = suite.get_user_task_by_id(ut_id)
            attacks = {s: make_attack(s, suite) for s in STRATEGIES}
            for inj_id in sorted(suite.injection_tasks):
                inj = suite.get_injection_task_by_id(inj_id)
                sec_attack, sec_benign = evaluator_sane(suite, ut, inj)
                per_strategy = {}
                for s, attack in attacks.items():
                    if isinstance(attack, Exception):
                        per_strategy[s] = {'error': f'{type(attack).__name__}: {attack}'[:160]}
                        continue
                    try:
                        injections = attack.attack(ut, inj)
                    except Exception as exc:  # noqa: BLE001
                        per_strategy[s] = {'error': f'{type(exc).__name__}: {str(exc)[:120]}'}
                        continue
                    per_strategy[s] = {
                        'n_placeholders': len(injections),
                        'delivery_tools': delivery_tools(suite, ut, injections),
                    }
                rows.append({
                    'suite': suite_name, 'user_task': ut_id,
                    'injection_task': inj_id, 'goal': inj.GOAL[:160],
                    'evaluator_attack_gt': sec_attack,
                    'evaluator_benign_gt': sec_benign,
                    'strategies': per_strategy,
                })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'dev_feasibility_scan.json').write_text(json.dumps(rows, indent=1))

    md = ['# Dev attack-grid feasibility scan (zero API)', '',
          '| suite | user_task | injection_task | evaluator ok | delivered strategies | '
          'tools | tool_knowledge plan |', '|---|---|---|---|---|---|---|']
    for r in rows:
        ok = r['evaluator_attack_gt'] and not r['evaluator_benign_gt']
        delivered, tools, tk = [], set(), 'yes'
        for s, v in r['strategies'].items():
            if 'error' in v:
                if s == 'tool_knowledge':
                    tk = 'no: ' + v['error'][:60]
                continue
            if v['delivery_tools']:
                delivered.append(s)
                tools.update(v['delivery_tools'])
        md.append(f"| {r['suite']} | {r['user_task']} | {r['injection_task']} | "
                  f"{'ok' if ok else 'CHECK'} | {len(delivered)}/{len(STRATEGIES)} "
                  f"({', '.join(sorted(delivered)) or '-'}) | "
                  f"{', '.join(sorted(tools)) or '-'} | {tk} |")
    (OUT / 'dev_feasibility_scan.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md[:12]))
    print(f'... {len(rows)} rows -> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
