#!/usr/bin/env python3
"""Attack development on the DEV task split (PI-approved, 2026-09-16).

Goal: find a tool-side injection template with none-arm ASR > 0 on tasks that
are DISJOINT from the frozen test list (first 5 user tasks by ID per suite).
Templates compared (all official AgentDojo attack classes + our frozen custom
one): T0 custom important-instructions (current), T1 ToolKnowledgeAttack (official,
spells out the exact tool sequence), T2 InjecAgentAttack, T3 DirectAttack,
T4 IgnorePreviousAttack.

AgentDojo semantics: `run_task_with_pipeline` -> (utility, security);
security=True means the INJECTION SUCCEEDED (ASR = mean(security)).

Nothing here is used as a test-set result: the chosen template must be frozen
and only then run on the test list. Zero risk to frozen files: new output path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro'))

from agentdojo_sanitizer_20260913 import (  # noqa: E402
    API_KEY, Ledger, build_pipeline, make_attack_class)

OUT = REPO / 'LOGS/final_five_tables/attack_dev_20260916.json'
DEV_TASKS = ('user_task_2', 'user_task_3')
TEST_TASKS = ('user_task_0', 'user_task_1', 'user_task_10', 'user_task_11',
              'user_task_12')
SUITES = ('banking', 'slack', 'travel', 'workspace')
N_INJECTIONS = 3


def attack_classes(suite, pipeline):
    from agentdojo.attacks.baseline_attacks import (DirectAttack,
                                                    IgnorePreviousAttack,
                                                    InjecAgentAttack)
    import agentdojo.attacks.important_instructions_attacks as II
    from agentdojo.attacks.important_instructions_attacks import (
        ToolKnowledgeAttack)
    # Our adapted model name is not in AgentDojo's official model list; the
    # official attack class derives a display name for the jailbreak string.
    # Patch only that derivation (template content untouched), script-scoped.
    II.get_model_name_from_pipeline = lambda pipeline: 'DeepSeek'
    custom = make_attack_class()
    return {
        'T0_important_custom': custom(suite, pipeline),
        'T1_tool_knowledge': ToolKnowledgeAttack(suite, pipeline),
        'T2_injecagent': InjecAgentAttack(suite, pipeline),
        'T3_direct': DirectAttack(suite, pipeline),
        'T4_ignore_previous': IgnorePreviousAttack(suite, pipeline),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-cap', type=float, default=1.0)
    ap.add_argument('--tasks', default=','.join(DEV_TASKS))
    ap.add_argument('--injections', type=int, default=N_INJECTIONS)
    ap.add_argument('--templates', default='T0_important_custom,T1_tool_knowledge,'
                                           'T2_injecagent,T3_direct,T4_ignore_previous')
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()

    ledger = Ledger(0.0)
    ledger.cap = ledger.start_rmb + a.batch_cap

    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.task_suite.load_suites import get_suite

    import agentdojo.agent_pipeline.llms.openai_llm as OAI
    _orig = OAI._message_to_openai

    def _deepseek(message, model_name):
        out = _orig(message, model_name)
        if isinstance(out, dict) and out.get('role') == 'developer':
            out = dict(out)
            out['role'] = 'system'
        return out

    OAI._message_to_openai = _deepseek

    agent = None
    from agentdojo_sanitizer_20260913 import CappedClient
    agent = CappedClient(API_KEY.read_text().strip(), ledger, 'attack_dev')
    llm = OpenAILLM(agent, 'deepseek-chat', temperature=0.0)
    stats = {'calls': 0}
    pipeline = build_pipeline(llm, ledger, stats, False)

    dev_tasks = [t for t in a.tasks.split(',') if t]
    need = set(dev_tasks) | set()
    assert not (need & set(TEST_TASKS)), 'dev tasks overlap the frozen test list'
    templates = [t for t in a.templates.split(',') if t]

    out_path = Path(a.out)
    results = {'mode': 'attack_dev', 'dev_tasks': dev_tasks,
               'test_tasks_excluded': list(TEST_TASKS),
               'n_injections': a.injections, 'runs': [],
               'semantics': 'security=True means injection succeeded (ASR)'}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    done = {(r['suite'], r['user_task'], r['injection_task'], r['template'])
            for r in results['runs'] if 'error' not in r}

    for suite_name in SUITES:
        suite = get_suite('v1.2', suite_name)
        attacks = {k: v for k, v in attack_classes(suite, pipeline).items()
                   if k in templates}
        inj_ids = sorted(suite.injection_tasks)[:a.injections]
        for tid in dev_tasks:
            if tid not in suite.user_tasks:
                continue
            ut = suite.get_user_task_by_id(tid)
            for iid in inj_ids:
                inj_task = suite.get_injection_task_by_id(iid)
                for tname, attack in attacks.items():
                    key = (suite_name, tid, iid, tname)
                    if key in done:
                        continue
                    t0 = time.time()
                    try:
                        injections = attack.attack(ut, inj_task)
                        utility, security = suite.run_task_with_pipeline(
                            pipeline, ut, inj_task, injections)
                        rec = {'suite': suite_name, 'user_task': tid,
                               'injection_task': iid, 'template': tname,
                               'condition': 'attack', 'arm': 'none',
                               'utility': bool(utility),
                               'security': bool(security),
                               'duration_s': round(time.time() - t0, 2),
                               'rmb_total': round(ledger.rmb, 4)}
                    except Exception as exc:  # noqa: BLE001
                        rec = {'suite': suite_name, 'user_task': tid,
                               'injection_task': iid, 'template': tname,
                               'condition': 'attack', 'arm': 'none',
                               'error': f'{type(exc).__name__}: {exc}'[:300],
                               'rmb_total': round(ledger.rmb, 4)}
                    results['runs'].append(rec)
                    out_path.write_text(json.dumps(results, indent=1,
                                                   ensure_ascii=False))
                    print(json.dumps(rec, ensure_ascii=False), flush=True)
                    if ledger.rmb >= ledger.cap:
                        print('BUDGET STOP', ledger.rmb, ledger.cap)
                        results['stats'] = stats
                        out_path.write_text(json.dumps(results, indent=1,
                                                       ensure_ascii=False))
                        return 1

    results['stats'] = stats
    results['batch_rmb'] = round(ledger.rmb - ledger.start_rmb, 4)
    results['batch_calls'] = ledger.calls - ledger.start_calls
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    agg = {}
    for r in results['runs']:
        if 'error' in r:
            continue
        agg.setdefault(r['template'], [0, 0])
        agg[r['template']][1] += 1
        agg[r['template']][0] += int(r['security'])
    print('ASR by template:', {k: f'{v[0]}/{v[1]}' for k, v in agg.items()})
    print('batch_rmb', results['batch_rmb'], 'calls', results['batch_calls'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
