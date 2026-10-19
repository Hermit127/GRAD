#!/usr/bin/env python3
"""AuthGraph online arm on the frozen Flash matched cases (Table 2).

Same frozen case list as the none/sanitizer/grad arms, same agent config
(deepseek-flash, non-thinking, T=0, max 8192), same attack class and evaluator.
AuthGraph runs as a pre-execution check on the visible prefix and blocks
violating calls (the agent continues). Budget-gated by the dedicated Flash
client (pre-call reservation + hard cap from the manifest).

Usage:
  /path/agentdojo-python run_authgraph_online.py            # print manifest
  ... run_authgraph_online.py --run --pilot                 # first 3 runs
  ... run_authgraph_online.py --run                         # full frozen list (resume-safe)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/final_five_tables'))
sys.path.insert(0, str(ROOT / 'code/repro/final_table2'))
import grad_agentdojo_hook as grad
import agentdojo_sanitizer_20260913 as san
from flash_budget import FlashClient, BudgetStop
from authgraph_agentdojo_hook import (FlashLLM, OnlineAuthGraph,
                                      catalog_from_functions,
                                      make_authgraph_executor)

OUT = ROOT / 'LOGS/final_table2/authgraph_online_20260917'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--pilot', action='store_true', help='first 3 cases only')
    args = ap.parse_args()
    manifest = json.loads((OUT / 'manifest.json').read_text())
    manifest_sha = hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest()
    if not args.run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    # cases must remain identical to the source frozen list
    src = json.loads((ROOT / manifest['cases_source']).read_text())
    assert json.dumps(src['cases'], sort_keys=True) == \
        json.dumps(manifest['cases'], sort_keys=True), 'case list drifted'
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.functions_runtime import FunctionsRuntime
    assert importlib.metadata.version('agentdojo') == '0.1.35'
    grad.patch_deepseek_roles()
    grad.MODEL = san.MODEL = 'deepseek-flash'
    client = FlashClient(OUT, grad.API_KEY, cap=manifest['budget_cap_rmb'])
    llm = OpenAILLM(client, 'deepseek-flash', temperature=0.0)

    result_path = OUT / 'results.json'
    report = json.loads(result_path.read_text()) if result_path.exists() else {
        'manifest_sha256': manifest_sha, 'protocol': manifest['protocol'],
        'runs': []}
    assert report['manifest_sha256'] == manifest_sha, 'manifest changed'
    done = {r['case_id'] for r in report['runs']}
    cases = manifest['cases'][:3] if args.pilot else manifest['cases']
    new = 0
    errors = 0
    stop = False
    for case in cases:
        for arm in manifest['arms']:
            assert arm == 'authgraph'
            cid = '|'.join([case['suite'], case['user_task'],
                            str(case['injection_task']), arm])
            if cid in done:
                continue
            suite = get_suite('v1.2', case['suite'])
            ut = suite.get_user_task_by_id(case['user_task'])
            inj = suite.get_injection_task_by_id(case['injection_task']) \
                if case['injection_task'] else None
            catalog = catalog_from_functions(suite.tools)
            ag_llm = FlashLLM(client, max_tokens=2048)
            online = OnlineAuthGraph(ag_llm, catalog)
            executor = make_authgraph_executor(online)
            pipeline = grad.build_pipeline(llm, executor)
            attack = grad.make_attack_class()(suite, pipeline)
            injections = attack.attack(ut, inj) if inj else {}
            client.tag = cid
            t0 = time.perf_counter()
            rec = dict(case, arm=arm, case_id=cid, error=None,
                       protocol_version=manifest['protocol'])
            stop = False
            before = client.state['charged_upper_rmb']
            try:
                rec.update(grad.run_one(suite, ut, inj, injections, pipeline))
                errors = 0
            except Exception as exc:
                rec.update(error=f'{type(exc).__name__}: {str(exc)[:200]}',
                           utility=None, security=None)
                errors += 1
                stop = isinstance(exc, BudgetStop) or errors >= 3
            rec['wall_ms'] = round((time.perf_counter() - t0) * 1000, 1)
            rec['decisions'] = online.decisions
            rec['defense_module_ms'] = online.defense_module_ms()
            rec['planner_ms'] = round(online.planner_ms, 3)
            rec['replan_events'] = online.replan_events
            rec['authgraph_errors'] = online.errors
            rec['planner_error'] = online.planner_error
            rec['degraded'] = bool(online.planner_error) or bool(online.errors)
            rec['ag_n_steps'] = len(online.ag['steps']) if online.ag else 0
            rec['n_blocked'] = sum(1 for d in online.decisions if d['blocked'])
            rec['cost_upper_rmb'] = round(
                client.state['charged_upper_rmb'] - before, 6)
            report['runs'].append(rec)
            result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            new += 1
            print(json.dumps({k: rec.get(k) for k in
                              ('case_id', 'utility', 'security', 'error',
                               'n_blocked', 'degraded', 'cost_upper_rmb')})
                  + f' total_upper={client.state["charged_upper_rmb"]:.4f}',
                  flush=True)
            if stop:
                break
        if stop:
            break
    print(f'done: {new} new runs; batch upper '
          f'{client.state["charged_upper_rmb"]:.4f}/{client.state["cap_rmb"]}')


if __name__ == '__main__':
    main()
