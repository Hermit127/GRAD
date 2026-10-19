#!/usr/bin/env python3
"""Paid dev-set attack screening runner (stage A/B) for the Flash experiments.

Every run keeps: utility/security, full message trace (so payload delivery and
model refusal can be audited per run), defense decisions, and cost. Budget is
reserved before each API call and hard-stopped by FlashClient.
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
import grad_agentdojo_hook as grad  # noqa: E402
import agentdojo_sanitizer_20260913 as san  # noqa: E402
from flash_budget import FlashClient, BudgetStop  # noqa: E402

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.attacks.attack_registry import ATTACKS  # noqa: E402
from agentdojo.attacks.important_instructions_attacks import ToolKnowledgeAttack  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402

OUT = ROOT / 'LOGS/final_table2/dev_attacks_20260918'


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
        return DeepSeekToolKnowledge(suite)
    return ATTACKS[strategy](suite, None)


class TraceRecorder(BasePipelineElement):
    def __init__(self, inner):
        self.inner = inner
        self.messages = []

    def query(self, query, runtime, env=None, messages=(), extra_args={}):
        result = self.inner.query(query, runtime, env, messages, extra_args)
        self.messages = list(result[3])
        return result


def serialize(messages) -> list[dict]:
    out = []
    for m in messages:
        entry = {'role': m.get('role')}
        if m.get('content') is not None:
            entry['text'] = get_text_content_as_str(m['content'])[:6000]
        if m.get('tool_calls'):
            entry['tool_calls'] = [
                {'function': tc.function, 'args': str(tc.args)[:300]}
                for tc in m['tool_calls']]
        if m.get('role') == 'tool' and m.get('tool_call') is not None:
            entry['tool'] = m['tool_call'].function
        out.append(entry)
    return out


def case_id(case, strategy, arm):
    return '|'.join([case['suite'], case['user_task'],
                     case.get('injection_task') or 'benign', strategy, arm])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--stage', default='A')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--manifest', default=None)
    args = ap.parse_args()
    out = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert args.stage == manifest['stage'], (args.stage, manifest['stage'])
    grid = []
    for c in manifest['cases']:
        if c.get('strategy'):
            strategies = [c['strategy']]
        else:
            strategies = ([f"benign:{s}" for s in manifest['attack_strategies']]
                          if not c.get('injection_task')
                          else manifest['attack_strategies'])
        grid += [(c, s, a) for s in strategies for a in manifest['arms']]
    if args.dry_run or not args.run:
        for c, s, a in grid:
            print(case_id(c, s, a))
        print(f'{len(grid)} runs, cap {manifest["budget_cap_rmb"]} RMB')
        return 0

    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
    assert importlib.metadata.version('agentdojo') == '0.1.35'
    grad.patch_deepseek_roles()
    grad.MODEL = san.MODEL = 'deepseek-flash'
    client = FlashClient(out, grad.API_KEY, cap=manifest['budget_cap_rmb'])
    san.CappedClient = lambda *a, **kw: client
    llm = OpenAILLM(client, 'deepseek-flash', temperature=0.0)
    gate = None
    if 'grad' in manifest['arms']:
        gate = grad.GradGateClient(url='http://127.0.0.1:8779',
            threshold=manifest['grad_threshold'], fail_mode='closed', timeout=30)
        assert gate.probe()['status'] == 'ok', 'local gate unavailable'

    result_path = out / f'results_stage{args.stage}.json'
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if result_path.exists():
        report = json.loads(result_path.read_text())
        assert report['manifest_sha256'] == manifest_sha
    else:
        report = {'manifest_sha256': manifest_sha, 'runs': []}
    done = {r['case_id'] for r in report['runs']}
    new = 0
    stop = False
    for case, strategy, arm in grid:
        cid = case_id(case, strategy, arm)
        if cid in done:
            continue
        stats = {'calls': 0, 'was_safe_true': 0, 'was_safe_false': 0, 'errors': 0}
        if gate is not None:
            gate.decisions.clear()
        if arm == 'none':
            pipeline = grad.build_pipeline(llm, ToolsExecutor())
        elif arm == 'grad':
            pipeline = grad.build_pipeline(llm, grad.make_grad_executor(gate))
        elif arm == 'sanitizer':
            pipeline = san.build_pipeline(llm, None, stats, True)
        else:
            raise ValueError(arm)
        recorder = TraceRecorder(pipeline)
        suite = get_suite('v1.2', case['suite'])
        ut = suite.get_user_task_by_id(case['user_task'])
        inj = suite.get_injection_task_by_id(case['injection_task']) if case.get('injection_task') else None
        if inj is not None:
            if case.get('injections') is not None:
                injections = dict(case['injections'])
            else:
                attack = make_attack(strategy, suite)
                injections = attack.attack(ut, inj)
        else:
            injections = {}
        client.tag = cid
        t0 = time.perf_counter()
        rec = dict(case, arm=arm, strategy=strategy, case_id=cid, error=None)
        before = client.state['charged_upper_rmb']
        try:
            rec.update(grad.run_one(suite, ut, inj, injections, recorder))
        except Exception as exc:  # noqa: BLE001
            rec.update(error=f'{type(exc).__name__}: {str(exc)[:200]}',
                       utility=None, security=None)
            stop = isinstance(exc, BudgetStop)
        rec['wall_ms'] = (time.perf_counter() - t0) * 1000
        trace = serialize(recorder.messages)
        rec['trace'] = trace
        rec['payload_delivered_in_run'] = any(
            norm(p) in norm(e.get('text', ''))
            for e in trace for p in injections.values())
        rec['tool_calls_in_run'] = [e['tool_calls'] for e in trace if e.get('tool_calls')]
        rec['n_tool_messages'] = sum(1 for e in trace if e['role'] == 'tool')
        rec['decisions'] = list(gate.decisions) if arm == 'grad' and gate else []
        rec['sanitizer_stats'] = stats if arm == 'sanitizer' else None
        rec['defense_module_ms'] = (
            sum(d['latency_ms'] + d.get('build_latency_ms', 0) for d in gate.decisions)
            if arm == 'grad' and gate else
            sum(stats.get('sanitizer_latency_ms', [])) if arm == 'sanitizer' else 0)
        rec['cost_upper_rmb'] = client.state['charged_upper_rmb'] - before
        report['runs'].append(rec)
        result_path.write_text(json.dumps(report, indent=2))
        new += 1
        print(json.dumps({k: rec.get(k) for k in
                          ('case_id', 'utility', 'security', 'error',
                           'payload_delivered_in_run', 'cost_upper_rmb')}) +
              f' total_upper={client.state["charged_upper_rmb"]:.4f}', flush=True)
        if stop or (args.limit and new >= args.limit):
            break
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
