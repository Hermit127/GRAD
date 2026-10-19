"""AuthGraph online pre-execution hook for AgentDojo (Flash matched panel).

Design (pre-registered in the AuthGraph online manifest):
  * At task start the Planner generates the authorization graph from the
    clean-context inputs only: ``{user_prompt, tool_catalog}`` (paper
    Property 1). No trajectory data reaches this call (the authgraph Planner
    carries a runtime isolation guard).
  * Every proposed tool call is checked BEFORE execution with the visible
    prefix only: the candidate call is appended with an empty observation, so
    it can never see its own future output. The prefix IRG is constructed
    deterministically from the visible calls/args/raw observation texts (the
    paper's checker consumes only the ordered INVOKE list plus raw
    observations; the LLM Builder construction is exercised offline and
    disclosed as not in the runtime loop). Detection semantics are identical
    to the batch checker (same ``DualPointerChecker``).
  * Blocked calls are not executed; the agent receives a tool-error style
    message and continues execution (same mechanic as the GRAD arm).
  * Replan steps: after a consumed step with ``replan=True`` and no block, the
    Planner generates a sub authorization graph constrained to the frozen
    ``replan_allowed_tools`` whitelist; the sub steps are spliced into the
    traversal and their tools added to S.
  * Errors (Planner/Checker LLM failures, malformed JSON, budget stop) are
    recorded per call and per task; the call under error executes (fail-open)
    and the trajectory is flagged ``degraded``. Errors are never counted as
    successful defense.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/repro/authgraph'))
sys.path.insert(0, str(ROOT / 'code/repro/final_five_tables'))

from checker import DualPointerChecker  # noqa: E402
from core import InvokeCall  # noqa: E402
from llm import JudgeLLM, extract_json_object  # noqa: E402
from planner import Planner  # noqa: E402
from schema import is_write_tool  # noqa: E402

BLOCK_VERDICTS = ('hard_block', 'suspicious', 'param_violation')


def catalog_from_functions(functions) -> list[dict]:
    """Compact tool catalog for the Planner (clean-context input).

    ``functions`` may be a dict (``runtime.functions``) or a list
    (``suite.tools``). ``security_critical`` = all parameters of write-class
    tools, so the completeness rule in ``validate_ag`` has teeth for AgentDojo
    tools whose security-critical fields are not published.
    """
    items = functions.items() if isinstance(functions, dict) \
        else ((getattr(f, 'name', ''), f) for f in functions)
    catalog = []
    for name, fn in items:
        try:
            schema = fn.parameters.model_json_schema()
            props = sorted((schema.get('properties') or {}).keys())
        except Exception:
            props = []
        write = is_write_tool(name)
        catalog.append({
            'name': name,
            'description': str(getattr(fn, 'description', ''))[:200],
            'kind': 'write' if write else 'read',
            'parameters': props,
            **({'security_critical': props} if write else {}),
        })
    return catalog


class FlashLLM:
    """``chat_json`` over the shared Flash budget client (same config as the
    agent model: deepseek-flash, non-thinking, T=0; reservations + hard cap
    enforced by the client)."""

    def __init__(self, client, max_tokens: int = 2048):
        self.client = client
        self.max_tokens = max_tokens
        self.ms = 0.0
        self.calls = 0

    def chat_json(self, system: str, user: str, name: str = 'default') -> dict:
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                messages=[{'role': 'system', 'content': system},
                          {'role': 'user', 'content': user}],
                max_tokens=self.max_tokens)
        finally:
            self.ms += (time.perf_counter() - t0) * 1000
            self.calls += 1
        text = resp.choices[0].message.content
        return extract_json_object(text)


def steps_from_trace(trace_entries: list[dict]) -> list[dict]:
    """grad trace entries -> authgraph prefix steps.

    Blocked entries keep their status but expose NO observation (a synthetic
    block message is not tool output and must not become a provenance
    source)."""
    steps = []
    for i, e in enumerate(trace_entries, start=1):
        blocked = e.get('status') == 'blocked'
        steps.append({'index': i, 'tool': e.get('tool'),
                      'args': dict(e.get('args') or {}),
                      'observation': '' if blocked else (e.get('result') or ''),
                      'status': 'blocked' if blocked else 'executed'})
    return steps


class OnlineAuthGraph:
    def __init__(self, llm: FlashLLM, catalog: list[dict]):
        self.llm = llm
        self.catalog = catalog
        self.user_prompt = ''
        self.ag: dict | None = None
        self.checker: DualPointerChecker | None = None
        self.planner_error: str | None = None
        self.planner_ms = 0.0
        self.replan_ms = 0.0
        self.replan_events: list[dict] = []
        self.decisions: list[dict] = []
        self.errors: list[dict] = []
        self.last_step: dict | None = None
        self.started = False

    # -- task lifecycle -----------------------------------------------------
    def ensure_started(self, user_prompt: str) -> None:
        """First query() supplies the exact visible user prompt; the Planner
        runs then, before any candidate call is judged."""
        if not self.started:
            self.start_task(user_prompt)

    def start_task(self, user_prompt: str) -> None:
        self.started = True
        self.user_prompt = user_prompt or ''
        t0 = time.perf_counter()
        try:
            planner = Planner(self.llm, self.catalog)
            ag = planner.plan(self.user_prompt)
            self.ag = ag
            self.checker = DualPointerChecker(
                ag, judge=JudgeLLM(self.llm), catalog=self.catalog)
        except Exception as exc:  # noqa: BLE001 - recorded, fail-open
            self.planner_error = f'{type(exc).__name__}: {str(exc)[:200]}'
            self.ag = None
            self.checker = None
        finally:
            self.planner_ms = (time.perf_counter() - t0) * 1000

    # -- pre-execution check ------------------------------------------------
    def check(self, tool: str, args: dict, prefix_steps: list[dict]) -> dict:
        """Returns {'blocked': bool, 'decisions': [...], 'degraded': bool}."""
        if self.checker is None:
            self.errors.append({'stage': 'check', 'tool': tool,
                                'error': self.planner_error or 'no_ag'})
            return {'blocked': False, 'decisions': [], 'degraded': True}
        call = InvokeCall(call_index=len(prefix_steps) + 1, tool=tool,
                          tool_node='', observation_node='',
                          params={k: str(v) for k, v in (args or {}).items()})
        trace = {'user_prompt': self.user_prompt, 'tool_catalog': self.catalog,
                 'steps': list(prefix_steps) + [
                     {'index': len(prefix_steps) + 1, 'tool': tool,
                      'args': dict(args or {}), 'observation': ''}]}
        t0 = time.perf_counter()
        try:
            new = self.checker.process(call, trace)
            self.last_step = self.checker.last_consumed_step
        except Exception as exc:  # noqa: BLE001 - recorded, fail-open
            self.errors.append({'stage': 'check', 'tool': tool,
                                'error': f'{type(exc).__name__}: {str(exc)[:200]}'})
            return {'blocked': False, 'decisions': [], 'degraded': True}
        latency = (time.perf_counter() - t0) * 1000
        recs = [{'tool': d.tool, 'layer': d.layer, 'verdict': d.verdict,
                 'param': d.param, 'ag_step': d.ag_step,
                 'blocked': d.verdict in BLOCK_VERDICTS,
                 'reason': d.reason, 'evidence': dict(d.evidence),
                 'latency_ms': round(latency, 3),
                 'build_latency_ms': 0.0, 'degraded': False} for d in new]
        self.decisions.extend(recs)
        return {'blocked': any(r['blocked'] for r in recs),
                'decisions': recs, 'degraded': False}

    # -- post-execution (replan) -------------------------------------------
    def post_execute(self, tool: str, args: dict, observation: str,
                     prefix_steps: list[dict]) -> None:
        step = self.last_step
        self.last_step = None
        if not step or not step.get('replan') or self.checker is None:
            return
        allowed = list(step.get('replan_allowed_tools') or [])
        if not allowed:
            return
        t0 = time.perf_counter()
        event = {'tool': tool, 'step_index': step.get('step_index'),
                 'allowed_tools': allowed, 'status': 'ok', 'error': None}
        try:
            planner = Planner(self.llm, self.catalog)
            sub = planner.replan(self.user_prompt, observation, allowed,
                                 next_step_index=step.get('step_index', 1) + 1,
                                 trigger_tool=tool)
            substeps = sub.get('steps') or []
            for k, s in enumerate(substeps):
                s['step_index'] = step.get('step_index', 1) + 1 + k
            self.checker.splice_substeps(step.get('step_index'), substeps)
            event['n_substeps'] = len(substeps)
            event['tools'] = [s.get('expected_tool') for s in substeps]
        except Exception as exc:  # noqa: BLE001 - recorded, fail-open
            event['status'] = 'error'
            event['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            self.errors.append({'stage': 'replan', 'tool': tool,
                                'error': event['error']})
        finally:
            event['latency_ms'] = round((time.perf_counter() - t0) * 1000, 3)
            self.replan_ms += event['latency_ms']
            self.replan_events.append(event)

    # -- reporting ----------------------------------------------------------
    def defense_module_ms(self) -> float:
        return round(self.planner_ms + self.replan_ms
                     + sum(d['latency_ms'] for d in self.decisions), 3)

    def blocked_text(self, tool: str, decisions: list[dict]) -> str:
        first = decisions[0] if decisions else {}
        return (f'[BLOCKED by AuthGraph] blocked_by_authgraph tool={tool} '
                f"layer={first.get('layer')} verdict={first.get('verdict')} "
                f"reason={str(first.get('reason'))[:300]}")


def make_authgraph_executor(online: OnlineAuthGraph):
    """Build the pre-execution gating ``ToolsExecutor`` (AgentDojo imported
    lazily so this module stays testable without agentdojo)."""
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
    import grad_agentdojo_hook as grad

    class AuthGraphToolsExecutor(ToolsExecutor):
        name = 'authgraph'

        def query(self, query, runtime, env=None, messages=(), extra_args=None):
            from agentdojo.functions_runtime import EmptyEnv
            from agentdojo.types import (ChatToolResultMessage,
                                         text_content_block_from_string)
            env = env if env is not None else EmptyEnv()
            extra_args = extra_args or {}
            if not messages or messages[-1].get('role') != 'assistant':
                return super().query(query, runtime, env, messages, extra_args)
            tool_calls = messages[-1].get('tool_calls') or []
            if not tool_calls:
                return super().query(query, runtime, env, messages, extra_args)
            processed: list[dict] = []
            out_msgs = []
            if tool_calls:
                online.ensure_started(grad.user_task_text(messages))
            for tc in tool_calls:
                args = dict(tc.args or {})
                t_build = time.perf_counter()
                trace_entries = grad.prefix_trace(messages) + processed
                prefix = steps_from_trace(trace_entries)
                build_ms = round((time.perf_counter() - t_build) * 1000, 3)
                rec = online.check(tc.function, args, prefix)
                for d in rec['decisions']:
                    d['build_latency_ms'] = build_ms
                if rec['blocked']:
                    processed.append({'tool': tc.function, 'args': args,
                                      'status': 'blocked', 'result': None})
                    out_msgs.append(ChatToolResultMessage(
                        role='tool',
                        content=[text_content_block_from_string(
                            online.blocked_text(tc.function, rec['decisions']))],
                        tool_call_id=tc.id, tool_call=tc,
                        error='blocked_by_authgraph'))
                    continue
                single = dict(messages[-1])
                single['tool_calls'] = [tc]
                msgs = list(messages[:-1]) + [single]
                _, _, _, res, _ = super().query(query, runtime, env, msgs,
                                                extra_args)
                out_msgs.append(res[-1])
                content = res[-1].get('content')
                text = (content if isinstance(content, str)
                        else '\n'.join(b.get('content', '') for b in (content or [])
                                       if isinstance(b, dict)))
                processed.append({'tool': tc.function, 'args': args,
                                  'status': 'executed', 'result': text})
                online.post_execute(tc.function, args, text, prefix)
            return query, runtime, env, list(messages) + out_msgs, extra_args

    return AuthGraphToolsExecutor()
