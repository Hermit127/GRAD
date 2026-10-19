"""AuthGraph (arXiv:2605.26497) pre-execution arm adapted to the v10 host.

Faithfulness notes (frozen in the protocol manifest):
  * The Planner runs once at task start on clean-context inputs only
    ``{user_prompt, tool_catalog}`` (the authgraph Planner isolation guard is
    active). It runs before the poisoned payload is delivered.
  * Every candidate action is checked BEFORE execution with the visible prefix
    only; the candidate is appended with an empty observation so it can never
    see its own future output. The prefix IRG is built deterministically from
    the visible calls/args/observation texts, identical to the AgentDojo online
    hook (same ``DualPointerChecker``).
  * Deterministic layers (L1 S-set, L2 in-S routing, L3 verbatim / cross-source)
    never call an LLM; ambiguous cases require the judge. With no judge wired
    in the checker raises ``JudgeRequired`` and the arm fails open, recording
    ``degraded`` -- an error, never a silent pass.
  * Blocked calls are not executed; the agent receives the block message and
    continues (replan bounded by the runner), same mechanic as GRAD/Monitor.
  * Offline/mock runs use a frozen AG and an optional stub judge; live runs use
    the shared budget client for Planner + Judge LLM calls.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/authgraph'))
sys.path.insert(0, str(REPO / 'code/repro/checkpoint_loop'))

from checker import DualPointerChecker, JudgeRequired  # noqa: E402
from core import InvokeCall  # noqa: E402
from llm import JudgeLLM, extract_json_object  # noqa: E402
from planner import Planner  # noqa: E402

from branch import tool_catalog  # noqa: E402

BLOCK_VERDICTS = ('hard_block', 'suspicious', 'param_violation')
SOURCE_TOOLS = {'shop': 'search', 'workspace': 'scan', 'finance': 'lookup',
                'cloud': 'scan'}


class BudgetChatJSON:
    """``chat_json`` adapter over the two-vendor budget client (same ledger,
    same caps, no retries)."""

    def __init__(self, client, max_tokens: int = 2048):
        self.client = client
        self.max_tokens = max_tokens
        self.calls = 0
        self.cost_rmb = 0.0

    def chat_json(self, system: str, user: str, name: str = 'default') -> dict:
        resp = self.client.chat.completions.create(
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user', 'content': user}],
            max_tokens=self.max_tokens)
        self.calls += 1
        self.cost_rmb += float(getattr(resp, 'cost_rmb', 0.0) or 0.0)
        return extract_json_object(str(getattr(resp, 'content', '') or ''))


def steps_from_v10_trace(trace: dict) -> list[dict]:
    """Executed v10 events -> authgraph prefix steps.

    Only executed tools are in ``trace['events']`` (blocked candidates are
    never appended), so a blocked call cannot become a provenance source."""
    steps = []
    for ev in trace.get('events', []):
        tool = ev.get('tool')
        if tool in (None, '', 'end'):
            continue
        steps.append({'index': len(steps) + 1, 'tool': tool,
                      'args': {'arg': str(ev.get('arg', ''))},
                      'observation': str(ev.get('observation', '')),
                      'status': 'executed'})
    return steps


def frozen_ag(env_name: str, allowed_source: str = 'observation_direct',
              source_tool: str | None = None) -> dict:
    """Deterministic offline Planner output for the v10 catalog."""
    action = tool_catalog(env_name)[-1]['name']
    source_tool = source_tool or SOURCE_TOOLS[env_name]
    policy = {'arg': {'allowed_source': allowed_source}}
    if allowed_source in ('observation_direct', 'observation_nl'):
        policy['arg']['source_tools'] = [source_tool]
    return {'reasoning': f'offline deterministic planner ({env_name})',
            'steps': [{'step_index': 1, 'expected_tool': action,
                       'replan': False, 'param_policies': policy}]}


class AuthGraphV10Arm:
    def __init__(self, env_name: str, chat_json=None, ag: dict | None = None,
                 judge=None, catalog: list[dict] | None = None):
        self.env_name = env_name
        self.catalog = catalog or tool_catalog(env_name)
        self.chat_json = chat_json
        self.ag = ag
        self.judge = judge
        self.checker: DualPointerChecker | None = None
        self.user_prompt = ''
        self.planner_error: str | None = None
        self.degraded = False
        self.errors: list[dict] = []
        self.planner_ms = 0.0
        self.check_ms = 0.0
        self.llm_cost_rmb = 0.0
        self.decisions: list[dict] = []
        self.blocked_calls: list[dict] = []
        self.prefix_audit: list[dict] = []

    # -- lifecycle ----------------------------------------------------------
    def start_task(self, user_prompt: str) -> None:
        self.user_prompt = user_prompt or ''
        t0 = time.perf_counter()
        try:
            ag = self.ag
            if ag is None:
                if self.chat_json is None:
                    raise RuntimeError('no planner llm wired in')
                ag = Planner(self.chat_json, self.catalog).plan(self.user_prompt)
                self.ag = ag
            self.checker = DualPointerChecker(ag, judge=self.judge,
                                              catalog=self.catalog)
        except Exception as exc:  # noqa: BLE001 - recorded, fail-open
            self.planner_error = f'{type(exc).__name__}: {str(exc)[:200]}'
            self.checker = None
            self.degraded = True
            self.errors.append({'stage': 'planner',
                                'error': self.planner_error})
        finally:
            self.planner_ms = (time.perf_counter() - t0) * 1000
            self._sync_llm_cost()

    # -- pre-execution check ------------------------------------------------
    def check(self, trace: dict, tool: str, arg) -> dict:
        if self.checker is None:
            return {'blocked': False, 'decisions': [], 'degraded': True,
                    'error': self.planner_error or 'no_ag', 'latency_ms': 0.0,
                    'prefix_steps': 0}
        steps = steps_from_v10_trace(trace)
        call = InvokeCall(call_index=len(steps) + 1, tool=tool, tool_node='',
                          observation_node='',
                          params={'arg': str(arg)})
        ctrace = {
            'user_prompt': self.user_prompt,
            'tool_catalog': self.catalog,
            'steps': steps + [{'index': len(steps) + 1, 'tool': tool,
                               'args': {'arg': str(arg)}, 'observation': ''}],
        }
        self.prefix_audit.append({
            'candidate': {'tool': tool, 'arg': str(arg)},
            'prefix_len': len(steps),
            'candidate_observation': ctrace['steps'][-1]['observation'],
        })
        t0 = time.perf_counter()
        try:
            new = self.checker.process(call, ctrace)
        except JudgeRequired as exc:
            return self._degraded(tool, f'JudgeRequired: {str(exc)[:200]}',
                                  t0, len(steps))
        except Exception as exc:  # noqa: BLE001 - recorded, fail-open
            return self._degraded(tool, f'{type(exc).__name__}: '
                                       f'{str(exc)[:200]}', t0, len(steps))
        lat = round((time.perf_counter() - t0) * 1000, 3)
        self.check_ms += lat
        recs = [{'tool': d.tool, 'layer': d.layer, 'verdict': d.verdict,
                 'param': d.param, 'ag_step': d.ag_step,
                 'blocked': d.verdict in BLOCK_VERDICTS,
                 'reason': str(d.reason), 'evidence': dict(d.evidence),
                 'latency_ms': lat} for d in new]
        self.decisions.extend(recs)
        self._sync_llm_cost()
        blocked = any(r['blocked'] for r in recs)
        if blocked:
            first = next(r for r in recs if r['blocked'])
            self.blocked_calls.append({'tool': tool, 'arg': str(arg),
                                       'layer': first['layer'],
                                       'verdict': first['verdict'],
                                       'reason': first['reason']})
        return {'blocked': blocked, 'decisions': recs, 'degraded': False,
                'error': '', 'latency_ms': lat, 'prefix_steps': len(steps)}

    def _degraded(self, tool: str, err: str, t0: float, n_steps: int) -> dict:
        lat = round((time.perf_counter() - t0) * 1000, 3)
        self.check_ms += lat
        self.degraded = True
        self.errors.append({'stage': 'check', 'tool': tool, 'error': err})
        self._sync_llm_cost()
        return {'blocked': False, 'decisions': [], 'degraded': True,
                'error': err, 'latency_ms': lat, 'prefix_steps': n_steps}

    def _sync_llm_cost(self) -> None:
        if hasattr(self.chat_json, 'cost_rmb'):
            self.llm_cost_rmb = float(self.chat_json.cost_rmb or 0.0)

    # -- reporting ----------------------------------------------------------
    def defense_module_ms(self) -> float:
        return round(self.planner_ms + self.check_ms, 3)

    @staticmethod
    def blocked_text(tool: str, decisions: list[dict]) -> str:
        first = decisions[0] if decisions else {}
        return (f'[BLOCKED by AuthGraph] blocked_by_authgraph tool={tool} '
                f'layer={first.get("layer")} verdict={first.get("verdict")} '
                f'reason={str(first.get("reason"))[:300]}')
