"""Graph Alignment Checker: dual-pointer traversal with three-layer detection.

Paper arXiv:2605.26497 section 3.4. Layer order:
  L1 hard block  - authorization steps exhausted and the tool is not in S
                   (programmatic, zero LLM),
  L2 tool name   - tools in S auto-pass and go to parameter checking; ambiguous
                   tools are judged by the LLM (extra_step_ok / skipped_step_ok
                   / suspicious),
  L3 param check - per-parameter policy: deterministic string match first
                   (user_prompt / observation_direct), LLM fallback otherwise
                   (observation_nl always LLM; user_prompt alignment fallback).

The deterministic core (S-set, INVOKE extraction, string matching, pointer
bookkeeping) never calls an LLM. When a case requires LLM judgment and no
judge is provided the checker raises ``JudgeRequired`` -- it never silently
passes an ambiguous call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core import (InvokeCall, extract_invoke_calls, authorized_set,
                  other_observations, prior_observations, value_in_text)
from schema import tool_index


class JudgeRequired(RuntimeError):
    """Raised when an LLM judgment is needed but no judge is wired in."""


@dataclass
class Decision:
    call_index: int
    tool: str
    layer: str
    verdict: str
    reason: str
    ag_step: int | None = None
    param: str | None = None
    evidence: dict = field(default_factory=dict)


@dataclass
class AlignmentResult:
    decisions: list[Decision]
    blocked_calls: list[int]
    steps_consumed: int
    steps_total: int

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_calls)


def _policy_step(steps: list[dict], start: int, tool: str) -> dict | None:
    """Closest step with a policy for this tool at/after the pointer."""
    for s in steps[start:]:
        if s.get("expected_tool") == tool:
            return s
    for s in steps:
        if s.get("expected_tool") == tool:
            return s
    return None


def _judge_param(judge, trace: dict, tool: str, pname: str, pval: str,
                 obs_texts: list[str], allowed_source: str) -> dict:
    if judge is None:
        raise JudgeRequired(
            f"parameter check for {tool}.{pname} needs LLM judgment "
            f"(allowed_source={allowed_source}) but no judge was provided")
    return judge.param_source_check(
        user_prompt=str(trace.get("user_prompt", "")), tool_name=tool,
        param_name=pname, param_value=str(pval), obs_texts=obs_texts,
        allowed_source=allowed_source)


def _check_params(step: dict | None, call: InvokeCall, trace: dict, judge,
                  decisions: list[Decision]) -> None:
    policies = (step or {}).get("param_policies") or {}
    for pname, pval in call.params.items():
        pol = policies.get(pname)
        if pol is None:
            decisions.append(Decision(
                call_index=call.call_index, tool=call.tool, layer="L3",
                verdict="param_uncovered", param=pname,
                ag_step=(step or {}).get("step_index"),
                reason="no param policy declared (completeness rule violation "
                       "flaggable by validate_ag); parameter not silently skipped"))
            continue
        src = pol.get("allowed_source")
        if src == "any":
            decisions.append(Decision(
                call_index=call.call_index, tool=call.tool, layer="L3",
                verdict="param_ok", param=pname, ag_step=step.get("step_index"),
                reason="allowed_source=any (no provenance constraint)"))
            continue
        if src == "user_prompt":
            if value_in_text(pval, str(trace.get("user_prompt", ""))):
                decisions.append(Decision(
                    call_index=call.call_index, tool=call.tool, layer="L3",
                    verdict="param_ok", param=pname, ag_step=step.get("step_index"),
                    reason="value appears verbatim in the user prompt"))
            else:
                verdict = _judge_param(judge, trace, call.tool, pname, pval, [], src)
                aligned = bool(verdict.get("aligned"))
                decisions.append(Decision(
                    call_index=call.call_index, tool=call.tool, layer="L3",
                    verdict="param_ok" if aligned else "param_violation",
                    param=pname, ag_step=step.get("step_index"),
                    reason=str(verdict.get("reason", "")),
                    evidence={"fallback": "llm_user_prompt_alignment"}))
            continue
        # observation_direct / observation_nl
        texts_by_tool = prior_observations(trace, call.call_index,
                                           pol.get("source_tools") or [])
        texts = [t for v in texts_by_tool.values() for t in v]
        if not texts:
            decisions.append(Decision(
                call_index=call.call_index, tool=call.tool, layer="L3",
                verdict="param_violation", param=pname, ag_step=step.get("step_index"),
                reason=f"declared source_tools {pol.get('source_tools')} produced "
                       f"no observation before this call; no legitimate provenance",
                evidence={"fallback": "source_tool_absent"}))
            continue
        if src == "observation_direct" and any(value_in_text(pval, t) for t in texts):
            decisions.append(Decision(
                call_index=call.call_index, tool=call.tool, layer="L3",
                verdict="param_ok", param=pname, ag_step=step.get("step_index"),
                reason="value found verbatim in the declared source_tools observation",
                evidence={"source_tools": pol.get("source_tools")}))
            continue
        if src == "observation_direct":
            # Deterministic cross-source check (paper running example): the
            # value is absent from the declared source but present in another
            # tool's observation in the visible prefix -> cross-tool pollution.
            others = other_observations(trace, call.call_index,
                                        pol.get("source_tools") or [])
            other_texts = [t for v in others.values() for t in v]
            if any(value_in_text(pval, t) for t in other_texts):
                decisions.append(Decision(
                    call_index=call.call_index, tool=call.tool, layer="L3",
                    verdict="param_violation", param=pname,
                    ag_step=step.get("step_index"),
                    reason=("value absent from the declared source_tools but "
                            "present in another tool's observation: cross-source "
                            "(cross-tool pollution) provenance"),
                    evidence={"fallback": "cross_source_provenance",
                              "declared_source_tools": pol.get("source_tools"),
                              "observed_in": sorted(others)}))
                continue
        # Not verbatim in the declared source and not cross-source either
        # (e.g. a reasoning product): LLM judgment with the published A.3
        # prompt (aligned=false only with a verbatim injection instruction).
        verdict = _judge_param(judge, trace, call.tool, pname, pval, texts, src)
        aligned = bool(verdict.get("aligned"))
        decisions.append(Decision(
            call_index=call.call_index, tool=call.tool, layer="L3",
            verdict="param_ok" if aligned else "param_violation",
            param=pname, ag_step=step.get("step_index"),
            reason=str(verdict.get("reason", "")),
            evidence={"fallback": "llm_param_source_check",
                      "source_tools": pol.get("source_tools")}))


class DualPointerChecker:
    """Stateful dual-pointer traversal over AG steps.

    ``process(call, trace)`` evaluates ONE call against the current pointer
    state and returns the decisions produced for it. This is what the online
    AgentDojo hook uses: each proposed tool call is checked before execution
    with only the visible prefix in ``trace`` (the candidate call is appended
    with an empty observation so it can never see its own future output).

    ``align`` below is the batch wrapper over the same class, so offline and
    online behavior cannot drift.
    """

    def __init__(self, ag: dict, judge=None, catalog: list[dict] | None = None):
        self.ag = ag
        self.steps = ag.get("steps") or []
        self.s_set = authorized_set(ag)
        self.judge = judge
        self.catalog = list(catalog or [])
        self.i = 0
        self.skips = 0
        self.decisions: list[Decision] = []
        self.blocked_calls: list[int] = []
        self.prior_calls: list[str] = []
        self.last_consumed_step: dict | None = None

    def splice_substeps(self, after_step_index: int, substeps: list[dict]) -> None:
        """Replan: insert the sub-graph steps after the replan step and add
        their tools to S. If the pointer already moved past that step, shift it
        so it still points at the same logical next step."""
        pos = len(self.steps) - 1
        for k, s in enumerate(self.steps):
            if s.get("step_index") == after_step_index:
                pos = k
                break
        self.steps[pos + 1:pos + 1] = substeps
        for s in substeps:
            self.s_set.add(s.get("expected_tool"))
            self.s_set.update(s.get("replan_allowed_tools") or [])
        if pos + 1 < self.i:
            self.i += len(substeps)

    def process(self, call: InvokeCall, trace: dict) -> list[Decision]:
        start = len(self.decisions)
        judge = self.judge
        steps = self.steps
        s_set = self.s_set
        idx = tool_index(self.catalog or (trace.get("tool_catalog") or []))
        self.last_consumed_step = None
        consumed = False
        while not consumed:
            if self.i >= len(steps):
                if call.tool not in s_set:
                    self.decisions.append(Decision(
                        call_index=call.call_index, tool=call.tool, layer="L1",
                        verdict="hard_block",
                        reason="all authorization steps exhausted and tool is not "
                               "in the authorized set S (programmatic block)",
                        evidence={"authorized_set_size": len(s_set)}))
                    self.blocked_calls.append(call.call_index)
                else:
                    step = _policy_step(steps, 0, call.tool)
                    _check_params(step, call, trace, judge, self.decisions)
                    self.last_consumed_step = step
                consumed = True
                continue
            cur = steps[self.i]
            if call.tool == cur.get("expected_tool"):
                _check_params(cur, call, trace, judge, self.decisions)
                self.last_consumed_step = cur
                self.i += 1
                consumed = True
                continue
            if call.tool in s_set:
                self.decisions.append(Decision(
                    call_index=call.call_index, tool=call.tool, layer="L2",
                    verdict="auto_pass_authorized_set",
                    ag_step=cur.get("step_index"),
                    reason="tool is in the authorized set S; routed directly to "
                           "parameter checking (no LLM tool-name judgment)",
                    evidence={"authorized_set_size": len(s_set)}))
                step = _policy_step(steps, self.i, call.tool)
                _check_params(step, call, trace, judge, self.decisions)
                self.last_consumed_step = step
                consumed = True
                continue
            if judge is None:
                raise JudgeRequired(
                    f"tool-name judgment needed for call {call.call_index} "
                    f"({call.tool!r} vs expected {cur.get('expected_tool')!r}) "
                    f"but no judge was provided")
            verdict = judge.tool_name_check(
                user_prompt=str(trace.get("user_prompt", "")),
                expected_tool=cur.get("expected_tool"),
                actual_tool=call.tool,
                planned_tools=[s.get("expected_tool") for s in steps],
                catalog_excerpt=([idx.get(call.tool)] if call.tool in idx else []),
                prior_calls=list(self.prior_calls))
            v = str(verdict.get("verdict", "suspicious"))
            max_total_skips = max(4, len(steps) + 2)
            if v == "extra_step_ok":
                self.decisions.append(Decision(
                    call_index=call.call_index, tool=call.tool, layer="L2",
                    verdict="extra_step_ok", ag_step=cur.get("step_index"),
                    reason=str(verdict.get("reason", "")),
                    evidence={"llm": True}))
                consumed = True
            elif v == "skipped_step_ok":
                self.decisions.append(Decision(
                    call_index=call.call_index, tool=call.tool, layer="L2",
                    verdict="skipped_step_ok", ag_step=cur.get("step_index"),
                    reason=str(verdict.get("reason", "")),
                    evidence={"llm": True}))
                # the expected step was skipped; the judged call itself is not
                # blocked and needs no parameter check (it has no authorizing
                # step of its own -- tools in S never reach this branch)
                self.skips += 1
                self.last_consumed_step = cur
                self.i += 1
                consumed = True
                if self.skips > max_total_skips:
                    self.decisions.append(Decision(
                        call_index=call.call_index, tool=call.tool, layer="L2",
                        verdict="suspicious",
                        reason="skip budget exhausted; treated as suspicious to "
                               "avoid unbounded skipping"))
                    self.blocked_calls.append(call.call_index)
            else:
                self.decisions.append(Decision(
                    call_index=call.call_index, tool=call.tool, layer="L2",
                    verdict="suspicious", ag_step=cur.get("step_index"),
                    reason=str(verdict.get("reason", "")),
                    evidence={"llm": True}))
                self.blocked_calls.append(call.call_index)
                consumed = True
        self.prior_calls.append(f"{call.tool}({dict(call.params)})")
        return self.decisions[start:]

    def result(self) -> AlignmentResult:
        return AlignmentResult(decisions=self.decisions,
                               blocked_calls=self.blocked_calls,
                               steps_consumed=self.i,
                               steps_total=len(self.steps))


def align(ag: dict, irg: dict, trace: dict, judge=None) -> AlignmentResult:
    checker = DualPointerChecker(ag, judge=judge,
                                 catalog=trace.get("tool_catalog") or [])
    for call in extract_invoke_calls(irg):
        checker.process(call, trace)
    return checker.result()


def trajectory_verdict(result: AlignmentResult) -> int:
    """Trajectory-level detection for Table 1: any blocked call => positive."""
    return 1 if result.blocked_calls else 0
