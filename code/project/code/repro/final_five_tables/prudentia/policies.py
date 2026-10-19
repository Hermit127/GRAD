"""Tool-call security policies (P-T / P-F) from the Prudentia paper.

Paper anchors:
  - Prudentia section 2, "Policies on tool calls": check `l <= pi_f` and
    `l_i <= pi_i` before every tool call; P-T (trusted action) requires a
    trusted context for the call and its trusted arguments; P-F (permitted
    flow) requires every recipient to be allowed to read egressed data;
    non-consequential calls are always permitted.
  - FIDES section 4.3 "Policy Checking": consequential actions and
    confidentiality as the two fundamental policies.

Assumption A3 (disclosed in the plan): the per-tool policy map for AgentDojo
is not published by either paper. Freeze a JSON map here and review it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import FrozenSet, Mapping, Optional, Sequence

from .ifc_labels import ContextLabel, Readers


class PolicyKind(Enum):
    TRIVIAL = "trivial"
    P_T = "P-T"
    P_F = "P-F"
    P_T_OR_P_F = "P-T or P-F"


@dataclass(frozen=True)
class ToolPolicy:
    """Policy record attached to one tool.

    kind: P-T / P-F / P-T or P-F / trivial.
    trusted_args: argument names that must carry a trusted label (P-T).
    egress_args: argument names that egress data (P-F).
    recipients_from: argument name holding the recipient set for P-F.
    result_integrity: "U" (box the result into variables, default) or "T"
        (results may enter the context directly). Frozen per tool in
        tool_policies.json from installed tool signatures/descriptions.
    """

    kind: PolicyKind = PolicyKind.TRIVIAL
    trusted_args: FrozenSet[str] = field(default_factory=frozenset)
    egress_args: FrozenSet[str] = field(default_factory=frozenset)
    recipients_from: Optional[str] = None
    result_integrity: str = "U"

    def describe(self) -> str:
        if self.kind is PolicyKind.TRIVIAL:
            return "ALWAYS ALLOWED (non-consequential)"
        if self.kind is PolicyKind.P_T:
            return "ONLY HIGH INTEGRITY (consequential)"
        if self.kind is PolicyKind.P_F:
            return "PERMITTED FLOW (egress)"
        return "ONLY HIGH INTEGRITY OR PERMITTED FLOW (egress)"


@dataclass(frozen=True)
class PolicyCheck:
    allowed: bool
    kind: PolicyKind
    reason: str
    failing_args: Sequence[str] = ()


def _trusted_args_ok(policy: ToolPolicy, context_label: ContextLabel,
                     arg_labels: Mapping[str, ContextLabel]) -> tuple[bool, tuple[str, ...]]:
    failing = []
    if not context_label.is_trusted():
        failing.append("<context>")
    for name in policy.trusted_args:
        label = arg_labels.get(name)
        if label is None or not label.is_trusted():
            failing.append(name)
    return (not failing), tuple(failing)


def _permitted_flow_ok(policy: ToolPolicy, arg_labels: Mapping[str, ContextLabel],
                       recipients: Optional[FrozenSet[str]]) -> tuple[bool, tuple[str, ...]]:
    if recipients is None:
        return False, ("<recipients>",)
    failing = []
    for name in policy.egress_args:
        label = arg_labels.get(name)
        if label is None:
            # Argument is a literal typed in the (trusted) prompt context, not
            # data read from a variable: no confidentiality constraint to check.
            continue
        if label.is_trusted() and label.readers is None:
            # Literal value in a HIGH INTEGRITY context (see pipeline
            # _literal_label): nothing to constrain.
            continue
        if label.readers is None:
            failing.append(name)
            continue
        if not label.readers.leq(Readers(frozenset(recipients))):
            failing.append(name)
    return (not failing), tuple(failing)


def check_tool_call(policy: ToolPolicy, context_label: ContextLabel,
                    arg_labels: Mapping[str, ContextLabel],
                    recipients: Optional[FrozenSet[str]] = None) -> PolicyCheck:
    """Deterministic pre-call policy check (Prudentia section 2)."""

    if policy.kind is PolicyKind.TRIVIAL:
        return PolicyCheck(True, policy.kind, "non-consequential; always allowed")

    if policy.kind is PolicyKind.P_T:
        ok, failing = _trusted_args_ok(policy, context_label, arg_labels)
        return PolicyCheck(ok, policy.kind,
                           "trusted context and arguments" if ok else
                           "P-T failed: untrusted context/arguments",
                           failing)

    if policy.kind is PolicyKind.P_F:
        ok, failing = _permitted_flow_ok(policy, arg_labels, recipients)
        return PolicyCheck(ok, policy.kind,
                           "all recipients may read egressed data" if ok else
                           "P-F failed: recipients not authorized for data",
                           failing)

    ok_t, failing_t = _trusted_args_ok(policy, context_label, arg_labels)
    if ok_t:
        return PolicyCheck(True, policy.kind, "P-T satisfied")
    ok_f, failing_f = _permitted_flow_ok(policy, arg_labels, recipients)
    if ok_f:
        return PolicyCheck(True, policy.kind, "P-F satisfied (robust declassification)")
    return PolicyCheck(False, policy.kind, "P-T or P-F failed",
                       tuple(dict.fromkeys((*failing_t, *failing_f))))


def load_policy_map(path: Path) -> dict[str, ToolPolicy]:
    """Load a frozen tool->policy JSON map.

    Schema (produced by gen_tool_policies.py)::

        {"tools": {"<tool>": {
            "consequential": bool, "egress": bool, "always_allowed": bool,
            "argument_source_requirements": {"<arg>": "trusted"|"any"},
            "kind": "P-T"|"P-F"|"P-T or P-F"|"trivial",
            "trusted_args": [...], "egress_args": [...],
            "recipients_from": "<arg>"|null, "result_integrity": "U"|"T"}}}

    A flat mapping without the "tools" wrapper is also accepted (legacy
    skeleton format). The map itself is a frozen artifact of the reproduction
    (plan section 9, assumption A3); this loader is pure and calls no API.
    """

    raw = json.loads(Path(path).read_text())
    tools = raw.get("tools", raw) if isinstance(raw, dict) else {}
    out: dict[str, ToolPolicy] = {}
    for tool, spec in tools.items():
        if tool.startswith("_") or not isinstance(spec, dict):
            continue
        out[tool] = ToolPolicy(
            kind=PolicyKind(spec.get("kind", "trivial")),
            trusted_args=frozenset(spec.get("trusted_args", ())),
            egress_args=frozenset(spec.get("egress_args", ())),
            recipients_from=spec.get("recipients_from"),
            result_integrity=str(spec.get("result_integrity", "U")),
        )
    return out
