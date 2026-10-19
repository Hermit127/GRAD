"""Frozen HITL simulation: approval/endorsement oracle + intervention ledger.

Invariants (plan section 5, PI approval 2026-09-16):
  - The oracle receives ONLY IFC-visible information (tool, argument labels,
    context label, policy-check outcome, variable names). Its signatures must
    never gain access to the run condition, attack/injection metadata,
    ground truth, or the AgentDojo `security` return value.
  - Oracle modes are frozen before a batch and never changed mid-run.
  - Every request is recorded exactly once; approvals and endorsements are
    counted separately so that both metric conventions of Prudentia section 3
    can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence


class HITLKind(Enum):
    APPROVAL = "approval"
    ENDORSEMENT = "endorsement"


class OracleMode(Enum):
    APPROVE_ALL = "approve_all"
    REJECT_ALL = "reject_all"


class EndorsementMode(Enum):
    ENDORSE_ALL = "endorse_all"
    DENY_ALL = "deny_all"


@dataclass(frozen=True)
class HITLRequest:
    """A pending human-intervention request (IFC-visible fields only)."""

    task_key: str
    turn: int
    kind: HITLKind
    tool: str
    policy_reason: str = ""
    variables: Sequence[str] = ()
    args_summary: str = ""


@dataclass(frozen=True)
class HITLDecision:
    request: HITLRequest
    approved: bool
    oracle_mode: str
    intervention_index: int


class FrozenApprovalOracle:
    """Deterministic approve/reject oracle with frozen modes.

    The pre-declared human model is exactly:

        decide(request) = approval_mode  if request.kind == APPROVAL
                          endorsement_mode otherwise

    i.e. the decision depends *only* on the (IFC-visible) request kind and on
    the mode frozen before the batch. Request content (tool, arguments,
    labels, policy reason) is never inspected, and the oracle structurally
    cannot read run conditions, injection metadata, ground truth, or the
    AgentDojo ``security`` value: those names do not exist in this module and
    the AST test ``test_hitl_module_never_references_attack_labels`` enforces
    it. "Approves unless the IFC policy check flags the call" is equivalent
    to APPROVE_ALL here because a request is only created when the check
    failed.

    Track-O (optimistic): APPROVE_ALL + ENDORSE_ALL, matching the paper's
    assumption that a human approves policy-check failures in successful
    traces (section 3).
    Track-P (pessimistic): REJECT_ALL + DENY_ALL, giving the security lower
    bound (no policy-violating call is ever executed).
    """

    def __init__(self, approval_mode: OracleMode = OracleMode.APPROVE_ALL,
                 endorsement_mode: EndorsementMode = EndorsementMode.ENDORSE_ALL):
        self.approval_mode = approval_mode
        self.endorsement_mode = endorsement_mode

    @property
    def name(self) -> str:
        return f"{self.approval_mode.value}+{self.endorsement_mode.value}"

    def decide(self, request: HITLRequest) -> bool:
        if request.kind is HITLKind.APPROVAL:
            return self.approval_mode is OracleMode.APPROVE_ALL
        return self.endorsement_mode is EndorsementMode.ENDORSE_ALL


@dataclass
class InterventionLedger:
    """Ordered event log for HITL decisions within one run."""

    task_key: str
    events: list[HITLDecision] = field(default_factory=list)
    _next_index: int = 1

    def record(self, request: HITLRequest, approved: bool,
               oracle: FrozenApprovalOracle) -> HITLDecision:
        decision = HITLDecision(request=request, approved=approved,
                                oracle_mode=oracle.name,
                                intervention_index=self._next_index)
        self._next_index += 1
        self.events.append(decision)
        return decision

    def count(self, kind: Optional[HITLKind] = None) -> int:
        if kind is None:
            return len(self.events)
        return sum(1 for e in self.events if e.request.kind is kind)

    @property
    def approvals(self) -> int:
        return self.count(HITLKind.APPROVAL)

    @property
    def endorsements(self) -> int:
        return self.count(HITLKind.ENDORSEMENT)

    def to_records(self) -> list[dict]:
        return [
            {
                "turn": e.request.turn,
                "kind": e.request.kind.value,
                "tool": e.request.tool,
                "variables": list(e.request.variables),
                "policy_check": e.request.policy_reason,
                "oracle": e.oracle_mode,
                "approved": e.approved,
                "intervention_index": e.intervention_index,
            }
            for e in self.events
        ]
