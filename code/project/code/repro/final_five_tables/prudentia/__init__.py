"""Prudentia reproduction (AgentDojo v1.2 panel).

Paper: Kolluri et al., "Optimizing Agent Planning for Security and Autonomy",
arXiv:2602.11416v1. Plan: LOGS/final_five_tables/prudentia_repro_plan_20260916.md.

Components:
  - ifc_labels / policies / variables / hitl / metrics / prompts: pure logic,
    importable without agentdojo;
  - quarantine: quarantined LLM (`ifc_quarantined_llm`);
  - pipeline: AgentDojo 0.1.35 wiring (lazy agentdojo imports);
  - run_panel: panel runner + shared budget ledger.

No network calls happen at import time; paid calls only occur when
``run_panel.py`` is deliberately executed with a ``--batch-cap``.
"""

from .ifc_labels import ContextLabel, Integrity, Readers, join_labels
from .policies import PolicyCheck, PolicyKind, ToolPolicy, check_tool_call
from .variables import VariableRef, VariableStore
from .hitl import (
    EndorsementMode,
    FrozenApprovalOracle,
    HITLDecision,
    HITLKind,
    HITLRequest,
    InterventionLedger,
    OracleMode,
)
from .metrics import RunRecord, hitl_load, tcr_at_k, tcr_curve
from .quarantine import QuarantineResult, QuarantinedLLM
from .pipeline import (
    PrudentiaConfig,
    PrudentiaState,
    PrudentiaToolbox,
    build_none_pipeline,
    build_prudentia_pipeline,
    make_oracle,
)

__all__ = [
    "ContextLabel",
    "Integrity",
    "Readers",
    "join_labels",
    "PolicyCheck",
    "PolicyKind",
    "ToolPolicy",
    "check_tool_call",
    "VariableRef",
    "VariableStore",
    "EndorsementMode",
    "FrozenApprovalOracle",
    "HITLDecision",
    "HITLKind",
    "HITLRequest",
    "InterventionLedger",
    "OracleMode",
    "RunRecord",
    "hitl_load",
    "tcr_at_k",
    "tcr_curve",
    "QuarantineResult",
    "QuarantinedLLM",
    "PrudentiaConfig",
    "PrudentiaState",
    "PrudentiaToolbox",
    "build_none_pipeline",
    "build_prudentia_pipeline",
    "make_oracle",
]
