"""Quarantined LLM (`ifc_quarantined_llm`) component.

Paper anchors:
  - Prudentia section 2 "Dual LLM and IFC": variables can be passed to a
    quarantined LLM that processes queries in isolation; its content stays
    hidden from the planner.
  - Section 5.1: the same model is used for planning and information
    extraction in quarantined LLM queries (assumption A7).
  - Appendix D.3 (verbatim planner prompt): the quarantined LLM receives a
    single question plus all relevant variables and an expected output type;
    its answer is stored in a fresh ``#quarantined_llm-result-N#`` variable
    labelled UNTRUSTED; the literal sentinel "I need more information/
    variables." is *not* wrapped in a variable (it is trusted).

Assumption A8 (disclosed): neither the Prudentia paper (appendix D) nor the
FIDES paper publishes the quarantined LLM's own system prompt; only the
planner-facing prompt is published. The frozen extraction prompt used here is
``prompts.QUARANTINED_LLM_EXTRACTION_PROMPT`` and is recorded in every run's
provenance.

No agentdojo import happens at module import time: the adapter to an agentdojo
LLM pipeline element imports lazily inside ``complete`` so this module (and the
pure unit tests) stay usable outside the agentdojo venv. Offline tests can pass
any object exposing ``complete(system_prompt, user_prompt) -> str``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from .ifc_labels import ContextLabel, Integrity, Readers, join_labels
from .prompts import QUARANTINE_SENTINEL, QUARANTINED_LLM_EXTRACTION_PROMPT
from .variables import VariableRef, VariableStore, render_value

DEFAULT_READER_IDENTITY = frozenset({
    "emma johnson",
    "emma.johnson@bluesparrowtech.com",
})


@runtime_checkable
class QuarantineBackend(Protocol):
    """Minimal LLM interface used by the quarantined LLM."""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class AgentDojoLLMBackend:
    """Adapter calling an agentdojo LLM pipeline element with an isolated,
    tool-less message list (never the planner context)."""

    def __init__(self, llm: Any):
        self._llm = llm

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from agentdojo.functions_runtime import FunctionsRuntime
        from agentdojo.types import (
            ChatSystemMessage,
            ChatUserMessage,
            get_text_content_as_str,
            text_content_block_from_string,
        )

        messages = [
            ChatSystemMessage(role="system", content=[
                text_content_block_from_string(system_prompt)]),
            ChatUserMessage(role="user", content=[
                text_content_block_from_string(user_prompt)]),
        ]
        _, _, _, out_messages, _ = self._llm.query(
            "", FunctionsRuntime([]), None, messages, {})
        last = out_messages[-1]
        content = last.get("content") or []  # type: ignore[union-attr]
        return get_text_content_as_str(content)


@dataclass(frozen=True)
class QuarantineResult:
    variable: Optional[VariableRef] = None
    raw_output: Any = None
    needs_more_information: bool = False


class QuarantinedLLM:
    """Isolated LLM over variable contents; no tools, no context access."""

    def __init__(self, llm: Any, store: VariableStore, *,
                 reader_identity: frozenset = DEFAULT_READER_IDENTITY,
                 system_prompt: str = QUARANTINED_LLM_EXTRACTION_PROMPT):
        self._llm = llm
        self._store = store
        self._readers = Readers(frozenset(r.lower() for r in reader_identity))
        self._system_prompt = system_prompt

    def _render_variables(self, variables: Sequence[str]) -> str:
        lines = []
        for name in variables:
            try:
                value = self._store.get(name).value
            except KeyError:
                value = "<unknown variable>"
            lines.append(f"{name}:\n{render_value(value)}")
        return "\n\n".join(lines) if lines else "(no variables provided)"

    def query(self, query: str, variables: Sequence[str],
              expected_output_type: str, call_index: int) -> QuarantineResult:
        """Run one quarantined query and wrap the output as a variable.

        The message list contains only the frozen extraction prompt and the
        variable contents + question; the planner context is never passed.
        Output is labelled UNTRUSTED and rooted so that
        ``VariableStore.expand(endorse=False)`` never expands it (D.3).
        """

        user_prompt = (
            f"Query: {query}\n\n"
            f"Variable contents:\n{self._render_variables(variables)}\n\n"
            f"Expected output type: {expected_output_type}"
        )
        raw = str(self._llm.complete(self._system_prompt, user_prompt)).strip()
        if raw.startswith(QUARANTINE_SENTINEL):
            return QuarantineResult(variable=None, raw_output=QUARANTINE_SENTINEL,
                                    needs_more_information=True)
        name = f"#quarantined_llm-result-{call_index}#"
        ref = VariableRef(
            name=name,
            value=raw,
            label=ContextLabel(Integrity.UNTRUSTED, self._readers),
            root=False,
        )
        self._store.add(ref)
        return QuarantineResult(variable=ref, raw_output=raw)


def quarantined_label(labels: Sequence[ContextLabel]) -> ContextLabel:
    return join_labels(labels) if labels else ContextLabel(Integrity.UNTRUSTED, None)
