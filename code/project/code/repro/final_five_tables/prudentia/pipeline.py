"""AgentDojo pipeline wiring for Prudentia (agentdojo 0.1.35, v1.2 panel).

Implements the four Prudentia components from arXiv:2602.11416v1 section 4 on
top of the AgentDojo interface used by ``code/repro/agentdojo_sanitizer_20260913.py``:

  1. Policy/label awareness: tool descriptions are annotated with their frozen
     policy (``tool_policies.json``); every tool call is checked against the
     current context label and the argument labels (P-T / P-F).
  2. Strategic variable expansion: the ``ifc_plan`` tool records the plan; the
     executor intercepts it (no side effects, no HITL, no violation).
  3. Endorsement vs approval: ``ifc_expand_variables(ask_endorsement=True)``
     asks the frozen HITL oracle for endorsement and, when granted, relabels
     the named variables TRUSTED *without* tainting the context;
     ``ask_endorsement=False`` expands ALL root variables and permanently
     taints the context. Policy failures on regular tool calls go to the
     oracle for approval; the call executes only if approved.
  4. Declassification is not offered (paper section 4).

Interface contract (mirrors the v1.2 runner):

    AgentPipeline([SystemMessage(prompt), InitQuery(), llm,
                   ToolsExecutionLoop([<IFC executor>, llm])])

with ``suite.run_task_with_pipeline(..., runtime_class=pipeline.runtime_class)``.
``build_prudentia_pipeline`` attaches ``runtime_class`` (annotated tools +
``ifc_plan`` / ``ifc_expand_variables`` / ``ifc_quarantined_llm`` schemas) and
``prudentia_state`` to the returned pipeline.

Attack/condition labels never reach this module: the executor and the oracle
only see tool calls, labels, and policy outcomes. All agentdojo imports are
lazy so the module imports without agentdojo (offline unit tests).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from .hitl import (
    EndorsementMode,
    FrozenApprovalOracle,
    HITLKind,
    HITLRequest,
    InterventionLedger,
    OracleMode,
)
from .ifc_labels import ContextLabel, Integrity, Readers
from .policies import PolicyKind, ToolPolicy, check_tool_call
from .prompts import (
    POLICY_ANNOTATION_TEMPLATE,
    POLICY_ANNOTATION_TRIVIAL_TEMPLATE,
    PRUDENTIA_SYSTEM_PROMPT,
)
from .quarantine import (
    DEFAULT_READER_IDENTITY,
    AgentDojoLLMBackend,
    QuarantinedLLM,
)
from .variables import VariableRef, VariableStore, render_value

SPECIAL_TOOLS = ("ifc_plan", "ifc_expand_variables", "ifc_quarantined_llm")
_VAR_TOKEN_RE = re.compile(r"#[A-Za-z0-9_.\-]+#")


@dataclass
class PrudentiaConfig:
    policy_map: Mapping[str, ToolPolicy]
    model: str = "deepseek-chat"
    temperature: float = 0.0
    system_prompt_override: Optional[str] = None
    quarantine_system_prompt: Optional[str] = None
    reader_identity: frozenset = DEFAULT_READER_IDENTITY
    tasks_per_suite: int = 5
    injections: str = "first"


@dataclass
class PrudentiaState:
    store: VariableStore = field(default_factory=VariableStore)
    ledger: Optional[InterventionLedger] = None
    context_label: ContextLabel = field(default_factory=ContextLabel)
    context_integrity_tainted: bool = False
    plan_notes: list[str] = field(default_factory=list)
    task_key: str = ""
    turn: int = 0
    policy_violations: int = 0
    quarantine_calls: int = 0
    result_counters: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def next_result_index(self, tool_name: str) -> int:
        index = self.result_counters.get(tool_name, 0)
        self.result_counters[tool_name] = index + 1
        return index

    def next_quarantine_index(self) -> int:
        self.quarantine_calls += 1
        return self.quarantine_calls


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def extract_variable_names(text: str) -> tuple[str, ...]:
    """Extract ``#...#`` variable identifiers, order-preserving, deduplicated."""

    seen: dict[str, None] = {}
    for match in _VAR_TOKEN_RE.findall(str(text)):
        seen.setdefault(match, None)
    return tuple(seen)


def parse_recipients(value: Any) -> Optional[frozenset]:
    """Best-effort recipient set extraction from a resolved argument value."""

    if isinstance(value, str):
        return frozenset({value.strip().lower()}) if value.strip() else None
    if isinstance(value, (list, tuple, set, frozenset)):
        items = {str(v).strip().lower() for v in value if str(v).strip()}
        return frozenset(items) if items else None
    if isinstance(value, Mapping):
        items = {str(v).strip().lower() for v in value.values()
                 if isinstance(v, str) and v.strip()}
        return frozenset(items) if items else None
    return None


def summarize_args(args: Mapping[str, Any], limit: int = 80) -> str:
    """Short, label-free summary of a tool call for HITL event logs."""

    parts = []
    for name, value in args.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            text = f"[{len(value)} items]"
        elif isinstance(value, Mapping):
            text = "{" + ",".join(value) + "}"
        else:
            text = str(value)
        if len(text) > limit:
            text = text[:limit] + "..."
        parts.append(f"{name}={text}")
    return "; ".join(parts)


def annotate_tool_descriptions(functions: Iterable[Any],
                               policy_map: Mapping[str, ToolPolicy]) -> list[Any]:
    """Return copies of agentdojo ``Function`` objects with policy annotations.

    The annotation is appended to the tool description (Prudentia section 4:
    "tool descriptions are annotated with the tool policy"); the frozen
    template is ``POLICY_ANNOTATION_TEMPLATE`` (assumption A5).
    """

    annotated = []
    for function in functions:
        policy = policy_map.get(getattr(function, "name", ""))
        if policy is None:
            annotated.append(function)
            continue
        template = (POLICY_ANNOTATION_TRIVIAL_TEMPLATE
                    if policy.kind is PolicyKind.TRIVIAL
                    else POLICY_ANNOTATION_TEMPLATE)
        annotation = template.format(description=policy.describe())
        description = (getattr(function, "description", "") or "") + annotation
        if hasattr(function, "model_copy"):
            annotated.append(function.model_copy(update={"description": description}))
        else:  # pragma: no cover - defensive fallback
            function.description = description
            annotated.append(function)
    return annotated


def _literal_label(context: ContextLabel, reader_identity: frozenset) -> ContextLabel:
    """Label of a literal argument typed by the planner.

    In a HIGH INTEGRITY context a literal carries no confidentiality
    restriction (readers=None). Once the context is tainted, literals inherit
    the taint and the user's readers so that egress to other recipients still
    requires approval (no-trust-through-copying guard).
    """

    if context.is_trusted():
        return ContextLabel(Integrity.TRUSTED, None)
    return ContextLabel(Integrity.UNTRUSTED,
                        Readers(frozenset(r.lower() for r in reader_identity)))


def _make_tool_message(tool_call: Any, text: str,
                       error: Optional[str] = None) -> Any:
    from agentdojo.types import ChatToolResultMessage, text_content_block_from_string

    return ChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string(text or "")],
        tool_call_id=getattr(tool_call, "id", None),
        tool_call=tool_call,
        error=error,
    )


def _render_variable_list(refs: Mapping[str, VariableRef]) -> str:
    lines = "\n".join(refs)
    return ("Tool result stored in variables (contents hidden):\n" + lines
            if lines else "Tool returned no data.")


# ---------------------------------------------------------------------------
# Prudentia-only tools (plan / expand / quarantine)
# ---------------------------------------------------------------------------

class PrudentiaToolbox:
    """Implementation of ``ifc_plan`` / ``ifc_expand_variables`` /
    ``ifc_quarantined_llm`` bound to one run's state, oracle and ledger."""

    def __init__(self, state: PrudentiaState, oracle: FrozenApprovalOracle,
                 ledger: InterventionLedger,
                 quarantine: Optional[QuarantinedLLM] = None,
                 reader_identity: frozenset = DEFAULT_READER_IDENTITY):
        self.state = state
        self.oracle = oracle
        self.ledger = ledger
        self.quarantine = quarantine
        self.reader_identity = frozenset(r.lower() for r in reader_identity)

    def record_plan(self, plan: str) -> str:
        self.state.plan_notes.append(str(plan))
        return "Plan recorded."

    def untrusted_label(self) -> ContextLabel:
        return ContextLabel(Integrity.UNTRUSTED, Readers(self.reader_identity))

    def expand_variables(self, variables_string: str,
                         ask_endorsement: bool) -> str:
        store = self.state.store
        requested = extract_variable_names(variables_string)
        known = [n for n in requested if n in store.names()]
        quarantined = [n for n in known if not store.get(n).root]
        unknown = [n for n in requested if n not in store.names()]

        if quarantined:
            return (f"Cannot expand quarantined-LLM output variables: "
                    f"{', '.join(quarantined)}. Expand the root variables "
                    f"instead (appendix D.3).")
        if not known:
            return ("No valid root variables were provided; nothing was "
                    f"expanded. Unknown: {', '.join(unknown) or 'none'}.")

        if ask_endorsement:
            request = HITLRequest(
                task_key=self.state.task_key, turn=self.state.turn,
                kind=HITLKind.ENDORSEMENT, tool="ifc_expand_variables",
                policy_reason="endorsement requested for variable expansion",
                variables=tuple(known), args_summary=summarize_args(
                    {"variables_string": variables_string,
                     "ask_endorsement": True}))
            approved = self.oracle.decide(request)
            self.ledger.record(request, approved, self.oracle)
            if not approved:
                return ("The user declined to endorse the variables. They "
                        "remain hidden and the context label is unchanged.")
            store.endorse(known)
            body = "\n".join(
                f"{n}: {render_value(store.get(n).value)}" for n in known)
            return ("The user endorsed the following variables (relabelled "
                    "trusted; context stays HIGH INTEGRITY):\n" + body)

        # endorse=False: expand ALL root variables, taint the context forever.
        outcome = store.expand([], endorse=False)
        self.state.context_label = ContextLabel(
            Integrity.UNTRUSTED, self.state.context_label.readers)
        self.state.context_integrity_tainted = True
        body = "\n".join(
            f"{name}: {render_value(store.get(name).value)}"
            for name in outcome.expanded_names)
        return ("Variables expanded WITHOUT endorsement; the context is now "
                "permanently LOW INTEGRITY:\n" + body)

    def quarantined_llm(self, query: str, variables: str,
                        expected_output_type: str) -> str:
        if self.quarantine is None:  # pragma: no cover - wiring guard
            raise RuntimeError("quarantined LLM is not configured")
        names = extract_variable_names(variables)
        result = self.quarantine.query(
            query, names, expected_output_type,
            call_index=self.state.next_quarantine_index())
        if result.variable is None:
            return str(result.raw_output)
        return result.variable.name


def build_special_tool_functions(toolbox: PrudentiaToolbox) -> list[Any]:
    """Create agentdojo ``Function`` schemas for the three Prudentia tools."""

    from agentdojo.functions_runtime import make_function

    def ifc_plan(plan: str) -> str:
        """Record the current plan and the security reasoning behind it.

        Args:
            plan: The intended tool calls with their security policies, what changed and why, and (before expansion) why expansion is needed.
        """
        return toolbox.record_plan(plan)

    def ifc_expand_variables(variables_string: str, ask_endorsement: bool) -> str:
        """Expand root variables into the context, optionally requesting user endorsement.

        Args:
            variables_string: Variable identifiers to expand, e.g. "[#read_email-result-0.body#]".
            ask_endorsement: True asks the user to endorse the named variables (context stays HIGH INTEGRITY); False expands ALL root variables and permanently taints the context.
        """
        return toolbox.expand_variables(variables_string, ask_endorsement)

    def ifc_quarantined_llm(query: str, variables: str,
                            expected_output_type: str) -> str:
        """Ask the quarantined LLM one question about hidden variable contents.

        Args:
            query: A single precise question about the variable contents.
            variables: Variable identifiers to provide, e.g. "[#read_email-result-0.body#]".
            expected_output_type: The expected response type, e.g. "string" or "integer".
        """
        return toolbox.quarantined_llm(query, variables, expected_output_type)

    return [make_function(ifc_plan),
            make_function(ifc_expand_variables),
            make_function(ifc_quarantined_llm)]


def make_prudentia_runtime_class(policy_map: Mapping[str, ToolPolicy],
                                 toolbox: PrudentiaToolbox,
                                 state: PrudentiaState) -> Any:
    """Runtime class that annotates tools and registers the Prudentia tools."""

    from agentdojo.functions_runtime import FunctionsRuntime

    specials = build_special_tool_functions(toolbox)

    class PrudentiaRuntime(FunctionsRuntime):
        def __init__(self, functions: Sequence[Any] = ()) -> None:
            annotated = annotate_tool_descriptions(functions, policy_map)
            super().__init__([*annotated, *specials])

    return PrudentiaRuntime


# ---------------------------------------------------------------------------
# IFC tool executor
# ---------------------------------------------------------------------------

class IFCToolExecutor:
    """Implements the policy check + HITL gate + variable wrapping.

    Mixed with ``agentdojo...ToolsExecutor`` at build time (see
    ``_build_executor``): ``query`` fully replaces the parent loop so that
    every single tool call is policy-checked before execution.
    """

    def __init__(self, config: PrudentiaConfig, oracle: FrozenApprovalOracle,
                 ledger: Optional[InterventionLedger], state: PrudentiaState,
                 toolbox: PrudentiaToolbox):
        self.config = config
        self.oracle = oracle
        self.ledger = ledger
        self.state = state
        self.toolbox = toolbox

    # -- public pipeline element interface ---------------------------------
    def query(self, query: str, runtime: Any, env: Any = None,
              messages: Any = (), extra_args: Any = None) -> Any:
        from agentdojo.functions_runtime import EmptyEnv

        env = env if env is not None else EmptyEnv()
        extra_args = extra_args or {}
        out_messages = list(messages)
        if not out_messages:
            return query, runtime, env, out_messages, extra_args
        last = out_messages[-1]
        if last.get("role") != "assistant" or not last.get("tool_calls"):
            return query, runtime, env, out_messages, extra_args

        self.state.turn += 1
        for tool_call in last["tool_calls"]:
            try:
                out_messages.append(
                    self._execute_tool_call(tool_call, runtime, env))
            except Exception as exc:  # noqa: BLE001 - one bad call must not kill the run
                self.state.warnings.append(
                    f"{getattr(tool_call, 'function', '?')}: "
                    f"{type(exc).__name__}: {exc}")
                out_messages.append(_make_tool_message(
                    tool_call, "",
                    error=f"{type(exc).__name__}: {exc}"[:300]))
        return query, runtime, env, out_messages, extra_args

    # -- internals ---------------------------------------------------------
    def _execute_tool_call(self, tool_call: Any, runtime: Any, env: Any) -> Any:
        name = getattr(tool_call, "function", "") or ""
        raw_args = dict(getattr(tool_call, "args", {}) or {})

        if not name:
            return _make_tool_message(
                tool_call, "",
                error="Empty function name provided. Provide a valid function name.")

        if name == "ifc_plan":
            if "plan" not in raw_args:
                return _make_tool_message(tool_call, "", error="Missing argument: plan")
            return _make_tool_message(
                tool_call, self.toolbox.record_plan(str(raw_args["plan"])))

        if name == "ifc_expand_variables":
            text = self.toolbox.expand_variables(
                str(raw_args.get("variables_string", "")),
                bool(raw_args.get("ask_endorsement", False)))
            return _make_tool_message(tool_call, text)

        if name == "ifc_quarantined_llm":
            text = self.toolbox.quarantined_llm(
                str(raw_args.get("query", "")),
                str(raw_args.get("variables", "")),
                str(raw_args.get("expected_output_type", "")))
            return _make_tool_message(tool_call, text)

        if name not in getattr(runtime, "functions", {}):
            return _make_tool_message(
                tool_call, "", error=f"Invalid tool {name} provided.")

        policy = self.config.policy_map.get(name)
        if policy is None:
            self.state.warnings.append(
                f"tool {name!r} missing from policy map; treated as TRIVIAL")
            policy = ToolPolicy(kind=PolicyKind.TRIVIAL)

        resolved, arg_labels = self._resolve_args(raw_args)
        recipients = None
        if policy.recipients_from is not None:
            recipients = parse_recipients(resolved.get(policy.recipients_from))

        check = check_tool_call(policy, self.state.context_label, arg_labels,
                                recipients=recipients)
        if not check.allowed:
            request = HITLRequest(
                task_key=self.state.task_key, turn=self.state.turn,
                kind=HITLKind.APPROVAL, tool=name,
                policy_reason=check.reason,
                variables=tuple(sorted(self._variable_args(raw_args))),
                args_summary=summarize_args(resolved))
            approved = self.oracle.decide(request)
            self.ledger.record(request, approved, self.oracle)
            # v(tau): every call that cannot be determined to comply with
            # policy counts, whether or not the human approves it (section 3).
            self.state.policy_violations += 1
            if not approved:
                return _make_tool_message(
                    tool_call, "",
                    error=("User rejected approval request "
                           f"({check.reason}). Tool call not executed.")[:300])

        resolved = self._prepare_run_args(resolved)
        result, error = runtime.run_function(env, name, resolved)
        if error is not None:
            return _make_tool_message(tool_call, "", error=error)

        if policy.result_integrity.upper().startswith("U"):
            refs = self.state.store.wrap_result(
                name, self.state.next_result_index(name), result,
                self.toolbox.untrusted_label(), root=True)
            if refs:
                return _make_tool_message(tool_call,
                                          _render_variable_list(refs))
        text = self._format_result(result)
        return _make_tool_message(tool_call, text)

    def _format_result(self, result: Any) -> str:
        from agentdojo.agent_pipeline.tool_execution import tool_result_to_str

        try:
            return tool_result_to_str(result)
        except (TypeError, ValueError):
            return str(result)

    def _prepare_run_args(self, args: Mapping[str, Any]) -> dict:
        from agentdojo.agent_pipeline.tool_execution import is_string_list

        prepared: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and is_string_list(value):
                from ast import literal_eval

                prepared[key] = literal_eval(value)
            else:
                prepared[key] = value
        return prepared

    def _resolve_value(self, value: Any) -> tuple[Any, ContextLabel]:
        store = self.state.store
        if isinstance(value, str) and value in store.names():
            ref = store.get(value)
            return ref.value, ref.label
        if isinstance(value, str) and _VAR_TOKEN_RE.fullmatch(value):
            # Variable-looking literal that is not in the store: treat as
            # untrusted so that P-T checks fail safely.
            return value, self.toolbox.untrusted_label()
        if isinstance(value, Mapping):
            resolved: dict = {}
            labels = []
            for key, item in value.items():
                resolved[key], label = self._resolve_value(item)
                labels.append(label)
            return resolved, _join(labels, self.state.context_label)
        if isinstance(value, (list, tuple)):
            items = []
            labels = []
            for item in value:
                resolved_item, label = self._resolve_value(item)
                items.append(resolved_item)
                labels.append(label)
            return items, _join(labels, self.state.context_label)
        return value, _literal_label(self.state.context_label,
                                     self.config.reader_identity)

    def _resolve_args(self, args: Mapping[str, Any]) -> tuple[dict, dict]:
        resolved: dict = {}
        labels: dict = {}
        for key, value in args.items():
            resolved[key], labels[key] = self._resolve_value(value)
        return resolved, labels

    def _variable_args(self, args: Mapping[str, Any]) -> set:
        found: set = set()

        def walk(value: Any) -> None:
            if isinstance(value, str):
                if value in self.state.store.names():
                    found.add(value)
            elif isinstance(value, Mapping):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        walk(args)
        return found


def _join(labels: Sequence[ContextLabel], fallback: ContextLabel) -> ContextLabel:
    result = fallback
    for label in labels:
        result = result.join(label)
    return result


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_none_pipeline(llm: Any) -> Any:
    """Plain AgentDojo pipeline (the ``none`` arm), same as the v1.2 runner."""

    from agentdojo.agent_pipeline.agent_pipeline import (AgentPipeline,
                                                         load_system_message)
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.tool_execution import (ToolsExecutionLoop,
                                                         ToolsExecutor)

    loop = ToolsExecutionLoop([ToolsExecutor(), llm])
    pipeline = AgentPipeline([SystemMessage(load_system_message(None)),
                              InitQuery(), llm, loop])
    pipeline.name = "none"
    return pipeline


def _build_executor(config: PrudentiaConfig, oracle: FrozenApprovalOracle,
                    ledger: Optional[InterventionLedger],
                    state: PrudentiaState,
                    toolbox: PrudentiaToolbox) -> Any:
    from agentdojo.agent_pipeline.tool_execution import (ToolsExecutor,
                                                         tool_result_to_str)

    class _Executor(IFCToolExecutor, ToolsExecutor):
        def __init__(self) -> None:
            ToolsExecutor.__init__(self, tool_result_to_str)
            IFCToolExecutor.__init__(self, config, oracle, ledger, state, toolbox)

    return _Executor()


def build_prudentia_pipeline(llm: Any, config: PrudentiaConfig,
                             oracle: FrozenApprovalOracle,
                             ledger: InterventionLedger,
                             state: Optional[PrudentiaState] = None,
                             task_key: str = "") -> Any:
    """Build the AgentDojo ``AgentPipeline`` for Prudentia (0.1.35).

    Returns a pipeline with two extra attributes:
      - ``runtime_class``: pass to ``suite.run_task_with_pipeline`` so the
        runtime exposes annotated tools plus the three Prudentia tools.
      - ``prudentia_state``: the per-run ``PrudentiaState`` (store, taint,
        policy violation counter, warnings).
    """

    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop

    state = state if state is not None else PrudentiaState()
    state.ledger = ledger
    if task_key:
        state.task_key = task_key
    state.context_integrity_tainted = (
        not state.context_label.is_trusted())

    backend = AgentDojoLLMBackend(llm)
    quarantine_kwargs: dict = {"reader_identity": config.reader_identity}
    if config.quarantine_system_prompt is not None:
        quarantine_kwargs["system_prompt"] = config.quarantine_system_prompt
    quarantine = QuarantinedLLM(backend, state.store, **quarantine_kwargs)
    toolbox = PrudentiaToolbox(state, oracle, ledger, quarantine=quarantine,
                               reader_identity=config.reader_identity)
    runtime_class = make_prudentia_runtime_class(config.policy_map, toolbox,
                                                 state)
    executor = _build_executor(config, oracle, ledger, state, toolbox)
    loop = ToolsExecutionLoop([executor, llm])
    pipeline = AgentPipeline([
        SystemMessage(config.system_prompt_override or PRUDENTIA_SYSTEM_PROMPT),
        InitQuery(), llm, loop])
    pipeline.name = config.model
    pipeline.runtime_class = runtime_class
    pipeline.prudentia_state = state
    pipeline.prudentia_toolbox = toolbox
    return pipeline


def make_oracle(track: str = "optimistic") -> FrozenApprovalOracle:
    """Frozen oracle factories (pre-declared per batch, never changed mid-run)."""

    if track == "optimistic":
        return FrozenApprovalOracle(OracleMode.APPROVE_ALL,
                                    EndorsementMode.ENDORSE_ALL)
    if track == "pessimistic":
        return FrozenApprovalOracle(OracleMode.REJECT_ALL,
                                    EndorsementMode.DENY_ALL)
    raise ValueError(f"unknown oracle track {track!r}")
