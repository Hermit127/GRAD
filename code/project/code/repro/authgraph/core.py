"""Deterministic core of AuthGraph: INVOKE extraction, S-set, text matching.

No LLM calls here; this module is the zero-API half of the pipeline and is
unit-testable on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InvokeCall:
    call_index: int
    tool: str
    tool_node: str
    observation_node: str
    params: dict[str, str] = field(default_factory=dict)
    param_nodes: dict[str, str] = field(default_factory=dict)


def infer_calls_from_edges(irg: dict) -> list[InvokeCall]:
    """Fallback extraction when the Builder omitted the redundant `calls`
    index: group INVOKE edges by target observation and use node roles."""
    nodes = {n["id"]: n for n in irg.get("nodes", [])}
    by_obs: dict[str, list[str]] = {}
    for e in irg.get("edges", []):
        if e.get("type") == "INVOKE":
            by_obs.setdefault(e["target"], []).append(e["source"])
    calls: list[InvokeCall] = []
    for i, (obs, sources) in enumerate(by_obs.items(), start=1):
        tool_node = None
        params: dict[str, str] = {}
        param_nodes: dict[str, str] = {}
        plain: list[str] = []
        for sid in sources:
            n = nodes.get(sid, {})
            is_param = n.get("role") == "param" or bool(n.get("param_name"))
            if is_param:
                pname = n.get("param_name") or sid
                params[pname] = str(n.get("value", n.get("content", "")))
                param_nodes[pname] = sid
            else:
                plain.append(sid)
                if n.get("role") == "tool" and tool_node is None:
                    tool_node = sid
        if tool_node is None and plain:
            tool_node = plain[0]
        tool = str(nodes.get(tool_node, {}).get("tool") or
                   nodes.get(tool_node, {}).get("content", ""))
        calls.append(InvokeCall(call_index=i, tool=tool, tool_node=tool_node or "",
                                observation_node=obs, params=params,
                                param_nodes=param_nodes))
    calls.sort(key=lambda c: c.call_index)
    return calls


def extract_invoke_calls(irg: dict) -> list[InvokeCall]:
    """Ordered tool-call list from the IRG (paper Eq. 1, Figure 1 step 3).

    Uses the Builder's validated `calls` index when present; otherwise falls
    back to grouping INVOKE edges by observation node. Never relies on
    DERIVE/EXTRACT edges (the paper explicitly excludes them from detection).
    """
    calls_in = irg.get("calls")
    if not calls_in:
        return infer_calls_from_edges(irg)
    calls: list[InvokeCall] = []
    for c in calls_in:
        params = {}
        param_nodes = {}
        for pname, p in (c.get("params") or {}).items():
            params[pname] = str(p.get("value")) if isinstance(p, dict) else str(p)
            if isinstance(p, dict) and p.get("node"):
                param_nodes[pname] = p["node"]
        calls.append(InvokeCall(
            call_index=int(c["call_index"]), tool=str(c["tool"]),
            tool_node=str(c.get("tool_node") or ""),
            observation_node=str(c.get("observation_node") or ""),
            params=params, param_nodes=param_nodes))
    calls.sort(key=lambda c: c.call_index)
    return calls


def authorized_set(ag: dict) -> set[str]:
    """S = all expected_tool entries + union of replan_allowed_tools
    (+ sub authorization graph tools, merged into the AG by the Planner)."""
    s: set[str] = set()
    for step in ag.get("steps", []):
        s.add(step.get("expected_tool"))
        s.update(step.get("replan_allowed_tools") or [])
        for sub in step.get("subgraph_steps") or []:
            s.add(sub.get("expected_tool"))
            s.update(sub.get("replan_allowed_tools") or [])
    s.discard(None)
    return s


_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return _WS.sub(" ", str(s)).strip().lower()


def value_in_text(value: str, text: str) -> bool:
    """Deterministic first pass for parameter provenance: verbatim token
    (whitespace/case-normalized) inside the authoritative text corpus."""
    if value is None or text is None:
        return False
    v = normalize_text(value)
    if not v:
        return False
    return v in normalize_text(text)


def trace_steps(trace: dict) -> list[dict]:
    return list(trace.get("steps") or [])


def prior_observations(trace: dict, call_index: int, source_tools: list[str]) -> dict[str, list[str]]:
    """Original observation texts from the declared source tools that occurred
    strictly before this call, keyed by tool. The checker must use the raw
    trajectory text, not Builder summaries (paper 3.4)."""
    want = set(source_tools or [])
    out: dict[str, list[str]] = {}
    for step in trace_steps(trace):
        try:
            idx = int(step.get("index"))
        except (TypeError, ValueError):
            continue
        if idx >= call_index:
            break
        if step.get("tool") in want and step.get("observation"):
            out.setdefault(step["tool"], []).append(str(step["observation"]))
    return out


def observations_for_tool(trace: dict, tool: str) -> list[str]:
    return [str(s.get("observation", "")) for s in trace_steps(trace)
            if s.get("tool") == tool and s.get("observation")]


def other_observations(trace: dict, call_index: int,
                       source_tools: list[str]) -> dict[str, list[str]]:
    """Observations from tools OTHER than the declared source_tools, strictly
    before this call. Used for the deterministic cross-source (cross-tool
    pollution) check: a value that exists only in a non-declared tool's output
    has no legitimate provenance for this parameter."""
    want = set(source_tools or [])
    out: dict[str, list[str]] = {}
    for step in trace_steps(trace):
        try:
            idx = int(step.get("index"))
        except (TypeError, ValueError):
            continue
        if idx >= call_index:
            break
        if step.get("tool") not in want and step.get("observation"):
            out.setdefault(step["tool"], []).append(str(step["observation"]))
    return out
