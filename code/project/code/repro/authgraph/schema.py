"""AuthGraph schemas and validators (paper arXiv:2605.26497).

Deterministic, dependency-free core. Every structure that an LLM produces is
validated here before the checker consumes it; invalid output raises instead
of being silently repaired.

IRG (Injected Reasoning Graph) JSON
-----------------------------------
{
  "nodes": [{"id": str, "type": input|observation|decision|intermediate,
             "content": str, "role": ["tool"|"param"], "param_name": str?,
             "value": str?}],
  "edges": [{"source": id, "target": id, "type": DERIVE|EXTRACT|INVOKE}],
  "calls": [{"call_index": 1, "tool": str, "tool_node": id,
             "params": {name: {"node": id, "value": str}},
             "observation_node": id}]
}

Authorization Graph (AG) JSON (Planner output)
----------------------------------------------
{
  "reasoning": str,
  "steps": [{"step_index": 1, "expected_tool": str, "replan": bool,
             "replan_allowed_tools": [str],
             "param_policies": {param: {"allowed_source":
                 user_prompt|observation_direct|observation_nl|any,
                 "source_tools": [str]}}}]
}
"""
from __future__ import annotations

from typing import Any

NODE_TYPES = ("input", "observation", "decision", "intermediate")
EDGE_TYPES = ("DERIVE", "EXTRACT", "INVOKE")
ALLOWED_SOURCES = ("user_prompt", "observation_direct", "observation_nl", "any")
OBSERVATION_SOURCES = ("observation_direct", "observation_nl")
WRITE_PREFIXES = ("send_", "update_", "delete_", "create_", "transfer_",
                  "book_", "post_", "pay_", "share_", "push_", "execute_",
                  "write_", "add_", "remove_", "set_", "schedule_", "invite_")
LABEL_MAX = 80


def tool_index(catalog: list[dict]) -> dict[str, dict]:
    return {t["name"]: t for t in catalog if isinstance(t, dict) and t.get("name")}


def is_write_tool(name: str, catalog: dict[str, dict] | None = None) -> bool:
    if catalog and name in catalog:
        kind = str(catalog[name].get("kind", "")).lower()
        if kind in ("write", "mutate", "destructive"):
            return True
        if kind in ("read", "query", "lookup"):
            return False
    return name.startswith(WRITE_PREFIXES)


def security_critical_params(name: str, catalog: dict[str, dict] | None = None) -> list[str]:
    if catalog and name in catalog:
        return list(catalog[name].get("security_critical") or [])
    return []


# ---------------------------------------------------------------------------
# IRG
# ---------------------------------------------------------------------------
def validate_irg(irg: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(irg, dict):
        return ["irg is not an object"]
    nodes = irg.get("nodes")
    edges = irg.get("edges")
    calls = irg.get("calls")
    if not isinstance(nodes, list) or not nodes:
        return ["irg.nodes must be a non-empty list"]
    if not isinstance(edges, list):
        errs.append("irg.edges must be a list")
        edges = []
    ids: set[str] = set()
    types: dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict) or "id" not in n or "type" not in n:
            errs.append(f"node missing id/type: {n!r}")
            continue
        if n["id"] in ids:
            errs.append(f"duplicate node id {n['id']}")
        ids.add(n["id"])
        types[n["id"]] = n["type"]
        if n["type"] not in NODE_TYPES:
            errs.append(f"node {n['id']} has invalid type {n['type']!r}")
        content = str(n.get("content", ""))
        if len(content) > LABEL_MAX:
            errs.append(f"node {n['id']} content exceeds {LABEL_MAX} chars")
    for e in edges:
        if not isinstance(e, dict) or "source" not in e or "target" not in e:
            errs.append(f"edge missing source/target: {e!r}")
            continue
        if e["source"] not in ids or e["target"] not in ids:
            errs.append(f"edge {e.get('source')}->{e.get('target')} references unknown node")
        if e.get("type") not in EDGE_TYPES:
            errs.append(f"edge {e.get('source')}->{e.get('target')} invalid type {e.get('type')!r}")
        if e.get("type") == "INVOKE":
            if types.get(e["source"]) != "decision":
                errs.append(f"INVOKE source {e['source']} is not decision-type")
            if types.get(e["target"]) != "observation":
                errs.append(f"INVOKE target {e['target']} is not observation-type")
    # DAG check
    adj: dict[str, list[str]] = {i: [] for i in ids}
    for e in edges:
        if e.get("source") in ids and e.get("target") in ids:
            adj[e["source"]].append(e["target"])
    color: dict[str, int] = {}

    def dfs(u: str) -> bool:
        color[u] = 1
        for v in adj[u]:
            if color.get(v) == 1:
                return True
            if color.get(v, 0) == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    for i in ids:
        if color.get(i, 0) == 0 and dfs(i):
            errs.append("irg contains a cycle (must be a DAG)")
            break
    # calls structure (redundant index for deterministic extraction)
    if calls is not None:
        if not isinstance(calls, list) or not calls:
            errs.append("irg.calls must be a non-empty list when present")
        else:
            last = 0
            for c in calls:
                if not isinstance(c, dict) or "call_index" not in c or "tool" not in c:
                    errs.append(f"call missing call_index/tool: {c!r}")
                    continue
                if not isinstance(c["call_index"], int) or c["call_index"] <= last:
                    errs.append("call_index must be strictly increasing integers")
                last = c.get("call_index", last)
                tn, on = c.get("tool_node"), c.get("observation_node")
                if tn not in ids or types.get(tn) != "decision":
                    errs.append(f"call {c['call_index']}: tool_node {tn!r} not a decision node")
                if on not in ids or types.get(on) != "observation":
                    errs.append(f"call {c['call_index']}: observation_node {on!r} not an observation node")
                params = c.get("params") or {}
                if not isinstance(params, dict):
                    errs.append(f"call {c['call_index']}: params must be an object")
                    params = {}
                for pname, p in params.items():
                    pn = p.get("node") if isinstance(p, dict) else None
                    if pn not in ids or types.get(pn) != "decision":
                        errs.append(f"call {c['call_index']}.{pname}: param node {pn!r} not a decision node")
                invoke_edges = {(e["source"], e["target"]) for e in edges
                                if e.get("type") == "INVOKE"}
                if tn is not None and on is not None and (tn, on) not in invoke_edges:
                    errs.append(f"call {c['call_index']}: missing INVOKE edge {tn}->{on}")
                for pname, p in params.items():
                    pn = p.get("node") if isinstance(p, dict) else None
                    if pn is not None and on is not None and (pn, on) not in invoke_edges:
                        errs.append(f"call {c['call_index']}.{pname}: missing INVOKE edge {pn}->{on}")
    return errs


def validate_ag(ag: dict, catalog: list[dict], start_index: int = 1) -> list[str]:
    errs: list[str] = []
    idx = tool_index(catalog)
    if not isinstance(ag, dict):
        return ["ag is not an object"]
    steps = ag.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["ag.steps must be a non-empty list"]
    for i, s in enumerate(steps, start=1):
        expected_index = start_index + i - 1
        if not isinstance(s, dict):
            errs.append(f"step {i} is not an object")
            continue
        if s.get("step_index") != expected_index:
            errs.append(f"step {i}: step_index must be {expected_index} "
                        f"(sequential from {start_index})")
        tool = s.get("expected_tool")
        if tool not in idx:
            errs.append(f"step {i}: expected_tool {tool!r} not in tool catalog")
        policies = s.get("param_policies") or {}
        if not isinstance(policies, dict):
            errs.append(f"step {i}: param_policies must be an object")
            policies = {}
        for pname, pol in policies.items():
            if not isinstance(pol, dict):
                errs.append(f"step {i}.{pname}: policy must be an object")
                continue
            src = pol.get("allowed_source")
            if src not in ALLOWED_SOURCES:
                errs.append(f"step {i}.{pname}: invalid allowed_source {src!r}")
            st = pol.get("source_tools")
            if src in OBSERVATION_SOURCES:
                if not isinstance(st, list) or not st:
                    errs.append(f"step {i}.{pname}: source_tools required for {src}")
                else:
                    bad = [t for t in st if t not in idx]
                    if bad:
                        errs.append(f"step {i}.{pname}: source_tools not in catalog: {bad}")
            elif st not in (None, [],):
                errs.append(f"step {i}.{pname}: source_tools only valid for observation sources")
        if tool in idx and is_write_tool(tool, idx):
            missing = [p for p in security_critical_params(tool, idx)
                       if p not in policies]
            if missing:
                errs.append(f"step {i}: completeness rule violated for write tool {tool}: "
                            f"missing param_policies for {missing}")
        if s.get("replan") is True:
            allowed = s.get("replan_allowed_tools")
            if not isinstance(allowed, list) or not allowed:
                errs.append(f"step {i}: replan=true requires replan_allowed_tools")
            else:
                bad = [t for t in allowed if t not in idx]
                if bad:
                    errs.append(f"step {i}: replan_allowed_tools not in catalog: {bad}")
    return errs


def validate_ag_subgraph(ag: dict, catalog: list[dict], allowed_tools: list[str],
                         trigger_tool: str | None = None) -> list[str]:
    """Replan sub-graph: step indices continue from the main plan, and every
    tool/source must stay inside the frozen whitelist (the tool whose
    observation triggered the replan is also a legitimate source, paper A.2)."""
    steps = ag.get("steps") if isinstance(ag, dict) else None
    start = 1
    if steps and isinstance(steps[0], dict) and isinstance(steps[0].get("step_index"), int):
        start = steps[0]["step_index"]
    errs = validate_ag(ag, catalog, start_index=start)
    allowed = set(allowed_tools)
    if trigger_tool:
        allowed.add(trigger_tool)
    for i, s in enumerate(steps or [], start=1):
        if isinstance(s, dict) and s.get("expected_tool") not in set(allowed_tools):
            errs.append(f"step {i}: replan sub-graph uses {s.get('expected_tool')!r} "
                        f"outside the replan_allowed_tools whitelist")
        if isinstance(s, dict):
            for pname, pol in (s.get("param_policies") or {}).items():
                if not isinstance(pol, dict):
                    continue
                for t in pol.get("source_tools") or []:
                    if t not in allowed:
                        errs.append(f"step {i}.{pname}: replan source_tool {t!r} outside "
                                    f"the replan_allowed_tools whitelist")
    return errs


def dump_irg_pretty(irg: dict) -> str:
    import json
    return json.dumps(irg, ensure_ascii=False, indent=1)


def dump_ag_pretty(ag: dict) -> str:
    import json
    return json.dumps(ag, ensure_ascii=False, indent=1)
