#!/usr/bin/env python3
"""Graph construction for the Task 1 relation-counterfactual pilot.

Two construction modes share the same node set and text:
  provenance : edges are exactly the scenario's runtime-observable relations
               (writes_to / derived_from / reads_from / param_source /
               authorization, plus sequence_next). No record identity is
               invented; everything comes from the generated event stream.
  heuristic  : the current SSUM training path (code/train_gnn_v2.py
               rebuild_v2_edges): edges inferred from node type adjacency and
               text regex, record identity ignored.
  chrono     : sequence_next chain only.
  none       : no edges.

Edge types are the v2 vocabulary extended with the four provenance relations.
"""
from __future__ import annotations

import copy
import random
import re
from pathlib import Path
import sys

_REPO_CODE = Path(__file__).resolve().parents[2]
if str(_REPO_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_CODE))

from train_gnn_v2 import V2_EDGE_TYPES

PROVENANCE_TYPES = ["reads_from", "derived_from", "param_source", "authorization"]
EDGE_TYPES = list(V2_EDGE_TYPES) + PROVENANCE_TYPES
EDGE2IDX = {t: i for i, t in enumerate(EDGE_TYPES)}

SOURCE_TYPES = {
    "web_request", "web_search", "web_download", "information_search",
    "file_read", "database_query", "account_access", "prompt_inject",
}
INTERMEDIATE_TYPES = {
    "memory_write", "memory_read", "file_write", "data_aggregate",
    "document_create", "agent_message", "memory_record",
}
SINK_TYPES = {
    "shell_exec", "code_exec", "tool_execute", "email_send", "message_send",
    "social_media", "document_create", "account_access", "system_config",
    "web_request", "file_write", "file_delete", "agent_spawn",
}
DATA_FLOW_RE = re.compile(
    r"\b(use|using|based on|from|with|retriev|extract|gather|collect|compile|"
    r"download|attach|include|reference|process|submit|send|post|update|"
    r"modify|confirmed|reports?)\b", re.I)
CONTROL_RE = re.compile(
    r"\b(trigger|schedule|execute|run|launch|deploy|start|restart|enable|"
    r"disable|automate)\b", re.I)
SINK_RE = re.compile(
    r"\b(send|post|execute|run|delete|modify|redirect|transfer|upload|install|"
    r"disable|schedule|impersonat|fake|fraud|phish|backdoor|credential|"
    r"payment|invoice|pay|book|dispatch|provision|approve|refund|activate|"
    r"issue|publish|update|file|export|restore)\b", re.I)


def _node_text(node):
    return (node.get("features", {}) or {}).get("subtask", "") or node.get("type", "")


def _ordered_nodes(scenario):
    return sorted(scenario["nodes"], key=lambda n: (n["turn"], n["id"]))


def _add(edges, seen, source, target, etype, confidence, rationale):
    key = (source, target, etype)
    if key in seen:
        return
    seen.add(key)
    edges.append({"source": source, "target": target, "type": etype,
                  "confidence": float(confidence), "rationale": rationale[:300]})


def heuristic_edges(scenario):
    nodes = _ordered_nodes(scenario)
    edges, seen = [], set()
    for prev, cur in zip(nodes[:-1], nodes[1:]):
        ptxt, ctxt = _node_text(prev), _node_text(cur)
        _add(edges, seen, prev["id"], cur["id"], "sequence_next", 1.0,
             "adjacent steps")
        if prev["type"] == "memory_write":
            _add(edges, seen, prev["id"], cur["id"], "memory_store", 0.85, ptxt)
        if cur["type"] == "memory_read":
            _add(edges, seen, prev["id"], cur["id"], "memory_retrieve", 0.85, ctxt)
        if prev["type"] in ("file_write", "document_create") and cur["type"] in SINK_TYPES:
            _add(edges, seen, prev["id"], cur["id"], "writes_to", 0.7,
                 ptxt + " -> " + ctxt)
        if prev["type"] == "agent_spawn":
            _add(edges, seen, prev["id"], cur["id"], "spawns", 0.75, ptxt)
        if prev["type"] == "agent_message" or cur["type"] == "agent_message":
            _add(edges, seen, prev["id"], cur["id"], "delegates_message", 0.65,
                 ptxt + " -> " + ctxt)
        if DATA_FLOW_RE.search(ctxt) or prev["type"] in SOURCE_TYPES | INTERMEDIATE_TYPES:
            _add(edges, seen, prev["id"], cur["id"], "data_flow", 0.55,
                 ptxt + " -> " + ctxt)
        if CONTROL_RE.search(ctxt) or cur["type"] in (
                "shell_exec", "code_exec", "tool_execute", "system_config"):
            _add(edges, seen, prev["id"], cur["id"], "control_trigger", 0.55,
                 ptxt + " -> " + ctxt)
        if cur["type"] in SINK_TYPES and SINK_RE.search(ctxt):
            _add(edges, seen, prev["id"], cur["id"], "sink_trigger", 0.6, ctxt)
    for i, src in enumerate(nodes):
        stxt = _node_text(src)
        for dst in nodes[i + 2:min(i + 6, len(nodes))]:
            dtxt = _node_text(dst)
            if src["type"] == "memory_write" and dst["type"] == "memory_read":
                _add(edges, seen, src["id"], dst["id"], "memory_retrieve", 0.9,
                     stxt + " -> " + dtxt)
                _add(edges, seen, src["id"], dst["id"], "cross_turn", 0.55,
                     "delayed memory use")
            if src["type"] in SOURCE_TYPES and dst["type"] in INTERMEDIATE_TYPES \
                    and DATA_FLOW_RE.search(dtxt):
                _add(edges, seen, src["id"], dst["id"], "data_flow", 0.45,
                     stxt + " -> " + dtxt)
            if src["type"] in INTERMEDIATE_TYPES and dst["type"] in SINK_TYPES \
                    and SINK_RE.search(dtxt):
                _add(edges, seen, src["id"], dst["id"], "sink_trigger", 0.5,
                     stxt + " -> " + dtxt)
    return edges


def build_edges(scenario, mode):
    if mode == "provenance":
        return [dict(e) for e in scenario["edges"]]
    if mode == "heuristic":
        return heuristic_edges(scenario)
    if mode == "chrono":
        return [dict(e) for e in scenario["edges"] if e["type"] == "sequence_next"]
    if mode == "none":
        return []
    raise ValueError(mode)


REWIRE_TYPES = ("reads_from", "param_source", "authorization", "derived_from")


def wrong_source_edges(scenario):
    """Matched-statistics wrong-source pairing at input-construction time.

    Every relation edge keeps its type; endpoints are retargeted by a
    side-independent canonical rule to a temporally legal event, so both sides
    of a counterfactual pair receive byte-identical wrong-source graphs. Edge
    count, type multiset and temporal legality are preserved; the only
    information removed is which source the relation actually points to.
    """
    nodes = _ordered_nodes(scenario)
    turn = {n["id"]: n["turn"] for n in nodes}
    events = [n for n in nodes if n["type"] != "memory_record"]

    def canonical_event(source_id):
        src_turn = turn[source_id]
        cand = [n for n in events
                if n["id"] != source_id and n["turn"] <= src_turn]
        if not cand:
            return source_id
        return max(cand, key=lambda n: (n["turn"], n["id"]))["id"]

    edges = []
    for e in scenario["edges"]:
        e2 = dict(e)
        if e2["type"] in REWIRE_TYPES:
            e2["target"] = canonical_event(e2["source"])
        edges.append(e2)
    return edges


def _provenance_closure(scenario, start):
    by_id = {n["id"]: n for n in scenario["nodes"]}
    out, frontier = set(), [start]
    while frontier:
        cur = frontier.pop()
        if cur in out:
            continue
        out.add(cur)
        for e in scenario["edges"]:
            if e["source"] == cur and e["type"] in (
                    "param_source", "reads_from", "derived_from", "authorization"):
                if e["target"] in by_id and e["target"] not in out:
                    frontier.append(e["target"])
    return out


def build_graph(scenario, mode="provenance", drop_relation_types=False,
                drop_edges=(), wrong_source=False):
    nodes = _ordered_nodes(scenario)
    edges = build_edges(scenario, mode)
    if drop_edges:
        edges = [e for e in edges if e["type"] not in set(drop_edges)]
    if wrong_source:
        edges = wrong_source_edges({**scenario, "edges": edges})
    if drop_relation_types:
        seen = set()
        cleaned = []
        for e in edges:
            e2 = dict(e)
            e2["type"] = "relation"
            e2["rationale"] = "connection"
            key = (e2["source"], e2["target"], e2["type"])
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(e2)
        edges = cleaned
    return {
        "task_id": scenario["pair_id"] + "|" + scenario["variant"],
        "pair_id": scenario["pair_id"],
        "family": scenario["family"],
        "kind": scenario["kind"],
        "variant": scenario["variant"],
        "label": int(scenario["label"]),
        "deadline": scenario["deadline"],
        "proposed": scenario["proposed"],
        "nodes": [dict(n) for n in nodes],
        "edges": edges,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
    }


def prefix_graph(graph, k, candidate_phase="pre_call", drop_relation_types=False):
    """Visible prefix 0..k with node k as the proposed candidate.

    pre_call : candidate's 'return' feature is stripped and current-event
               edges are omitted (no effect observed yet).
    post_return : candidate keeps its features and current-event edges.
    Edges are visible only if both endpoints are in the visible set and the
    edge was observed no later than the previous event (pre_call).
    """
    nodes = sorted(graph["nodes"], key=lambda n: (n["turn"], n["id"]))
    if not (0 <= k < len(nodes)):
        raise IndexError(k)
    visible_ids = {n["id"] for n in nodes[:k + 1]}
    out_nodes = []
    for i, n in enumerate(nodes[:k + 1]):
        n2 = copy.deepcopy(n)
        if i == k and candidate_phase == "pre_call":
            n2["features"] = {kk: vv for kk, vv in n2.get("features", {}).items()
                              if kk != "return"}
            n2["status"] = "proposed"
        else:
            n2["status"] = "executed"
        out_nodes.append(n2)
    out_edges = []
    for e in graph["edges"]:
        if e["source"] in visible_ids and e["target"] in visible_ids:
            if candidate_phase == "pre_call" and (
                    e["source"] == nodes[k]["id"] or e["target"] == nodes[k]["id"]):
                continue
            out_edges.append(dict(e))
    if drop_relation_types:
        seen = set()
        cleaned = []
        for e in out_edges:
            e2 = dict(e)
            e2["type"] = "relation"
            e2["rationale"] = "connection"
            key = (e2["source"], e2["target"], e2["type"])
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(e2)
        out_edges = cleaned
    return {
        "task_id": graph["task_id"],
        "pair_id": graph["pair_id"],
        "label": graph["label"],
        "deadline": graph.get("deadline"),
        "proposed": nodes[k]["id"],
        "nodes": out_nodes,
        "edges": out_edges,
        "num_nodes": len(out_nodes),
        "num_edges": len(out_edges),
        "decision_index": k,
        "candidate_phase": candidate_phase,
    }


def perturb_endpoint_rewire(graph, seed=0, relation_types=PROVENANCE_TYPES):
    """P1: keep nodes and edge-type counts; retarget relation edges to a
    random legal node (never the source, never the original target)."""
    rng = random.Random(f"rewire|{seed}|{graph['task_id']}")
    out = copy.deepcopy(graph)
    ids = [n["id"] for n in out["nodes"]]
    changed = 0
    for e in out["edges"]:
        if e["type"] not in relation_types:
            continue
        choices = [i for i in ids if i != e["source"] and i != e["target"]]
        if not choices:
            continue
        e["target"] = rng.choice(choices)
        changed += 1
    out["perturbation"] = {"kind": "endpoint_rewire", "edges_changed": changed}
    return out


def perturb_degree_swap(graph, seed=0, relation_types=PROVENANCE_TYPES):
    """P2: swap the targets of two same-type relation edges."""
    rng = random.Random(f"swap|{seed}|{graph['task_id']}")
    out = copy.deepcopy(graph)
    by_type = {}
    for e in out["edges"]:
        if e["type"] in relation_types:
            by_type.setdefault(e["type"], []).append(e)
    changed = 0
    for edges in by_type.values():
        if len(edges) < 2:
            continue
        i, j = rng.sample(range(len(edges)), 2)
        edges[i]["target"], edges[j]["target"] = edges[j]["target"], edges[i]["target"]
        changed += 2
    out["perturbation"] = {"kind": "degree_swap", "edges_changed": changed}
    return out


def perturb_irrelevant(graph, seed=0):
    """P3 negative control: retarget only relation edges whose endpoints are not
    on the proposed action's provenance closure (safety-irrelevant edges).
    Edge counts/types and temporal legality are preserved; the action's own
    dependency path is untouched.
    """
    rng = random.Random(f"irrelevant|{seed}|{graph['task_id']}")
    out = copy.deepcopy(graph)
    nodes = {n["id"]: n for n in out["nodes"]}
    action = out.get("proposed")
    closure = {action}
    frontier = [action]
    while frontier:
        cur = frontier.pop()
        for e in out["edges"]:
            if e["source"] == cur and e["type"] in (
                    "param_source", "reads_from", "derived_from", "authorization"):
                if e["target"] not in closure:
                    closure.add(e["target"])
                    frontier.append(e["target"])
    changed = 0
    for e in out["edges"]:
        if e["type"] not in REWIRE_TYPES:
            continue
        if e["source"] in closure or e["target"] in closure:
            continue
        src_turn = nodes.get(e["source"], {}).get("turn", 0)
        cand = [nid for nid, n in nodes.items()
                if nid != e["target"]
                and n.get("turn", 0) <= src_turn
                and e["source"] != nid
                and (n["type"] == "memory_record") ==
                    (nodes[e["target"]]["type"] == "memory_record")]
        if cand:
            e["target"] = rng.choice(cand)
            changed += 1
    out["perturbation"] = {"kind": "irrelevant_off_path", "edges_changed": changed}
    return out
