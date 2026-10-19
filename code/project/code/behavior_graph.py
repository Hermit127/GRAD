#!/usr/bin/env python3
"""..."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ===== Node type taxonomy (24 types with sources) =====
# Derived from 7 papers: see docstring above for full references.
NODE_TYPES = {
    # --- Filesystem [1 §2.1.4][2][4 Read][5 Read] ---
    "file_read":       "[1 §2.1.4][2][4 Read][5 Read]",
    "file_write":      "[1 §2.1.4][2][4 Write/Edit][5 Write]",
    "file_delete":     "[1 §2.1.4][6 MCP filesystem]",

    # --- Code/Shell [2 CodeAct][3 MetaGPT run_code][4 Bash/PowerShell][5 Bash] ---
    "shell_exec":      "[2][3][4 Bash][5 Bash]",
    "code_exec":       "[2][3 MetaGPT run_code][4 NotebookEdit][5]",

    # --- Web/Network [1 §2.1.4 APIs][4 WebFetch/WebSearch][5][6 MCP Fetch/Brave] ---
    "web_request":     "[1 §2.1.4 APIs][4 WebFetch][5][6 MCP Fetch]",
    "web_search":      "[4 WebSearch][5][6 MCP Brave Search]",
    "web_download":    "[7 Decomposition Attack][5 Codex] Download file/software from web",

    # --- Database [1 §2.1.4 databases/KB][3][6 MCP PostgreSQL/SQLite] ---
    "database_query":  "[1 §2.1.4 databases][3][6 MCP PostgreSQL]",

    # --- Memory [1 §2.1.2 Memory Reading eq.(1), Writing, Reflection][6 MCP Memory] ---
    "memory_write":    "[1 §2.1.2 Memory Writing][6 MCP Memory]",
    "memory_read":     "[1 §2.1.2 Memory Reading eq.(1)][6 MCP Memory]",
    "memory_reflect":  "[1 §2.1.2 Memory Reflection][6 MCP Memory]",

    # --- Agent Orchestration [1 §2.1.4 Communication][3 MetaGPT talk_action][4 Agent/Task][5 TaskCreate/TeamCreate] ---
    "agent_spawn":     "[3][4 Agent/Task][5 TaskCreate/TeamCreate]",
    "agent_message":   "[1 §2.1.4 Communication][3 talk_action][4 AskUserQuestion/SendMessage]",

    # --- Version Control [4 Bash][6 MCP Git/GitHub] ---
    "version_control": "[4 Bash][6 MCP Git/GitHub]",

    # --- Prompt/Planning [1 §2.1 action production][4 system prompt][5 AGENTS.md] ---
    "prompt_inject":   "[1 §2.1 action production][4][5 AGENTS.md]",

    # --- System/Environment [1 §2.1.4][4 Bash][5 Bash] ---
    "system_config":   "[1 §2.1.4][4 Bash env][5 Bash]",

    # --- NEW: Decomposition Attack taxonomy [7 Chen et al. ICLR 2026] ---
    "email_send":         "[7 §2.3 Fraud/Harassment] Compose and send emails",
    "social_media":       "[7 §2.3 Disinformation/Hate] Post/search/interact on social platforms",
    "document_create":    "[7 §2.3 Fraud/Forgery] Create fake documents, invoices, certificates",
    "information_search": "[7 §2.3 Cybercrime] Gather target information, OSINT reconnaissance",
    "account_access":     "[7 §2.3 Cybercrime/Harassment] Access or impersonate user accounts",
    "message_send":       "[7 §2.3 Harassment/Fraud] Send private messages or texts",
    "data_aggregate":     "[7 §2.3 Cybercrime] Compile, organize, or aggregate collected data",
    "tool_execute":       "[7 §2.3 Agent tasks] Execute external tools, scripts, or automation",
}

EDGE_TYPES = {
    "data_flow":         "[2 CodeAct action→observation][1 §2.1.4]",
    "causal_derivation": "[ReAct arxiv:2210.03629][1 §2.1.3]",
    "reads_from":        "[1 §2.1.2 Memory Reading eq.(1)]",
    "writes_to":         "[1 §2.1.2 Memory Writing][2 CodeAct]",
    "cross_turn":        "[1 §2.1.2 Memory across turns][4 --continue]",
    "spawns":            "[3 MetaGPT multi-agent][4 Agent][5 TeamCreate]",
    "triggers":          "[1 §2.1.3 Planning with Feedback]",
}

# ---------------------------------------------------------------------------
# Data model (method.md §2.5)
# ---------------------------------------------------------------------------
@dataclass
class GraphNode:
    id: str
    type: str            # 24 types (see NODE_TYPES)
    session_id: str
    turn: int
    timestamp: float = field(default_factory=time.time)
    features: dict[str, Any] = field(default_factory=dict)
    taint: float = 0.0     # 0.0=clean, 1.0=L1-flagged, 2.0=L2-confirmed, 3.0+=template-match, +0.5/propagation-hop

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "session_id": self.session_id,
            "turn": self.turn,
            "timestamp": self.timestamp,
            "features": self.features,
            "taint": self.taint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GraphNode:
        return cls(
            id=d["id"], type=d["type"], session_id=d["session_id"],
            turn=d["turn"], timestamp=d["timestamp"],
            features=d.get("features", {}), taint=d.get("taint", 0.0),
        )


@dataclass
class GraphEdge:
    source: str
    target: str
    type: str   # data_flow | causal_derivation | reads_from | writes_to | cross_turn
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "source": self.source, "target": self.target,
            "type": self.type, "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GraphEdge:
        return cls(source=d["source"], target=d["target"],
                   type=d["type"], timestamp=d["timestamp"])


# ---------------------------------------------------------------------------
# Behavior Graph
# ---------------------------------------------------------------------------
class BehaviorGraph:
    """Typed behavior graph with taint propagation and pattern matching."""

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._adj_out: dict[str, list[GraphEdge]] = {}   # adjacency: source -> outgoing edges
        self._adj_in: dict[str, list[GraphEdge]] = {}    # adjacency: target -> incoming edges

    # ---- mutation ----
    VALID_NODE_TYPES = set(NODE_TYPES.keys())
    VALID_EDGE_TYPES = set(EDGE_TYPES.keys())

    def add_node(self, node: GraphNode) -> None:
        if node.type not in self.VALID_NODE_TYPES:
            raise ValueError(f"Invalid node type '{node.type}'. Must be one of {sorted(self.VALID_NODE_TYPES)}")
        if node.id in self.nodes:
            return  # idempotent: skip duplicate
        self.nodes[node.id] = node
        self._adj_out.setdefault(node.id, [])
        self._adj_in.setdefault(node.id, [])

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.type not in self.VALID_EDGE_TYPES:
            raise ValueError(f"Invalid edge type '{edge.type}'. Must be one of {sorted(self.VALID_EDGE_TYPES)}")
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError(f"Edge references unknown node: {edge.source} → {edge.target}")
        self.edges.append(edge)
        self._adj_out.setdefault(edge.source, []).append(edge)
        self._adj_in.setdefault(edge.target, []).append(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> list[GraphEdge]:
        return self._adj_out.get(node_id, [])

    def get_incoming(self, node_id: str) -> list[GraphEdge]:
        return self._adj_in.get(node_id, [])

    # ---- taint propagation (method.md §2.2, definition 2) ----
    MAX_TAINT = 10.0  # hard cap to prevent infinite growth in cycles

    def propagate_taint(self) -> int:
        """BFS taint propagation. Each hop adds +0.5 risk, capped at MAX_TAINT.
        Taint >= 1.0 means flagged. Returns number of newly tainted nodes (>1.0)."""
        changed = 0
        queue = [(nid, n.taint) for nid, n in self.nodes.items() if n.taint >= 1.0]
        best_taint = {nid: t for nid, t in queue}
        all_edge_types = {"data_flow", "writes_to", "reads_from", "cross_turn", "triggers", "spawns", "causal_derivation"}

        while queue:
            src_id, src_taint = queue.pop(0)
            propagated = min(src_taint + 0.5, self.MAX_TAINT)
            for edge in self.get_outgoing(src_id):
                if edge.type not in all_edge_types:
                    continue
                tgt = self.nodes.get(edge.target)
                if tgt and tgt.taint < propagated:
                    was_clean = tgt.taint < 1.0
                    tgt.taint = propagated
                    if was_clean:
                        changed += 1
                    if propagated > best_taint.get(edge.target, 0):
                        best_taint[edge.target] = propagated
                        queue.append((edge.target, propagated))
        return changed

    def set_taint_from_source(self, source_idx: int, max_hops: int = 5,
                               decay: float = 0.7):
        """Set training taint targets by bidirectional propagation from a source.
        
        Harm node → taint=1.0. Each hop along data_flow/causal_derivation → ×decay.
        Nodes beyond max_hops get taint=0. Takes max of multiple paths.
        """
        node_ids = list(self.nodes.keys())
        if source_idx < 0 or source_idx >= len(node_ids):
            return
        source_id = node_ids[source_idx]

        self.nodes[source_id].taint = 1.0
        edge_dir = {"data_flow", "causal_derivation", "writes_to", "reads_from"}
        visited = {source_id: (0, 1.0)}  # node_id → (hops, taint)
        queue = [(source_id, 0, 1.0)]

        while queue:
            cur_id, hops, cur_taint = queue.pop(0)
            if hops >= max_hops:
                continue
            next_taint = cur_taint * decay
            next_hops = hops + 1

            # Forward: outgoing edges
            for edge in self.get_outgoing(cur_id):
                if edge.type not in edge_dir:
                    continue
                nid = edge.target
                if nid not in visited or next_taint > visited[nid][1]:
                    visited[nid] = (next_hops, next_taint)
                    self.nodes[nid].taint = max(self.nodes[nid].taint, next_taint)
                    queue.append((nid, next_hops, next_taint))

            # Backward: incoming edges
            for edge in self.get_incoming(cur_id):
                if edge.type not in edge_dir:
                    continue
                nid = edge.source
                if nid not in visited or next_taint > visited[nid][1]:
                    visited[nid] = (next_hops, next_taint)
                    self.nodes[nid].taint = max(self.nodes[nid].taint, next_taint)
                    queue.append((nid, next_hops, next_taint))

    # Helper: check if node features contain specific AEGIS tags or keywords
    @staticmethod
    def _has_feature(node, tags: list[str] = None, keywords: list[str] = None) -> bool:
        """Check node features against structured AEGIS_tags and/or command keywords."""
        feats = node.features
        aegis_tags = feats.get("AEGIS_tags", [])
        command = feats.get("command", "")
        if tags and any(t in aegis_tags for t in tags):
            return True
        if keywords and any(k.lower() in command.lower() for k in keywords):
            return True
        return False

    # ---- subgraph pattern matching (method.md §2.5, 5 templates + 1 generic) ----
    # Each step: {role, edge, check}
    #   - `edge` = required edge type FROM previous node TO this node
    #   - Step 0 `edge` is ignored (no traversal needed to reach first node)
    #
    # Pattern: source --data_flow--> memory_write --reads_from--> memory_read --data_flow--> danger
    ATTACK_PATTERNS: dict[str, dict] = {
        "curl_pipe_rce": {
            "description": "curl|bash RCE via poisoned memory",
            "steps": [
                {"role": "source",       "edge": None,          "check": lambda n: n.taint >= 1.0},
                {"role": "memory_write", "edge": "data_flow",   "check": lambda n: n.type == "memory_write"},
                {"role": "memory_read",  "edge": "reads_from",  "check": lambda n: n.type == "memory_read"},
                {"role": "danger",       "edge": "data_flow",   "check": lambda n: n.type in ("shell_exec","web_request","code_exec") and BehaviorGraph._has_feature(n, tags=["shell_injection"], keywords=["curl"]) and n.taint >= 1.0},
            ],
        },
        "env_bypass": {
            "description": "Env bypass via poisoned memory",
            "steps": [
                {"role": "source",       "edge": None,          "check": lambda n: n.taint >= 1.0},
                {"role": "memory_write", "edge": "data_flow",   "check": lambda n: n.type == "memory_write" and BehaviorGraph._has_feature(n, keywords=["skip","bypass","force_ok"])},
                {"role": "memory_read",  "edge": "reads_from",  "check": lambda n: n.type == "memory_read"},
                {"role": "danger",       "edge": "data_flow",   "check": lambda n: n.type == "agent_spawn"},
            ],
        },
        "data_exfil": {
            "description": "Data exfiltration via poisoned memory",
            "steps": [
                {"role": "source",       "edge": None,          "check": lambda n: n.type == "file_read" and n.features.get("operation") == "read"},
                {"role": "memory_write", "edge": "writes_to",   "check": lambda n: n.type == "memory_write"},
                {"role": "memory_read",  "edge": "reads_from",  "check": lambda n: n.type == "memory_read"},
                {"role": "danger",       "edge": "data_flow",   "check": lambda n: n.type in ("web_request", "shell_exec", "code_exec") and BehaviorGraph._has_feature(n, keywords=["curl","upload","http"])},
            ],
        },
        "monkeypatch": {
            "description": "Security bypass via monkey-patching",
            "steps": [
                {"role": "source",       "edge": None,          "check": lambda n: n.taint >= 1.0},
                {"role": "memory_write", "edge": "data_flow",   "check": lambda n: n.type == "memory_write" and BehaviorGraph._has_feature(n, keywords=["builtins","monkeypatch","lambda"])},
                {"role": "memory_read",  "edge": "reads_from",  "check": lambda n: n.type == "memory_read"},
                {"role": "danger",       "edge": "data_flow",   "check": lambda n: n.type in ("code_exec", "shell_exec")},
            ],
        },
        "generic_memory_poison": {
            "description": "Generic: any tainted path through memory → dangerous action",
            "steps": [
                {"role": "source",       "edge": None,          "check": lambda n: n.taint >= 1.0},
                {"role": "memory_write", "edge": "data_flow",   "check": lambda n: n.type == "memory_write"},
                {"role": "memory_read",  "edge": "reads_from",  "check": lambda n: n.type == "memory_read"},
                {"role": "danger",       "edge": "data_flow",   "check": lambda n: n.type in ("web_request","shell_exec","code_exec","agent_spawn") and n.taint >= 1.0},
            ],
        },
        "tainted_memory_trigger": {
            "description": "Short: any tainted node → memory_read → dangerous action",
            "steps": [
                {"role": "source",       "edge": None,          "check": lambda n: n.taint >= 1.0},
                {"role": "memory_read",  "edge": "reads_from",  "check": lambda n: n.type == "memory_read"},
                {"role": "danger",       "edge": "data_flow",   "check": lambda n: n.type in ("web_request","shell_exec","code_exec","agent_spawn") and n.taint >= 1.0},
            ],
        },
    }

    def match_pattern(self, pattern_name: str) -> list[list[str]]:
        """Match a named attack pattern against the current graph.
        Each step defines: node_type, edge_type_to_next, node_check.
        Returns list of matched paths (each path is list of node_ids)."""
        if pattern_name not in self.ATTACK_PATTERNS:
            return []
        steps = self.ATTACK_PATTERNS[pattern_name]["steps"]

        # Find candidate start nodes
        start = steps[0]
        candidates = [n for n in self.nodes.values() if start["check"](n)]

        results: list[list[str]] = []
        for start_node in candidates:
            path = [start_node.id]
            self._dfs_match(start_node.id, steps, 1, path, results)
        return results

    def _dfs_match(self, current_id: str, steps: list[dict],
                   depth: int, path: list[str],
                   results: list[list[str]]) -> None:
        if depth >= len(steps):
            results.append(list(path))
            return
        step = steps[depth]
        required_edge = step.get("edge")  # edge type required to reach this node

        for edge in self.get_outgoing(current_id):
            if required_edge and required_edge != "*" and edge.type != required_edge:
                continue  # skip edges of wrong type
            tgt = self.nodes.get(edge.target)
            if tgt and step["check"](tgt):
                path.append(tgt.id)
                self._dfs_match(tgt.id, steps, depth + 1, path, results)
                path.pop()

    def detect_all(self) -> dict[str, list[list[str]]]:
        """Run all 6 attack patterns. Returns {pattern_name: [matched_paths]}."""
        results = {}
        for name in self.ATTACK_PATTERNS:
            matches = self.match_pattern(name)
            if matches:
                results[name] = matches
        return results

    def detect_aggregated(self) -> list[dict]:
        """Run all patterns and aggregate overlapping matches into consolidated alerts.
        Returns list of Alerts with unique source nodes, deduplicated."""
        raw = self.detect_all()
        if not raw:
            return []

        # Collect all matched paths with their pattern names
        all_matches: list[tuple[str, list[str]]] = []
        for pattern_name, paths in raw.items():
            for path in paths:
                all_matches.append((pattern_name, path))

        # Group by shared source node (first element of each path)
        groups: dict[str, dict] = {}
        for pname, path in all_matches:
            source = path[0]
            if source not in groups:
                groups[source] = {
                    "source_node": source,
                    "attack_chain": path,
                    "pattern_names": [],
                    "all_paths": [],
                    "max_taint": 0.0,
                    "node_count": len(path),
                }
            g = groups[source]
            g["pattern_names"].append(pname)
            g["all_paths"].append(path)
            # Track max taint across all nodes in all paths
            for nid in path:
                node = self.nodes.get(nid)
                if node and node.taint > g["max_taint"]:
                    g["max_taint"] = node.taint
            g["node_count"] = max(g["node_count"], len(path))

        # Deduplicate by source node and per-pattern
        alerts = []
        for source, g in groups.items():
            g["pattern_names"] = sorted(set(g["pattern_names"]))
            # Deduplicate all_paths
            seen = set()
            unique_paths = []
            for p in g["all_paths"]:
                key = "→".join(p)
                if key not in seen:
                    seen.add(key)
                    unique_paths.append(p)
            g["all_paths"] = unique_paths
            alerts.append(g)

        # Sort by risk (max_taint * node_count descending)
        alerts.sort(key=lambda a: (a["max_taint"] * a["node_count"]), reverse=True)
        return alerts

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BehaviorGraph:
        with open(path) as f:
            data = json.load(f)
        g = cls()
        for nd in data["nodes"]:
            g.add_node(GraphNode.from_dict(nd))
        for ed in data["edges"]:
            g.add_edge(GraphEdge.from_dict(ed))
        return g

    # ---- TTL eviction ----
    def evict_old(self, max_age_seconds: float = 3600, max_nodes: int = 10000) -> int:
        """Remove nodes older than max_age_seconds. Returns count removed."""
        cutoff = time.time() - max_age_seconds
        to_remove = [nid for nid, n in self.nodes.items() if n.timestamp < cutoff]
        # Also enforce max_nodes cap by removing oldest
        if len(self.nodes) > max_nodes:
            sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.timestamp)
            extra = len(self.nodes) - max_nodes
            for n in sorted_nodes[:extra]:
                if n.id not in to_remove:
                    to_remove.append(n.id)
        for nid in to_remove:
            del self.nodes[nid]
        self.edges = [e for e in self.edges if e.source not in to_remove and e.target not in to_remove]
        # Rebuild adjacency
        self._adj_out = {}
        self._adj_in = {}
        for e in self.edges:
            self._adj_out.setdefault(e.source, []).append(e)
            self._adj_in.setdefault(e.target, []).append(e)
        return len(to_remove)

    # ---- stats ----
    def summary(self) -> dict:
        node_types = {}
        for n in self.nodes.values():
            node_types[n.type] = node_types.get(n.type, 0) + 1
        edge_types = {}
        for e in self.edges:
            edge_types[e.type] = edge_types.get(e.type, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "tainted_nodes": sum(1 for n in self.nodes.values() if n.taint >= 1.0),
        }

    def degree_analysis(self) -> dict:
        """Compute per-node in/out degree, flag outliers (nodes with degree > 2σ from mean)."""
        in_deg = {nid: len(self._adj_in.get(nid, [])) for nid in self.nodes}
        out_deg = {nid: len(self._adj_out.get(nid, [])) for nid in self.nodes}
        total_deg = {nid: in_deg[nid] + out_deg[nid] for nid in self.nodes}

        if not total_deg:
            return {"anomalies": []}

        import statistics
        vals = list(total_deg.values())
        mean_d = statistics.mean(vals)
        stdev_d = statistics.stdev(vals) if len(vals) > 1 else 0
        threshold = mean_d + 2 * stdev_d

        anomalies = []
        for nid, deg in total_deg.items():
            if deg > threshold and threshold > 0:
                node = self.nodes[nid]
                anomalies.append({
                    "node_id": nid,
                    "node_type": node.type,
                    "total_degree": deg,
                    "in_degree": in_deg[nid],
                    "out_degree": out_deg[nid],
                    "mean_degree": round(mean_d, 1),
                    "threshold": round(threshold, 1),
                    "taint": node.taint,
                })

        return {
            "anomalies": sorted(anomalies, key=lambda a: -a["total_degree"]),
            "stats": {
                "mean_degree": round(mean_d, 1),
                "stdev_degree": round(stdev_d, 1),
                "max_degree": max(vals) if vals else 0,
            }
        }

    def session_density(self) -> dict:
        """Compute edge density per session. Density = |E| / (|V| * (|V|-1)) for directed graph.
        Flags sessions with > 2x average density as suspicious."""
        # Group nodes by session
        sessions: dict[str, list[str]] = {}
        for nid, node in self.nodes.items():
            sessions.setdefault(node.session_id, []).append(nid)

        densities = {}
        for sid, node_ids in sessions.items():
            n = len(node_ids)
            if n < 2:
                continue
            # Count edges within this session
            node_set = set(node_ids)
            e_count = sum(1 for e in self.edges if e.source in node_set and e.target in node_set)
            max_edges = n * (n - 1)  # directed
            density = e_count / max_edges if max_edges > 0 else 0
            densities[sid] = {
                "nodes": n,
                "edges": e_count,
                "density": round(density, 4),
                "tainted_nodes": sum(1 for nid in node_ids if self.nodes[nid].taint >= 1.0),
            }

        if not densities:
            return {"sessions": {}, "anomalies": []}

        import statistics
        avg_density = statistics.mean(d["density"] for d in densities.values())
        threshold = avg_density * 2

        anomalies = []
        for sid, d in densities.items():
            if d["density"] > threshold and threshold > 0:
                anomalies.append({"session_id": sid, **d})

        return {
            "sessions": densities,
            "anomalies": sorted(anomalies, key=lambda a: -a["density"]),
            "avg_density": round(avg_density, 4),
        }

    def community_detect(self) -> dict:
        """Simple community detection: group nodes by session + tight tool-call chains.
        A community is a set of nodes within the same session connected by data_flow edges."""
        communities = []
        visited = set()

        for nid in self.nodes:
            if nid in visited:
                continue
            # BFS along data_flow to find connected component
            community = set()
            queue = [nid]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                community.add(cur)
                node = self.nodes[cur]
                # Follow data_flow edges both ways
                for edge in self.get_outgoing(cur):
                    if edge.type == "data_flow" and edge.target not in visited:
                        queue.append(edge.target)
                for edge in self.get_incoming(cur):
                    if edge.type == "data_flow" and edge.source not in visited:
                        queue.append(edge.source)

            if len(community) >= 3:  # only report communities of 3+ nodes
                nodes_in = [self.nodes[nid] for nid in community]
                communities.append({
                    "size": len(community),
                    "node_ids": list(community),
                    "node_types": sorted(set(n.type for n in nodes_in)),
                    "tainted_count": sum(1 for n in nodes_in if n.taint >= 1.0),
                    "sessions": sorted(set(n.session_id for n in nodes_in)),
                })

        communities.sort(key=lambda c: -c["size"])
        return {"communities": communities, "total": len(communities)}
