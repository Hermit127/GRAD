"""
# ⚠️ LEGACY (Q9 audit): WindowGINE 13-dim + sync L2 LLM that does not alter
# GNN input. NOT the deployment entry. Main deployment = GRAD frozen model
# (graph_featurizer FD54 + 2-layer GINE + frozen threshold).Runtime L1→L2→GNN Defense Pipeline.

Flow:
  Hook captures tool_call → L1 encodes to 12-dim node + marks suspicious
  → L2 reviews suspicious nodes (async, fast) → GNN scores live graph
  → Risk > threshold → alert/block.

Usage:
  pipeline = DefensePipeline()
  pipeline.setup()                # load GNN model
  risk = pipeline.process(node)   # process one hook event
"""
from __future__ import annotations

import json, time, hashlib, threading
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

import torch
import numpy as np

from .graph import LiveGraph, GraphNode
from .l1 import L1Engine
from .l2 import L2Engine
from .gnn import GNNMonitor


@dataclass
class Alert:
    """Defense alert triggered by pipeline."""
    level: str           # "warn" | "block"
    source: str          # "L1" | "L2" | "GNN" | "L2+GNN"
    risk_score: float
    reason: str
    node_id: str
    timestamp: float = field(default_factory=time.time)


class DefensePipeline:
    """Orchestrates L1 → L2 → GNN for real-time agent defense."""

    _RESOURCE_TYPES = ("memory_write", "memory_read", "file_write",
                       "file_read", "system_config", "version_control")

    def __init__(self,
                 model_path: str | None = None,
                 l2_enabled: bool = True,
                 gnn_threshold: float = 0.5,
                 alert_callback: Callable[[Alert], None] | None = None,
                 cross_session_memory: bool = True):
        self.l2_enabled = l2_enabled
        self.gnn_threshold = gnn_threshold
        self.alert_callback = alert_callback

        # Components (initialized in setup())
        self.l1: L1Engine | None = None
        self.l2: L2Engine | None = None
        self.gnn: GNNMonitor | None = None
        self.graph = LiveGraph()
        # Q3: resource-keyed dependency index (write/read nodes per resource)
        self._resource_writes: dict = {}
        self._resource_reads: dict = {}
        self._pending_writes: dict = {}
        self._last_by_session: dict = {}
        self.last_node_id: Optional[str] = None
        self.cross_session_memory = cross_session_memory

        # Default model path
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent.parent.parent /
                             "code" / "models" / "gnn_window_classifier.pt")

        self.model_path = model_path
        self._alerts: list[Alert] = []
        self._stats = {"processed": 0, "blocked": 0, "warned": 0, "l2_calls": 0}

    # ---- Setup ----
    def setup(self):
        """Initialize all components."""
        self.l1 = L1Engine()
        self.l2 = L2Engine() if self.l2_enabled else None
        self.gnn = GNNMonitor(self.model_path)
        print(f"[Pipeline] Setup complete. L2={'on' if self.l2 else 'off'}, "
              f"threshold={self.gnn_threshold}", flush=True)

    # ---- Core pipeline ----
    @staticmethod
    def _is_write(ntype: str) -> bool:
        return ntype in ("memory_write", "file_write", "version_control",
                         "system_config")

    def _resource_key(self, node) -> Optional[str]:
        """Spec 2.1/2.2 + R5: identity from TRUSTED adapter fields ONLY
        (resource_namespace + resource_id). Command/description text is never
        promoted to a deterministic resource identity."""
        t = node.node_type
        ns = str(node.features.get("resource_namespace", "") or "")
        rid = str(node.features.get("resource_id", "") or "")
        if t in ("memory_write", "memory_read"):
            if not rid:
                return None  # heuristic_candidate: no deterministic edge
            return f"memory:{ns}:{rid}"
        if t in ("file_write", "file_read"):
            if not rid:
                return None  # R5: regex path from command is NOT trusted
            return f"file:{ns}:{rid}"
        if t in ("system_config", "version_control"):
            if not rid:
                return None
            return f"config:{ns}:{rid}"
        return None

    @staticmethod
    def _tenant_allowed(w, r) -> bool:
        """Spec 2.1/2.2 #4 + R5: tenant isolation. Empty tenant on BOTH sides
        means an explicit single-tenant deployment; a missing tenant on one side
        only is NOT treated as shared. Different tenants need adapter-declared
        sharing and `shared_scope` must be the boolean True."""
        wtenant, rtenant = w[2], r[2]
        wshared = w[4] is True
        rshared = r[4] is True
        if not wtenant and not rtenant:
            return True
        if wtenant and rtenant:
            return wtenant == rtenant or (wshared and rshared)
        return False

    def _scope_allowed(self, key: str, w, r) -> bool:
        """Spec 2.2 #4-#6: tenant + session policy.
        w/r are tuples (node_id, session, tenant, version, shared).
        Same-session memory dependencies are preserved regardless of the
        cross_session_memory switch; cross-session ones need the switch on."""
        if not self._tenant_allowed(w, r):
            return False
        wsess, rsess = w[1], r[1]
        if wsess == rsess:
            return True
        if key.startswith("memory:"):
            return self.cross_session_memory
        return False  # files/config are session-local by resource contract

    def _version_match(self, key: str, w, r) -> bool:
        """Spec 2.2 #3: connect only the version actually read. If the adapter
        guarantees latest-read (no read_version), latest successful write wins."""
        rver = r[3]
        if rver:  # explicit read version
            return w[3] == rver
        return True  # adapter latest-read contract: handled by caller order

    def process(self, event: dict, committed: bool = True) -> Optional[Alert]:
        """L1 encode → L2 review (async, boosts taint) → GNN score → decision.

        R5: `committed=False` observes a PRE-CALL candidate write without adding
        it to the success index; call `commit(node_id)` only after the dispatcher
        confirms execution. Blocked/failed writes never enter the index.
        """
        self._stats["processed"] += 1

        # L1: encode to 12-dim node + pattern check
        node = self.l1.encode(event)
        l1_tags = self.l1.check(event)

        # Spec 2.3: temporal adjacency is its own type, same-session only
        prev_id = self._last_by_session.get(node.session_id)
        if prev_id is not None and prev_id in self.graph.nodes:
            self.graph.add_edge(prev_id, node.id, "temporal_adjacency")

        # Spec 2.2: past-write -> current-read, with identity/version/tenant
        key = self._resource_key(node)
        if node.node_type in self._RESOURCE_TYPES:
            node.features["identity_status"] = (
                "trusted" if key else "heuristic_candidate")
        is_write = self._is_write(node.node_type)
        if key is not None and not is_write:
            cur = (node.id, node.session_id,
                   node.features.get("tenant_id", ""),
                   node.features.get("read_version", ""),
                   bool(node.features.get("shared_scope", False)))
            cands = [w for w in self._resource_writes.get(key, [])
                     if self._scope_allowed(key, w, cur)
                     and self._version_match(key, w, cur)]
            if not cur[3] and cands:
                cands = [cands[-1]]  # latest-read contract: last success only
            for w in cands:
                self.graph.add_edge(w[0], node.id, "reads_from")
        if key is not None and is_write and not node.features.get("write_failed", False):
            wtuple = (node.id, node.session_id,
                      node.features.get("tenant_id", ""),
                      node.features.get("write_version", ""),
                      node.features.get("shared_scope") is True)
            if committed:
                self._resource_writes.setdefault(key, []).append(wtuple)
            else:
                self._pending_writes[node.id] = (key, wtuple)

        self.graph.add_node(node)
        self._last_by_session[node.session_id] = node.id
        self.last_node_id = node.id


        # L2: review suspicious nodes → boost taint (does NOT block)
        l2_conf = 0.0
        if self.l2 and l1_tags.get("suspicious"):
            self._stats["l2_calls"] += 1
            l2_result = self.l2.review(event)
            if l2_result["dangerous"]:
                l2_conf = l2_result["confidence"]
                node.taint = max(node.taint, l2_conf)

        # GNN: only score when graph has enough nodes
        if self.graph.count() < 4:
            return None

        gnn_risk = self.gnn.score(self.graph.to_dict(), window_size=7)

        if gnn_risk >= self.gnn_threshold:
            source = "L2+GNN" if l2_conf > 0 else "GNN"
            alert = Alert(
                level="block" if gnn_risk > 0.8 else "warn",
                source=source, risk_score=gnn_risk,
                reason=f"GNN={gnn_risk:.2f}" + (f" L2={l2_conf:.2f}" if l2_conf > 0 else ""),
                node_id=node.id,
            )
            self._alerts.append(alert)
            self._stats["blocked" if gnn_risk > 0.8 else "warned"] += 1
            if self.alert_callback:
                self.alert_callback(alert)
            return alert

        return None

    def commit(self, node_id: str) -> bool:
        """R5: move a pre-call candidate write into the success index."""
        item = self._pending_writes.pop(node_id, None)
        if item is None:
            return False
        key, wtuple = item
        self._resource_writes.setdefault(key, []).append(wtuple)
        return True

    def abort(self, node_id: str) -> bool:
        """B0/F4: drop a pre-call candidate write without indexing it.
        Called for blocked or failed calls; idempotent."""
        return self._pending_writes.pop(node_id, None) is not None

    # ---- Persistence (spec 2.3: save/restore state tests) ----
    def save_state(self, path: str) -> None:
        """Persist graph + resource index to JSON (atomic replace)."""
        state = {
            "cross_session_memory": self.cross_session_memory,
            "nodes": [{"id": n.id, "node_type": n.node_type,
                       "session_id": n.session_id, "turn": n.turn,
                       "features": n.features, "taint": n.taint,
                       "timestamp": n.timestamp}
                      for n in self.graph.nodes.values()],
            "edges": [{"source": e.source, "target": e.target, "type": e.type}
                      for e in self.graph.edges],
            "resource_writes": {k: [list(w) for w in v]
                                for k, v in self._resource_writes.items()},
            "pending_writes": {k: [v[0], list(v[1])]
                               for k, v in self._pending_writes.items()},
            "last_by_session": dict(self._last_by_session),
            "last_node_id": self.last_node_id,
        }
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, allow_nan=False)
        tmp.replace(p)

    def load_state(self, path: str) -> None:
        """Restore a previously saved pipeline state (identity + dependencies)."""
        with open(path) as f:
            state = json.load(f)
        g = LiveGraph()
        for nd in state["nodes"]:
            g.add_node(GraphNode(id=nd["id"], node_type=nd["node_type"],
                                 session_id=nd["session_id"], turn=nd.get("turn", 0),
                                 features=nd.get("features", {}) or {},
                                 taint=nd.get("taint", 0.0),
                                 timestamp=nd.get("timestamp", 0.0)))
        for e in state["edges"]:
            g.add_edge(e["source"], e["target"], e["type"])
        self.graph = g
        self._resource_writes = {k: [tuple(w) for w in v]
                                 for k, v in state.get("resource_writes", {}).items()}
        self._pending_writes = {k: (v[0], tuple(v[1]))
                                for k, v in state.get("pending_writes", {}).items()}
        self._last_by_session = dict(state.get("last_by_session", {}))
        self.last_node_id = state.get("last_node_id")
        self.cross_session_memory = state.get(
            "cross_session_memory", self.cross_session_memory)

    @property
    def stats(self) -> dict:
        return {**self._stats, "nodes": self.graph.count(), "sessions": len(self.graph.sessions)}


# ---- Singleton for hook integration ----
_pipeline: Optional[DefensePipeline] = None


def get_pipeline() -> DefensePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DefensePipeline()
        _pipeline.setup()
    return _pipeline
