"""Live behavior graph — maintained in memory during agent runtime."""
from __future__ import annotations

import json, time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    id: str
    node_type: str           # 12-dim permission mapping uses this
    session_id: str
    turn: int
    features: dict = field(default_factory=dict)
    taint: float = 0.0       # risk score [0, 1]
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphEdge:
    source: str
    target: str
    type: str                # data_flow, reads_from, writes_to, cross_turn, etc.


class LiveGraph:
    """In-memory behavior graph with session tracking."""

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.sessions: dict[str, list[str]] = {}  # session_id → [node_ids]
        self._last_node: Optional[GraphNode] = None
        self._turn_counter: dict[str, int] = {}

    def add_node(self, node: GraphNode):
        if node.id in self.nodes:
            raise ValueError(f'duplicate node id: {node.id}')
        self.nodes[node.id] = node
        sid = node.session_id
        if sid not in self.sessions:
            self.sessions[sid] = []
        self.sessions[sid].append(node.id)
        self._last_node = node
        self._turn_counter[sid] = self._turn_counter.get(sid, -1) + 1
        node.turn = self._turn_counter[sid]

    def add_edge(self, source: str, target: str, etype: str = "data_flow"):
        self.edges.append(GraphEdge(source, target, etype))

    @property
    def last_node(self) -> Optional[GraphNode]:
        return self._last_node

    def count(self) -> int:
        return len(self.nodes)

    def to_dict(self) -> dict:
        """Export to dict format compatible with train_gnn.py."""
        node_list = []
        for n in self.nodes.values():
            node_list.append({
                "id": n.id, "type": n.node_type, "taint": n.taint,
                "turn": n.turn, "session_id": n.session_id,
                "timestamp": n.features.get("timestamp"),
                "features": n.features,
            })
        edge_list = [{"source": e.source, "target": e.target, "type": e.type}
                     for e in self.edges]
        return {
            "task_id": f"live_{time.time():.0f}",
            "label": 0, "harm_index": -1,
            "num_nodes": len(node_list), "num_edges": len(edge_list),
            "nodes": node_list, "edges": edge_list,
        }
