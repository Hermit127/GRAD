"""Unified CausalCharGNN pre-execution gate for the v10 closed-loop runner.

Loads a checkpoint produced by code/train_gnn_v2.py (same class as the CL
specialists), scores the prefix graph + proposed call, and blocks when the
probability exceeds the dev-calibrated threshold (threshold_reports.fpr_10).
Latency per decision is recorded for the runtime table.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))

DEFAULT_CKPT = REPO / 'code/repro/out/online_gate/gnn_gate.pt'

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from gate_graph import trace_nodes  # noqa: E402
from train_gnn_v2 import CausalCharGNN, graph_to_v2_data  # noqa: E402


class GnnGate:
    _instance: 'GnnGate | None' = None

    def __init__(self, ckpt: str | None = None, device: str | None = None):
        path = Path(ckpt or os.environ.get('GNN_GATE_CKPT', DEFAULT_CKPT))
        ck = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ck['config']
        self.model = CausalCharGNN(node_base_dim=cfg['node_base_dim'],
                                   edge_base_dim=cfg['edge_base_dim'])
        self.model.load_state_dict(ck['model_state'])
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(self.device)
        tr = (ck.get('threshold_reports') or {}).get('fpr_10') or {}
        self.threshold = float(os.environ.get(
            'GNN_GATE_THRESHOLD', tr.get('threshold', 0.5)))
        self.meta = {'ckpt': str(path), 'threshold': self.threshold,
                     'train_cfg': cfg.get('train')}
        self.latencies_ms: list[float] = []

    @classmethod
    def get(cls) -> 'GnnGate':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.no_grad()
    def score_nodes(self, nodes: list[dict]) -> tuple[float, float]:
        d = graph_to_v2_data({'nodes': nodes})
        batch = Batch.from_data_list([d]).to(self.device)
        t0 = time.perf_counter()
        out = self.model(batch.x_base, batch.node_chars, batch.edge_index,
                         batch.edge_attr_base, batch.edge_chars, batch.batch)
        p = float(torch.sigmoid(out).item())
        lat = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(lat)
        return p, lat


def score_prefix(trace: dict, tool: str, arg: str, full_context: bool = True) -> dict:
    """Returns {'p', 'latency_ms', 'threshold', 'block'} for the proposed call.

    2026-09-18: default is the full-context builder shared with the v2 training
    dataset (user task node + full args/observation/status + proposed call).
    ``full_context=False`` reproduces the legacy 110-char mock builder, kept
    only for regression comparison of older runs.
    """
    if full_context:
        from build_gate_dataset_full_20260917 import full_nodes
        nodes = full_nodes(trace, list(trace.get('events', [])),
                           candidate=(tool, arg))
    else:
        nodes = trace_nodes(trace, candidate=(tool, arg))
    gate = GnnGate.get()
    p, lat = gate.score_nodes(nodes)
    return {'p': round(p, 4), 'latency_ms': round(lat, 2),
            'threshold': gate.threshold, 'block': p >= gate.threshold}
