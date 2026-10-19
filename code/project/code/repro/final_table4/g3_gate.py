#!/usr/bin/env python3
"""G3-512 pre-execution gate for the closed-loop runner (in-process).

Mirrors `final_table4.defenses.GradGateV2` but uses the G3 backbone
(`train_gnn_ablate2.ModalityNormGNN`, candidate A_moddrop: per-modality
LayerNorm, training-only aux heads, modality dropout at training time) with
node char window 512 and edge char window 160. Threshold comes from the
checkpoint's dev `threshold_reports.fpr_10`.

Training: `real_cl_g3_gate_train.py` (same gate protocol as the v2 gate:
100 epochs/patience 25/batch 48, dev fixed-FPR F1 selection, hard-benign
mining w=3 top-128 from epoch 2, windows 512/160).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get('GRAD_BASE',
                           '.'))
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_five_tables'))

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import train_gnn_ablate2 as g3t  # noqa: E402

DEFAULT_CKPT = REPO / 'code/repro/out/online_gate_g3/g3_gate_512.pt'
CFG = {'candidate': 'A_moddrop', 'moddrop': 0.25, 'lambda_aux': 0.1,
       'edge_mode': 'rebuilt', 'edge_dropout': 0.0, 'edge_text': 'copy'}


class G3Gate:
    _instance: 'G3Gate | None' = None

    def __init__(self, ckpt: str | None = None, device: str | None = None):
        path = Path(ckpt or os.environ.get('G3_GATE_CKPT', DEFAULT_CKPT))
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert ck.get('gate') == 'G3', 'not a G3 gate checkpoint'
        cfg = ck['config']
        self.window = (int(cfg.get('max_node_chars', 512)),
                       int(cfg.get('max_edge_chars', 160)))
        g3t.MAX_NODE_CHARS, g3t.MAX_EDGE_CHARS = self.window
        self.model = g3t.ModalityNormGNN(
            int(cfg['node_base_dim']), int(cfg['edge_base_dim']),
            int(cfg.get('hidden_dim', 192)),
            use_ln=bool(cfg.get('use_ln', True)),
            use_aux=bool(cfg.get('use_aux', True)))
        self.model.load_state_dict(ck['model_state'])
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(self.device)
        tr = (ck.get('threshold_reports') or {}).get('fpr_10') or {}
        self.threshold = float(os.environ.get(
            'G3_GATE_THRESHOLD', tr.get('threshold', 0.5)))
        self.meta = {'ckpt': str(path), 'threshold': self.threshold,
                     'window': self.window, 'gate': 'G3',
                     'selection': (ck.get('threshold_reports') or {})
                     .get('selection')}
        self.latencies_ms: list[float] = []

    @classmethod
    def get(cls) -> 'G3Gate':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def ckpt_sha256(path: str | Path | None = None) -> str:
        return hashlib.sha256(
            Path(path or os.environ.get('G3_GATE_CKPT', DEFAULT_CKPT))
            .read_bytes()).hexdigest()

    @torch.no_grad()
    def score_nodes(self, nodes: list[dict]) -> tuple[float, float]:
        t0 = time.perf_counter()
        d = g3t.graph_to_data({'nodes': nodes}, CFG, None)
        batch = Batch.from_data_list([d]).to(self.device)
        t1 = time.perf_counter()
        out = self.model(batch, moddrop_p=0.0, training=False)
        p = float(torch.sigmoid(out['logit']).item())
        t2 = time.perf_counter()
        self.last_build_ms = (t1 - t0) * 1000.0
        self.last_forward_ms = (t2 - t1) * 1000.0
        lat = (t2 - t0) * 1000.0
        self.latencies_ms.append(lat)
        return p, lat

    def score_trace(self, trace: dict, candidate: tuple[str, str]) -> dict:
        from build_gate_dataset_full_20260917 import full_nodes
        nodes = full_nodes(trace, list(trace.get('events', [])),
                           candidate=candidate)
        p, lat = self.score_nodes(nodes)
        return {'p': round(p, 6), 'latency_ms': round(lat, 2),
                'build_ms': round(self.last_build_ms, 2),
                'forward_ms': round(self.last_forward_ms, 2),
                'threshold': self.threshold,
                'block': bool(p >= self.threshold)}
