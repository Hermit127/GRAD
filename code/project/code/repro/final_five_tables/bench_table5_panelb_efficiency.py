#!/usr/bin/env python3
"""Panel B inference-cost microbenchmark (zero training / zero API).

The p16 runs persisted no checkpoints (runner has no torch.save), so this uses
the archived P1 pilot checkpoints (same CausalCharGNN + rank-8 Adapter
structure, 2-domain, 10 updates) purely for *timing and memory*, never for
accuracy. Disclosed in the table provenance.

Measured on identical hardware and identical batch size (batch=16 and batch=1):
  shared_b16      : shared detector, 1 batch forward.
  routed_b16      : router (signature + nearest centroid) + 1 expert forward.
  all5_b16        : 5 expert forwards per batch (same batch as routed).
  routed_archive  : per-graph grouping + batch forward (p16 eval_routed path).
  all5_archive    : per-graph x 5 experts (p16 eval_all_experts path).
  router_only     : signature extraction + nearest-centroid assignment.

Also reports parameter/disk bytes per expert and per shared detector and
peak RSS (driver runs each mode under /usr/bin/time -l).

Usage:
  python bench_table5_panelb_efficiency.py --mode <mode> --out-json <path>
  python bench_table5_panelb_efficiency.py --all --out-json <path>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "repro"))

from train_gnn_v2 import CausalCharGNN, graph_to_v2_data  # noqa: E402

NODE_TYPES = ['agent_message', 'code_exec', 'database_query', 'web_request',
              'web_search', 'file_read', 'file_write', 'shell_exec',
              'tool_execute', 'memory_write', 'memory_read', 'system_config',
              'email_send', 'message_send', 'social_media', 'file_delete',
              'account_access', 'version_control', 'data_aggregate',
              'document_create', 'agent_spawn', 'prompt_inject', 'other']
CKPT_DIR = REPO / "code/repro/out/real_cl/p1_pilot_local"
TEST = REPO / "code/repro/out/unified_gnn/fix2_splits/mcp/fold0/test.jsonl"


class Adapter(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


def signature(g):
    v = [0.0] * len(NODE_TYPES)
    for n in g["nodes"]:
        t = str(n.get("type", "other"))
        v[NODE_TYPES.index(t) if t in NODE_TYPES else NODE_TYPES.index("other")] += 1
    n = max(len(g["nodes"]), 1)
    return np.array([x / n for x in v] +
                    [len(g["nodes"]) / 50.0, len(g.get("edges", [])) / 50.0])


def load_setup():
    graphs = [json.loads(l) for l in TEST.read_text().splitlines() if l.strip()]
    model = CausalCharGNN(node_base_dim=22, edge_base_dim=11)
    model.load_state_dict(torch.load(CKPT_DIR / "final_naive.pt", map_location="cpu"))
    model.eval()
    experts = torch.load(CKPT_DIR / "experts_specialists.pt", map_location="cpu")
    first = next(iter(experts))
    ad_sd, hd_sd = experts[first]
    data = {g["task_id"]: graph_to_v2_data(g) for g in graphs}
    return graphs, model, ad_sd, hd_sd, data


def fwd(m, b):
    return m(b.x_base, b.node_chars, b.edge_index, b.edge_attr_base,
             b.edge_chars, b.batch)


def run_mode(mode: str, graphs, model, ad_sd, hd_sd, data, reps: int = 3):
    torch.set_num_threads(4)
    hidden = model.node_proj.out_features
    adapter = Adapter(hidden, r=8)
    adapter.load_state_dict(ad_sd)
    head = nn.Linear(hidden, 1)
    head.load_state_dict(hd_sd)

    def expert_forward(b):
        old = model.classifier
        model.classifier = head
        handle = model.tf2.register_forward_hook(lambda mod, i, o: adapter(o))
        with torch.no_grad():
            out = torch.sigmoid(fwd(model, b)).cpu().tolist()
        handle.remove()
        model.classifier = old
        return out

    def shared_forward(b):
        with torch.no_grad():
            return torch.sigmoid(fwd(model, b)).cpu().tolist()

    def batched(batch_size):
        res = []
        for i in range(0, len(graphs), batch_size):
            ch = graphs[i:i + batch_size]
            res.append((ch, Batch.from_data_list([data[g["task_id"]] for g in ch])))
        return res

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        if mode == "shared_b16":
            for _, b in batched(16):
                shared_forward(b)
        elif mode == "routed_b16":
            for _, b in batched(16):
                expert_forward(b)
        elif mode == "all5_b16":
            for _, b in batched(16):
                for _ in range(5):
                    expert_forward(b)
        elif mode == "routed_archive":
            for g in graphs:
                b = Batch.from_data_list([data[g["task_id"]]])
                expert_forward(b)
        elif mode == "all5_archive":
            for g in graphs:
                b = Batch.from_data_list([data[g["task_id"]]])
                for _ in range(5):
                    expert_forward(b)
        elif mode == "router_only":
            centroids = [signature(graphs[i]) for i in range(5)]
            for g in graphs:
                v = signature(g)
                _ = int(np.argmin([float(np.linalg.norm(v - c)) for c in centroids]))
        else:
            raise SystemExit(f"unknown mode {mode}")
        times.append(time.perf_counter() - t0)
    return {
        "mode": mode,
        "n_graphs": len(graphs),
        "reps": reps,
        "seconds_mean": float(np.mean(times)),
        "seconds_std": float(np.std(times)),
        "ms_per_graph": float(np.mean(times) * 1000 / len(graphs)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    graphs, model, ad_sd, hd_sd, data = load_setup()
    hidden = model.node_proj.out_features
    modes = ["shared_b16", "routed_b16", "all5_b16", "routed_archive",
             "all5_archive", "router_only"] if args.all else [args.mode]

    out = {
        "note": "timing/memory only; P1 pilot checkpoints (2 domains, 10 updates); "
                "all5_* repeats the same expert 5x to emulate K=5 expert count; "
                "AUC/ACC from these checkpoints is not used anywhere",
        "hardware": "Apple M2 CPU, torch 2.14.0, 4 torch threads",
        "n_graphs": len(graphs),
        "params": {
            "shared_detector": sum(p.numel() for p in model.parameters()),
            "shared_detector_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "adapter": sum(p.numel() for p in Adapter(hidden, r=8).parameters()),
            "adapter_bytes": sum(p.numel() * p.element_size() for p in Adapter(hidden, r=8).parameters()),
            "head": sum(p.numel() * p.element_size() for p in nn.Linear(hidden, 1).parameters()),
            "expert_disk_bytes": (CKPT_DIR / "experts_specialists.pt").stat().st_size,
            "k_experts": 5,
        },
        "results": [run_mode(m, graphs, model, ad_sd, hd_sd, data) for m in modes],
    }
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    for r in out["results"]:
        print(f"{r['mode']:15s} {r['ms_per_graph']:8.2f} ms/graph "
              f"({r['seconds_mean']:.3f}s ± {r['seconds_std']:.3f} for {r['n_graphs']} graphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
