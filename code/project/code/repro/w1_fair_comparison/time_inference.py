#!/usr/bin/env python3
"""W1 cost audit: params, checkpoint size, per-decision CPU latency (batch=1)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "LOGS/w1_fair_comparison"
TEST = OUT / "data/obs1/test.jsonl"
_FT = Path(__file__).resolve().parents[1] / "final_five_tables"
sys.path.insert(0, str(_FT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

torch.set_num_threads(4)
import train_grad_repair as g  # noqa: E402
import train_gnn_variant_repair as v  # noqa: E402
from run_temporal import TemporalCharNet  # noqa: E402


def measure(model, batch_fn, batches) -> dict:
    model.eval()
    with torch.no_grad():
        for b in batches[:5]:
            batch_fn(model, b)
        times = []
        for b in batches:
            t0 = time.perf_counter()
            batch_fn(model, b)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    return {"n": n, "median_ms": times[n // 2], "p95_ms": times[int(n * 0.95)],
            "mean_ms": sum(times) / n}


def batches_variant(arm: str, ckpt: Path):
    v.VARIANT = "text_only"
    v.MAX_NODE_CHARS = int(ckpt["config"].get("max_node_chars", 512))
    ds = v.V2GraphDataset(str(TEST))
    loader = v.DataLoader(ds, batch_size=1)
    if arm == "TEMP":
        model = TemporalCharNet(1, len(v.V2_EDGE_TYPES) + 1,
                                hidden_dim=ckpt["config"].get("hidden_dim", 192))
    else:
        model = v.CausalCharGNN(1, len(v.V2_EDGE_TYPES) + 1,
                                hidden_dim=ckpt["config"].get("hidden_dim", 192))
    model.load_state_dict(ckpt["model_state"])
    model.cpu()
    return model, v.batch_logits, list(loader)


def batches_grad(ckpt: Path):
    cfg = {"candidate": "A_moddrop", "edge_mode": "rebuilt", "edge_dropout": 0.0,
           "edge_text": "copy", "max_node_chars": 512, "max_edge_chars": 160}
    ds = g.CData(str(TEST), cfg, None)
    loader = g.DataLoader(ds, batch_size=1)
    model = g.ModalityNormGNN(len(g.PERMISSION) + 1 + len(g.CONTENT_KEYWORDS),
                              len(g.V2_EDGE_TYPES) + 1, 192,
                              use_ln=True, use_aux=True, direct_text_add=False)
    model.load_state_dict(ckpt["model_state"])
    model.cpu()
    return model, (lambda m, b: g.batch_logits(m, b, cfg)), list(loader)


def main() -> int:
    report = {}
    for arm, seed in (("GRAD", 211), ("TXT", 211), ("TEMP", 211)):
        d = OUT / "runs" / f"{arm}_s{seed}"
        ck = sorted(d.glob("model.pt")) or sorted(d.glob("gnn_ablate_*.pt"))
        if not ck:
            print(f"{arm}: no checkpoint yet")
            continue
        ckp = ck[-1]
        state = torch.load(ckp, map_location="cpu", weights_only=False)
        if arm == "GRAD":
            model, fn, batches = batches_grad(state)
        else:
            model, fn, batches = batches_variant(arm, state)
        params = sum(p.numel() for p in model.parameters())
        res = measure(model, fn, batches)
        res.update({"params": int(params), "checkpoint_bytes": ckp.stat().st_size,
                    "checkpoint": str(ckp.relative_to(REPO))})
        report[arm] = res
        print(arm, json.dumps(res), flush=True)
    (OUT / "cost.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
