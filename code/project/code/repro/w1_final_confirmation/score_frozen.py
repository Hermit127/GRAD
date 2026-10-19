#!/usr/bin/env python3
"""W5 one-pass scoring of frozen checkpoints on the confirmation corpus."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("W1_PATCH", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
_FT = REPO / "code/repro/final_five_tables"
sys.path.insert(0, str(_FT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "code/repro/w1_fair_comparison"))
import train_grad_repair as g  # noqa: E402
import train_gnn_variant_repair as v  # noqa: E402
from run_temporal import TemporalCharNet  # noqa: E402

torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_grad(data: Path, ckpt: Path) -> list[dict]:
    cfg = {"candidate": "A_moddrop", "edge_mode": "rebuilt", "edge_dropout": 0.0,
           "edge_text": "copy", "max_node_chars": 512, "max_edge_chars": 160}
    ds = g.CData(str(data), cfg, None)
    model = g.ModalityNormGNN(len(g.PERMISSION) + 1 + len(g.CONTENT_KEYWORDS),
                              len(g.V2_EDGE_TYPES) + 1, 192,
                              use_ln=True, use_aux=True, direct_text_add=False).to(DEVICE)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    loader = g.DataLoader(ds, batch_size=48)
    out = []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            logit = g.batch_logits(model, b, cfg)
            if isinstance(logit, dict):
                logit = logit["logit"]
            for p in torch.sigmoid(logit).cpu().tolist():
                out.append(float(p))
    assert len(out) == len(ds.graphs)
    return [{"task_id": ds.graphs[i]["task_id"], "label": int(ds.graphs[i]["label"]),
             "score": out[i]} for i in range(len(out))]


def score_variant(arm: str, data: Path, ckpt: Path) -> list[dict]:
    v.VARIANT = "text_only"
    v.MAX_NODE_CHARS = 512
    ds = v.V2GraphDataset(str(data))
    if arm == "TXT":
        model = v.CausalCharGNN(1, len(v.V2_EDGE_TYPES) + 1, hidden_dim=192).to(DEVICE)
    else:
        model = TemporalCharNet(1, len(v.V2_EDGE_TYPES) + 1, hidden_dim=192).to(DEVICE)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    loader = v.DataLoader(ds, batch_size=48)
    out = []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            out.extend(torch.sigmoid(v.batch_logits(model, b)).cpu().tolist())
    return [{"task_id": ds.graphs[i]["task_id"], "label": int(ds.graphs[i]["label"]),
             "score": float(out[i])} for i in range(len(out))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["GRAD", "TXT", "TEMP"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = (score_grad if args.arm == "GRAD" else
            (lambda d, c: score_variant(args.arm, d, c)))(Path(args.data), Path(args.ckpt))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.arm}: {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
