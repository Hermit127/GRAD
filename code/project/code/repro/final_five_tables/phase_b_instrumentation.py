#!/usr/bin/env python3
"""Phase B instrumentation: keyword channel norms, gradient mass, sensitivity.

Trains the frozen `full` model for a short diagnostic budget (default 15 epochs;
NOT the frozen selection rule) and records, per epoch:
  - pre-fusion input norms of the structure block (dims 0:13), keyword block
    (13:22) and char-text block (22:86);
  - gradient norms of node_proj.weight rows for each block;
  - dev AUROC with (a) normal input, (b) keyword block zeroed,
    (c) keyword block permuted across graphs (same marginal, association broken).

Dev only (no test file is read). Sensitivity probes are diagnostics and are never
used as an ablation substitute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))

import train_gnn_ablate as abl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abl.set_seed(args.seed)
    train_ds = abl.V2GraphDataset(args.train)
    val_ds = abl.V2GraphDataset(args.val)
    train_loader = abl.DataLoader(train_ds, batch_size=48, shuffle=True)
    val_loader = abl.DataLoader(val_ds, batch_size=48)
    pos = sum(1 for g in train_ds.graphs if int(g.get("label", 0)) == 1)
    neg = train_ds.len() - pos
    pos_weight = neg / max(pos, 1)

    model = abl.CausalCharGNN(22, len(abl.V2_EDGE_TYPES) + 1, hidden_dim=192).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    opt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=10)

    norms = {}
    grads = {}

    def fwd_hook(module, inputs):
        x = inputs[0]
        norms.setdefault("struct", []).append(float(x[:, :13].norm(dim=-1).mean()))
        norms.setdefault("keyword", []).append(float(x[:, 13:22].norm(dim=-1).mean()))
        norms.setdefault("char", []).append(float(x[:, 22:].norm(dim=-1).mean()))

    def grad_hook(module, grad_in, grad_out):
        g = grad_out[0]
        grads.setdefault("struct", []).append(float(g[:, :13].norm(dim=-1).mean()))
        grads.setdefault("keyword", []).append(float(g[:, 13:22].norm(dim=-1).mean()))
        grads.setdefault("char", []).append(float(g[:, 22:].norm(dim=-1).mean()))

    h1 = model.node_proj.register_forward_pre_hook(fwd_hook)
    h2 = model.node_proj.register_full_backward_hook(grad_hook)

    def clean_eval(mask_mode="normal"):
        """Evaluate with an input-level mask applied per graph (no batching changes)."""
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev)
                if mask_mode == "zero":
                    batch.x_base[:, 13:22] = 0.0
                elif mask_mode == "permute":
                    rng = np.random.default_rng(args.seed)
                    b = batch.batch
                    ng = int(b.max().item()) + 1
                    perm = torch.tensor(rng.permutation(ng), device=batch.x_base.device)
                    mask_rows = perm[b]
                    batch.x_base[:, 13:22] = batch.x_base[:, 13:22][mask_rows]
                logits = abl.batch_logits(model, batch)
                probs.extend(torch.sigmoid(logits).cpu().tolist())
        labels = [int(g.get("label", 0)) for g in val_ds.graphs]
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")

    history = []
    best_state, best_auroc, best_epoch = None, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            logits = abl.batch_logits(model, batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch.y, pos_weight=torch.tensor(pos_weight, device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        auroc = clean_eval("normal")
        opt.step(1 - auroc)
        if auroc > best_auroc:
            best_auroc, best_epoch = auroc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc": auroc,
               "norm_struct": float(np.mean(norms.get("struct", [0]))),
               "norm_keyword": float(np.mean(norms.get("keyword", [0]))),
               "norm_char": float(np.mean(norms.get("char", [0]))),
               "grad_struct": float(np.mean(grads.get("struct", [0]))),
               "grad_keyword": float(np.mean(grads.get("keyword", [0]))),
               "grad_char": float(np.mean(grads.get("char", [0])))}
        history.append(row)
        norms.clear(); grads.clear()
        print(json.dumps(row), flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    result = {
        "config": vars(args),
        "diagnostic_only": True,
        "selection_rule": "best dev AUROC over the diagnostic budget (NOT the frozen rule)",
        "history": history,
        "sensitivity_dev_auroc": {
            "normal": clean_eval("normal"),
            "keyword_zeroed": clean_eval("zero"),
            "keyword_permuted": clean_eval("permute"),
        },
        "best_epoch": best_epoch,
        "note": "test file not read; masking/permutation are sensitivity probes, not ablations",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instrumentation.json").write_text(json.dumps(result, indent=1))
    print("saved", out / "instrumentation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
