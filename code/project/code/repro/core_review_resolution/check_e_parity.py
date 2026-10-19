#!/usr/bin/env python3
"""E-arm parity check: wrapper-retrained E vs the frozen W1 GRAD checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", default="LOGS/core_review_resolution/out/b_train")
    ap.add_argument("--frozen-root", default="LOGS/w1_fair_comparison/runs_srv")
    ap.add_argument("--seeds", default="211,212,213")
    args = ap.parse_args()
    out = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        new_ck = REPO / args.new_root / f"E_s{seed}/model.pt"
        old_ck = REPO / args.frozen_root / f"GRAD_s{seed}/model.pt"
        a = torch.load(new_ck, map_location="cpu", weights_only=False)["model_state"]
        b = torch.load(old_ck, map_location="cpu", weights_only=False)["model_state"]
        assert set(a) == set(b), "state dict keys differ"
        maxdiff = max(float((a[k] - b[k]).abs().max()) for k in a)
        neq = sum(int((a[k] != b[k]).sum()) for k in a)
        new_js = sorted((REPO / args.new_root / f"E_s{seed}").glob("phase2_*.json"))[-1]
        old_js = sorted((REPO / args.frozen_root / f"GRAD_s{seed}").glob("phase2_*.json"))[-1]
        rn = json.loads(new_js.read_text())
        ro = json.loads(old_js.read_text())
        sd = {}
        for split in ("val", "test"):
            sn = [r["score"] for r in rn["scores"][split]]
            so = [r["score"] for r in ro["scores"][split]]
            d = max(abs(x - y) for x, y in zip(sn, so))
            sd[split] = {"max_abs_diff": d, "n": len(sn),
                         "auc_new": rn[split]["auroc"], "auc_old": ro[split]["auroc"]}
        out[str(seed)] = {"weight_max_abs_diff": maxdiff,
                          "weight_unequal_params": neq, "scores": sd}
    print(json.dumps(out, indent=1))
    ok = all(v["weight_max_abs_diff"] == 0 for v in out.values()) and all(
        x["max_abs_diff"] == 0 for v in out.values() for x in v["scores"].values())
    print("E_PARITY_BITWISE" if ok else "E_PARITY_MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
