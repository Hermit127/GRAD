#!/usr/bin/env python3
"""Score one E/Z/R GRAD checkpoint with the frozen W1 scoring path.

Wraps `code/repro/w1_final_confirmation/score_frozen.py::score_grad`, applying
the same edge transform used at training time (env `CRR_EDGE_ARM`). No frozen
file is modified.

Usage:
  python code/repro/core_review_resolution/score_edge_arm.py \
      --edge-arm Z --ckpt <model.pt> --data <split.jsonl> --out <rows.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))
sys.path.insert(0, str(REPO / "code/repro/w1_final_confirmation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import edge_arms  # noqa: E402
import score_frozen  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-arm", choices=["E", "Z", "R"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ[edge_arms.ARM_ENV] = args.edge_arm
    arm = edge_arms.install_patch(score_frozen.g)
    rows = score_frozen.score_grad(Path(args.data), Path(args.ckpt))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"edge arm {arm}: {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
