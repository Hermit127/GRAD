#!/usr/bin/env python3
"""Train one E/Z/R edge arm through the FROZEN W1 GRAD trainer.

Byte-identical training protocol to the frozen W1 runs (`train_det_wrapper.py`
+ `train_grad_repair.py --candidate A_moddrop --lambda-aux 0.1 --moddrop 0.25
--max-node-chars 512 --max-edge-chars 160 --epochs 60 --patience 15`); the only
difference is the edge transform selected by --edge-arm (E = no-op).

Usage (server):
  python code/repro/core_review_resolution/train_edge_arm.py --edge-arm Z <rest...>
"""
from __future__ import annotations

import argparse
import hashlib
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
_FT = REPO / "code/repro/final_five_tables"
sys.path.insert(0, str(_FT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import edge_arms  # noqa: E402
import train_grad_repair as t  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--edge-arm", choices=["E", "Z", "R"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args, rest = ap.parse_known_args()

    os.environ[edge_arms.ARM_ENV] = args.edge_arm
    arm = edge_arms.install_patch(t)

    sys.argv = [sys.argv[0], *rest, "--out-dir", args.out_dir, "--seed", str(args.seed)]
    rc = t.main()

    out_dir = Path(args.out_dir)
    sidecar = {
        "edge_arm": arm,
        "rewire_salt": edge_arms.REWIRE_SALT,
        "rewire_rule": "per-type target permutation; sources and target multiset kept",
        "seed": args.seed,
        "wrapper_sha256": sha256_file(Path(__file__)),
        "edge_arms_sha256": sha256_file(Path(edge_arms.__file__)),
        "frozen_trainer_sha256": sha256_file(_FT / "train_grad_repair.py"),
        "argv": sys.argv,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crr_edge_arm.json").write_text(json.dumps(sidecar, indent=1))
    print(f"edge arm {arm}: rc={rc} sidecar -> {out_dir / 'crr_edge_arm.json'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
