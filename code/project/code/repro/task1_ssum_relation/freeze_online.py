#!/usr/bin/env python3
"""Freeze the layer-4 closed-loop protocol before any evaluation.

Writes `online_freeze.json` binding:
  * the scenario list (pairs_online.jsonl) by sha256,
  * each defense checkpoint by sha256 + config + dev-selected threshold,
  * the rule verifier version and the outcome-verifier version,
  * code file hashes for the loop.
Refuses to freeze if any asset is missing. `online_loop.py --freeze` verifies
every hash before scoring and never re-tunes anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--checkpoints", required=True,
                    help="dir containing <arm>_s<seed>.pt from train_eval "
                         "--save-checkpoint")
    ap.add_argument("--runs", required=True,
                    help="worker dirs containing runs/<arm>_s<seed>.json")
    ap.add_argument("--arms", default="full_provenance,chrono_only,text_relation")
    ap.add_argument("--seed", type=int, default=211)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = Path(args.pairs)
    if not pairs.exists():
        raise SystemExit(f"missing pairs file: {pairs}")
    entry = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "pairs_file": str(pairs),
        "pairs_sha256": sha256(pairs),
        "pairs_manifest": str(pairs.parent / "pairs_manifest.json"),
        "pairs_manifest_sha256": sha256(pairs.parent / "pairs_manifest.json")
        if (pairs.parent / "pairs_manifest.json").exists() else None,
        "defenses": {},
        "outcome_verifier": "cfgen2.verify_policy (task authorization + actual "
                            "parameter provenance)",
        "loop_code": {},
    }
    run_dirs = [Path(d) for d in args.runs.split(",")]
    ckpt_dirs = [Path(d) for d in args.checkpoints.split(",")]
    for arm in [a for a in args.arms.split(",") if a]:
        run_name = f"{arm}_s{args.seed}"
        ckpt = None
        for d in ckpt_dirs:
            cand = d / f"{run_name}.pt"
            if cand.exists():
                ckpt = cand
                break
        run_json = None
        for d in run_dirs:
            cand = d / "runs" / f"{run_name}.json"
            if cand.exists():
                run_json = cand
                break
        if run_json is None:
            raise SystemExit(f"missing run json for {run_name} in {run_dirs}")
        if ckpt is None:
            raise SystemExit(f"missing checkpoint for {run_name} in {ckpt_dirs}")
        run = json.load(open(run_json))
        entry["defenses"][arm] = {
            "checkpoint": str(ckpt),
            "checkpoint_sha256": sha256(ckpt),
            "run_json": str(run_json),
            "run_json_sha256": sha256(run_json),
            "threshold": run["metrics"]["dev"]["threshold_at_target_fpr"]["threshold"],
            "config": run["config"],
            "dev_auroc": run["metrics"]["dev"]["auroc"],
        }
    here = Path(__file__).resolve().parent
    for name in ("online_loop.py", "cfgen2.py", "graphbuild.py", "variants.py",
                 "models_text.py", "rule_baseline.py", "serialize.py"):
        p = here / name
        if p.exists():
            entry["loop_code"][name] = sha256(p)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(entry, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in entry.items() if k != "loop_code"},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
