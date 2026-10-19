#!/usr/bin/env python3
"""Aggregate per-seed layer-4 closed-loop results (mean±sd over seeds)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(Path().glob(args.glob)) if "*" in args.glob else [Path(args.glob)]
    per_arm = defaultdict(list)
    seeds = []
    for f in files:
        d = json.load(open(f))
        seeds.append(f.stem)
        for arm, m in d["arms"].items():
            per_arm[arm].append(m)
    summary = {"seeds": seeds, "arms": {}}
    lines = ["# Layer-4 closed loop (mean±sd over frozen seeds)", "",
             "| arm | ASR↓ | UA↑ | BU↑ | false-block↓ | lat p50 (ms) |",
             "|---|---|---|---|---|---|"]
    for arm, ms in per_arm.items():
        def msd(key):
            vals = [m[key] for m in ms]
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        summary["arms"][arm] = {
            "n_seeds": len(ms),
            "asr_mean": float(np.mean([m["asr"] for m in ms])),
            "asr_sd": float(np.std([m["asr"] for m in ms])),
            "ua_mean": float(np.mean([m["ua"] for m in ms])),
            "bu_mean": float(np.mean([m["bu"] for m in ms])),
            "false_block_mean": float(np.mean([m["false_block_rate"] for m in ms])),
            "latency_p50_mean": float(np.mean([m["latency_ms_p50"] for m in ms])),
            "latency_p95_mean": float(np.mean([m["latency_ms_p95"] for m in ms])),
        }
        lines.append(f"| {arm} | {msd('asr')} | {msd('ua')} | {msd('bu')} | "
                     f"{msd('false_block_rate')} | {msd('latency_ms_p50')} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
