#!/usr/bin/env python3
"""Flatten per-example score jsonl files into one CSV (reporting only)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def R(p):
        p = Path(p)
        return p if p.is_absolute() else REPO / p

    scores_dir, meta_path, out = R(args.scores_dir), R(args.meta), R(args.out)
    gmap = {}
    for mp in str(meta_path).split(","):
        for line in open(R(mp)):
            r = json.loads(line)
            gmap[r["task_id"]] = str(r["group"])
    rows = []
    for p in sorted(scores_dir.glob("*.jsonl")):
        parts = p.stem.split("_")
        if len(parts) == 3:
            arm, split, seed = parts[0], parts[1], parts[2].lstrip("s")
        else:
            arm, split, seed = parts[0], "confirm2", parts[1].lstrip("s")
        for r in (json.loads(l) for l in open(p) if l.strip()):
            rows.append({"arm": arm, "split": split, "seed": seed,
                         "task_id": r["task_id"], "group": gmap.get(r["task_id"], ""),
                         "label": int(r["label"]), "score": float(r["score"])})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "split", "seed", "task_id",
                                          "group", "label", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
