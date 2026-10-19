#!/usr/bin/env python3
"""Build the sealed group-isolated MCP split for Phase 2 confirmation.

Never used during candidate development. Groups follow static_big_table_v1's
group_mcp (metadata.task stem, _benign stripped). Deterministic 60/15/25 split
over groups with seed 20260918; pair-preserving (whole groups move together).
Writes train/dev/test JSONL + manifest (sha256, group counts) to
code/repro/out/table5_candidates/sealed_mcp/.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIX2 = REPO / "code/repro/out/unified_gnn/fix2_splits/mcp"
OUT = REPO / "code/repro/out/table5_candidates/sealed_mcp"
SEED = 20260918


def group_of(g: dict) -> str:
    md = g.get("metadata", {}) or {}
    task = str(md.get("task") or "")
    if task.endswith("_benign"):
        task = task[: -len("_benign")]
    return task or str(g.get("task_id"))


def main() -> int:
    by_id = {}
    for fold in sorted(FIX2.glob("fold*")):
        for split in ("train", "dev", "test"):
            for line in (fold / f"{split}.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                g = json.loads(line)
                by_id[g["task_id"]] = g
    groups = defaultdict(list)
    for g in by_id.values():
        groups[group_of(g)].append(g)
    names = sorted(groups)
    rng = random.Random(SEED)
    rng.shuffle(names)
    n = len(names)
    n_tr, n_dv = int(n * 0.60), int(n * 0.15)
    role = {}
    for name in names[:n_tr]:
        role[name] = "train"
    for name in names[n_tr:n_tr + n_dv]:
        role[name] = "dev"
    for name in names[n_tr + n_dv:]:
        role[name] = "test"
    out = {"train": [], "dev": [], "test": []}
    for name in names:
        out[role[name]].extend(groups[name])
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "n_groups": n, "roles": {}}
    for split, gs in out.items():
        p = OUT / f"{split}.jsonl"
        p.write_text("\n".join(json.dumps(g, sort_keys=True) for g in gs) + "\n")
        manifest["roles"][split] = {
            "n_graphs": len(gs), "n_pos": sum(int(g["label"]) for g in gs),
            "n_groups": len({group_of(g) for g in gs}),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    group_sets = {s: {group_of(g) for g in gs} for s, gs in out.items()}
    manifest["group_overlap"] = {
        "train_x_dev": len(group_sets["train"] & group_sets["dev"]),
        "train_x_test": len(group_sets["train"] & group_sets["test"]),
        "dev_x_test": len(group_sets["dev"] & group_sets["test"]),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
