"""Materialize fix2 group-splits into jsonl subsets for the unified CausalCharGNN trainer.

Reads:  code/repro/out/static_big_table_v1/fix2_20260910/split_ids/<bench>.json
Writes: code/repro/out/unified_gnn/fix2_splits/<bench>/{train,dev,test}.jsonl + manifest.json

Role map (fix2 protocol preserved):
  train        -> train
  dev          -> dev
  test         -> test
  gate_fit     -> excluded (fix2 used it to fit the removed gate; documented in manifest)
  threshold_cal-> excluded (fix2 threshold calibration only; GNN selects on dev)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT_DIR = ROOT / "code" / "repro" / "out" / "static_big_table_v1" / "fix2_20260910" / "split_ids"
OUT_ROOT = ROOT / "code" / "repro" / "out" / "unified_gnn" / "fix2_splits"
MINJA_DIR = ROOT / "code" / "repro" / "static_big_table_v1"
KEEP = {"train": "train", "dev": "dev", "test": "test"}


def load_source(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_minja_rows() -> list[dict]:
    sys.path.insert(0, str(MINJA_DIR))
    import minja_merged  # noqa: E402

    rows = minja_merged.load_merged()
    for i, row in enumerate(rows):
        if row["_source_index"] != i:
            raise RuntimeError(f"minja merged index mismatch at {i}")
    return rows


def materialize_fold(bench: str, fold: int, spec: dict, group_role: dict) -> dict:
    by_role: dict[str, list[dict]] = {r: [] for r in KEEP.values()}
    excluded: dict[str, int] = {}
    for s in spec["samples"]:
        role = group_role[s["group"]]
        mapped = KEEP.get(role)
        if mapped is None:
            excluded[role] = excluded.get(role, 0) + 1
            continue
        by_role[mapped].append(s)

    out_dir = OUT_ROOT / bench / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[dict]] = {}
    merged = load_minja_rows() if bench == "minja" else None
    manifest_roles: dict[str, dict] = {}
    for role, samples in by_role.items():
        recs = []
        for s in samples:
            if merged is not None:
                rec = dict(merged[s["source_index"]])
            else:
                src = s["source_file"]
                if src not in cache:
                    cache[src] = load_source(str(ROOT / src))
                rec = dict(cache[src][s["source_index"]])
            rec["sample_id"] = s["sample_id"]
            rec["group"] = s["group"]
            recs.append(rec)
        path = out_dir / f"{role}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pos = sum(1 for r in recs if int(r.get("label", 0)) == 1)
        manifest_roles[role] = {"samples": len(recs), "pos": pos, "neg": len(recs) - pos,
                                "groups": len({r["group"] for r in recs})}

    sha = hashlib.sha256(open(SPLIT_DIR / f"{bench}.json", "rb").read()).hexdigest()
    manifest = {
        "benchmark": bench,
        "fold": fold,
        "split_source": str(SPLIT_DIR / f"{bench}.json"),
        "split_sha256": sha,
        "roles": manifest_roles,
        "excluded": excluded,
        "note": "gate_fit/threshold_cal excluded; fix2 gate removed under unified CausalCharGNN protocol",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def materialize(bench: str) -> list[dict]:
    spec = json.load(open(SPLIT_DIR / f"{bench}.json", encoding="utf-8"))
    if "group_role" in spec:
        return [materialize_fold(bench, 0, spec, spec["group_role"])]
    return [materialize_fold(bench, f["fold"], spec, f["group_role"]) for f in spec["folds"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="all", help="decomp|mg|minja|mcp|all")
    args = ap.parse_args()
    benches = ["decomp", "mg", "minja", "mcp"] if args.bench == "all" else [args.bench]
    for bench in benches:
        for m in materialize(bench):
            print(bench, f"fold{m['fold']}", json.dumps(m["roles"]), "excluded=", m["excluded"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
