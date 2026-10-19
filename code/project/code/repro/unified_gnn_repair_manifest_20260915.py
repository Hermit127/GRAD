"""Verify (and optionally fix) unified sweep run filenames against internal config.

Identity is authoritative from each result JSON's config:
  fix2:  .../fix2_splits/<bench>/fold<k>/train.jsonl          -> <bench>_f<k>_s<seed>.json
  mrt:   .../results/mrt_cross_task_v1/fold_<task>/train.jsonl -> mrt_<task>_s<seed>.json

Usage:
  python code/repro/unified_gnn_repair_manifest_20260915.py --src <runs_dir>          # check only
  python code/repro/unified_gnn_repair_manifest_20260915.py --src <runs_dir> --apply  # rename mismatches
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def identity_from_config(cfg: dict) -> tuple[str, str, int] | None:
    train = str(cfg.get("train", ""))
    seed = int(cfg.get("seed", -1))
    parts = Path(train).parts
    if "fix2_splits" in parts:
        i = parts.index("fix2_splits")
        bench = parts[i + 1]
        fold = parts[i + 2]  # e.g. fold0
        if not fold.startswith("fold"):
            return None
        return bench, fold[len("fold"):], seed
    if "mrt_cross_task_v1" in parts:
        i = parts.index("mrt_cross_task_v1")
        fold = parts[i + 1]  # e.g. fold_american_airlines_2
        if not fold.startswith("fold_"):
            return None
        return "mrt", fold[len("fold_"):], seed
    return None


def expected_name(ident: tuple[str, str, int]) -> str:
    bench, fold, seed = ident
    if bench == "mrt":
        return f"mrt_{fold}_s{seed}.json"
    return f"{bench}_f{fold}_s{seed}.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    rows, mismatches, duplicates = [], [], {}
    seen: dict[tuple[str, str, int], str] = {}
    for path in sorted(src.glob("*.json")):
        r = json.loads(path.read_text())
        ident = identity_from_config(r.get("config", {}))
        if ident is None:
            rows.append((path.name, "UNPARSED", ""))
            continue
        exp = expected_name(ident)
        rows.append((path.name, exp, ident))
        if exp != path.name:
            mismatches.append((path.name, exp))
        if ident in seen:
            duplicates.setdefault(ident, [seen[ident]]).append(path.name)
        seen[ident] = path.name

    print(f"runs={len(rows)} mismatches={len(mismatches)} duplicates={len(duplicates)}")
    for old, exp in mismatches:
        print(f"  MISMATCH {old} -> {exp}")
    for ident, names in duplicates.items():
        print(f"  DUP {ident}: {names}")

    if args.apply and mismatches:
        for old, exp in mismatches:
            src.joinpath(old).rename(src / f".tmprepair_{exp}")
        for old, exp in mismatches:
            src.joinpath(f".tmprepair_{exp}").rename(src / exp)
        print(f"renamed {len(mismatches)} files; residual check:")
        return main_quiet(src)
    return 0


def main_quiet(src: Path) -> int:
    bad = 0
    for path in sorted(src.glob("*.json")):
        r = json.loads(path.read_text())
        ident = identity_from_config(r.get("config", {}))
        if ident is None or expected_name(ident) != path.name:
            bad += 1
            print("  still bad:", path.name)
    print(f"post-check mismatches={bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
