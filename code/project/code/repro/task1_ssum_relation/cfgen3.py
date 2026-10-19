#!/usr/bin/env python3
"""DISC-2026W39-002 §3: new-family isolated scenario corpus for the controlled pairing.

Frozen rules: pair construction and labels come from `cfgen2.make_pair_v2` unchanged
(monkeypatched family table only); `cfgen.check_pair` invariants must pass for every
pair. New family indices are offset by 100 so tokens/pair_ids never collide with the
v1/v2/online corpora. Train split is a byte-copy of the frozen v2 train file.

dev families : 100-103  (maritime, wildfire, waterquality, metro)
test families: 104-111  (the remaining 8; do not generate before the dev-based
               size decision is recorded; see prereg_controlled_20260923.md §3)

Usage:
  python cfgen3.py gen --out OUT_DIR --split dev --train-from pairs_v2 --instances 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cfgen2  # noqa: E402
from cfgen import CONFIRMATION_FAMILIES, FAMILIES, check_pair  # noqa: E402

FAMILY_OFFSET = 100
NEW_FAMILIES = [
    ("maritime", "web_request", "shell_exec", "reroute vessel", "voyage"),
    ("wildfire", "tool_return", "message_send", "issue evacuation", "zone"),
    ("waterquality", "web_request", "file_write", "post advisory", "sample"),
    ("metro", "tool_return", "system_config", "adjust headway", "service"),
    ("postal", "web_request", "document_create", "release hold", "parcel"),
    ("veterinary", "tool_return", "message_send", "approve treatment", "animal"),
    ("court", "web_request", "document_create", "file motion", "docket"),
    ("waste", "tool_return", "shell_exec", "schedule haul", "route"),
    ("desalination", "web_request", "file_write", "set salinity", "plant"),
    ("datacenter", "tool_return", "system_config", "throttle rack", "cluster"),
    ("laboratory", "web_request", "document_create", "release result", "specimen"),
    ("seedbank", "tool_return", "file_write", "log accession", "accession"),
]
DEV_FAMILIES = list(range(FAMILY_OFFSET, FAMILY_OFFSET + 4))
TEST_FAMILIES = list(range(FAMILY_OFFSET + 4, FAMILY_OFFSET + len(NEW_FAMILIES)))
KINDS = ["direct", "read_hop", "chained", "authorized"]


def _install_family_map() -> None:
    old_domains = {f[0] for f in list(FAMILIES) + list(CONFIRMATION_FAMILIES)
                   + list(cfgen2.ONLINE_FAMILIES)}
    overlap = old_domains & {f[0] for f in NEW_FAMILIES}
    assert not overlap, f"new family domains overlap existing: {overlap}"

    def _family(idx):
        assert FAMILY_OFFSET <= idx < FAMILY_OFFSET + len(NEW_FAMILIES), idx
        return NEW_FAMILIES[idx - FAMILY_OFFSET]

    cfgen2._family = _family


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gen(out_dir: Path, split: str, instances: int, gen_seed: int, train_from: Path | None) -> dict:
    _install_family_map()
    families = DEV_FAMILIES if split == "dev" else TEST_FAMILIES
    rows = []
    kind_counts: dict[str, dict[str, int]] = {}
    for fam in families:
        for i in range(instances):
            kind = KINDS[i % len(KINDS)]
            harm, safe = cfgen2.make_pair_v2(fam, i, kind, gen_seed)
            assert check_pair(harm, safe), (fam, i, kind)
            rows.extend([harm, safe])
            kc = kind_counts.setdefault(kind, {"harm": 0, "safe": 0})
            kc["harm"] += int(harm["label"] == 1)
            kc["safe"] += int(safe["label"] == 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pairs_{split}.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    manifest = {
        "generator": "cfgen3.py (cfgen2.make_pair_v2 frozen; new family table)",
        "gen_seed": gen_seed,
        "split": split, "families": families, "instances_per_family": instances,
        "pairs": len(rows) // 2, "graphs": len(rows),
        "n_pos": sum(int(r["label"]) for r in rows),
        "kinds": kind_counts,
        "path": str(path),
        "sha256": _sha256(path),
    }
    if train_from is not None:
        src = train_from / "pairs_train.jsonl"
        if src.exists():
            dst = out_dir / "pairs_train.jsonl"
            if not dst.exists():
                shutil.copyfile(src, dst)
            manifest["train_from"] = {"src": str(src), "sha256": _sha256(dst)}
        man = train_from / "pairs_manifest.json"
        if man.exists():
            manifest["train_source_manifest"] = json.loads(man.read_text()).get("seed")
    # maintain combined manifest expected by train_eval.py (pairs_manifest.json)
    combined_path = out_dir / "pairs_manifest.json"
    combined = json.loads(combined_path.read_text()) if combined_path.exists() else {"seed": gen_seed, "splits": {}}
    combined.setdefault("splits", {})[split] = manifest
    combined["seed"] = gen_seed
    combined_path.write_text(json.dumps(combined, indent=1, sort_keys=True))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--split", choices=["dev", "test"], required=True)
    g.add_argument("--instances", type=int, default=12)
    g.add_argument("--gen-seed", type=int, default=211)
    g.add_argument("--train-from", default="code/repro/out/task1_ssum_relation/pairs_v2")
    args = ap.parse_args()
    out_dir = Path(args.out)
    train_from = Path(args.train_from) if args.train_from else None
    manifest = gen(out_dir, args.split, args.instances, args.gen_seed, train_from)
    mpath = out_dir / f"pairs_manifest_{args.split}.json"
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
